#!/usr/bin/env python3
"""
Full MAML training: adapt ALL parameters in the inner loop.

Comparison to ANIL (head-only adaptation). Same data, same hyperparams,
same architecture. The only difference: the inner loop updates trunk +
bottleneck + head, not just the head.

This is first-order MAML (FOMAML) since we don't compute second-order
gradients through the inner loop (same as the ANIL implementation).

Usage:
    cd ~/code/maml-dasg
    python train_full_maml.py \
        --data-dir processed_combined_flat \
        --db-path combined_openings.sqlite \
        --out-dir runs/full_maml_combined \
        --meta-iters 5000

    # Chess only:
    python train_full_maml.py \
        --data-dir processed_chess_flat \
        --db-path combined_openings.sqlite \
        --out-dir runs/full_maml_chess \
        --meta-iters 5000
"""

import argparse
import os
import signal
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.func import functional_call

from spec import UnifiedSpec, num_channels
from model_v2 import ValueNet
from task_sampler_v2 import ValueTaskSampler
from maml_anil import value_loss

ROOT = os.path.dirname(os.path.abspath(__file__))


def inner_adapt_full(model, base_params, Xs, ys_val, inner_lr, inner_steps):
    """Full MAML inner loop: adapt ALL parameters on the support set."""
    params = {k: v.clone() for k, v in base_params.items()}

    for step in range(inner_steps):
        params_grad = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
        v_pred = functional_call(model, params_grad, (Xs,))
        loss = value_loss(v_pred, ys_val)

        all_tensors = [params_grad[k] for k in sorted(params_grad.keys())]
        grads = torch.autograd.grad(loss, all_tensors, create_graph=False)

        grad_map = dict(zip(sorted(params_grad.keys()), grads))
        params = {k: (params_grad[k] - inner_lr * grad_map[k]).detach() for k in params_grad}

    return params


def save_checkpoint(filepath, model, optimizer, iteration, best_val, config, histories):
    torch.save({
        "iteration": iteration,
        "best_val_meta": best_val,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        **histories,
    }, filepath)


def main():
    parser = argparse.ArgumentParser(description="Full MAML (all-param inner loop)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--out-dir", default="./runs/full_maml")
    parser.add_argument("--meta-iters", type=int, default=5000)
    parser.add_argument("--meta-batch-size", type=int, default=128)
    parser.add_argument("--k-support", type=int, default=16)
    parser.add_argument("--k-query", type=int, default=16)
    parser.add_argument("--inner-lr", type=float, default=0.005)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--outer-lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--trunk-hidden", type=int, default=64)
    parser.add_argument("--bottleneck-dim", type=int, default=64)
    parser.add_argument("--value-hidden", type=int, default=64)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--val-tasks", type=int, default=128)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--max-hours", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--min-positions", type=int, default=32)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu")
    spec = UnifiedSpec()
    C = num_channels(spec)

    # Model
    model = ValueNet(
        in_channels=C,
        trunk_hidden=args.trunk_hidden,
        bottleneck_dim=args.bottleneck_dim,
        value_hidden=args.value_hidden,
    ).to(device)
    print(f"Model: {model.total_param_count():,} params (ALL adapted in inner loop)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.outer_lr, eps=1e-4)

    # Data
    sampler = ValueTaskSampler(
        data_dir=os.path.join(ROOT, args.data_dir),
        db_path=os.path.join(ROOT, args.db_path),
        task_mode="opening",
        train_frac=args.train_frac,
        seed=args.seed,
        min_positions_per_task=args.min_positions,
    )

    config = vars(args)
    config["n_channels"] = C
    config["total_params"] = model.total_param_count()
    config["adapted_params"] = model.total_param_count()  # ALL params adapted
    config["method"] = "full_maml"

    with open(os.path.join(args.out_dir, "config.txt"), "w") as f:
        for k, v in sorted(config.items()):
            f.write(f"{k}={v}\n")

    # Signal handling
    stop_requested = [False]
    def handle_signal(sig, frame):
        if stop_requested[0]:
            sys.exit(1)
        print("\nStopping after current iteration.")
        stop_requested[0] = True
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Resume
    start_iter = 0
    train_hist, val_hist, val_x = [], [], []
    best_val = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iter = ckpt["iteration"]
        best_val = ckpt["best_val_meta"]
        train_hist = ckpt.get("train_meta_history", [])
        val_hist = ckpt.get("val_meta_history", [])
        val_x = ckpt.get("val_meta_x", [])
        print(f"Resumed at iteration {start_iter}")

    histories = lambda: {
        "train_meta_history": train_hist,
        "val_meta_history": val_hist,
        "val_meta_x": val_x,
    }

    t0 = time.time()
    print(f"\nTraining: {args.meta_iters} iters, {args.meta_batch_size} tasks/iter")
    print(f"  inner: {args.inner_steps} steps @ lr={args.inner_lr} (ALL {model.total_param_count():,} params)")
    print()

    for it in range(start_iter + 1, args.meta_iters + 1):
        if stop_requested[0]:
            break
        if time.time() - t0 > args.max_hours * 3600:
            print(f"Reached max_hours={args.max_hours}")
            break

        iter_t0 = time.time()

        # Sample meta-batch
        tasks = sampler.sample_meta_batch(
            args.meta_batch_size, args.k_support, args.k_query, split="train"
        )

        # Meta-step
        optimizer.zero_grad(set_to_none=True)
        base_params = {n: p for n, p in model.named_parameters()}
        accum_grads = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        meta_loss_sum = 0.0

        for sX, sy_val, qX, qy_val, task_id in tasks:
            sX_t = torch.tensor(sX, dtype=torch.float32, device=device)
            sy_t = torch.tensor(sy_val, dtype=torch.float32, device=device)
            qX_t = torch.tensor(qX, dtype=torch.float32, device=device)
            qy_t = torch.tensor(qy_val, dtype=torch.float32, device=device)

            adapted = inner_adapt_full(
                model, base_params, sX_t, sy_t,
                args.inner_lr, args.inner_steps,
            )

            # Query loss with adapted params
            adapted_req = {k: v.detach().clone().requires_grad_(True) for k, v in adapted.items()}
            q_v = functional_call(model, adapted_req, (qX_t,))
            q_loss = value_loss(q_v, qy_t)

            q_grads = torch.autograd.grad(
                q_loss, tuple(adapted_req.values()),
                create_graph=False, allow_unused=True,
            )
            for (name, _), g in zip(adapted_req.items(), q_grads):
                if g is not None:
                    accum_grads[name] += g.detach()

            meta_loss_sum += q_loss.detach().item()

        # Average and apply
        n_tasks = len(tasks)
        for name, p in model.named_parameters():
            p.grad = accum_grads[name] / n_tasks

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

        optimizer.step()

        meta_loss = meta_loss_sum / n_tasks
        train_hist.append(meta_loss)
        iter_ms = (time.time() - iter_t0) * 1000
        elapsed = time.time() - t0
        print(f"[it {it:5d}] meta_loss={meta_loss:.4f} | {iter_ms:.0f}ms | {elapsed:.0f}s")

        # Validation
        if it % args.val_every == 0:
            val_tasks = sampler.sample_meta_batch(
                args.val_tasks, args.k_support, args.k_query, split="val"
            )
            val_loss_sum = 0.0
            val_base = {n: p.detach().clone() for n, p in model.named_parameters()}
            for sX, sy_val, qX, qy_val, task_id in val_tasks:
                sX_t = torch.tensor(sX, dtype=torch.float32, device=device)
                sy_t = torch.tensor(sy_val, dtype=torch.float32, device=device)
                qX_t = torch.tensor(qX, dtype=torch.float32, device=device)
                qy_t = torch.tensor(qy_val, dtype=torch.float32, device=device)

                adapted = inner_adapt_full(
                    model, val_base, sX_t, sy_t,
                    args.inner_lr, args.inner_steps,
                )
                with torch.no_grad():
                    q_v = functional_call(model, {k: v.detach() for k, v in adapted.items()}, (qX_t,))
                    val_loss_sum += value_loss(q_v, qy_t).item()

            val_loss = val_loss_sum / len(val_tasks)
            val_hist.append(val_loss)
            val_x.append(it)

            improved = " *" if val_loss < best_val else ""
            print(f"  [VAL] meta_loss={val_loss:.4f} (best={min(best_val, val_loss):.4f}){improved}")

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(os.path.join(args.out_dir, "best.pt"),
                                model, optimizer, it, best_val, config, histories())

            save_checkpoint(os.path.join(args.out_dir, "latest.pt"),
                            model, optimizer, it, best_val, config, histories())

        if it % args.ckpt_every == 0:
            save_checkpoint(os.path.join(args.out_dir, f"ckpt_it{it}.pt"),
                            model, optimizer, it, best_val, config, histories())

    # Final
    save_checkpoint(os.path.join(args.out_dir, "final.pt"),
                    model, optimizer, it if 'it' in dir() else 0, best_val, config, histories())

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_hist) + 1), train_hist, alpha=0.3, label="Train")
    if val_hist:
        plt.plot(val_x, val_hist, "o-", label="Val", markersize=3)
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.title("Full MAML (all params adapted)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "loss.png"), dpi=200)
    plt.close()

    print(f"\nDone. Best val: {best_val:.4f}")


if __name__ == "__main__":
    main()
