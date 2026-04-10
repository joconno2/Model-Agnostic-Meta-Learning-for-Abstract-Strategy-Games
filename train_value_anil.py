#!/usr/bin/env python3
"""
Value-only ANIL training for MAML on abstract strategy games.

Usage:
    # Game-as-task (original framing, limited signal):
    python train_value_anil.py --data-dir ./processed_chess_flat

    # Opening-as-task (richer signal — recommended):
    python train_value_anil.py --data-dir ./processed_chess_flat \
        --db-path ./lichess_2016-03_elo1800_base300.sqlite \
        --task-mode opening

    # Player-as-task:
    python train_value_anil.py --data-dir ./processed_chess_flat \
        --db-path ./lichess_2016-03_elo1800_base300.sqlite \
        --task-mode player

Key differences from train_maml_chess.py:
    1. Value-only (no policy head) — cleaner meta-learning signal
    2. ANIL — only value head adapted in inner loop (4,225 params vs ~350K)
    3. Bottleneck between trunk and value head (5184 → 64)
    4. 5 inner steps (up from 1) — room for real adaptation
    5. meta_batch_size=32 (up from 8) — smoother meta-gradients
    6. Gradient clipping — prevents loss spikes
    7. Support for opening-as-task and player-as-task
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from spec import UnifiedSpec, num_channels
from model_v2 import ValueNet
from task_sampler_v2 import ValueTaskSampler
from maml_anil import meta_step_anil, inner_adapt_anil, value_loss


def evaluate_meta_loss(model, sampler, device, head_param_names, args, num_tasks=32):
    """Validation meta-loss: adapt on support, evaluate on query, no model update."""
    model.eval()
    tasks = sampler.sample_meta_batch(num_tasks, args.k_support, args.k_query, split="val")

    total_loss = 0.0
    for sX, sy_val, qX, qy_val, task_id in tasks:
        sX = torch.tensor(sX, dtype=torch.float32, device=device)
        sy_val = torch.tensor(sy_val, dtype=torch.float32, device=device)
        qX = torch.tensor(qX, dtype=torch.float32, device=device)
        qy_val = torch.tensor(qy_val, dtype=torch.float32, device=device)

        base_params = {n: p for n, p in model.named_parameters()}
        adapted = inner_adapt_anil(
            model=model,
            base_params=base_params,
            head_param_names=head_param_names,
            Xs=sX,
            ys_val=sy_val,
            inner_lr=args.inner_lr,
            inner_steps=args.inner_steps,
        )

        with torch.no_grad():
            from torch.func import functional_call
            q_v = functional_call(model, adapted, (qX,))
            loss = value_loss(q_v, qy_val)
        total_loss += loss.item()

    model.train()
    return total_loss / num_tasks


def save_checkpoint(filepath, model, optimizer, iteration, best_val, config, histories):
    torch.save({
        "iteration": iteration,
        "best_val_meta": best_val,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        **histories,
    }, filepath)
    print(f"  Saved: {filepath}")


def save_loss_plot(train_hist, val_hist, val_x, outpath):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_hist) + 1), train_hist, alpha=0.4, label="Train meta-loss")
    # Smoothed training curve
    if len(train_hist) > 20:
        window = min(50, len(train_hist) // 5)
        smoothed = [
            sum(train_hist[max(0, i - window):i + 1]) / len(train_hist[max(0, i - window):i + 1])
            for i in range(len(train_hist))
        ]
        plt.plot(range(1, len(smoothed) + 1), smoothed, label=f"Train (smoothed, w={window})")
    if val_hist:
        plt.plot(val_x, val_hist, "o-", label="Val meta-loss", markersize=3)
    plt.xlabel("Iteration")
    plt.ylabel("Value MSE (meta-loss)")
    plt.title("ANIL Value-Only Meta-Learning")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"  Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Value-only ANIL training")
    parser.add_argument("--data-dir", default="./processed_chess_flat")
    parser.add_argument("--db-path", default=None, help="SQLite DB path (required for opening/player mode)")
    parser.add_argument("--task-mode", default="game", choices=["game", "opening", "player"])
    parser.add_argument("--out-dir", default="./runs/value_anil")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--min-positions", type=int, default=32)

    # Meta-learning hyperparameters
    parser.add_argument("--meta-iters", type=int, default=5000)
    parser.add_argument("--meta-batch-size", type=int, default=32)
    parser.add_argument("--k-support", type=int, default=16)
    parser.add_argument("--k-query", type=int, default=16)
    parser.add_argument("--inner-lr", type=float, default=0.005)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--outer-lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)

    # Architecture
    parser.add_argument("--trunk-hidden", type=int, default=64)
    parser.add_argument("--bottleneck-dim", type=int, default=64)
    parser.add_argument("--value-hidden", type=int, default=64)

    # Logging
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--val-tasks", type=int, default=32)
    parser.add_argument("--ckpt-every", type=int, default=500)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    sampler = ValueTaskSampler(
        data_dir=args.data_dir,
        db_path=args.db_path,
        task_mode=args.task_mode,
        train_frac=args.train_frac,
        seed=args.seed,
        min_positions_per_task=args.min_positions,
    )

    # Model
    spec = UnifiedSpec()
    C = num_channels(spec)

    model = ValueNet(
        in_channels=C,
        trunk_hidden=args.trunk_hidden,
        bottleneck_dim=args.bottleneck_dim,
        value_hidden=args.value_hidden,
    ).to(device)

    print(f"Model: {model.total_param_count()} total params, "
          f"{model.head_param_count()} adapted in inner loop")

    # Identify head params for ANIL
    head_param_names = set(n for n, _ in model.head_params())
    print(f"Head params: {sorted(head_param_names)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.outer_lr)

    # Save config
    config = vars(args)
    config["n_channels"] = C
    config["total_params"] = model.total_param_count()
    config["head_params"] = model.head_param_count()

    with open(os.path.join(args.out_dir, "config.txt"), "w") as f:
        for k, v in sorted(config.items()):
            f.write(f"{k}={v}\n")

    # History
    train_hist = []
    val_hist = []
    val_x = []
    best_val = float("inf")

    histories = lambda: {
        "train_meta_history": train_hist,
        "val_meta_history": val_hist,
        "val_meta_x": val_x,
    }

    t0 = time.time()

    for it in range(1, args.meta_iters + 1):
        task_batch = sampler.sample_meta_batch(
            args.meta_batch_size, args.k_support, args.k_query, split="train"
        )

        meta_loss, grad_norm = meta_step_anil(
            model=model,
            optimizer=optimizer,
            task_batch=task_batch,
            device=device,
            inner_lr=args.inner_lr,
            inner_steps=args.inner_steps,
            head_param_names=head_param_names,
            max_grad_norm=args.max_grad_norm,
        )

        train_hist.append(meta_loss)

        elapsed = time.time() - t0
        print(f"[it {it:5d}] meta_loss={meta_loss:.4f} grad_norm={grad_norm:.4f} ({elapsed:.0f}s)")

        if it % args.val_every == 0:
            val_meta = evaluate_meta_loss(
                model, sampler, device, head_param_names, args, num_tasks=args.val_tasks
            )
            val_hist.append(val_meta)
            val_x.append(it)
            improved = " *" if val_meta < best_val else ""
            print(f"  [VAL] meta_loss={val_meta:.4f} (best={min(best_val, val_meta):.4f}){improved}")

            if val_meta < best_val:
                best_val = val_meta
                save_checkpoint(
                    os.path.join(args.out_dir, "best.pt"),
                    model, optimizer, it, best_val, config, histories(),
                )

            save_checkpoint(
                os.path.join(args.out_dir, "latest.pt"),
                model, optimizer, it, best_val, config, histories(),
            )

            save_loss_plot(
                train_hist, val_hist, val_x,
                os.path.join(args.out_dir, "loss.png"),
            )

        if it % args.ckpt_every == 0:
            save_checkpoint(
                os.path.join(args.out_dir, f"ckpt_it{it}.pt"),
                model, optimizer, it, best_val, config, histories(),
            )

    # Final save
    save_checkpoint(
        os.path.join(args.out_dir, "final.pt"),
        model, optimizer, args.meta_iters, best_val, config, histories(),
    )
    save_loss_plot(
        train_hist, val_hist, val_x,
        os.path.join(args.out_dir, "loss_final.png"),
    )

    total_time = time.time() - t0
    print(f"\nDone. {args.meta_iters} iterations in {total_time:.0f}s. Best val={best_val:.4f}")


if __name__ == "__main__":
    main()
