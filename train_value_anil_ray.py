#!/usr/bin/env python3
"""
Distributed value-only ANIL training on the Ray cluster.

The driver (this machine) holds the model and optimizer. Each meta-iteration:
  1. Serialize model weights to flat numpy array
  2. Fan out task slices to Ray actors (each runs inner adapt + query grads)
  3. Collect and average gradients
  4. Apply to model via optimizer.step()

Data must be pre-distributed to workers at --data-dir (e.g. /tmp/maml-chess/).

Usage:
    # Ensure Ray tunnel is open:
    python ~/research/work-tools/infra/ray-tunnel.py --head 136.244.224.136

    # Run:
    python train_value_anil_ray.py \
        --data-dir /tmp/maml-chess/processed_chess_flat \
        --ray-address ray://127.0.0.1:10001 \
        --meta-batch-size 128 \
        --max-actors 50

    # With opening-as-task (requires SQLite DB on each worker):
    python train_value_anil_ray.py \
        --data-dir /tmp/maml-chess/processed_chess_flat \
        --db-path /tmp/maml-chess/lichess.sqlite \
        --task-mode opening \
        --ray-address ray://127.0.0.1:10001
"""

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from spec import UnifiedSpec, num_channels
from model_v2 import ValueNet


def model_to_flat(model: torch.nn.Module) -> np.ndarray:
    parts = [p.detach().cpu().numpy().ravel() for p in model.parameters()]
    return np.concatenate(parts).astype(np.float32)


def flat_to_model(model: torch.nn.Module, flat: np.ndarray):
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(torch.from_numpy(
            flat[offset:offset + numel].reshape(p.shape).copy()
        ))
        offset += numel


def save_checkpoint(filepath, model, optimizer, iteration, best_val, config, histories):
    torch.save({
        "iteration": iteration,
        "best_val_meta": best_val,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        **histories,
    }, filepath)


def save_loss_plot(train_hist, val_hist, val_x, outpath):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_hist) + 1), train_hist, alpha=0.3, label="Train meta-loss")
    if len(train_hist) > 20:
        w = min(50, len(train_hist) // 5)
        smoothed = [
            sum(train_hist[max(0, i - w):i + 1]) / len(train_hist[max(0, i - w):i + 1])
            for i in range(len(train_hist))
        ]
        plt.plot(range(1, len(smoothed) + 1), smoothed, label=f"Train (smoothed, w={w})")
    if val_hist:
        plt.plot(val_x, val_hist, "o-", label="Val meta-loss", markersize=3)
    plt.xlabel("Iteration")
    plt.ylabel("Value MSE (meta-loss)")
    plt.title("Distributed ANIL Value-Only Meta-Learning")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Distributed value-only ANIL on Ray")

    # Data
    parser.add_argument("--data-dir", required=True, help="Path to processed_chess_flat on WORKERS (not this machine)")
    parser.add_argument("--db-path", default=None, help="SQLite DB path on WORKERS (for opening/player mode)")
    parser.add_argument("--task-mode", default="game", choices=["game", "opening", "player"])
    parser.add_argument("--min-positions", type=int, default=32)

    # Ray
    parser.add_argument("--ray-address", default="ray://127.0.0.1:10001")
    parser.add_argument("--max-actors", type=int, default=None)

    # Output
    parser.add_argument("--out-dir", default="./runs/value_anil_ray")

    # Meta-learning
    parser.add_argument("--meta-iters", type=int, default=5000)
    parser.add_argument("--meta-batch-size", type=int, default=128, help="Total tasks per meta-step (distributed)")
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

    # Schedule
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--val-tasks", type=int, default=128)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--max-hours", type=float, default=24.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Connect to Ray ----
    import ray
    from maml_ray_pool import MAMLRayPool

    repo_root = str(Path(__file__).parent.resolve())
    print(f"Connecting to Ray at {args.ray_address} ...")
    ray.init(
        address=args.ray_address,
        logging_level="WARNING",
        runtime_env={
            "working_dir": repo_root,
            "excludes": [".git", ".venv", "runs", "__pycache__", "*.sqlite", "processed_*"],
            # Workers need torch + numpy for the ANIL inner loop.
            # These are installed on workers via install_worker_deps.py
            # (one-time, not per-run). Don't use runtime_env pip for torch
            # — it's 3GB and would install on every actor startup.
        },
    )
    cluster_res = ray.cluster_resources()
    print(f"Cluster: {int(cluster_res.get('CPU', 0))} CPUs")

    # ---- Build model (driver-side) ----
    spec = UnifiedSpec()
    C = num_channels(spec)

    model = ValueNet(
        in_channels=C,
        trunk_hidden=args.trunk_hidden,
        bottleneck_dim=args.bottleneck_dim,
        value_hidden=args.value_hidden,
    )
    print(f"Model: {model.total_param_count():,} params ({model.head_param_count():,} adapted)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.outer_lr)
    head_param_names = set(n for n, _ in model.head_params())

    # ---- Start actor pool ----
    print("Starting MAML actor pool ...")
    pool = MAMLRayPool(
        data_dir=args.data_dir,
        db_path=args.db_path,
        task_mode=args.task_mode,
        in_channels=C,
        trunk_hidden=args.trunk_hidden,
        bottleneck_dim=args.bottleneck_dim,
        value_hidden=args.value_hidden,
        train_frac=args.train_frac,
        seed=args.seed,
        min_positions_per_task=args.min_positions,
        max_actors=args.max_actors,
    )
    pool.start()
    print(f"Pool ready: {pool.actor_count} actors across {len(pool.describe_capacity())} hosts")
    print(f"  {pool.describe_capacity()}")

    # ---- Config ----
    config = vars(args)
    config["n_channels"] = C
    config["total_params"] = model.total_param_count()
    config["head_params"] = model.head_param_count()
    config["n_actors"] = pool.actor_count

    with open(os.path.join(args.out_dir, "config.txt"), "w") as f:
        for k, v in sorted(config.items()):
            f.write(f"{k}={v}\n")

    # ---- Signal handling ----
    stop_requested = [False]
    def handle_signal(sig, frame):
        if stop_requested[0]:
            sys.exit(1)
        print("\nSignal received — stopping after current iteration.")
        stop_requested[0] = True
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ---- Training loop ----
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
    max_seconds = args.max_hours * 3600

    print(f"\nTraining: {args.meta_iters} iters, {args.meta_batch_size} tasks/iter "
          f"(distributed across {pool.actor_count} actors)")
    print(f"  inner: {args.inner_steps} steps @ lr={args.inner_lr}")
    print(f"  outer: Adam lr={args.outer_lr}, grad_clip={args.max_grad_norm}")
    print()

    for it in range(1, args.meta_iters + 1):
        if stop_requested[0]:
            break
        if time.time() - t0 > max_seconds:
            print(f"Reached max_hours={args.max_hours} — stopping.")
            break

        iter_t0 = time.time()

        # Serialize weights
        weights_flat = model_to_flat(model)

        # Distributed meta-step
        grad_flat, meta_loss = pool.meta_step(
            weights_flat=weights_flat,
            total_tasks=args.meta_batch_size,
            k_support=args.k_support,
            k_query=args.k_query,
            inner_lr=args.inner_lr,
            inner_steps=args.inner_steps,
        )

        # Apply gradients to model
        optimizer.zero_grad(set_to_none=True)
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            p.grad = torch.from_numpy(
                grad_flat[offset:offset + numel].reshape(p.shape).copy()
            ).float()
            offset += numel

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

        optimizer.step()

        train_hist.append(meta_loss)
        iter_ms = (time.time() - iter_t0) * 1000
        elapsed = time.time() - t0

        print(f"[it {it:5d}] meta_loss={meta_loss:.4f} | {iter_ms:.0f}ms | {elapsed:.0f}s")

        # Validation
        if it % args.val_every == 0:
            val_weights = model_to_flat(model)
            val_loss = pool.val_loss(
                weights_flat=val_weights,
                total_tasks=args.val_tasks,
                k_support=args.k_support,
                k_query=args.k_query,
                inner_lr=args.inner_lr,
                inner_steps=args.inner_steps,
            )
            val_hist.append(val_loss)
            val_x.append(it)
            improved = " *" if val_loss < best_val else ""
            print(f"  [VAL] meta_loss={val_loss:.4f} (best={min(best_val, val_loss):.4f}){improved}")

            if val_loss < best_val:
                best_val = val_loss
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
        model, optimizer, it if 'it' in dir() else 0, best_val, config, histories(),
    )
    save_loss_plot(
        train_hist, val_hist, val_x,
        os.path.join(args.out_dir, "loss_final.png"),
    )

    total_time = time.time() - t0
    print(f"\nDone. {len(train_hist)} iterations in {total_time:.0f}s. Best val={best_val:.4f}")

    pool.shutdown()
    ray.shutdown()


if __name__ == "__main__":
    main()
