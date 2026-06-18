#!/usr/bin/env python3
"""
ANIL/MAML training with game-disjoint support/query sampling.

Key difference from original training: support and query positions within
each meta-learning task come from DIFFERENT games sharing the same opening.
This prevents within-game memorization and forces the trunk to learn features
that generalize across games.

Supports both ANIL (head-only adaptation) and full MAML (all-param adaptation).

Usage:
    cd ~/code/maml-dasg

    # ANIL, cross-game:
    python train_disjoint.py \
        --data-dir processed_combined_flat \
        --db-path combined_openings.sqlite \
        --out-dir runs/disjoint_anil_combined \
        --mode anil --seed 42

    # ANIL, chess-only:
    python train_disjoint.py \
        --data-dir processed_chess_flat \
        --db-path combined_openings.sqlite \
        --out-dir runs/disjoint_anil_chess \
        --mode anil --seed 42

    # Full MAML, cross-game:
    python train_disjoint.py \
        --data-dir processed_combined_flat \
        --db-path combined_openings.sqlite \
        --out-dir runs/disjoint_maml_combined \
        --mode maml --seed 42

    # Multiple seeds:
    for s in 42 123 456 789 1337; do
        python train_disjoint.py \
            --data-dir processed_combined_flat \
            --db-path combined_openings.sqlite \
            --out-dir runs/disjoint_anil_combined_s$s \
            --mode anil --seed $s
    done
"""

import argparse
import os
import signal
import sqlite3
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.func import functional_call

from spec import UnifiedSpec, num_channels
from model_v2 import ValueNet
from maml_anil import inner_adapt_anil, value_loss

ROOT = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Game-disjoint task sampler (training version)
# ============================================================

class DisjointTrainSampler:
    """
    Task sampler for meta-training with game-disjoint support/query.

    Memory strategy: builds lightweight index at init (~50MB), caches
    shard data on first access. After warmup, all data is in memory
    for fast random access.
    """

    def __init__(self, data_dir, db_path, train_frac=0.8, seed=42,
                 min_games=10, max_pos_per_game=50):
        self.rng = np.random.RandomState(seed)
        self.py_rng = __import__("random").Random(seed)

        shard_files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith(".npz")
        )
        self.shard_files = shard_files

        # Build game_id -> (shard_idx, local_row) index
        game_locs = defaultdict(list)
        total_pos = 0
        for shard_idx, path in enumerate(shard_files):
            with np.load(path) as d:
                gids = d["game_id"].astype(np.int64)
            total_pos += len(gids)
            for local_row, gid in enumerate(gids):
                game_locs[int(gid)].append((shard_idx, local_row))

        # Load opening codes
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT id, eco FROM games WHERE eco IS NOT NULL AND eco != ''"
        ).fetchall()
        conn.close()
        game_to_eco = {int(r[0]): r[1] for r in rows}

        # Group: opening -> game_id -> [(shard_idx, local_row)]
        opening_games = defaultdict(lambda: defaultdict(list))
        for gid, locs in game_locs.items():
            eco = game_to_eco.get(gid)
            if eco is None:
                continue
            if len(locs) > max_pos_per_game:
                locs = [locs[i] for i in
                        self.rng.choice(len(locs), max_pos_per_game, replace=False)]
            opening_games[eco][gid] = locs

        # Filter by min games
        self.opening_games = {}
        self.task_ids = []
        for eco, games in opening_games.items():
            if len(games) >= min_games:
                self.opening_games[eco] = dict(games)
                self.task_ids.append(eco)
        self.task_ids.sort()

        # Train/val split
        ids = self.task_ids[:]
        self.rng.shuffle(ids)
        split = int(len(ids) * train_frac)
        self.train_task_ids = ids[:split]
        self.val_task_ids = ids[split:]

        # Shard cache
        self._cache = {}

        chess_t = sum(1 for t in self.train_task_ids if t.startswith("chess_"))
        shogi_t = sum(1 for t in self.train_task_ids if not t.startswith("chess_"))
        chess_v = sum(1 for t in self.val_task_ids if t.startswith("chess_"))
        shogi_v = sum(1 for t in self.val_task_ids if not t.startswith("chess_"))

        print(f"[DisjointTrainSampler] {total_pos:,} positions, {len(game_locs):,} games, "
              f"{len(shard_files)} shards")
        print(f"  {len(self.task_ids)} openings with >= {min_games} games")
        print(f"  Train: {len(self.train_task_ids)} ({chess_t} chess, {shogi_t} shogi)")
        print(f"  Val:   {len(self.val_task_ids)} ({chess_v} chess, {shogi_v} shogi)")

    def _load(self, locs):
        """Load positions from (shard_idx, row) tuples."""
        by_shard = defaultdict(list)
        for si, row in locs:
            by_shard[si].append(row)

        X_parts, yv_parts = [], []
        for si, rows in by_shard.items():
            if si not in self._cache:
                d = np.load(self.shard_files[si])
                self._cache[si] = (d["X"], d["y_value"])
            X_s, yv_s = self._cache[si]
            arr = np.array(rows, dtype=np.int64)
            X_parts.append(X_s[arr])
            yv_parts.append(yv_s[arr])

        return np.concatenate(X_parts), np.concatenate(yv_parts)

    def sample_task(self, k_support, k_query, split="train"):
        """Sample one task with game-disjoint support/query."""
        pool = self.train_task_ids if split == "train" else self.val_task_ids

        for _ in range(200):
            eco = self.py_rng.choice(pool)
            games = self.opening_games[eco]
            game_ids = list(games.keys())
            if len(game_ids) < 2:
                continue

            self.py_rng.shuffle(game_ids)
            mid = max(1, len(game_ids) // 2)
            s_games = game_ids[:mid]
            q_games = game_ids[mid:]

            s_pool = []
            for gid in s_games:
                s_pool.extend(games[gid])
            q_pool = []
            for gid in q_games:
                q_pool.extend(games[gid])

            if len(s_pool) < k_support or len(q_pool) < k_query:
                continue

            s_pick = [s_pool[i] for i in
                      self.rng.choice(len(s_pool), k_support, replace=False)]
            q_pick = [q_pool[i] for i in
                      self.rng.choice(len(q_pool), k_query, replace=False)]

            sX, sy = self._load(s_pick)
            qX, qy = self._load(q_pick)
            return sX, sy, qX, qy, eco

        raise RuntimeError("Could not sample a valid task after 200 attempts")

    def sample_meta_batch(self, batch_size, k_support, k_query, split="train"):
        return [self.sample_task(k_support, k_query, split) for _ in range(batch_size)]


# ============================================================
# Inner loop variants
# ============================================================

def inner_adapt_full(model, base_params, Xs, ys, inner_lr, inner_steps):
    """Full MAML: adapt ALL parameters."""
    params = {k: v.clone() for k, v in base_params.items()}
    for _ in range(inner_steps):
        pgrad = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
        v_pred = functional_call(model, pgrad, (Xs,))
        loss = value_loss(v_pred, ys)
        grads = torch.autograd.grad(loss, [pgrad[k] for k in sorted(pgrad)],
                                    create_graph=False)
        gmap = dict(zip(sorted(pgrad.keys()), grads))
        params = {k: (pgrad[k] - inner_lr * gmap[k]).detach() for k in pgrad}
    return params


# ============================================================
# Training loop
# ============================================================

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
    parser = argparse.ArgumentParser(description="ANIL/MAML with game-disjoint sampling")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=["anil", "maml"], default="anil")
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
    parser.add_argument("--max-hours", type=float, default=48.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--min-games", type=int, default=10)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu")
    spec = UnifiedSpec()
    C = num_channels(spec)

    model = ValueNet(
        in_channels=C,
        trunk_hidden=args.trunk_hidden,
        bottleneck_dim=args.bottleneck_dim,
        value_hidden=args.value_hidden,
    ).to(device)

    adapted_count = model.head_param_count() if args.mode == "anil" else model.total_param_count()
    print(f"Model: {model.total_param_count():,} params, "
          f"{adapted_count:,} adapted ({args.mode.upper()})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.outer_lr, eps=1e-4)
    head_param_names = set(n for n, _ in model.head_params())

    # Data
    sampler = DisjointTrainSampler(
        data_dir=os.path.join(ROOT, args.data_dir),
        db_path=os.path.join(ROOT, args.db_path),
        train_frac=args.train_frac,
        seed=args.seed,
        min_games=args.min_games,
    )

    config = vars(args)
    config["n_channels"] = C
    config["total_params"] = model.total_param_count()
    config["adapted_params"] = adapted_count
    config["disjoint_support_query"] = True
    config["min_games_per_opening"] = args.min_games

    with open(os.path.join(args.out_dir, "config.txt"), "w") as f:
        for k, v in sorted(config.items()):
            f.write(f"{k}={v}\n")

    # Signal handling
    stop = [False]
    def handler(sig, frame):
        if stop[0]:
            sys.exit(1)
        print("\nStopping after current iteration.")
        stop[0] = True
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

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
    print(f"  mode: {args.mode.upper()}, inner: {args.inner_steps} steps @ lr={args.inner_lr}")
    print(f"  game-disjoint support/query, min {args.min_games} games/opening")
    print()

    for it in range(start_iter + 1, args.meta_iters + 1):
        if stop[0]:
            break
        if time.time() - t0 > args.max_hours * 3600:
            print(f"Reached max_hours={args.max_hours}")
            break

        iter_t0 = time.time()

        tasks = sampler.sample_meta_batch(
            args.meta_batch_size, args.k_support, args.k_query, split="train"
        )

        optimizer.zero_grad(set_to_none=True)
        base_params = {n: p for n, p in model.named_parameters()}
        accum_grads = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        meta_loss_sum = 0.0

        for sX, sy, qX, qy, task_id in tasks:
            sX_t = torch.tensor(sX, dtype=torch.float32, device=device)
            sy_t = torch.tensor(sy, dtype=torch.float32, device=device)
            qX_t = torch.tensor(qX, dtype=torch.float32, device=device)
            qy_t = torch.tensor(qy, dtype=torch.float32, device=device)

            if args.mode == "anil":
                adapted = inner_adapt_anil(
                    model, base_params, head_param_names,
                    sX_t, sy_t, args.inner_lr, args.inner_steps,
                )
            else:
                adapted = inner_adapt_full(
                    model, base_params,
                    sX_t, sy_t, args.inner_lr, args.inner_steps,
                )

            adapted_req = {k: v.detach().clone().requires_grad_(True)
                           for k, v in adapted.items()}
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

        # Validation (also game-disjoint)
        if it % args.val_every == 0:
            val_tasks = sampler.sample_meta_batch(
                args.val_tasks, args.k_support, args.k_query, split="val"
            )
            val_base = {n: p.detach().clone() for n, p in model.named_parameters()}
            val_loss_sum = 0.0

            for sX, sy, qX, qy, task_id in val_tasks:
                sX_t = torch.tensor(sX, dtype=torch.float32, device=device)
                sy_t = torch.tensor(sy, dtype=torch.float32, device=device)
                qX_t = torch.tensor(qX, dtype=torch.float32, device=device)
                qy_t = torch.tensor(qy, dtype=torch.float32, device=device)

                if args.mode == "anil":
                    adapted = inner_adapt_anil(
                        model, val_base, head_param_names,
                        sX_t, sy_t, args.inner_lr, args.inner_steps,
                    )
                else:
                    adapted = inner_adapt_full(
                        model, val_base,
                        sX_t, sy_t, args.inner_lr, args.inner_steps,
                    )
                with torch.no_grad():
                    q_v = functional_call(model,
                                          {k: v.detach() for k, v in adapted.items()},
                                          (qX_t,))
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
    final_it = it if "it" in dir() else 0
    save_checkpoint(os.path.join(args.out_dir, "final.pt"),
                    model, optimizer, final_it, best_val, config, histories())

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_hist) + 1), train_hist, alpha=0.3, label="Train")
    if len(train_hist) > 20:
        w = min(50, len(train_hist) // 5)
        sm = [sum(train_hist[max(0, i-w):i+1]) / len(train_hist[max(0, i-w):i+1])
              for i in range(len(train_hist))]
        plt.plot(range(1, len(sm) + 1), sm, label=f"Train (w={w})")
    if val_hist:
        plt.plot(val_x, val_hist, "o-", label="Val", markersize=3)
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.title(f"{args.mode.upper()} Game-Disjoint (seed {args.seed})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "loss.png"), dpi=200)
    plt.close()

    print(f"\nDone. {len(train_hist)} iters, best val: {best_val:.4f}")


if __name__ == "__main__":
    main()
