#!/usr/bin/env python3
"""
Stockfish-centipawn evaluation suite for MAML-DASG.

Forked from eval_reviewer_suite.py. The reviewer suite tested game-disjoint
(chess vs shogi) generalization on the W/L target. The Stockfish centipawn
signal is chess-only, so the generalization axis here is OPENING-DISJOINT:
adapt on some openings, test on held-out openings, with support and query
drawn from different games within each opening.

Target: y_value = Stockfish centipawn eval squashed to [-1, 1].

Experiments:
  1. Adaptation curve (0, 1, 3, 5, 10 inner steps) on held-out openings
  2. Random trunk + adapted head (lower bound)
  3. Meta-learned trunk + random head (isolates trunk vs head init)
  4. CCA-style pre/post adaptation prediction correlation

The gate: inner-loop adaptation (steps > 0) must beat zero-shot (steps 0)
on held-out openings, and the meta trunk must beat the random trunk.

Usage:
    cd ~/maml-dasg
    CUDA_VISIBLE_DEVICES= ~/evo-distill/.venv/bin/python eval_sf_suite.py \
        --tasks-per-cell 600 --seeds 42,123,456,789,1337
"""

import argparse
import json
import os
import sqlite3
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call

from model_v2 import ValueNet
from maml_anil import inner_adapt_anil, value_loss

ROOT = os.path.dirname(os.path.abspath(__file__))

# Match the SF k=64 run config (runs/disjoint_anil_sf_k64_s42/config.txt)
N_CHANNELS = 45
TRUNK_HIDDEN = 64
BOTTLENECK_DIM = 64
VALUE_HIDDEN = 64
TRAIN_FRAC = 0.8
K_SUPPORT = 64
K_QUERY = 64
INNER_LR = 0.005
MIN_GAMES_PER_OPENING = 10

SF_CKPT = "runs/disjoint_anil_sf_k64_s42/best.pt"
SF_DATA = "processed_sf_chess_small"
SF_DB = "sf_openings.sqlite"


# ============================================================
# Opening-disjoint task sampler (support/query from different games)
# ============================================================

class DisjointTaskSampler:
    """
    Sample support and query positions from different games within the same
    opening (ECO code). Filters to openings with >= min_games distinct games.
    Builds a lightweight shard index at init; loads positions on demand.
    """

    def __init__(self, data_dir, db_path, train_frac=0.8, seed=42,
                 min_games=10, max_pos_per_game=50):
        self.rng = np.random.RandomState(seed)
        self.min_games = min_games
        self.data_dir = data_dir

        self.shard_files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith(".npz")
        )

        game_locs = defaultdict(list)
        total_pos = 0
        for shard_idx, path in enumerate(self.shard_files):
            with np.load(path) as d:
                gids = d["game_id"].astype(np.int64)
            total_pos += len(gids)
            for local_row, gid in enumerate(gids):
                game_locs[int(gid)].append((shard_idx, local_row))

        print(f"[DisjointSampler] indexed {total_pos} positions, "
              f"{len(game_locs)} games, {len(self.shard_files)} shards")

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT id, eco FROM games WHERE eco IS NOT NULL AND eco != ''"
        ).fetchall()
        conn.close()
        game_to_eco = {int(r[0]): r[1] for r in rows}

        opening_games_raw = defaultdict(lambda: defaultdict(list))
        for gid, locs in game_locs.items():
            eco = game_to_eco.get(gid)
            if eco is None:
                continue
            if len(locs) > max_pos_per_game:
                locs = [locs[i] for i in self.rng.choice(len(locs), max_pos_per_game, replace=False)]
            opening_games_raw[eco][gid] = locs

        self.opening_games = {}
        self.task_ids = []
        for eco, games in opening_games_raw.items():
            if len(games) >= min_games:
                self.opening_games[eco] = dict(games)
                self.task_ids.append(eco)
        self.task_ids.sort()

        ids_shuffled = self.task_ids[:]
        self.rng.shuffle(ids_shuffled)
        split = int(len(ids_shuffled) * train_frac)
        self.train_task_ids = sorted(ids_shuffled[:split])
        self.val_task_ids = sorted(ids_shuffled[split:])

        self._shard_cache = {}
        self._cache_order = []
        self._max_cache = len(self.shard_files) + 1

        print(f"[DisjointSampler] {len(self.task_ids)} openings with >= {min_games} games")
        print(f"  Train: {len(self.train_task_ids)} | Val: {len(self.val_task_ids)}")

    def _load_positions(self, locs):
        by_shard = defaultdict(list)
        for shard_idx, local_row in locs:
            by_shard[shard_idx].append(local_row)

        X_parts, yv_parts = [], []
        for shard_idx, rows in by_shard.items():
            if shard_idx not in self._shard_cache:
                if len(self._shard_cache) >= self._max_cache:
                    evict = self._cache_order.pop(0)
                    del self._shard_cache[evict]
                data = np.load(self.shard_files[shard_idx])
                self._shard_cache[shard_idx] = (data["X"], data["y_value"])
                self._cache_order.append(shard_idx)

            X_shard, yv_shard = self._shard_cache[shard_idx]
            rows_arr = np.array(rows, dtype=np.int64)
            X_parts.append(X_shard[rows_arr])
            yv_parts.append(yv_shard[rows_arr])

        return np.concatenate(X_parts, axis=0), np.concatenate(yv_parts, axis=0)

    def sample_disjoint(self, eco, k_support, k_query):
        games = self.opening_games.get(eco)
        if games is None:
            return None
        game_ids = list(games.keys())
        if len(game_ids) < 2:
            return None

        self.rng.shuffle(game_ids)
        mid = max(1, len(game_ids) // 2)
        support_games = game_ids[:mid]
        query_games = game_ids[mid:]

        support_locs = []
        for gid in support_games:
            support_locs.extend(games[gid])
        query_locs = []
        for gid in query_games:
            query_locs.extend(games[gid])

        if len(support_locs) < k_support or len(query_locs) < k_query:
            return None

        s_pick = [support_locs[i] for i in self.rng.choice(len(support_locs), k_support, replace=False)]
        q_pick = [query_locs[i] for i in self.rng.choice(len(query_locs), k_query, replace=False)]

        sX, sy = self._load_positions(s_pick)
        qX, qy = self._load_positions(q_pick)
        return sX, sy, qX, qy


# ============================================================
# Model / evaluation
# ============================================================

def load_model(ckpt_path, n_channels, device="cpu"):
    model = ValueNet(
        in_channels=n_channels,
        trunk_hidden=TRUNK_HIDDEN,
        bottleneck_dim=BOTTLENECK_DIM,
        value_hidden=VALUE_HIDDEN,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def evaluate_disjoint(model, sampler, task_ids, inner_steps,
                      tasks_per_cell, base_params=None, device="cpu"):
    head_param_names = set(n for n, _ in model.head_params())
    if base_params is None:
        base_params = {n: p.detach().clone() for n, p in model.named_parameters()}

    losses = []
    episodes = 0
    max_attempts = tasks_per_cell * 3

    for _ in range(max_attempts):
        if episodes >= tasks_per_cell:
            break
        eco = task_ids[sampler.rng.randint(len(task_ids))]
        sample = sampler.sample_disjoint(eco, K_SUPPORT, K_QUERY)
        if sample is None:
            continue

        sX, sy, qX, qy = sample
        sX_t = torch.tensor(sX, dtype=torch.float32, device=device)
        sy_t = torch.tensor(sy, dtype=torch.float32, device=device)
        qX_t = torch.tensor(qX, dtype=torch.float32, device=device)
        qy_t = torch.tensor(qy, dtype=torch.float32, device=device)

        if inner_steps > 0:
            adapted = inner_adapt_anil(
                model=model, base_params=base_params,
                head_param_names=head_param_names,
                Xs=sX_t, ys_val=sy_t,
                inner_lr=INNER_LR, inner_steps=inner_steps,
            )
        else:
            adapted = {k: v.detach() for k, v in base_params.items()}

        with torch.no_grad():
            q_v = functional_call(model, adapted, (qX_t,))
            loss = value_loss(q_v, qy_t)
        losses.append(loss.item())
        episodes += 1

    return losses


def make_random_head_params(model):
    head_param_names = set(n for n, _ in model.head_params())
    params = {}
    for n, p in model.named_parameters():
        if n in head_param_names:
            params[n] = torch.randn_like(p) * 0.01
        else:
            params[n] = p.detach().clone()
    return params


def make_random_trunk_params(model):
    params = {}
    for n, p in model.named_parameters():
        if "trunk" in n or "bottleneck" in n:
            if p.dim() >= 2:
                params[n] = torch.empty_like(p)
                nn.init.kaiming_normal_(params[n])
            else:
                params[n] = torch.randn_like(p) * 0.01
        else:
            params[n] = torch.randn_like(p) * 0.01
    return params


def compute_cca_similarity(model, sampler, task_ids, n_tasks=100, device="cpu"):
    head_param_names = set(n for n, _ in model.head_params())
    base_params = {n: p.detach().clone() for n, p in model.named_parameters()}

    similarities = []
    for _ in range(n_tasks * 3):
        if len(similarities) >= n_tasks:
            break
        eco = task_ids[sampler.rng.randint(len(task_ids))]
        sample = sampler.sample_disjoint(eco, K_SUPPORT, K_QUERY)
        if sample is None:
            continue

        sX, sy, qX, qy = sample
        sX_t = torch.tensor(sX, dtype=torch.float32, device=device)
        sy_t = torch.tensor(sy, dtype=torch.float32, device=device)
        qX_t = torch.tensor(qX, dtype=torch.float32, device=device)

        pre_params = {k: v.detach() for k, v in base_params.items()}
        adapted = inner_adapt_anil(
            model=model, base_params=base_params,
            head_param_names=head_param_names,
            Xs=sX_t, ys_val=sy_t,
            inner_lr=INNER_LR, inner_steps=5,
        )
        with torch.no_grad():
            pre_v = functional_call(model, pre_params, (qX_t,)).numpy()
            adapted_d = {k: v.detach() for k, v in adapted.items()}
            post_v = functional_call(model, adapted_d, (qX_t,)).numpy()

        if pre_v.std() > 1e-8 and post_v.std() > 1e-8:
            corr = np.corrcoef(pre_v.flatten(), post_v.flatten())[0, 1]
            similarities.append(corr)

    return {
        "mean_corr": float(np.mean(similarities)) if similarities else None,
        "std_corr": float(np.std(similarities)) if similarities else None,
        "n": len(similarities),
        "note": "Pearson correlation between pre- and post-adaptation value "
                "predictions. Low correlation = adaptation changes predictions "
                "= the inner loop matters.",
    }


def compute_ci(values, confidence=0.95):
    n = len(values)
    if n == 0:
        return None, None, None
    mean = float(np.mean(values))
    if n < 3:
        return mean, mean, mean
    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(2000):
        sample = rng.choice(values, n, replace=True)
        boot_means.append(np.mean(sample))
    boot_means = sorted(boot_means)
    alpha = (1 - confidence) / 2
    ci_low = boot_means[int(alpha * 2000)]
    ci_high = boot_means[int((1 - alpha) * 2000)]
    return mean, float(ci_low), float(ci_high)


def paired_ttest(losses_a, losses_b):
    n = min(len(losses_a), len(losses_b))
    if n < 3:
        return None
    diffs = np.array(losses_a[:n]) - np.array(losses_b[:n])
    denom = (diffs.std(ddof=1) / np.sqrt(n))
    if denom == 0:
        return None
    t_stat = diffs.mean() / denom
    try:
        from scipy import stats as sp_stats
        return float(sp_stats.t.sf(abs(t_stat), n - 1) * 2)
    except ImportError:
        from math import erf, sqrt
        return float(2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2)))))


# ============================================================
# Suite
# ============================================================

def run_suite(tasks_per_cell, seed, data_dir, db_path, ckpt_path):
    device = torch.device("cpu")

    results = {"seed": seed, "tasks_per_cell": tasks_per_cell,
               "min_games_per_opening": MIN_GAMES_PER_OPENING,
               "disjoint_support_query": True,
               "target": "stockfish_centipawn (y_value, [-1,1])",
               "k_support": K_SUPPORT, "k_query": K_QUERY}

    model, ckpt = load_model(ckpt_path, N_CHANNELS, device)
    results["ckpt_val_meta"] = ckpt.get("best_val_meta", None)

    print(f"\nBuilding opening-disjoint sampler (min {MIN_GAMES_PER_OPENING} games/opening)...")
    sampler = DisjointTaskSampler(
        data_dir=data_dir, db_path=db_path,
        train_frac=TRAIN_FRAC, seed=seed,
        min_games=MIN_GAMES_PER_OPENING,
    )
    results["val_openings"] = len(sampler.val_task_ids)

    step_counts = [0, 1, 3, 5, 10]

    # 1. Adaptation curve on held-out openings
    print(f"\n{'='*60}\n  1. ADAPTATION CURVE (held-out openings)\n{'='*60}")
    results["adaptation_curve"] = {}
    step_losses = {}
    for steps in step_counts:
        t0 = time.time()
        losses = evaluate_disjoint(
            model, sampler, sampler.val_task_ids,
            inner_steps=steps, tasks_per_cell=tasks_per_cell, device=device,
        )
        step_losses[steps] = losses
        elapsed = time.time() - t0
        mean, ci_lo, ci_hi = compute_ci(losses)
        results["adaptation_curve"][f"steps_{steps}"] = {
            "mse": mean, "ci_low": ci_lo, "ci_high": ci_hi,
            "std": float(np.std(losses)) if losses else 0.0,
            "n": len(losses), "elapsed_s": round(elapsed, 1),
        }
        marker = " <-- default" if steps == 5 else ""
        ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]" if ci_lo is not None else "N/A"
        print(f"  steps={steps:2d}: MSE={mean:.4f} 95%CI={ci_str} "
              f"(n={len(losses)}, {elapsed:.1f}s){marker}")

    # Gate: does adaptation beat zero-shot?
    p_adapt = paired_ttest(step_losses[0], step_losses[5])
    results["adaptation_vs_zeroshot_p"] = p_adapt
    if results["adaptation_curve"]["steps_0"]["mse"] and results["adaptation_curve"]["steps_5"]["mse"]:
        delta = results["adaptation_curve"]["steps_0"]["mse"] - results["adaptation_curve"]["steps_5"]["mse"]
        results["adaptation_gain_mse"] = delta
        print(f"\n  GATE: 0-step MSE - 5-step MSE = {delta:+.4f} "
              f"(paired p={p_adapt:.2e})  {'PASS' if delta > 0 else 'FAIL'}")

    # 2 & 3. Baselines
    print(f"\n{'='*60}\n  2. BASELINES\n{'='*60}")
    results["baselines"] = {}

    print("\n  Random trunk + random head (lower bound):")
    random_params = make_random_trunk_params(model)
    for steps in [0, 5, 10]:
        losses = evaluate_disjoint(
            model, sampler, sampler.val_task_ids,
            inner_steps=steps, tasks_per_cell=tasks_per_cell,
            base_params=random_params, device=device,
        )
        mean, ci_lo, ci_hi = compute_ci(losses)
        results["baselines"][f"random_trunk_steps_{steps}"] = {
            "mse": mean, "ci_low": ci_lo, "ci_high": ci_hi, "n": len(losses),
        }
        ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]" if ci_lo is not None else "N/A"
        print(f"    steps={steps:2d}: MSE={mean:.4f} 95%CI={ci_str} (n={len(losses)})")

    print("\n  Meta-learned trunk + random head (isolates trunk quality):")
    random_head_params = make_random_head_params(model)
    for steps in [0, 5, 10]:
        losses = evaluate_disjoint(
            model, sampler, sampler.val_task_ids,
            inner_steps=steps, tasks_per_cell=tasks_per_cell,
            base_params=random_head_params, device=device,
        )
        mean, ci_lo, ci_hi = compute_ci(losses)
        results["baselines"][f"meta_trunk_random_head_steps_{steps}"] = {
            "mse": mean, "ci_low": ci_lo, "ci_high": ci_hi, "n": len(losses),
        }
        ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]" if ci_lo is not None else "N/A"
        print(f"    steps={steps:2d}: MSE={mean:.4f} 95%CI={ci_str} (n={len(losses)})")

    # 4. CCA / pre-post correlation
    print(f"\n{'='*60}\n  3. REPRESENTATION ANALYSIS\n{'='*60}")
    cca_result = compute_cca_similarity(
        model, sampler, sampler.val_task_ids,
        n_tasks=min(100, tasks_per_cell), device=device,
    )
    results["cca"] = cca_result
    if cca_result["mean_corr"] is not None:
        print(f"  Pre/post adaptation value correlation: "
              f"{cca_result['mean_corr']:.4f} +/- {cca_result['std_corr']:.4f} "
              f"(n={cca_result['n']})")
        print(f"  ({cca_result['note']})")

    return results


def main():
    parser = argparse.ArgumentParser(description="MAML-DASG Stockfish centipawn eval")
    parser.add_argument("--tasks-per-cell", type=int, default=600)
    parser.add_argument("--seeds", default="42,123,456,789,1337")
    parser.add_argument("--data", default=None, help=f"Path to {SF_DATA}")
    parser.add_argument("--db", default=None, help=f"Path to {SF_DB}")
    parser.add_argument("--ckpt", default=None, help=f"Path to {SF_CKPT}")
    args = parser.parse_args()

    data_dir = args.data or os.path.join(ROOT, SF_DATA)
    db_path = args.db or os.path.join(ROOT, SF_DB)
    ckpt_path = args.ckpt or os.path.join(ROOT, SF_CKPT)

    seeds = [int(s) for s in args.seeds.split(",")]

    all_results = []
    for seed in seeds:
        print(f"\n{'#'*70}\n  SEED {seed}\n{'#'*70}")
        all_results.append(run_suite(args.tasks_per_cell, seed, data_dir, db_path, ckpt_path))

    out_path = os.path.join(ROOT, "runs", "sf_eval.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print(f"\n{'#'*70}\n  CROSS-SEED SUMMARY ({len(seeds)} seeds)\n{'#'*70}")
    print(f"\n  Adaptation curve (mean across seeds):")
    print(f"    {'Steps':>5s}  {'MSE':>14s}")
    for steps in [0, 1, 3, 5, 10]:
        key = f"steps_{steps}"
        vals = [r["adaptation_curve"][key]["mse"]
                for r in all_results if r["adaptation_curve"][key]["mse"] is not None]
        m = np.mean(vals) if vals else float("inf")
        s = np.std(vals) if len(vals) > 1 else 0
        print(f"    {steps:5d}  {m:.4f}+/-{s:.4f}")

    gains = [r.get("adaptation_gain_mse") for r in all_results if r.get("adaptation_gain_mse") is not None]
    if gains:
        print(f"\n  Adaptation gain (0-step - 5-step MSE): "
              f"{np.mean(gains):+.4f} +/- {np.std(gains):.4f}")

    print(f"\n  Baselines (5 inner steps):")
    for bname in ["random_trunk_steps_5", "meta_trunk_random_head_steps_5"]:
        vals = [r["baselines"][bname]["mse"]
                for r in all_results if r["baselines"].get(bname, {}).get("mse") is not None]
        if vals:
            label = bname.replace("_steps_5", "").replace("_", " ")
            print(f"    {label:35s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    corrs = [r["cca"]["mean_corr"] for r in all_results if r["cca"]["mean_corr"] is not None]
    if corrs:
        print(f"\n  Pre/post adaptation correlation: {np.mean(corrs):.4f} +/- {np.std(corrs):.4f}")


if __name__ == "__main__":
    main()
