#!/usr/bin/env python3
"""Cross-game evaluation for the JOINT chess+shogi SF meta-learner.

Two results:
  1. Per-game opening-disjoint adaptation curve (chess vs shogi, held-out openings).
  2. Held-out-game transfer probe on shogi openings:
       - joint model (saw shogi in meta-train)
       - chess-only model (never saw shogi)  -> chess->shogi transfer
       - random-init lower bound
     If joint >> chess-only ~ random, shogi must be in meta-train (no zero-shot
     cross-game transfer). If chess-only > random, the learned-to-adapt capability
     transfers across games.

Reuses the sampler / eval machinery from eval_sf_suite.

Usage:
    cd ~/maml-dasg
    CUDA_VISIBLE_DEVICES= ~/evo-distill/.venv/bin/python eval_joint_suite.py \
        --joint runs/joint_sf_k64_s42/best.pt \
        --chess-only runs/disjoint_anil_sf_k64_s42/best.pt \
        --data processed_sf_combined --db sf_combined_openings.sqlite \
        --tasks-per-cell 400 --seeds 42,123,456
"""

import argparse
import json
import os

import numpy as np
import torch

from eval_sf_suite import (
    DisjointTaskSampler, load_model, evaluate_disjoint,
    make_random_trunk_params, compute_ci, paired_ttest,
    N_CHANNELS, MIN_GAMES_PER_OPENING, TRAIN_FRAC,
)

ROOT = os.path.dirname(os.path.abspath(__file__))


def is_chess(t):
    return t.startswith("chess_")


def is_shogi(t):
    return t.startswith("shogi_")


def curve(model, sampler, ids, tasks, base_params=None, device="cpu",
          steps=(0, 1, 3, 5, 10)):
    out = {}
    losses_by_step = {}
    for s in steps:
        losses = evaluate_disjoint(model, sampler, ids, inner_steps=s,
                                   tasks_per_cell=tasks, base_params=base_params,
                                   device=device)
        losses_by_step[s] = losses
        m, lo, hi = compute_ci(losses)
        out[f"steps_{s}"] = {"mse": m, "ci_low": lo, "ci_high": hi, "n": len(losses)}
    return out, losses_by_step


def run_seed(seed, joint_path, chess_path, data_dir, db_path, tasks, shogi_only_path=None):
    device = torch.device("cpu")
    res = {"seed": seed}

    sampler = DisjointTaskSampler(data_dir=data_dir, db_path=db_path,
                                  train_frac=TRAIN_FRAC, seed=seed,
                                  min_games=MIN_GAMES_PER_OPENING)
    val = sampler.val_task_ids
    chess_val = [t for t in val if is_chess(t)]
    shogi_val = [t for t in val if is_shogi(t)]
    res["val_counts"] = {"chess": len(chess_val), "shogi": len(shogi_val)}
    print(f"  val openings: {len(chess_val)} chess, {len(shogi_val)} shogi")

    joint, jck = load_model(joint_path, N_CHANNELS, device)
    res["joint_ckpt_val"] = jck.get("best_val_meta", None)

    # 1. Per-game opening-disjoint adaptation curve (joint model)
    print("  [1] joint per-game adaptation curve")
    res["joint_chess"], jc = curve(joint, sampler, chess_val, tasks, device=device)
    res["joint_shogi"], js = curve(joint, sampler, shogi_val, tasks, device=device)
    res["gate_chess_p"] = paired_ttest(jc[0], jc[5])
    res["gate_shogi_p"] = paired_ttest(js[0], js[5])

    # 2. Held-out-game probe on shogi openings
    print("  [2] chess->shogi transfer probe")
    probe = {}
    # joint already computed (joint_shogi steps 0/5)
    probe["joint"] = {k: res["joint_shogi"][k]["mse"] for k in ("steps_0", "steps_5")}
    # chess-only model on shogi openings
    chess_only, _ = load_model(chess_path, N_CHANNELS, device)
    co, _ = curve(chess_only, sampler, shogi_val, tasks, device=device, steps=(0, 5))
    probe["chess_only"] = {k: co[k]["mse"] for k in ("steps_0", "steps_5")}
    # random-init lower bound
    rnd = make_random_trunk_params(joint)
    rb, _ = curve(joint, sampler, shogi_val, tasks, base_params=rnd, device=device, steps=(0, 5))
    probe["random_init"] = {k: rb[k]["mse"] for k in ("steps_0", "steps_5")}
    res["shogi_transfer_probe"] = probe

    # 3. Joint vs shogi-only on the SAME held-out shogi openings (matched split).
    if shogi_only_path:
        print("  [3] joint vs shogi-only on matched shogi split")
        so_model, _ = load_model(shogi_only_path, N_CHANNELS, device)
        so, sos = curve(so_model, sampler, shogi_val, tasks, device=device)
        res["shogi_only"] = so
        res["shogi_only_paired_p"] = paired_ttest(sos[5], js[5])  # shogi-only vs joint @ 5 steps
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", required=True)
    ap.add_argument("--chess-only", default="runs/disjoint_anil_sf_k64_s42/best.pt")
    ap.add_argument("--data", default="processed_sf_combined")
    ap.add_argument("--db", default="sf_combined_openings.sqlite")
    ap.add_argument("--shogi-only", default=None, help="shogi-only ckpt for matched comparison")
    ap.add_argument("--tasks-per-cell", type=int, default=600)
    ap.add_argument("--seeds", default="42,123,456,789,1337")
    ap.add_argument("--out", default="runs/joint_eval.json")
    args = ap.parse_args()

    data_dir = os.path.join(ROOT, args.data)
    db_path = os.path.join(ROOT, args.db)
    seeds = [int(s) for s in args.seeds.split(",")]

    allr = []
    for sd in seeds:
        print(f"\n{'#'*60}\n SEED {sd}\n{'#'*60}")
        allr.append(run_seed(sd, args.joint, args.chess_only, data_dir, db_path,
                             args.tasks_per_cell, shogi_only_path=args.shogi_only))

    with open(os.path.join(ROOT, args.out), "w") as f:
        json.dump(allr, f, indent=2)

    def mean_step(key, step):
        vals = [r[key][f"steps_{step}"]["mse"] for r in allr
                if r.get(key, {}).get(f"steps_{step}", {}).get("mse") is not None]
        return (np.mean(vals), np.std(vals)) if vals else (float("nan"), 0)

    print(f"\n{'#'*60}\n CROSS-SEED SUMMARY ({len(seeds)} seeds)\n{'#'*60}")
    print("\n Joint per-game adaptation (MSE, mean+/-std):")
    print(f"   {'steps':>5} {'chess':>16} {'shogi':>16}")
    for s in (0, 1, 3, 5, 10):
        cm, cs = mean_step("joint_chess", s)
        sm, ss = mean_step("joint_shogi", s)
        print(f"   {s:>5} {cm:>8.4f}+/-{cs:.4f} {sm:>8.4f}+/-{ss:.4f}")

    print("\n Shogi transfer probe (MSE at 5 steps, mean+/-std):")
    for name in ("joint", "chess_only", "random_init"):
        vals = [r["shogi_transfer_probe"][name]["steps_5"] for r in allr
                if r.get("shogi_transfer_probe", {}).get(name)]
        if vals:
            print(f"   {name:14}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    if any("shogi_only" in r for r in allr):
        print("\n Joint vs shogi-only on matched shogi split (MSE, mean+/-std):")
        for s in (0, 5):
            jv = [r["joint_shogi"][f"steps_{s}"]["mse"] for r in allr]
            sv = [r["shogi_only"][f"steps_{s}"]["mse"] for r in allr if "shogi_only" in r]
            print(f"   steps {s}: joint {np.mean(jv):.4f}+/-{np.std(jv):.4f}  "
                  f"shogi-only {np.mean(sv):.4f}+/-{np.std(sv):.4f}")
        # paired t-test across seeds on 5-step seed-means
        jv5 = np.array([r["joint_shogi"]["steps_5"]["mse"] for r in allr])
        sv5 = np.array([r["shogi_only"]["steps_5"]["mse"] for r in allr if "shogi_only" in r])
        if len(jv5) == len(sv5) and len(jv5) >= 2:
            print(f"   joint helps shogi: delta {np.mean(sv5-jv5):+.4f}  paired-p {paired_ttest(list(sv5), list(jv5)):.2e}")
    print(f"\n Saved: {args.out}")


if __name__ == "__main__":
    main()
