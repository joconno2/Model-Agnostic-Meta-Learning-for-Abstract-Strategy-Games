#!/usr/bin/env python3
"""
Ablation study: inner-loop step count and support set size.

For each checkpoint (opening_v1, cross_game_v1), evaluates on validation
tasks with varying inner_steps and k_support. Reports mean MSE on query sets.

Run from the project root with the local .venv:
    .venv/bin/python ablation_steps_support.py
"""

import json
import os
import sys
import time
from itertools import product

import numpy as np
import torch
from torch.func import functional_call

# Project imports
from spec import UnifiedSpec, num_channels
from model_v2 import ValueNet
from task_sampler_v2 import ValueTaskSampler
from maml_anil import inner_adapt_anil, value_loss


# ---------- Config ----------

RUNS = {
    "opening_v1": {
        "ckpt": "runs/opening_v1/best.pt",
        "data_dir": "processed_chess_flat",
        "db_path": "combined_openings.sqlite",
        "best_val": 0.0366,
    },
    "cross_game_v1": {
        "ckpt": "runs/cross_game_v1/best.pt",
        "data_dir": "processed_combined_flat",
        "db_path": "combined_openings.sqlite",
        "best_val": 0.0516,
    },
}

INNER_STEPS = [1, 2, 3, 5, 7, 10]
K_SUPPORTS = [4, 8, 16, 32]
K_QUERY = 16
INNER_LR = 0.005
VAL_TASKS = 128
SEED = 42

# Architecture (same for both runs)
TRUNK_HIDDEN = 64
BOTTLENECK_DIM = 64
VALUE_HIDDEN = 64
TRAIN_FRAC = 0.8
MIN_POSITIONS = 32

ROOT = os.path.dirname(os.path.abspath(__file__))


def evaluate(model, sampler, k_support, k_query, inner_lr, inner_steps, n_tasks, device):
    """Run ANIL inner loop on val tasks and compute mean query MSE."""
    head_param_names = set(n for n, _ in model.head_params())
    base_params = {n: p.detach().clone() for n, p in model.named_parameters()}

    losses = []
    for _ in range(n_tasks):
        try:
            sX, sy_val, qX, qy_val, task_id = sampler.sample_task(
                k_support, k_query, split="val"
            )
        except RuntimeError:
            continue

        sX_t = torch.tensor(sX, dtype=torch.float32, device=device)
        sy_t = torch.tensor(sy_val, dtype=torch.float32, device=device)
        qX_t = torch.tensor(qX, dtype=torch.float32, device=device)
        qy_t = torch.tensor(qy_val, dtype=torch.float32, device=device)

        adapted = inner_adapt_anil(
            model=model,
            base_params=base_params,
            head_param_names=head_param_names,
            Xs=sX_t,
            ys_val=sy_t,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
        )

        with torch.no_grad():
            adapted_d = {k: v.detach() for k, v in adapted.items()}
            q_v = functional_call(model, adapted_d, (qX_t,))
            loss = value_loss(q_v, qy_t)

        losses.append(loss.item())

    return float(np.mean(losses)) if losses else float("inf")


def main():
    device = torch.device("cpu")
    spec = UnifiedSpec()
    C = num_channels(spec)

    all_results = {}

    for run_name, cfg in RUNS.items():
        print(f"\n{'='*60}")
        print(f"Run: {run_name}")
        print(f"{'='*60}")

        ckpt_path = os.path.join(ROOT, cfg["ckpt"])
        data_dir = os.path.join(ROOT, cfg["data_dir"])
        db_path = os.path.join(ROOT, cfg["db_path"])

        # Load model
        model = ValueNet(
            in_channels=C,
            trunk_hidden=TRUNK_HIDDEN,
            bottleneck_dim=BOTTLENECK_DIM,
            value_hidden=VALUE_HIDDEN,
        ).to(device)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        iteration = ckpt.get("iteration", "?")
        best_val_ckpt = ckpt.get("best_val_meta", "?")
        print(f"  Loaded checkpoint at iteration {iteration}, best_val={best_val_ckpt}")

        # Build sampler
        # We need to find the max k_support + k_query needed so min_positions filter works.
        max_total = max(K_SUPPORTS) + K_QUERY  # 32 + 16 = 48
        sampler = ValueTaskSampler(
            data_dir=data_dir,
            db_path=db_path,
            task_mode="opening",
            train_frac=TRAIN_FRAC,
            seed=SEED,
            min_positions_per_task=max_total,
        )

        print(f"  Val tasks available: {len(sampler.val_task_ids)}")
        print()

        run_results = {}
        for steps in INNER_STEPS:
            for k_sup in K_SUPPORTS:
                t0 = time.time()
                mse = evaluate(
                    model, sampler,
                    k_support=k_sup,
                    k_query=K_QUERY,
                    inner_lr=INNER_LR,
                    inner_steps=steps,
                    n_tasks=VAL_TASKS,
                    device=device,
                )
                elapsed = time.time() - t0
                key = f"steps={steps}_k={k_sup}"
                run_results[key] = {
                    "inner_steps": steps,
                    "k_support": k_sup,
                    "val_mse": round(mse, 6),
                    "elapsed_s": round(elapsed, 1),
                }
                marker = " <-- training default" if steps == 5 and k_sup == 16 else ""
                print(f"  steps={steps:2d}  k_support={k_sup:2d}  val_MSE={mse:.6f}  ({elapsed:.1f}s){marker}")

        all_results[run_name] = run_results

    # Save JSON
    out_json = os.path.join(ROOT, "runs", "ablation_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved JSON: {out_json}")

    # Print summary table
    print_summary_table(all_results)

    # Save markdown
    save_markdown_table(all_results)


def print_summary_table(results):
    """Print a formatted summary table to stdout."""
    for run_name, run_data in results.items():
        print(f"\n{'='*60}")
        print(f"  {run_name}")
        print(f"{'='*60}")

        # Build table
        header = f"{'steps':>5s}"
        for k in K_SUPPORTS:
            header += f"  k={k:>2d}"
        print(header)
        print("-" * len(header))

        for steps in INNER_STEPS:
            row = f"{steps:5d}"
            for k in K_SUPPORTS:
                key = f"steps={steps}_k={k}"
                mse = run_data[key]["val_mse"]
                row += f"  {mse:.4f}"
            print(row)


def save_markdown_table(results):
    """Save results as a markdown file for the vault."""
    md_path = os.path.expanduser(
        "~/Documents/Work/Work/Research/Projects/MAML-DASG/ablation_results.md"
    )
    os.makedirs(os.path.dirname(md_path), exist_ok=True)

    lines = [
        "# MAML-DASG Ablation: Inner Steps x Support Set Size",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "Evaluation: 128 val tasks per cell, k_query=16, inner_lr=0.005",
        "",
        "Training default: inner_steps=5, k_support=16",
        "",
    ]

    for run_name, run_data in results.items():
        cfg = RUNS[run_name]
        lines.append(f"## {run_name}")
        lines.append("")
        lines.append(f"- Checkpoint: `{cfg['ckpt']}`")
        lines.append(f"- Training best val MSE: {cfg['best_val']}")
        lines.append(f"- Data: `{cfg['data_dir']}`")
        lines.append("")

        # Markdown table
        header = "| Steps |"
        sep = "|-------|"
        for k in K_SUPPORTS:
            header += f" k={k} |"
            sep += "------|"
        lines.append(header)
        lines.append(sep)

        for steps in INNER_STEPS:
            row = f"| {steps} |"
            for k in K_SUPPORTS:
                key = f"steps={steps}_k={k}"
                mse = run_data[key]["val_mse"]
                marker = ""
                if steps == 5 and k == 16:
                    marker = " *"
                row += f" {mse:.4f}{marker} |"
            lines.append(row)

        lines.append("")
        lines.append("\\* = training default")
        lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved markdown: {md_path}")


if __name__ == "__main__":
    main()
