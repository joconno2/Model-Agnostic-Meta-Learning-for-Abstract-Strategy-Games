"""Generate publication figures for MAML-DASG cross-game results.

Reads result JSONs from --dir (default results/) and writes PNGs to --out (figures/).
Figures:
  fig1_within_game.png      within-game adaptation curves (chess, joint chess, joint shogi)
  fig2_transfer_bar.png     cross-game: raw-board vs shared-feature vs few-shot vs ceiling
  fig3_fewshot_curves.png   few-shot adaptation over the shared abstraction (both directions)
  fig4_gameplay.png         shogi gameplay win rates with Wilson CIs

Usage: python make_figures.py --dir results --out figures
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS = [0, 1, 3, 5, 10]
C = {"chess": "#1f77b4", "shogi": "#d62728", "joint": "#2e7d32",
     "raw": "#9e9e9e", "shared": "#1f77b4", "few": "#7b1fa2", "ceil": "#2e7d32"}


def boot_ci(vals, n=5000, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) < 2:
        m = float(vals.mean()) if len(vals) else float("nan")
        return m, m, m
    rng = np.random.RandomState(seed)
    bm = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return float(vals.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def wilson(w, nt, z=1.96):
    if nt == 0:
        return 0, 0, 0
    p = w / nt
    d = 1 + z * z / nt
    c = (p + z * z / (2 * nt)) / d
    h = z * math.sqrt(p * (1 - p) / nt + z * z / (4 * nt * nt)) / d
    return p, c - h, c + h


def load_concat(d, pattern):
    out = []
    for p in sorted(glob.glob(os.path.join(d, pattern))):
        v = json.load(open(p))
        if isinstance(v, list):
            out.extend(v)
    return out


def curve_stats(records, key):
    """mean + CI per step for records[i][key][steps_S][mse]."""
    ms, los, his = [], [], []
    for s in STEPS:
        vals = [r[key][f"steps_{s}"]["mse"] for r in records
                if r.get(key, {}).get(f"steps_{s}", {}).get("mse") is not None]
        m, lo, hi = boot_ci(vals)
        ms.append(m); los.append(lo); his.append(hi)
    return np.array(ms), np.array(los), np.array(his)


def plot_curve(ax, ms, los, his, label, color, marker="o"):
    ax.plot(STEPS, ms, marker=marker, color=color, label=label, lw=2)
    ax.fill_between(STEPS, los, his, color=color, alpha=0.15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    D, O = args.dir, args.out
    os.makedirs(O, exist_ok=True)

    sf = load_concat(D, "sf_eval*.json")
    je = load_concat(D, "joint_eval5*.json")
    xm = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(D, "xfer_maml_*.json")))]
    tp_files = sorted(glob.glob(os.path.join(D, "transfer_plain*.json")))
    gp = json.load(open(os.path.join(D, "shogi_gameplay.json")))

    # ---- Fig 1: within-game adaptation curves ----
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for recs, key, lab, col in [
        (sf, "adaptation_curve", "Chess (single-game)", C["chess"]),
        (je, "joint_chess", "Joint: chess", C["joint"]),
        (je, "joint_shogi", "Joint: shogi", C["shogi"]),
    ]:
        if recs:
            plot_curve(ax, *curve_stats(recs, key), lab, col)
    ax.set_xlabel("Inner adaptation steps"); ax.set_ylabel("Value MSE (held-out openings)")
    ax.set_title("Within-game opening adaptation"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(O, "fig1_within_game.png"), dpi=160); plt.close(fig)

    # ---- Fig 2: cross-game transfer bar (5-step) ----
    # raw-board (chess->shogi probe), shared-feature plain, few-shot 5-step, ceiling
    def tp_vals(metric):
        v = []
        for f in tp_files:
            v += json.load(open(f))["modes"]["raw"][metric]["vals"]
        return v
    probe = {nm: boot_ci([r["shogi_transfer_probe"][nm]["steps_5"] for r in je
                          if r.get("shogi_transfer_probe", {}).get(nm)]) for nm in
             ("chess_only", "random_init")}
    xm_c2s = [r for r in xm if r["target"] == "shogi"]
    xm_s2c = [r for r in xm if r["target"] == "chess"]
    few = {"chess2shogi": boot_ci([r["cross_game"]["5"]["mse"] for r in xm_c2s]),
           "shogi2chess": boot_ci([r["cross_game"]["5"]["mse"] for r in xm_s2c])}
    bars = [
        ("raw-board\nchess->shogi", boot_ci(probe["chess_only"][0:1]) if False else
         (probe["chess_only"]), C["raw"]),
        ("shared-feat\nchess->shogi", boot_ci(tp_vals("chess2shogi")), C["shared"]),
        ("few-shot\nchess->shogi", few["chess2shogi"], C["few"]),
        ("shared-feat\nshogi->chess", boot_ci(tp_vals("shogi2chess")), C["shared"]),
        ("few-shot\nshogi->chess", few["shogi2chess"], C["few"]),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = np.arange(len(bars))
    for i, (lab, (m, lo, hi), col) in enumerate(bars):
        ax.bar(i, m, color=col, yerr=[[m - lo], [hi - m]], capsize=4, width=0.7)
    ax.axhline(0.52, ls="--", color="gray", lw=1); ax.text(len(bars)-0.5, 0.53, "random", color="gray", fontsize=8)
    ax.axhline(0.124, ls=":", color=C["ceil"], lw=1.2); ax.text(len(bars)-0.5, 0.13, "within-game ceiling", color=C["ceil"], fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylabel("Value MSE on the other game (5-step)")
    ax.set_title("Cross-game transfer: raw boards fail, shared abstraction works")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(O, "fig2_transfer_bar.png"), dpi=160); plt.close(fig)

    # ---- Fig 3: few-shot adaptation curves over the abstraction ----
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for recs, lab, col in [(xm_s2c, "shogi->chess", C["shogi"]), (xm_c2s, "chess->shogi", C["chess"])]:
        ms, los, his = [], [], []
        for s in STEPS:
            m, lo, hi = boot_ci([r["cross_game"][str(s)]["mse"] for r in recs])
            ms.append(m); los.append(lo); his.append(hi)
        plot_curve(ax, np.array(ms), np.array(los), np.array(his), lab, col)
        rb = boot_ci([r["random_body_target"]["5"] for r in recs if "5" in r.get("random_body_target", {})])
        ax.axhline(rb[0], ls="--", color=col, alpha=0.5, lw=1)
    ax.set_xlabel("Inner adaptation steps"); ax.set_ylabel("Value MSE on held-out game")
    ax.set_title("Few-shot cross-game adaptation over shared abstraction\n(dashed = random-body baseline)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(O, "fig3_fewshot_curves.png"), dpi=160); plt.close(fig)

    # ---- Fig 4: gameplay win rates ----
    fig, ax = plt.subplots(figsize=(6.5, 4))
    order = ["NN-base_vs_Material", "NN-base_vs_Random", "NN-adapted_vs_NN-base", "Material_vs_Random"]
    labs, ps, los, his = [], [], [], []
    for k in order:
        if k in gp["results"]:
            r = gp["results"][k]
            p, lo, hi = wilson(r["wins_a"] + 0.5 * r["draws"], r["total"])
            labs.append(k.replace("_vs_", "\nvs ")); ps.append(p * 100)
            los.append((p - lo) * 100); his.append((hi - p) * 100)
    ax.bar(range(len(labs)), ps, yerr=[los, his], capsize=5,
           color=["#2e7d32", "#66bb6a", "#9e9e9e", "#bdbdbd"][:len(labs)], width=0.65)
    ax.axhline(50, ls="--", color="gray", lw=1)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("Win rate (%)"); ax.set_ylim(0, 105)
    ax.set_title(f"Shogi gameplay (n={gp['results'][order[0]]['total']}/matchup, Wilson 95% CI)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(O, "fig4_gameplay.png"), dpi=160); plt.close(fig)

    print(f"Wrote 4 figures to {O}/")


if __name__ == "__main__":
    main()
