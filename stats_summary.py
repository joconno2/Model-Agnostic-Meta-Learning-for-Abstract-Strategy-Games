"""Assemble all MAML-DASG cross-game results into one clean table with CIs + tests.

Ingests result JSONs from a directory (gather them from DGX/trx first):
  sf_eval.json            chess within-game adaptation (5 seeds)
  joint_eval5.json        joint per-game + transfer probe + joint-vs-shogi-only (5 seeds)
  transfer_plain.json     plain shared-feature transfer (5 seeds)
  xfer_maml_*2*_s*.json   few-shot cross-game adaptation (per seed/direction)
  shogi_gameplay.json     shogi gameplay win rates

Stats: mean +/- 95% bootstrap CI for MSE (from per-seed values), paired t across seeds
for model comparisons, Wilson 95% CI for win rates.

Usage: python stats_summary.py --dir runs_all
"""

import argparse
import glob
import json
import math
import os

import numpy as np


def boot_ci(vals, n=5000, seed=0):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        return (float(vals.mean()), float(vals.mean()), float(vals.mean())) if len(vals) else (float("nan"),) * 3
    rng = np.random.RandomState(seed)
    bm = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return float(vals.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def paired_t(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return float("nan")
    t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
    try:
        from scipy import stats
        return float(stats.t.sf(abs(t), n - 1) * 2)
    except ImportError:
        return float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))


def wilson(w, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def fmt(m, lo, hi):
    return f"{m:.4f} [{lo:.4f}, {hi:.4f}]"


def load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def load_concat(d, pattern):
    """Concatenate per-seed list JSONs matching pattern (e.g. sf_eval*.json)."""
    out = []
    for p in sorted(glob.glob(os.path.join(d, pattern))):
        v = json.load(open(p))
        if isinstance(v, list):
            out.extend(v)
    return out or None


def load_transfer(d):
    """Merge transfer_plain*.json: concat per-metric 'vals' across files."""
    files = sorted(glob.glob(os.path.join(d, "transfer_plain*.json")))
    if not files:
        return None
    base = json.load(open(files[0]))
    for p in files[1:]:
        x = json.load(open(p))
        for mode in base["modes"]:
            for k in base["modes"][mode]:
                base["modes"][mode][k]["vals"] += x["modes"][mode][k]["vals"]
        base["seeds"] += x.get("seeds", [])
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs_all")
    args = ap.parse_args()
    D = args.dir

    print("=" * 70)
    print("  MAML-DASG CROSS-GAME RESULTS  (mean [95% CI])")
    print("=" * 70)

    # E1 chess within-game (sf_eval*.json: list of per-seed dicts)
    sf = load_concat(D, "sf_eval*.json")
    if sf:
        print(f"\n[E1] Chess within-game adaptation (held-out openings) [{len(sf)} seeds]:")
        for s in (0, 1, 3, 5, 10):
            vals = [r["adaptation_curve"][f"steps_{s}"]["mse"] for r in sf]
            print(f"   steps {s:2d}: {fmt(*boot_ci(vals))}")

    # E2/E3/E4/E5 joint_eval5*.json
    je = load_concat(D, "joint_eval5*.json")
    if je:
        print(f"\n[E3] Joint per-game adaptation [{len(je)} seeds]:")
        for game in ("joint_chess", "joint_shogi"):
            print(f"   {game}:")
            for s in (0, 1, 3, 5, 10):
                vals = [r[game][f"steps_{s}"]["mse"] for r in je]
                print(f"     steps {s:2d}: {fmt(*boot_ci(vals))}")
        print("\n[E4] Chess->shogi transfer probe (5-step MSE):")
        for nm in ("joint", "chess_only", "random_init"):
            vals = [r["shogi_transfer_probe"][nm]["steps_5"] for r in je if r.get("shogi_transfer_probe", {}).get(nm)]
            if vals:
                print(f"   {nm:12}: {fmt(*boot_ci(vals))}")
        if any("shogi_only" in r for r in je):
            print("\n[E5] Joint vs shogi-only on matched shogi split (5-step):")
            jv = [r["joint_shogi"]["steps_5"]["mse"] for r in je]
            sv = [r["shogi_only"]["steps_5"]["mse"] for r in je if "shogi_only" in r]
            print(f"   joint      : {fmt(*boot_ci(jv))}")
            print(f"   shogi-only : {fmt(*boot_ci(sv))}")
            print(f"   joint helps shogi: delta {np.mean(np.array(sv)-np.array(jv)):+.4f}  paired-p {paired_t(sv, jv):.2e}")

    # E6 plain transfer
    tp = load_transfer(D)
    if tp:
        n = len(tp["modes"]["raw"]["chess2shogi"]["vals"])
        print(f"\n[E6] Plain shared-feature transfer (raw features, {n} seeds):")
        m = tp["modes"]["raw"]
        print(f"   variance baselines: chess {tp['variance']['chess']:.3f}  shogi {tp['variance']['shogi']:.3f}")
        for k in ("chess_within", "shogi_within", "chess2shogi", "shogi2chess"):
            print(f"   {k:14}: {fmt(*boot_ci(m[k]['vals']))}")

    # E7 few-shot cross-game adaptation
    xm = sorted(glob.glob(os.path.join(D, "xfer_maml_*.json")))
    if xm:
        print("\n[E7] Few-shot cross-game adaptation (meta-learned shared abstraction):")
        by_dir = {}
        for p in xm:
            r = json.load(open(p))
            by_dir.setdefault(f"{r['train_game']}->{r['target']}", []).append(r)
        for d, recs in by_dir.items():
            print(f"   {d}  ({len(recs)} seeds):")
            for s in ("0", "1", "3", "5", "10"):
                vals = [rec["cross_game"][s]["mse"] for rec in recs if s in rec["cross_game"]]
                if vals:
                    print(f"     steps {s:>2}: {fmt(*boot_ci(vals))}")
            rb = [rec["random_body_target"]["5"] for rec in recs if "5" in rec.get("random_body_target", {})]
            if rb:
                print(f"     random-body 5-step: {fmt(*boot_ci(rb))}")

    # E8 gameplay
    gp = load(D, "shogi_gameplay.json")
    if gp:
        print("\n[E8] Shogi gameplay (win rate, Wilson 95% CI):")
        for key, r in gp["results"].items():
            w = r["wins_a"] + 0.5 * r["draws"]
            p, lo, hi = wilson(w, r["total"])
            print(f"   {key:28}: {p:.1%} [{lo:.1%}, {hi:.1%}]  (n={r['total']})")


if __name__ == "__main__":
    main()
