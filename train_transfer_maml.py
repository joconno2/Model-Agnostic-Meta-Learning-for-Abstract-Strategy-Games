"""Spec A1+: meta-learning over the shared abstraction, with cross-game few-shot adaptation.

Tasks = openings. A small head (body 8->64->64, adapted final linear 64->1, tanh) is
meta-trained with ANIL on ONE game's openings. We then few-shot adapt to the OTHER
game's openings and ask: does adaptation close the zero-shot transfer gap
(chess->shogi 0.25 zero-shot vs 0.13 within-game ceiling)?

This turns "transferable shared features" into "meta-learning over a shared abstraction":
the body learns a representation good for fast adaptation; the head adapts per opening,
across games.

Compares, on the held-out game's openings:
  steps 0 (zero-shot transfer) / 1 / 3 / 5 / 10 inner steps
  vs random-body lower bound, vs the target game's own meta-head (ceiling).

Usage:
  python train_transfer_maml.py --train-game chess \
      --chess feat_chess.npz --shogi feat_shogi.npz \
      --chess-db sf_openings.sqlite --shogi-db sf_shogi_openings.sqlite \
      --meta-iters 3000
"""

import argparse
import sqlite3
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call

# Cap intra-op threads: these runs are launched many-wide; without this each
# process grabs all cores and oversubscribes the box (load >> ncores).
torch.set_num_threads(2)

INNER_LR = 0.05
K_SUPPORT = 32
K_QUERY = 32
MIN_POS = 64


def load_game(npz, db):
    d = np.load(npz)
    X, y, g = d["X"].astype(np.float32), d["y_value"].astype(np.float32), d["game_id"].astype(np.int64)
    conn = sqlite3.connect(db)
    gid2eco = {int(i): e for i, e in conn.execute("SELECT id, eco FROM games")}
    conn.close()
    by_open = defaultdict(list)
    for idx, gid in enumerate(g):
        e = gid2eco.get(int(gid))
        if e is not None:
            by_open[e].append(idx)
    # keep openings with enough positions
    openings = {e: np.array(ix) for e, ix in by_open.items() if len(ix) >= MIN_POS}
    return X, y, openings


def split_openings(openings, frac=0.8, seed=42):
    rng = np.random.RandomState(seed)
    keys = sorted(openings.keys())
    rng.shuffle(keys)
    k = int(len(keys) * frac)
    return keys[:k], keys[k:]


class Net(nn.Module):
    def __init__(self, d=8, h=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        return torch.tanh(self.head(self.body(x))).squeeze(-1)


def sample_task(X, y, openings, keys, rng):
    e = keys[rng.randint(len(keys))]
    idx = openings[e]
    pick = rng.choice(len(idx), min(K_SUPPORT + K_QUERY, len(idx)), replace=False)
    sup, qry = pick[:K_SUPPORT], pick[K_SUPPORT:K_SUPPORT + K_QUERY]
    s, q = idx[sup], idx[qry]
    return (torch.tensor(X[s]), torch.tensor(y[s]), torch.tensor(X[q]), torch.tensor(y[q]))


def adapt_head(model, base_params, sX, sy, steps, lr=INNER_LR, create_graph=True):
    """ANIL: adapt only head.* on support. create_graph=True for meta-train
    (second-order, grad flows to body+head-init); False for eval (first-order)."""
    head_names = sorted(n for n in base_params if n.startswith("head."))
    fast = dict(base_params)
    for _ in range(steps):
        if not create_graph:
            for n in head_names:
                fast[n] = fast[n].detach().requires_grad_(True)
        pred = functional_call(model, fast, (sX,))
        loss = ((pred - sy) ** 2).mean()
        grads = torch.autograd.grad(loss, [fast[n] for n in head_names], create_graph=create_graph)
        for n, gr in zip(head_names, grads):
            fast[n] = fast[n] - lr * gr
    return fast


def meta_train(X, y, openings, train_keys, meta_iters, seed=42, mbs=32, inner=5):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = Net(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for it in range(meta_iters):
        opt.zero_grad()
        base = {n: p for n, p in model.named_parameters()}
        meta_loss = 0.0
        for _ in range(mbs):
            sX, sy, qX, qy = sample_task(X, y, openings, train_keys, rng)
            fast = adapt_head(model, base, sX, sy, inner)
            qp = functional_call(model, fast, (qX,))
            meta_loss = meta_loss + ((qp - qy) ** 2).mean()
        meta_loss = meta_loss / mbs
        meta_loss.backward()
        opt.step()
        if (it + 1) % 500 == 0:
            print(f"  meta-iter {it+1}: meta_loss {meta_loss.item():.4f}")
    return model


def eval_curve(model, X, y, openings, keys, steps_list=(0, 1, 3, 5, 10), n_tasks=300, seed=7):
    rng = np.random.RandomState(seed)
    base = {n: p.detach() for n, p in model.named_parameters()}
    out = {}
    for steps in steps_list:
        losses = []
        for _ in range(n_tasks):
            sX, sy, qX, qy = sample_task(X, y, openings, keys, rng)
            fast = adapt_head(model, base, sX, sy, steps, create_graph=False) if steps > 0 else base
            with torch.no_grad():
                qp = functional_call(model, fast, (qX,))
                losses.append(float(((qp - qy) ** 2).mean()))
        out[steps] = (float(np.mean(losses)), float(np.std(losses) / np.sqrt(len(losses))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-game", choices=["chess", "shogi"], required=True)
    ap.add_argument("--chess", default="feat_chess.npz")
    ap.add_argument("--shogi", default="feat_shogi.npz")
    ap.add_argument("--chess-db", default="sf_openings.sqlite")
    ap.add_argument("--shogi-db", default="sf_shogi_openings.sqlite")
    ap.add_argument("--meta-iters", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Xc, yc, oc = load_game(args.chess, args.chess_db)
    Xs, ys, os_ = load_game(args.shogi, args.shogi_db)
    print(f"chess openings {len(oc)}  shogi openings {len(os_)}")

    if args.train_game == "chess":
        Xtr, ytr, otr = Xc, yc, oc
        Xte, yte, ote = Xs, ys, os_
        tgt = "shogi"
    else:
        Xtr, ytr, otr = Xs, ys, os_
        Xte, yte, ote = Xc, yc, oc
        tgt = "chess"

    tr_keys, val_keys = split_openings(otr, seed=args.seed)
    te_keys, _ = split_openings(ote, frac=1.0, seed=args.seed)  # all target openings for cross eval

    print(f"\nMeta-training ANIL head on {args.train_game} ({len(tr_keys)} openings) seed={args.seed}...")
    model = meta_train(Xtr, ytr, otr, tr_keys, args.meta_iters, seed=args.seed)

    print(f"\n[within-game] {args.train_game} held-out openings adaptation curve:")
    wc = eval_curve(model, Xtr, ytr, otr, val_keys)
    for s, (m, se) in wc.items():
        print(f"  steps {s:2d}: MSE {m:.4f} +/- {se:.4f}")

    print(f"\n[CROSS-GAME] {args.train_game} meta-head -> {tgt} openings adaptation curve:")
    cc = eval_curve(model, Xte, yte, ote, te_keys)
    for s, (m, se) in cc.items():
        print(f"  steps {s:2d}: MSE {m:.4f} +/- {se:.4f}")

    # random-body lower bound on target
    rnd = Net(Xtr.shape[1])
    rc = eval_curve(rnd, Xte, yte, ote, te_keys, steps_list=(0, 5))
    print(f"\n[lower bound] random-init body -> {tgt}: steps0 {rc[0][0]:.4f}  steps5 {rc[5][0]:.4f}")

    z, a = cc[0][0], cc[5][0]
    print(f"\nVERDICT: {args.train_game}->{tgt} zero-shot {z:.4f} -> 5-step {a:.4f}  "
          f"(delta {z-a:+.4f}); within-game ceiling {wc[5][0]:.4f}")

    import json, os
    rec = {"train_game": args.train_game, "target": tgt, "seed": args.seed,
           "meta_iters": args.meta_iters,
           "within_game": {str(s): {"mse": m, "se": se} for s, (m, se) in wc.items()},
           "cross_game": {str(s): {"mse": m, "se": se} for s, (m, se) in cc.items()},
           "random_body_target": {str(s): rc[s][0] for s in rc}}
    out = os.path.join("runs", f"xfer_maml_{args.train_game}2{tgt}_s{args.seed}.json")
    os.makedirs("runs", exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
