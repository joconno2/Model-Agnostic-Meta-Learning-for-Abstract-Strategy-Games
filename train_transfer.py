"""Spec A1: cross-game value transfer via shared game-general features.

Trains a small MLP value head on one game's shared features and tests it zero-shot
on the other game. The control is our raw-board result (chess-only on shogi = 0.749
MSE, worse than random 0.536). If a head on shared features transfers (chess->shogi
MSE near shogi's own within-game head), that confirms: transfer needs a shared
abstraction, not raw boards.

Reports for both raw-invariant features and per-game-standardized features.

Usage:
  python train_transfer.py --chess feat_chess.npz --shogi feat_shogi.npz
"""

import argparse
import numpy as np
import torch
import torch.nn as nn


def load(path):
    d = np.load(path)
    return d["X"].astype(np.float32), d["y_value"].astype(np.float32), d["game_id"].astype(np.int64)


def split_by_game(gid, frac=0.8, seed=42):
    rng = np.random.RandomState(seed)
    games = np.unique(gid)
    rng.shuffle(games)
    k = int(len(games) * frac)
    train_g = set(games[:k].tolist())
    tr = np.array([g in train_g for g in gid])
    return tr, ~tr


class Head(nn.Module):
    def __init__(self, d=8, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return torch.tanh(self.net(x)).squeeze(-1)


def train_head(X, y, epochs=30, bs=4096, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    model = Head(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xt = torch.tensor(X); yt = torch.tensor(y)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    return model


def mse(model, X, y):
    with torch.no_grad():
        p = model(torch.tensor(X))
        return float(((p - torch.tensor(y)) ** 2).mean())


def standardize(Xtr, *others):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    return [( (A - mu) / sd ).astype(np.float32) for A in (Xtr,) + others]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chess", default="feat_chess.npz")
    ap.add_argument("--shogi", default="feat_shogi.npz")
    args = ap.parse_args()

    Xc, yc, gc = load(args.chess)
    Xs, ys, gs = load(args.shogi)
    print(f"chess {Xc.shape}  shogi {Xs.shape}")
    print(f"variance (predict-mean MSE): chess {float((yc**2).mean()):.4f}  shogi {float((ys**2).mean()):.4f}")

    ctr, cval = split_by_game(gc)
    str_, sval = split_by_game(gs)

    for mode in ("raw", "per-game-standardized"):
        print(f"\n===== {mode} =====")
        if mode == "raw":
            Xc_tr, Xc_va = Xc[ctr], Xc[cval]
            Xs_tr, Xs_va = Xs[str_], Xs[sval]
            Xc_all, Xs_all = Xc, Xs
        else:
            Xc_tr, Xc_va, Xc_all = standardize(Xc[ctr], Xc[cval], Xc)
            Xs_tr, Xs_va, Xs_all = standardize(Xs[str_], Xs[sval], Xs)

        chess_head = train_head(Xc_tr, yc[ctr])
        shogi_head = train_head(Xs_tr, ys[str_])

        print(f"  chess head -> chess val : {mse(chess_head, Xc_va, yc[cval]):.4f}  (within-game)")
        print(f"  shogi head -> shogi val : {mse(shogi_head, Xs_va, ys[sval]):.4f}  (within-game)")
        print(f"  chess head -> SHOGI all : {mse(chess_head, Xs_all, ys):.4f}  (TRANSFER; raw-board was 0.749)")
        print(f"  shogi head -> CHESS all : {mse(shogi_head, Xc_all, yc):.4f}  (TRANSFER)")


if __name__ == "__main__":
    main()
