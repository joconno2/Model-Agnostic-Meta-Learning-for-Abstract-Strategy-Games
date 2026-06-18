#!/usr/bin/env python3
"""
Supervised baseline: same ValueNet architecture, standard training, no meta-learning.

Trains on all chess+shogi positions with plain MSE loss. No inner/outer loop,
no task structure. This is the "multi-task learning without meta-learning"
baseline that reviewer 65 requested.

Also supports chess-only and shogi-only modes for single-game baselines.

Usage:
    cd ~/code/maml-dasg
    python train_supervised.py --data-dir processed_combined_flat --out-dir runs/supervised_combined
    python train_supervised.py --data-dir processed_chess_flat --out-dir runs/supervised_chess
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from spec import UnifiedSpec, num_channels
from model_v2 import ValueNet

ROOT = os.path.dirname(os.path.abspath(__file__))


class PositionDataset(Dataset):
    """Flat dataset of all positions from .npz shards."""

    def __init__(self, data_dir, split="train", train_frac=0.8, seed=42):
        shard_files = sorted(
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")
        )
        if not shard_files:
            raise RuntimeError(f"No .npz shards in {data_dir}")

        X_parts, yv_parts = [], []
        for path in shard_files:
            with np.load(path) as d:
                X_parts.append(d["X"])
                yv_parts.append(d["y_value"])

        X_all = np.concatenate(X_parts, axis=0)
        yv_all = np.concatenate(yv_parts, axis=0)

        # Deterministic split by index
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(X_all))
        split_idx = int(len(indices) * train_frac)

        if split == "train":
            idx = indices[:split_idx]
        else:
            idx = indices[split_idx:]

        self.X = X_all[idx]
        self.yv = yv_all[idx]
        print(f"[PositionDataset] {split}: {len(self.X)} positions")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.float32),
            torch.tensor(self.yv[i], dtype=torch.float32),
        )


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for X, yv in loader:
            X, yv = X.to(device), yv.to(device)
            pred = model(X)
            loss = nn.functional.mse_loss(pred, yv, reduction="sum")
            total_loss += loss.item()
            n += len(yv)
    return total_loss / n if n > 0 else float("inf")


def main():
    parser = argparse.ArgumentParser(description="Supervised baseline for MAML-DASG")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="./runs/supervised")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--trunk-hidden", type=int, default=64)
    parser.add_argument("--bottleneck-dim", type=int, default=64)
    parser.add_argument("--value-hidden", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = UnifiedSpec()
    C = num_channels(spec)

    # Data
    train_ds = PositionDataset(args.data_dir, split="train", train_frac=args.train_frac, seed=args.seed)
    val_ds = PositionDataset(args.data_dir, split="val", train_frac=args.train_frac, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model (identical architecture to ANIL)
    model = ValueNet(
        in_channels=C,
        trunk_hidden=args.trunk_hidden,
        bottleneck_dim=args.bottleneck_dim,
        value_hidden=args.value_hidden,
    ).to(device)
    print(f"Model: {model.total_param_count():,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-4)

    # Save config
    config = vars(args)
    config["n_channels"] = C
    config["total_params"] = model.total_param_count()
    config["train_size"] = len(train_ds)
    config["val_size"] = len(val_ds)
    with open(os.path.join(args.out_dir, "config.txt"), "w") as f:
        for k, v in sorted(config.items()):
            f.write(f"{k}={v}\n")

    # Training loop
    train_hist = []
    val_hist = []
    best_val = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_n = 0

        for X, yv in train_loader:
            X, yv = X.to(device), yv.to(device)
            pred = model(X)
            loss = nn.functional.mse_loss(pred, yv)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item() * len(yv)
            epoch_n += len(yv)

        train_mse = epoch_loss / epoch_n
        val_mse = evaluate(model, val_loader, device)
        train_hist.append(train_mse)
        val_hist.append(val_mse)

        improved = ""
        if val_mse < best_val:
            best_val = val_mse
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_mse": best_val,
                "config": config,
                "train_history": train_hist,
                "val_history": val_hist,
            }, os.path.join(args.out_dir, "best.pt"))
            improved = " *"

        elapsed = time.time() - t0
        print(f"[epoch {epoch:3d}] train={train_mse:.4f} val={val_mse:.4f} "
              f"best={best_val:.4f} ({elapsed:.0f}s){improved}")

    # Final save
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_mse": best_val,
        "config": config,
        "train_history": train_hist,
        "val_history": val_hist,
    }, os.path.join(args.out_dir, "final.pt"))

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_hist) + 1), train_hist, label="Train MSE")
    plt.plot(range(1, len(val_hist) + 1), val_hist, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Supervised Baseline")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "loss.png"), dpi=200)
    plt.close()

    print(f"\nDone. Best val MSE: {best_val:.4f}")
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
