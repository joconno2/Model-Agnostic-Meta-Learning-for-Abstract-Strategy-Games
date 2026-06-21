"""Diagnostic: does opening adaptation change the greedy (1-ply) MOVE, or only
the value CALIBRATION? Measures base-vs-adapted move agreement over held-out
opening positions. High agreement explains a ~50% adapted-vs-base gameplay result:
adaptation lowers value MSE without changing move rankings.

Usage:
    python diag_move_agreement_shogi.py --checkpoint runs/shogi_sf_k64_s42/best.pt \
        --data processed_sf_shogi --openings sf_shogi_openings.sqlite --positions 600
"""
import argparse, glob, os, sqlite3
from collections import defaultdict
import numpy as np
import torch
from torch.func import functional_call
import shogi

from spec import UnifiedSpec, num_channels
from model_v2 import ValueNet
from encode_shogi import encode_sfen_to_unified
from maml_anil import inner_adapt_anil

ROOT = os.path.dirname(os.path.abspath(__file__))
TH, BD, VH, ILR = 64, 64, 64, 0.005


def greedy_move(board, model, params, spec, device):
    moves = list(board.legal_moves)
    if not moves:
        return None
    tens, mvs = [], []
    for mv in moves:
        board.push(mv)
        if board.is_game_over():
            board.pop(); return mv
        parts = board.sfen().split(" ")
        x, _ = encode_sfen_to_unified(parts[0], parts[1], parts[2] if len(parts) > 2 else "-", spec)
        tens.append(x); mvs.append(mv); board.pop()
    X = torch.tensor(np.stack(tens), dtype=torch.float32, device=device)
    with torch.no_grad():
        v = functional_call(model, params, (X,)).numpy()
    return mvs[int(np.argmin(v))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="processed_sf_shogi")
    ap.add_argument("--openings", default="sf_shogi_openings.sqlite")
    ap.add_argument("--positions", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    spec = UnifiedSpec(); C = num_channels(spec)

    model = ValueNet(in_channels=C, trunk_hidden=TH, bottleneck_dim=BD, value_hidden=VH)
    ck = torch.load(os.path.join(ROOT, args.checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    head_names = set(n for n, _ in model.head_params())

    # held-out openings with >=10 games
    conn = sqlite3.connect(os.path.join(ROOT, args.openings))
    eco_games = defaultdict(list)
    for gid, eco in conn.execute("SELECT id, eco FROM games"):
        eco_games[eco].append(int(gid))
    conn.close()
    ecos = sorted([e for e, g in eco_games.items() if len(g) >= 10])
    rng.shuffle(ecos)
    val_ecos = ecos[int(0.8 * len(ecos)):]  # held-out fold
    val_gids = set(g for e in val_ecos for g in eco_games[e])
    gid_eco = {g: e for e in val_ecos for g in eco_games[e]}

    # collect val positions (X,y) per opening from shards
    byX, byy = defaultdict(list), defaultdict(list)
    for p in sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz"))):
        d = np.load(p)
        gids = d["game_id"].astype(np.int64)
        mask = np.isin(gids, list(val_gids))
        if not mask.any():
            continue
        for xi, yi, gi in zip(d["X"][mask], d["y_value"][mask], gids[mask]):
            e = gid_eco.get(int(gi))
            byX[e].append(xi); byy[e].append(yi)

    # adapt per opening on support, then compare base vs adapted greedy move
    # over real opening start positions + short random rollouts.
    agree = same_val = 0
    total = 0
    for e in val_ecos:
        if e not in byX or len(byy[e]) < 16:
            continue
        X = np.array(byX[e], dtype=np.float32); y = np.array(byy[e], dtype=np.float32)
        idx = rng.choice(len(y), 16, replace=False)
        sX = torch.tensor(X[idx], dtype=torch.float32)
        sy = torch.tensor(y[idx], dtype=torch.float32)
        adapted = inner_adapt_anil(model, base, head_names, sX, sy, inner_lr=ILR, inner_steps=5)

        board_fp = e[len("shogi_"):] if e.startswith("shogi_") else e
        # a few positions: start + short random rollouts to diversify
        for _ in range(max(1, args.positions // max(1, len(val_ecos)))):
            board = shogi.Board(f"{board_fp} b - 5")
            for _r in range(rng.randint(0, 12)):
                lm = list(board.legal_moves)
                if not lm or board.is_game_over():
                    break
                board.push(lm[rng.randint(len(lm))])
            if board.is_game_over() or not list(board.legal_moves):
                continue
            mb = greedy_move(board, model, base, spec, device)
            ma = greedy_move(board, model, adapted, spec, device)
            if mb is not None and ma is not None:
                total += 1
                if mb == ma:
                    agree += 1

    print(f"positions compared: {total}")
    print(f"base vs adapted greedy-move AGREEMENT: {agree/total:.1%}" if total else "no positions")
    print("(high agreement => adaptation sharpens value calibration, not move choice "
          "=> ~50% adapted-vs-base gameplay is expected)")


if __name__ == "__main__":
    main()
