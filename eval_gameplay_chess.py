#!/usr/bin/env python3
"""Chess gameplay evaluation (mirror of eval_gameplay_shogi.py).

Per-opening: adapt the ANIL value head on that opening's support positions, then
play real chess games from a representative FEN of that opening (ply ~10, from
the labeled corpus). Compares NN-adapted vs NN-base, plus material/random baselines.
Answers: does opening adaptation change actual PLAY in chess (where the value-MSE
gain is large), unlike shogi (where it was not significant)?

Usage:
  python eval_gameplay_chess.py --checkpoint runs/disjoint_anil_sf_k64_s42/best.pt \
      --data processed_sf_chess --openings sf_openings.sqlite --fens eco_fens.json \
      --games 200 --depth 1 --adapt-openings 8
"""

import argparse
import glob
import json
import os
import random
import sqlite3
from collections import defaultdict

import numpy as np
import torch
from torch.func import functional_call

import chess

from spec import UnifiedSpec, num_channels
from model_v2 import ValueNet
from encode_chess import encode_fen_to_unified
from maml_anil import inner_adapt_anil

ROOT = os.path.dirname(os.path.abspath(__file__))
TRUNK_HIDDEN = BOTTLENECK_DIM = VALUE_HIDDEN = 64
INNER_LR = 0.005

PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
               chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
MAX_MATERIAL = 39.0


def board_to_tensor(board, spec):
    x, _ = encode_fen_to_unified(board.fen(), spec)
    return x


def material_eval(board):
    stm = board.turn
    score = 0
    for _, pc in board.piece_map().items():
        v = PIECE_VALUE[pc.piece_type]
        score += v if pc.color == stm else -v
    return score / MAX_MATERIAL


def nn_eval(board, model, params, spec, device="cpu"):
    X = torch.tensor(board_to_tensor(board, spec), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return functional_call(model, params, (X,)).item()


def negamax(board, depth, alpha, beta, eval_fn):
    if board.is_game_over():
        return -1.0e6 - depth, None
    if depth == 0:
        return eval_fn(board), None
    best, best_move = -float("inf"), None
    for mv in board.legal_moves:
        board.push(mv)
        val, _ = negamax(board, depth - 1, -beta, -alpha, eval_fn)
        val = -val
        board.pop()
        if val > best:
            best, best_move = val, mv
        alpha = max(alpha, val)
        if alpha >= beta:
            break
    return best, best_move


def select_move_search(board, eval_fn, depth):
    _, mv = negamax(board, depth, -float("inf"), float("inf"), eval_fn)
    if mv is None:
        ms = list(board.legal_moves)
        return ms[0] if ms else None
    return mv


def select_move_nn_batched(board, model, params, spec, device="cpu"):
    moves = list(board.legal_moves)
    if not moves:
        return None
    terms, tensors, child_moves = [], [], []
    for mv in moves:
        board.push(mv)
        if board.is_game_over():
            terms.append(mv)
        else:
            tensors.append(board_to_tensor(board, spec)); child_moves.append(mv)
        board.pop()
    if terms:
        return terms[0]
    X = torch.tensor(np.stack(tensors), dtype=torch.float32, device=device)
    with torch.no_grad():
        vals = functional_call(model, params, (X,)).numpy()
    return child_moves[int(np.argmin(vals))]


def select_move_random(board):
    ms = list(board.legal_moves)
    return random.choice(ms) if ms else None


def play_game(white_fn, black_fn, start_fen, max_moves=200):
    board = chess.Board(start_fen)
    for _ in range(max_moves):
        if board.is_game_over():
            return board.result()
        fn = white_fn if board.turn == chess.WHITE else black_fn
        mv = fn(board)
        if mv is None:
            return board.result()
        board.push(mv)
    return "1/2-1/2"


def run_match(fn_a, fn_b, n_games, start):
    res = {"wins_a": 0, "wins_b": 0, "draws": 0, "total": 0}
    half = n_games // 2
    for i in range(n_games):
        if i < half:
            r = play_game(fn_a, fn_b, start); a_win, b_win = "1-0", "0-1"
        else:
            r = play_game(fn_b, fn_a, start); a_win, b_win = "0-1", "1-0"
        if r == a_win:
            res["wins_a"] += 1
        elif r == b_win:
            res["wins_b"] += 1
        else:
            res["draws"] += 1
        res["total"] += 1
    res["win_rate_a"] = (res["wins_a"] + 0.5 * res["draws"]) / res["total"]
    return res


def load_support(data_dir, openings_db, fens, min_games, max_openings, seed):
    rng = np.random.RandomState(seed)
    conn = sqlite3.connect(openings_db)
    rows = conn.execute("SELECT id, eco FROM games").fetchall()
    conn.close()
    eco_games = defaultdict(list)
    for gid, eco in rows:
        eco_games[eco].append(int(gid))
    ecos = [e for e, g in eco_games.items() if len(g) >= min_games and e in fens]
    ecos.sort(); rng.shuffle(ecos); ecos = ecos[:max_openings]
    chosen = set()
    gid_to_eco = {}
    for e in ecos:
        for g in eco_games[e]:
            chosen.add(g); gid_to_eco[g] = e
    bx, by = defaultdict(list), defaultdict(list)
    for p in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        d = np.load(p)
        gids = d["game_id"].astype(np.int64)
        mask = np.isin(gids, list(chosen))
        if not mask.any():
            continue
        for xi, yi, gi in zip(d["X"][mask], d["y_value"][mask], gids[mask]):
            e = gid_to_eco.get(int(gi))
            if e:
                bx[e].append(xi); by[e].append(yi)
    support = {e: (np.array(bx[e], dtype=np.float32), np.array(by[e], dtype=np.float32))
               for e in ecos if len(by[e]) >= 16}
    return support


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="processed_sf_chess")
    ap.add_argument("--openings", default="sf_openings.sqlite")
    ap.add_argument("--fens", default="eco_fens.json")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--adapt-openings", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cpu")
    spec = UnifiedSpec(); C = num_channels(spec)

    model = ValueNet(in_channels=C, trunk_hidden=TRUNK_HIDDEN,
                     bottleneck_dim=BOTTLENECK_DIM, value_hidden=VALUE_HIDDEN).to(device)
    ckpt = torch.load(os.path.join(ROOT, args.checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    head_names = set(n for n, _ in model.head_params())

    fens = json.load(open(os.path.join(ROOT, args.fens)))
    support = load_support(os.path.join(ROOT, args.data), os.path.join(ROOT, args.openings),
                           fens, 10, args.adapt_openings, args.seed)
    print(f"Loaded {args.checkpoint}; {len(support)} openings with support+FEN")

    adapted = {}
    for e, (X, y) in support.items():
        k = min(16, len(y)); idx = np.random.choice(len(y), k, replace=False)
        adapted[e] = inner_adapt_anil(model, base, head_names,
                                      torch.tensor(X[idx]), torch.tensor(y[idx]),
                                      inner_lr=INNER_LR, inner_steps=5)

    def nn_move(params):
        def f(board):
            if args.depth == 1:
                return select_move_nn_batched(board, model, params, spec, device)
            return select_move_search(board, lambda b: nn_eval(b, model, params, spec, device), args.depth)
        return f
    base_move = nn_move(base)
    mat_move = lambda b: select_move_search(b, material_eval, args.depth)

    games_per = max(2, args.games // max(1, len(support)))
    agg = {k: defaultdict(int) for k in
           ("NN-adapted_vs_NN-base", "NN-base_vs_Material", "NN-base_vs_Random", "Material_vs_Random")}
    for e in support:
        start = fens[e]
        adapt_move = nn_move(adapted[e])
        for key, fa, fb in [
            ("NN-adapted_vs_NN-base", adapt_move, base_move),
            ("NN-base_vs_Material", base_move, mat_move),
            ("NN-base_vs_Random", base_move, select_move_random),
            ("Material_vs_Random", mat_move, select_move_random),
        ]:
            r = run_match(fa, fb, games_per, start)
            for k in ("wins_a", "wins_b", "draws", "total"):
                agg[key][k] += r[k]

    results = {}
    print(f"\n{'='*56}\n  CHESS GAMEPLAY (depth={args.depth}, {games_per} games/opening)\n{'='*56}")
    for key, r in agg.items():
        wr = (r["wins_a"] + 0.5 * r["draws"]) / max(1, r["total"])
        r["win_rate_a"] = wr; results[key] = dict(r)
        a, b = key.split("_vs_")
        print(f"  {a:>12s} vs {b:<10s}: {wr:.1%}  ({r['wins_a']}W/{r['draws']}D/{r['wins_b']}L, n={r['total']})")

    out = os.path.join(ROOT, "runs", "chess_gameplay.json")
    with open(out, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "depth": args.depth,
                   "games_per_opening": games_per, "results": results}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
