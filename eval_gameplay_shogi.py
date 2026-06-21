#!/usr/bin/env python3
"""
Shogi gameplay evaluation: plug the ANIL value function into alpha-beta search
and play real shogi games. The practical test the SMC reviewers asked for.

Compares, at fixed search depth, starting from real opening positions:
  1. Random move selection
  2. Material counting (shogi piece values, incl. pieces in hand)
  3. ANIL value function, no adaptation (0 inner steps)
  4. ANIL value function, adapted (5 inner steps on the opening's support positions)

Requires: python-shogi. Value net uses encode_shogi.encode_sfen_to_unified, so
the encoding matches training exactly.

Usage:
    python eval_gameplay_shogi.py --checkpoint runs/disjoint_anil_sf_shogi_s42/best.pt \
        --data processed_sf_shogi --openings sf_shogi_openings.sqlite \
        --games 40 --depth 2 --adapt-openings 8
"""

import argparse
import glob
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch.func import functional_call

import shogi

from spec import UnifiedSpec, num_channels, build_channel_index
from model_v2 import ValueNet
from encode_shogi import encode_sfen_to_unified
from maml_anil import inner_adapt_anil

ROOT = os.path.dirname(os.path.abspath(__file__))

TRUNK_HIDDEN = 64
BOTTLENECK_DIM = 64
VALUE_HIDDEN = 64
INNER_LR = 0.005

# Shogi material values (board + hand). King 0.
PIECE_VALUE = {
    "P": 1, "L": 3, "N": 4, "S": 5, "G": 6, "B": 8, "R": 10, "K": 0,
    "+P": 6, "+L": 6, "+N": 6, "+S": 6, "+B": 10, "+R": 12,
}
MAX_MATERIAL = 60.0


def board_to_tensor(board, spec):
    """python-shogi Board -> 45ch tensor via the training encoder."""
    parts = board.sfen().split(" ")
    bd, player, hands = parts[0], parts[1], (parts[2] if len(parts) > 2 else "-")
    x, _ = encode_sfen_to_unified(bd, player, hands, spec)
    return x


def material_eval(board):
    """Material from side-to-move perspective, parsed from SFEN. [-1,1]-ish."""
    parts = board.sfen().split(" ")
    bd, player = parts[0], parts[1]
    hands = parts[2] if len(parts) > 2 else "-"
    stm_black = (player == "b")

    score = 0
    i = 0
    # board pieces
    while i < len(bd):
        ch = bd[i]
        if ch == "/" or ch.isdigit():
            i += 1
            continue
        promoted = ""
        if ch == "+":
            promoted = "+"
            i += 1
            ch = bd[i]
        is_black = ch.isupper()
        val = PIECE_VALUE.get(promoted + ch.upper(), 0)
        score += val if (is_black == stm_black) else -val
        i += 1
    # hands
    if hands != "-":
        j = 0
        while j < len(hands):
            cnt = 0
            while j < len(hands) and hands[j].isdigit():
                cnt = cnt * 10 + int(hands[j]); j += 1
            if cnt == 0:
                cnt = 1
            if j >= len(hands):
                break
            ch = hands[j]; j += 1
            is_black = ch.isupper()
            val = PIECE_VALUE.get(ch.upper(), 0) * cnt
            score += val if (is_black == stm_black) else -val
    return score / MAX_MATERIAL


def nn_eval(board, model, params, spec, device="cpu"):
    X = torch.tensor(board_to_tensor(board, spec), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        v = functional_call(model, params, (X,))
    return v.item()


def negamax(board, depth, alpha, beta, eval_fn):
    """
    Negamax with alpha-beta. eval_fn returns value from the SIDE-TO-MOVE
    perspective, so we negate at each level. Returns (value, best_move) where
    value is from the perspective of the side to move at this node.
    """
    if board.is_game_over():
        # side to move has no escape -> losing for side to move
        return -1.0e6 - depth, None
    if depth == 0:
        return eval_fn(board), None
    best = -float("inf")
    best_move = None
    for move in board.legal_moves:
        board.push(move)
        val, _ = negamax(board, depth - 1, -beta, -alpha, eval_fn)
        val = -val
        board.pop()
        if val > best:
            best, best_move = val, move
        alpha = max(alpha, val)
        if alpha >= beta:
            break
    return best, best_move


def select_move_search(board, eval_fn, depth):
    _, move = negamax(board, depth, -float("inf"), float("inf"), eval_fn)
    if move is None:
        moves = list(board.legal_moves)
        return moves[0] if moves else None
    return move


def select_move_nn_batched(board, model, params, spec, device="cpu"):
    """
    Depth-1 greedy with a single batched forward pass over all legal children.
    negamax(depth 1): value(move) = -eval(child); pick argmax = argmin eval(child).
    ~100x faster than per-position eval for shogi's wide branching.
    """
    moves = list(board.legal_moves)
    if not moves:
        return None
    terminal_idx, child_tensors, child_moves = [], [], []
    for mv in moves:
        board.push(mv)
        if board.is_game_over():
            terminal_idx.append(mv)  # opponent has no move -> we just won
        else:
            child_tensors.append(board_to_tensor(board, spec))
            child_moves.append(mv)
        board.pop()
    if terminal_idx:
        return terminal_idx[0]  # immediate winning move
    X = torch.tensor(np.stack(child_tensors), dtype=torch.float32, device=device)
    with torch.no_grad():
        vals = functional_call(model, params, (X,)).numpy()  # child = opp perspective
    return child_moves[int(np.argmin(vals))]  # minimize opponent's value


def select_move_random(board):
    moves = list(board.legal_moves)
    return random.choice(moves) if moves else None


def play_game(black_fn, white_fn, start_sfen, max_moves=240):
    """Play one game from start_sfen. Returns 'b'/'w'/'draw'."""
    board = shogi.Board(start_sfen)
    for _ in range(max_moves):
        if board.is_game_over():
            # side to move has no escape -> loses
            return "w" if board.turn == shogi.BLACK else "b"
        fn = black_fn if board.turn == shogi.BLACK else white_fn
        move = fn(board)
        if move is None:
            return "w" if board.turn == shogi.BLACK else "b"
        board.push(move)
    return "draw"


def run_match(fn_a, fn_b, n_games, openings):
    """A vs B, alternating colors, cycling through opening start positions."""
    res = {"wins_a": 0, "wins_b": 0, "draws": 0, "total": 0}
    half = n_games // 2
    for i in range(n_games):
        start = openings[i % len(openings)]
        if i < half:
            winner = play_game(fn_a, fn_b, start)   # A black
            a_is = "b"
        else:
            winner = play_game(fn_b, fn_a, start)   # A white
            a_is = "w"
        if winner == "draw":
            res["draws"] += 1
        elif winner == a_is:
            res["wins_a"] += 1
        else:
            res["wins_b"] += 1
        res["total"] += 1
    res["win_rate_a"] = (res["wins_a"] + 0.5 * res["draws"]) / res["total"]
    return res


def load_opening_support(data_dir, openings_db, min_games=10, max_openings=8, seed=42):
    """
    Pick openings with >= min_games and return:
      start_sfens: list of full SFENs (board fingerprint + 'b - <ply+1>') to start games
      support: {eco: (X, y)} support tensors for adaptation
    """
    rng = np.random.RandomState(seed)
    conn = __import__("sqlite3").connect(openings_db)
    rows = conn.execute("SELECT id, eco FROM games").fetchall()
    conn.close()
    eco_games = defaultdict(list)
    for gid, eco in rows:
        eco_games[eco].append(int(gid))
    ecos = [e for e, g in eco_games.items() if len(g) >= min_games]
    ecos.sort()
    rng.shuffle(ecos)
    ecos = ecos[:max_openings]
    chosen_gids = set()
    for e in ecos:
        chosen_gids.update(eco_games[e])

    # gather positions for chosen game_ids from shards
    gid_to_eco = {}
    for e in ecos:
        for g in eco_games[e]:
            gid_to_eco[g] = e
    by_eco_X = defaultdict(list)
    by_eco_y = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        d = np.load(p)
        gids = d["game_id"].astype(np.int64)
        mask = np.isin(gids, list(chosen_gids))
        if not mask.any():
            continue
        X, y, g = d["X"][mask], d["y_value"][mask], gids[mask]
        for xi, yi, gi in zip(X, y, g):
            e = gid_to_eco.get(int(gi))
            if e is not None:
                by_eco_X[e].append(xi)
                by_eco_y[e].append(yi)

    support = {}
    start_sfens = []
    for e in ecos:
        if e not in by_eco_X or len(by_eco_y[e]) < 16:
            continue
        support[e] = (np.array(by_eco_X[e], dtype=np.float32),
                      np.array(by_eco_y[e], dtype=np.float32))
        # opening fingerprint = board part after "shogi_" prefix; start black to move
        board_fp = e[len("shogi_"):] if e.startswith("shogi_") else e
        start_sfens.append(f"{board_fp} b - 5")
    return start_sfens, support


def main():
    parser = argparse.ArgumentParser(description="Shogi gameplay evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="processed_sf_shogi")
    parser.add_argument("--openings", default="sf_shogi_openings.sqlite")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--adapt-openings", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    spec = UnifiedSpec()
    C = num_channels(spec)

    model = ValueNet(in_channels=C, trunk_hidden=TRUNK_HIDDEN,
                     bottleneck_dim=BOTTLENECK_DIM, value_hidden=VALUE_HIDDEN).to(device)
    ckpt = torch.load(os.path.join(ROOT, args.checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {args.checkpoint}")

    base_params = {n: p.detach().clone() for n, p in model.named_parameters()}
    head_param_names = set(n for n, _ in model.head_params())

    start_sfens, support = load_opening_support(
        os.path.join(ROOT, args.data), os.path.join(ROOT, args.openings),
        max_openings=args.adapt_openings, seed=args.seed)
    print(f"{len(start_sfens)} opening start positions, {len(support)} with support")

    # One adapted param set per opening (adapt the head on that opening's support).
    adapted_by_open = {}
    for e, (X, y) in support.items():
        k = min(16, len(y))
        idx = np.random.choice(len(y), k, replace=False)
        sX = torch.tensor(X[idx], dtype=torch.float32, device=device)
        sy = torch.tensor(y[idx], dtype=torch.float32, device=device)
        adapted_by_open[e] = inner_adapt_anil(model, base_params, head_param_names,
                                              sX, sy, inner_lr=INNER_LR, inner_steps=5)

    # Move functions. For adapted, pick params by the opening the game started from
    # is not tracked inside the search; instead we run per-opening matches.
    def make_nn_move(params):
        def f(board):
            if args.depth == 1:
                return select_move_nn_batched(board, model, params, spec, device)
            return select_move_search(
                board, lambda b: nn_eval(b, model, params, spec, device), args.depth)
        return f

    def material_move(board):
        return select_move_search(board, material_eval, args.depth)

    def random_move(board):
        return select_move_random(board)

    nn_base_move = make_nn_move(base_params)

    # Per-opening: NN-adapted vs NN-base (the key practical comparison), plus baselines.
    games_per = max(2, args.games // max(1, len(support)))
    agg = {"NN-adapted_vs_NN-base": defaultdict(int),
           "NN-base_vs_Material": defaultdict(int),
           "NN-base_vs_Random": defaultdict(int),
           "Material_vs_Random": defaultdict(int)}

    for e, (X, y) in support.items():
        board_fp = e[len("shogi_"):] if e.startswith("shogi_") else e
        opens = [f"{board_fp} b - 5"]
        nn_adapt_move = make_nn_move(adapted_by_open[e])
        for key, fa, fb in [
            ("NN-adapted_vs_NN-base", nn_adapt_move, nn_base_move),
            ("NN-base_vs_Material", nn_base_move, material_move),
            ("NN-base_vs_Random", nn_base_move, random_move),
            ("Material_vs_Random", material_move, random_move),
        ]:
            r = run_match(fa, fb, games_per, opens)
            for k in ("wins_a", "wins_b", "draws", "total"):
                agg[key][k] += r[k]

    all_results = {}
    print(f"\n{'='*56}\n  SHOGI GAMEPLAY (depth={args.depth}, {games_per} games/opening)\n{'='*56}")
    for key, r in agg.items():
        wr = (r["wins_a"] + 0.5 * r["draws"]) / max(1, r["total"])
        r["win_rate_a"] = wr
        all_results[key] = dict(r)
        a, b = key.split("_vs_")
        print(f"  {a:>12s} vs {b:<10s}: {wr:.1%}  ({r['wins_a']}W/{r['draws']}D/{r['wins_b']}L, n={r['total']})")

    out = os.path.join(ROOT, "runs", "shogi_gameplay.json")
    with open(out, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "depth": args.depth,
                   "games_per_opening": games_per, "results": all_results}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
