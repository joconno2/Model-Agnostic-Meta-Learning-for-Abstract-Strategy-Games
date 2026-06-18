#!/usr/bin/env python3
"""
Gameplay evaluation: plug ANIL value function into minimax search.

Tests whether the learned evaluation function produces useful gameplay
when combined with alpha-beta search. Compares:
  1. Random move selection
  2. Material counting (standard piece values)
  3. ANIL value function, no adaptation (0 inner steps)
  4. ANIL value function, adapted (5 inner steps on opening positions)

All at the same search depth. Games start from a given opening position
or from the standard starting position.

Requires: pip install chess
Optional: pip install python-shogi  (for shogi gameplay)

Usage:
    cd ~/code/maml-dasg
    pip install chess
    python eval_gameplay.py --checkpoint runs/opening_v1/best.pt --games 50 --depth 3
    python eval_gameplay.py --checkpoint runs/cross_game_v1/best.pt --games 50 --depth 3
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.func import functional_call

from spec import UnifiedSpec, num_channels, BOARD_PIECE_TYPES, build_channel_index
from model_v2 import ValueNet
from maml_anil import inner_adapt_anil

ROOT = os.path.dirname(os.path.abspath(__file__))

TRUNK_HIDDEN = 64
BOTTLENECK_DIM = 64
VALUE_HIDDEN = 64

# Standard piece values for material evaluation
MATERIAL_VALUES = {
    "P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0,
}

# Map python-chess piece types to our encoding
CHESS_PIECE_MAP = {
    1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K",
}


def board_to_tensor(board, spec, channel_idx):
    """Convert a python-chess Board to our 45-channel 9x9 tensor."""
    import chess
    C = len(channel_idx)
    tensor = np.zeros((C, 9, 9), dtype=np.float32)

    # Determine perspective: always encode from side-to-move's perspective
    flip = not board.turn  # flip if black to move

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue

        piece_type = CHESS_PIECE_MAP.get(piece.piece_type)
        if piece_type is None:
            continue

        # Board coordinates
        row = chess.square_rank(sq)
        col = chess.square_file(sq)

        if flip:
            row = 7 - row
            is_mine = piece.color == chess.BLACK
        else:
            is_mine = piece.color == chess.WHITE

        prefix = "my_board_" if is_mine else "opp_board_"
        ch_name = prefix + piece_type

        if ch_name in channel_idx:
            tensor[channel_idx[ch_name], row, col] = 1.0

    # Side to move plane
    if "side_to_move" in channel_idx:
        tensor[channel_idx["side_to_move"], :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    return tensor


def material_eval(board):
    """Simple material counting evaluation from side-to-move perspective."""
    import chess
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        pt = CHESS_PIECE_MAP.get(piece.piece_type, "")
        val = MATERIAL_VALUES.get(pt, 0)
        if piece.color == board.turn:
            score += val
        else:
            score -= val
    return score / 39.0  # normalize to roughly [-1, 1] (max material ~39)


def nn_eval(board, model, params, spec, channel_idx, device="cpu"):
    """Neural network evaluation from side-to-move perspective."""
    tensor = board_to_tensor(board, spec, channel_idx)
    X = torch.tensor(tensor, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        v = functional_call(model, params, (X,))
    return v.item()


def alpha_beta(board, depth, alpha, beta, eval_fn, maximizing=True):
    """Alpha-beta minimax search."""
    import chess
    if depth == 0 or board.is_game_over():
        return eval_fn(board), None

    best_move = None
    if maximizing:
        max_eval = -float("inf")
        for move in board.legal_moves:
            board.push(move)
            val, _ = alpha_beta(board, depth - 1, alpha, beta, eval_fn, False)
            board.pop()
            if val > max_eval:
                max_eval = val
                best_move = move
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float("inf")
        for move in board.legal_moves:
            board.push(move)
            val, _ = alpha_beta(board, depth - 1, alpha, beta, eval_fn, True)
            board.pop()
            if val < min_eval:
                min_eval = val
                best_move = move
            beta = min(beta, val)
            if beta <= alpha:
                break
        return min_eval, best_move


def select_move_search(board, eval_fn, depth):
    """Select best move using alpha-beta search."""
    _, move = alpha_beta(board, depth, -float("inf"), float("inf"), eval_fn, True)
    return move


def select_move_random(board):
    """Select a random legal move."""
    import chess
    moves = list(board.legal_moves)
    return random.choice(moves) if moves else None


def play_game(white_fn, black_fn, max_moves=200, start_fen=None):
    """
    Play a game between two move-selection functions.
    Returns: result string ("1-0", "0-1", "1/2-1/2"), move count.
    """
    import chess
    board = chess.Board(start_fen) if start_fen else chess.Board()

    for move_num in range(max_moves):
        if board.is_game_over():
            break

        if board.turn == chess.WHITE:
            move = white_fn(board)
        else:
            move = black_fn(board)

        if move is None:
            break
        board.push(move)

    result = board.result()
    if result == "*":
        result = "1/2-1/2"  # timeout = draw
    return result, board.fullmove_number


def run_match(name_a, fn_a, name_b, fn_b, n_games, start_fen=None):
    """Run a match: n_games/2 as white, n_games/2 as black."""
    results = {"wins_a": 0, "wins_b": 0, "draws": 0, "total": 0}
    half = n_games // 2

    for i in range(n_games):
        if i < half:
            # A plays white
            result, moves = play_game(fn_a, fn_b, start_fen=start_fen)
            if result == "1-0":
                results["wins_a"] += 1
            elif result == "0-1":
                results["wins_b"] += 1
            else:
                results["draws"] += 1
        else:
            # A plays black
            result, moves = play_game(fn_b, fn_a, start_fen=start_fen)
            if result == "0-1":
                results["wins_a"] += 1
            elif result == "1-0":
                results["wins_b"] += 1
            else:
                results["draws"] += 1
        results["total"] += 1

    win_rate = (results["wins_a"] + 0.5 * results["draws"]) / results["total"]
    results["win_rate_a"] = win_rate
    return results


def main():
    try:
        import chess
    except ImportError:
        print("python-chess not installed. Run: pip install chess")
        return

    parser = argparse.ArgumentParser(description="Gameplay evaluation for MAML-DASG")
    parser.add_argument("--checkpoint", default="runs/opening_v1/best.pt")
    parser.add_argument("--games", type=int, default=50, help="Games per matchup")
    parser.add_argument("--depth", type=int, default=3, help="Search depth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opening-eco", default=None, help="ECO code to adapt to (e.g. chess_B20)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    spec = UnifiedSpec()
    C = num_channels(spec)
    channel_idx = build_channel_index(spec)

    # Load model
    ckpt_path = os.path.join(ROOT, args.checkpoint)
    model = ValueNet(in_channels=C, trunk_hidden=TRUNK_HIDDEN,
                     bottleneck_dim=BOTTLENECK_DIM, value_hidden=VALUE_HIDDEN).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    base_params = {n: p.detach().clone() for n, p in model.named_parameters()}
    head_param_names = set(n for n, _ in model.head_params())

    # Optionally adapt to a specific opening
    adapted_params = None
    if args.opening_eco:
        from task_sampler_v2 import ValueTaskSampler
        # Determine which data dir to use
        if "cross_game" in args.checkpoint:
            data_dir = os.path.join(ROOT, "processed_combined_flat")
        else:
            data_dir = os.path.join(ROOT, "processed_chess_flat")

        sampler = ValueTaskSampler(
            data_dir=data_dir,
            db_path=os.path.join(ROOT, "combined_openings.sqlite"),
            task_mode="opening", train_frac=0.8, seed=args.seed,
            min_positions_per_task=32,
        )
        if args.opening_eco in sampler._task_data:
            X_all, yv_all = sampler._task_data[args.opening_eco]
            k = min(16, len(yv_all))
            idx = np.random.choice(len(yv_all), k, replace=False)
            sX = torch.tensor(X_all[idx], dtype=torch.float32, device=device)
            sy = torch.tensor(yv_all[idx], dtype=torch.float32, device=device)
            adapted_params = inner_adapt_anil(
                model, base_params, head_param_names,
                sX, sy, inner_lr=0.005, inner_steps=5,
            )
            print(f"Adapted to opening {args.opening_eco} ({k} support positions)")
        else:
            print(f"Warning: opening {args.opening_eco} not found in data. Using base params.")

    # Build evaluation functions
    def material_move(board):
        return select_move_search(board, material_eval, args.depth)

    def nn_base_move(board):
        eval_fn = lambda b: nn_eval(b, model, base_params, spec, channel_idx, device)
        return select_move_search(board, eval_fn, args.depth)

    def nn_adapted_move(board):
        params = adapted_params if adapted_params else base_params
        eval_fn = lambda b: nn_eval(b, model, params, spec, channel_idx, device)
        return select_move_search(board, eval_fn, args.depth)

    def random_move(board):
        return select_move_random(board)

    # Run matches
    all_results = {}
    matchups = [
        ("NN-base", nn_base_move, "Random", random_move),
        ("NN-base", nn_base_move, "Material", material_move),
        ("Material", material_move, "Random", random_move),
    ]
    if adapted_params:
        matchups.append(("NN-adapted", nn_adapted_move, "NN-base", nn_base_move))
        matchups.append(("NN-adapted", nn_adapted_move, "Material", material_move))

    for name_a, fn_a, name_b, fn_b in matchups:
        print(f"\n{name_a} vs {name_b} ({args.games} games, depth {args.depth})...")
        t0 = time.time()
        result = run_match(name_a, fn_a, name_b, fn_b, args.games)
        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 1)
        key = f"{name_a}_vs_{name_b}"
        all_results[key] = result
        print(f"  {name_a}: {result['wins_a']}W {result['draws']}D {result['wins_b']}L "
              f"(win rate {result['win_rate_a']:.1%}, {elapsed:.0f}s)")

    # Save results
    out_path = os.path.join(ROOT, "runs", "gameplay_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "depth": args.depth,
            "games_per_matchup": args.games,
            "opening": args.opening_eco,
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"  SUMMARY (depth={args.depth}, {args.games} games/matchup)")
    print(f"{'='*50}")
    for key, r in all_results.items():
        names = key.split("_vs_")
        print(f"  {names[0]:>12s} vs {names[1]:<12s}: {r['win_rate_a']:.1%} "
              f"({r['wins_a']}W/{r['draws']}D/{r['wins_b']}L)")


if __name__ == "__main__":
    main()
