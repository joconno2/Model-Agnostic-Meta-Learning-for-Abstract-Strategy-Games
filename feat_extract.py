"""Spec A: extract game-general value features from engine-labeled positions.

Same shared feature vector for chess and shogi, computed from the side-to-move
perspective. Pairs each position's features with its engine centipawn label
(tanh-normalized) so a value head trained on one game can be tested on the other.

Feature slots (all stm-relative AND game-invariant ratios, so "ahead by X" means
the same thing in chess and shogi -- the prerequisite for cross-game transfer):
  0 material_fraction  (my - opp) / (my + opp)        in [-1,1]
  1 phase              (my + opp) / start_total[game]  in [0,1] material remaining
  2 mobility_per_piece legal-move count / (my_pieces + 1)
  3 in_check           1.0 if stm in check
  4 piece_fraction     my_pieces / (my + opp pieces)
  5 hand_fraction      (my - opp hand value) / (my + opp board value)  (chess: 0)
  6 promoted_fraction  (my - opp promoted) / (my + opp pieces)         (chess: 0)
  7 center_fraction    (my - opp center pieces) / (center pieces + 1)

Usage:
  python feat_extract.py --game chess --db lichess_chess.sqlite --out feat_chess.npz --cp-scale 400
  python feat_extract.py --game shogi --db shogi_positions.sqlite --out feat_shogi.npz --cp-scale 800
"""

import argparse
import math
import sqlite3
import numpy as np

NDIM = 8
EPS = 1e-6
CHESS_START = 78.0   # 2*(8P + 2N + 2B + 2R + Q) = 2*39
SHOGI_START = 126.0  # 2*(9P+2L+2N+2S+2G+B+R) per side = 63

CHESS_VAL = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}  # P N B R Q K
CHESS_CENTER = {27, 28, 35, 36}  # d4 e4 d5 e5

# python-shogi piece_type ints
SHOGI_VAL = {1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 8, 7: 10, 8: 0,
             9: 6, 10: 6, 11: 6, 12: 6, 13: 10, 14: 12}  # incl promoted 9-14
SHOGI_PROMOTED = {9, 10, 11, 12, 13, 14}
SHOGI_HAND_VAL = {1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 8, 7: 10}


def _pack(my_val, opp_val, my_n, opp_n, my_c, opp_c, my_hand, opp_hand,
          my_prom, opp_prom, mob, in_check, start_total):
    tot_val = my_val + opp_val + EPS
    tot_n = my_n + opp_n + EPS
    tot_c = my_c + opp_c + 1.0
    f = np.zeros(NDIM, dtype=np.float32)
    f[0] = (my_val - opp_val) / tot_val
    f[1] = (my_val + opp_val) / start_total
    f[2] = mob / (my_n + 1.0)
    f[3] = 1.0 if in_check else 0.0
    f[4] = my_n / tot_n
    f[5] = (my_hand - opp_hand) / tot_val
    f[6] = (my_prom - opp_prom) / tot_n
    f[7] = (my_c - opp_c) / tot_c
    return f


def chess_features(fen):
    import chess
    b = chess.Board(fen)
    stm = b.turn
    my_val = opp_val = my_n = opp_n = my_c = opp_c = 0
    for sq, pc in b.piece_map().items():
        v = CHESS_VAL.get(pc.piece_type, 0)
        c = 1 if sq in CHESS_CENTER else 0
        if pc.color == stm:
            my_val += v; my_n += 1; my_c += c
        else:
            opp_val += v; opp_n += 1; opp_c += c
    mob = sum(1 for _ in b.legal_moves)
    return _pack(my_val, opp_val, my_n, opp_n, my_c, opp_c, 0, 0, 0, 0,
                 mob, b.is_check(), CHESS_START)


def shogi_features(sfen):
    import shogi
    b = shogi.Board(sfen)
    stm = b.turn  # BLACK=0 (sente), WHITE=1
    my_val = opp_val = my_n = opp_n = my_c = opp_c = my_prom = opp_prom = 0
    center_sqs = {r * 9 + c for r in (3, 4, 5) for c in (3, 4, 5)}
    for sq in range(81):
        pc = b.piece_at(sq)
        if pc is None:
            continue
        v = SHOGI_VAL.get(pc.piece_type, 0)
        prom = 1 if pc.piece_type in SHOGI_PROMOTED else 0
        c = 1 if sq in center_sqs else 0
        if pc.color == stm:
            my_val += v; my_n += 1; my_c += c; my_prom += prom
        else:
            opp_val += v; opp_n += 1; opp_c += c; opp_prom += prom
    my_hand = sum(SHOGI_HAND_VAL.get(pt, 0) * n for pt, n in b.pieces_in_hand[stm].items())
    opp_hand = sum(SHOGI_HAND_VAL.get(pt, 0) * n for pt, n in b.pieces_in_hand[1 - stm].items())
    mob = sum(1 for _ in b.legal_moves)
    return _pack(my_val, opp_val, my_n, opp_n, my_c, opp_c, my_hand, opp_hand,
                 my_prom, opp_prom, mob, b.is_check(), SHOGI_START)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=["chess", "shogi"], required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cp-scale", type=float, required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    if args.game == "chess":
        q = ("SELECT game_id, fen, stockfish_cp FROM positions "
             "WHERE stockfish_cp IS NOT NULL ORDER BY game_id, ply")
        featfn = chess_features
    else:
        q = ("SELECT game_id, full_sfen, engine_cp FROM board_states "
             "WHERE engine_cp IS NOT NULL ORDER BY game_id, turn")
        featfn = shogi_features
    if args.limit:
        q += f" LIMIT {args.limit}"

    X, Y, G = [], [], []
    n = skip = 0
    for gid, pos, cp in conn.execute(q):
        try:
            f = featfn(pos)
        except Exception:
            skip += 1
            continue
        X.append(f)
        Y.append(math.tanh(float(cp) / args.cp_scale))
        G.append(int(gid))
        n += 1
        if n % 50000 == 0:
            print(f"  {n} positions")
    conn.close()

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    G = np.array(G, dtype=np.int64)
    np.savez_compressed(args.out, X=X, y_value=Y, game_id=G)
    print(f"Done. {n} positions, {skip} skipped -> {args.out}  X={X.shape}")
    print(f"feature means: {np.round(X.mean(0),3)}")
    print(f"feature stds:  {np.round(X.std(0),3)}")


if __name__ == "__main__":
    main()
