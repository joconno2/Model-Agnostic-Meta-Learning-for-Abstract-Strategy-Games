"""Preprocess shogi board_states (with Fairy-Stockfish engine_cp) to npz shards.

Shogi analog of db_preprocess_chess_sf.py. Reads full_sfen + engine_cp, splits
the SFEN into board/stm/hands, encodes to the unified 45ch tensor, and stores
y_value = tanh(cp / scale). Also builds an opening DB (board fingerprint at
ply 4) matching what train_disjoint.py / eval_sf_suite.py expect: games(id, eco).

Usage:
    python db_preprocess_shogi_sf.py shogi_positions.sqlite processed_sf_shogi \
        --cp-scale 600
"""

import argparse
import math
import os
import sqlite3
from collections import defaultdict

import numpy as np

from spec import UnifiedSpec, num_channels
from encode_shogi import encode_sfen_to_unified

OPENING_PLY = 4


def split_sfen(full_sfen):
    parts = full_sfen.split(" ")
    board = parts[0]
    player = parts[1] if len(parts) > 1 else "b"
    hands = parts[2] if len(parts) > 2 else "-"
    return board, player, hands


def build_opening_db(conn, out_path):
    rows = conn.execute(
        "SELECT game_id, full_sfen FROM board_states WHERE turn = ? ORDER BY game_id",
        (OPENING_PLY,),
    ).fetchall()
    game_openings = {}
    for gid, sfen in rows:
        # "shogi_" prefix so the combined chess+shogi opening space is unambiguous
        game_openings[int(gid)] = "shogi_" + sfen.split(" ")[0]  # board fingerprint
    counts = defaultdict(int)
    for b in game_openings.values():
        counts[b] += 1
    print(f"[Openings] {len(game_openings)} games at ply {OPENING_PLY}, "
          f"{len(counts)} unique fingerprints")
    out = sqlite3.connect(out_path)
    out.execute("DROP TABLE IF EXISTS games")
    out.execute("CREATE TABLE games (id INTEGER PRIMARY KEY, eco TEXT NOT NULL)")
    out.executemany("INSERT INTO games (id, eco) VALUES (?, ?)",
                    list(game_openings.items()))
    out.commit()
    out.close()
    print(f"[Openings] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("out_dir")
    ap.add_argument("--cp-scale", type=float, default=600.0)
    ap.add_argument("--shard-size", type=int, default=50000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    spec = UnifiedSpec()
    C = num_channels(spec)

    conn = sqlite3.connect(args.db)

    opening_db = os.path.join(os.path.dirname(args.out_dir) or ".", "sf_shogi_openings.sqlite")
    build_opening_db(conn, opening_db)

    cur = conn.execute(
        "SELECT game_id, turn, full_sfen, engine_cp FROM board_states "
        "WHERE engine_cp IS NOT NULL ORDER BY game_id, turn"
    )

    shard_idx = 0
    n = 0
    X = np.zeros((args.shard_size, C, 9, 9), dtype=np.float32)
    y_value = np.zeros((args.shard_size,), dtype=np.float32)
    y_policy = np.zeros((args.shard_size,), dtype=np.int64)
    game_ids = np.zeros((args.shard_size,), dtype=np.int32)

    def flush(k):
        nonlocal shard_idx
        if k == 0:
            return
        path = os.path.join(args.out_dir, f"shogi_shard_{shard_idx:04d}.npz")
        np.savez_compressed(path, X=X[:k], y_value=y_value[:k],
                            y_policy=y_policy[:k], game_id=game_ids[:k])
        print(f"  wrote {path} ({k} rows)")
        shard_idx += 1

    total = 0
    skipped = 0
    cps = []
    for gid, turn, sfen, cp in cur:
        board, player, hands = split_sfen(sfen)
        try:
            x, _ = encode_sfen_to_unified(board, player, hands, spec)
        except Exception:
            skipped += 1
            continue
        X[n] = x
        y_value[n] = math.tanh(float(cp) / args.cp_scale)
        game_ids[n] = int(gid)
        cps.append(cp)
        n += 1
        total += 1
        if n >= args.shard_size:
            flush(n)
            n = 0
    flush(n)
    conn.close()

    cps = np.array(cps, dtype=np.float64)
    yv = np.tanh(cps / args.cp_scale)
    print(f"Done. {total} positions, {skipped} skipped, {shard_idx} shards.")
    print(f"cp: mean={cps.mean():.1f} std={cps.std():.1f} "
          f"p5={np.percentile(cps,5):.0f} p95={np.percentile(cps,95):.0f}")
    print(f"y_value (tanh, scale={args.cp_scale}): mean={yv.mean():.3f} std={yv.std():.3f} "
          f"|>0.95|={np.mean(np.abs(yv)>0.95)*100:.1f}%  (saturation check)")


if __name__ == "__main__":
    main()
