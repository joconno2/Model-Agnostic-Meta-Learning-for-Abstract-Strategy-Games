"""Preprocess SQLite positions to npz shards with Stockfish centipawn targets.

Like db_preprocess_chess.py but reads stockfish_cp column and stores
a normalized evaluation as y_value instead of game outcome z.

Normalization: tanh(cp / 400) maps centipawns to [-1, 1].
  - +400 cp -> ~0.76
  - +200 cp -> ~0.46
  - +100 cp -> ~0.24
  - mate   -> +/- 1.0
"""

import os
import math
import sqlite3
import numpy as np
from spec import UnifiedSpec, num_channels
from encode_chess import encode_fen_to_unified


def cp_to_value(cp: int) -> float:
    """Normalize centipawns to [-1, 1] via tanh."""
    return math.tanh(cp / 400.0)


def preprocess_chess_sqlite_to_npz(
    db_path: str,
    out_dir: str,
    table: str = "positions",
    shard_size: int = 50000,
    use_stockfish: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    spec = UnifiedSpec()
    C = num_channels(spec)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if use_stockfish:
        cur.execute(
            f"SELECT id, game_id, ply, turn, fen, action_id, z, stockfish_cp "
            f"FROM {table} WHERE stockfish_cp IS NOT NULL ORDER BY game_id, ply"
        )
    else:
        cur.execute(
            f"SELECT id, game_id, ply, turn, fen, action_id, z "
            f"FROM {table} ORDER BY game_id, ply"
        )

    shard_idx = 0
    rows_in_shard = 0

    X = np.zeros((shard_size, C, 9, 9), dtype=np.float32)
    y_policy = np.zeros((shard_size,), dtype=np.int64)
    y_value = np.zeros((shard_size,), dtype=np.float32)
    y_wdl = np.zeros((shard_size,), dtype=np.float32)
    game_ids = np.zeros((shard_size,), dtype=np.int32)

    def flush(n: int):
        nonlocal shard_idx, rows_in_shard
        if n == 0:
            return
        path = os.path.join(out_dir, f"chess_shard_{shard_idx:04d}.npz")
        np.savez_compressed(
            path,
            X=X[:n],
            y_policy=y_policy[:n],
            y_value=y_value[:n],
            y_wdl=y_wdl[:n],
            game_id=game_ids[:n],
        )
        print(f"Wrote {path} with {n} rows")
        shard_idx += 1
        rows_in_shard = 0

    for row in cur:
        fen = row["fen"]
        a = int(row["action_id"])
        z = float(row["z"])
        gid = int(row["game_id"])

        x, _info = encode_fen_to_unified(fen, spec)

        X[rows_in_shard] = x
        y_policy[rows_in_shard] = a
        y_wdl[rows_in_shard] = z
        game_ids[rows_in_shard] = gid

        if use_stockfish:
            cp = int(row["stockfish_cp"])
            y_value[rows_in_shard] = cp_to_value(cp)
        else:
            y_value[rows_in_shard] = z

        rows_in_shard += 1

        if rows_in_shard >= shard_size:
            flush(rows_in_shard)

    flush(rows_in_shard)
    conn.close()
    print("Done.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python db_preprocess_chess_sf.py <db_path> <out_dir> [--wdl]"
        )

    db_path = sys.argv[1]
    out_dir = sys.argv[2]
    use_sf = "--wdl" not in sys.argv

    preprocess_chess_sqlite_to_npz(db_path, out_dir, use_stockfish=use_sf)
