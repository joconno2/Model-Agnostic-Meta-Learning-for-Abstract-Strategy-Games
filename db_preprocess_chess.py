# db_preprocess_chess.py
import os
import sqlite3
import numpy as np
from typing import List, Dict, Any
from spec import UnifiedSpec, num_channels
from encode_chess import encode_fen_to_unified
from action_encoding_chess import N_ACTIONS_CHESS, legal_mask_fen

def preprocess_chess_sqlite_to_npz(
    db_path: str,
    out_dir: str,
    table: str = "positions",
    shard_size: int = 50000,
    include_legal_masks: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    spec = UnifiedSpec()
    C = num_channels(spec)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [r[0] for r in cur.fetchall()]
    print("Tables in DB:", tables)

    if table not in tables:
        raise RuntimeError(f"table {repr(table)} not found. Available: {tables}")

    cur.execute(f"SELECT id, game_id, ply, turn, fen, action_id, z FROM {table} ORDER BY game_id, ply")

    shard_idx = 0
    rows_in_shard = 0

    X = np.zeros((shard_size, C, 9, 9), dtype=np.float32)
    y_policy = np.zeros((shard_size,), dtype=np.int64)
    y_value = np.zeros((shard_size,), dtype=np.float32)
    game_ids = np.zeros((shard_size,), dtype=np.int32)
    legal = np.zeros((shard_size, N_ACTIONS_CHESS), dtype=np.uint8) if include_legal_masks else None

    def flush(n: int):
        nonlocal shard_idx, rows_in_shard
        if n == 0:
            return
        path = os.path.join(out_dir, f"chess_shard_{shard_idx:04d}.npz")
        arrays = {
            "X": X[:n],
            "y_policy": y_policy[:n],
            "y_value": y_value[:n],
            "game_id": game_ids[:n],
        }
        if include_legal_masks:
            arrays["legal_mask"] = legal[:n]
        np.savez_compressed(path, **arrays)
        print(f"Wrote {path} with {n} rows (legal_mask={'yes' if include_legal_masks else 'no'})")
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
        y_value[rows_in_shard] = z
        game_ids[rows_in_shard] = gid

        if include_legal_masks:
            m = legal_mask_fen(fen)  # list[bool] length 20480
            legal[rows_in_shard] = np.asarray(m, dtype=np.uint8)

        rows_in_shard += 1

        if rows_in_shard >= shard_size:
            flush(rows_in_shard)

    flush(rows_in_shard)

    conn.close()
    print("Done.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("db_path")
    ap.add_argument("out_dir")
    ap.add_argument("--no-legal-masks", action="store_true",
                    help="Skip legal move mask computation (not recommended; masking cuts policy loss from ~10 to ~3.5).")
    ap.add_argument("--shard-size", type=int, default=50000)
    args = ap.parse_args()

    preprocess_chess_sqlite_to_npz(
        args.db_path,
        args.out_dir,
        table="positions",
        shard_size=args.shard_size,
        include_legal_masks=not args.no_legal_masks,
    )
