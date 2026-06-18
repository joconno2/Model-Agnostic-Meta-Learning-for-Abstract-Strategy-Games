# db_preprocess_shogi.py
"""
Preprocess the Shogi SQLite DB into sharded npz files for MAML training.

Also builds a 'games' table with opening fingerprints (board state at a
fixed ply) for use as task groupings in opening-mode meta-learning.

Usage:
    python3 db_preprocess_shogi.py /path/to/shogi_board_states.db ./processed_shogi_flat

Output:
    processed_shogi_flat/
        shogi_shard_0000.npz   (X, y_policy, y_value, game_id arrays)
        shogi_shard_0001.npz
        ...
    shogi_openings.sqlite      (game_id -> opening fingerprint mapping)
"""

import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np

from spec import UnifiedSpec, num_channels
from encode_shogi import encode_sfen_to_unified

# Ply at which to snapshot the board for opening fingerprinting.
OPENING_PLY = 4


def _assign_game_ids(conn):
    """
    Map SHA1 game_id strings to sequential integers.
    Returns dict {hash_str: int_id} and list of unique hashes.
    """
    cur = conn.execute("SELECT DISTINCT game_id FROM board_states ORDER BY game_id")
    mapping = {}
    for i, row in enumerate(cur, start=1):
        mapping[row[0]] = i
    return mapping


def _build_opening_db(conn, game_id_map, out_path):
    """
    Build a small SQLite DB mapping game integer IDs to opening fingerprints.

    The fingerprint is the board state at OPENING_PLY. Games that reach the
    same board state at that ply share an opening.
    """
    # Collect board state at OPENING_PLY for each game
    cur = conn.execute(
        "SELECT game_id, board FROM board_states WHERE turn = ? ORDER BY game_id",
        (OPENING_PLY,),
    )
    game_openings = {}
    for row in cur:
        gid_str = row[0]
        if gid_str in game_id_map:
            game_openings[game_id_map[gid_str]] = row[1]

    # For games that don't have a position at exactly OPENING_PLY
    # (e.g., game ended before ply 4), skip them.
    print(f"[Openings] {len(game_openings)} / {len(game_id_map)} games have a position at ply {OPENING_PLY}")

    # Count games per opening
    opening_counts = defaultdict(int)
    for board in game_openings.values():
        opening_counts[board] += 1
    print(f"[Openings] {len(opening_counts)} unique opening fingerprints")

    # Write to SQLite
    out_conn = sqlite3.connect(out_path)
    out_conn.execute("DROP TABLE IF EXISTS games")
    out_conn.execute("""
        CREATE TABLE games (
            id INTEGER PRIMARY KEY,
            eco TEXT NOT NULL
        )
    """)
    for int_id, board in game_openings.items():
        out_conn.execute("INSERT INTO games (id, eco) VALUES (?, ?)", (int_id, board))
    out_conn.commit()
    out_conn.close()
    print(f"[Openings] Wrote {out_path}")


def preprocess_shogi_to_npz(db_path: str, out_dir: str, shard_size: int = 50000):
    os.makedirs(out_dir, exist_ok=True)

    spec = UnifiedSpec()
    C = num_channels(spec)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Check table exists
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(f"Tables in DB: {tables}")
    if "board_states" not in tables:
        raise RuntimeError(f"'board_states' table not found. Available: {tables}")

    # Map hash game_ids to integers
    print("Mapping game IDs ...")
    game_id_map = _assign_game_ids(conn)
    print(f"  {len(game_id_map)} unique games")

    # Build opening fingerprint DB
    opening_db_path = os.path.join(os.path.dirname(out_dir), "shogi_openings.sqlite")
    _build_opening_db(conn, game_id_map, opening_db_path)

    # Process positions
    print("Processing positions ...")
    cur = conn.execute(
        "SELECT game_id, board, player, hands, turn, winner "
        "FROM board_states ORDER BY game_id, turn"
    )

    shard_idx = 0
    rows_in_shard = 0

    X = np.zeros((shard_size, C, 9, 9), dtype=np.float32)
    y_policy = np.zeros((shard_size,), dtype=np.int64)
    y_value = np.zeros((shard_size,), dtype=np.float32)
    game_ids = np.zeros((shard_size,), dtype=np.int32)

    def flush(n):
        nonlocal shard_idx, rows_in_shard
        if n == 0:
            return
        path = os.path.join(out_dir, f"shogi_shard_{shard_idx:04d}.npz")
        np.savez_compressed(path, X=X[:n], y_policy=y_policy[:n], y_value=y_value[:n], game_id=game_ids[:n])
        print(f"  Wrote {path} with {n} rows")
        shard_idx += 1
        rows_in_shard = 0

    skipped = 0
    total = 0

    for row in cur:
        gid_str = row["game_id"]
        board = row["board"]
        player = row["player"]    # "b" or "w" (side to move)
        hand = row["hands"]
        winner = row["winner"]    # "b" or "w"

        if gid_str not in game_id_map:
            skipped += 1
            continue

        # Value from side-to-move perspective
        z = 1.0 if player == winner else -1.0

        # Encode
        try:
            x, _ = encode_sfen_to_unified(board, player, hand, spec)
        except Exception as e:
            skipped += 1
            continue

        X[rows_in_shard] = x
        y_policy[rows_in_shard] = 0   # no action labels in Shogi DB
        y_value[rows_in_shard] = z
        game_ids[rows_in_shard] = game_id_map[gid_str]

        rows_in_shard += 1
        total += 1

        if rows_in_shard >= shard_size:
            flush(rows_in_shard)

    flush(rows_in_shard)
    conn.close()

    print(f"Done. {total} positions processed, {skipped} skipped, {shard_idx} shards.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python3 db_preprocess_shogi.py <db_path> <out_dir>")

    preprocess_shogi_to_npz(sys.argv[1], sys.argv[2])
