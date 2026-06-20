"""Combine chess SF and shogi engine datasets into one joint meta-learning set.

Produces processed_sf_combined/ (all shards) + sf_combined_openings.sqlite with
game-prefixed opening ids so chess and shogi tasks are distinguishable:
  chess openings: "chess_" + ECO
  shogi openings: "shogi_" + board-fingerprint  (already prefixed by preprocessor)

Shogi game_ids are offset by OFFSET to avoid colliding with chess game_ids,
since the sampler indexes positions by game_id across all shards in the dir.

Usage:
    python combine_sf_datasets.py \
        --chess-data processed_sf_chess_small --chess-db sf_openings.sqlite \
        --shogi-data processed_sf_shogi --shogi-db sf_shogi_openings.sqlite \
        --out-data processed_sf_combined --out-db sf_combined_openings.sqlite
"""

import argparse
import os
import sqlite3
import numpy as np

OFFSET = 10_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chess-data", required=True)
    ap.add_argument("--chess-db", required=True)
    ap.add_argument("--shogi-data", required=True)
    ap.add_argument("--shogi-db", required=True)
    ap.add_argument("--out-data", required=True)
    ap.add_argument("--out-db", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_data, exist_ok=True)

    # Chess shards: copy through unchanged (game_id kept).
    chess_shards = sorted(f for f in os.listdir(args.chess_data) if f.endswith(".npz"))
    for f in chess_shards:
        d = np.load(os.path.join(args.chess_data, f))
        out = {k: d[k] for k in d.files}
        np.savez_compressed(os.path.join(args.out_data, f), **out)
    print(f"chess: copied {len(chess_shards)} shards")

    # Shogi shards: offset game_id, ensure y_value/y_policy present.
    shogi_shards = sorted(f for f in os.listdir(args.shogi_data) if f.endswith(".npz"))
    for f in shogi_shards:
        d = np.load(os.path.join(args.shogi_data, f))
        out = {k: d[k] for k in d.files}
        out["game_id"] = d["game_id"].astype(np.int64) + OFFSET
        np.savez_compressed(os.path.join(args.out_data, f), **out)
    print(f"shogi: copied {len(shogi_shards)} shards (game_id += {OFFSET})")

    # Merge opening DBs.
    out = sqlite3.connect(args.out_db)
    out.execute("DROP TABLE IF EXISTS games")
    out.execute("CREATE TABLE games (id INTEGER PRIMARY KEY, eco TEXT NOT NULL)")

    c = sqlite3.connect(args.chess_db)
    chess_rows = c.execute("SELECT id, eco FROM games WHERE eco IS NOT NULL").fetchall()
    c.close()
    out.executemany("INSERT INTO games (id, eco) VALUES (?, ?)",
                    [(gid, "chess_" + eco) for gid, eco in chess_rows])

    s = sqlite3.connect(args.shogi_db)
    shogi_rows = s.execute("SELECT id, eco FROM games WHERE eco IS NOT NULL").fetchall()
    s.close()
    # shogi eco already "shogi_"-prefixed by db_preprocess_shogi_sf.py
    out.executemany("INSERT INTO games (id, eco) VALUES (?, ?)",
                    [(gid + OFFSET, eco) for gid, eco in shogi_rows])
    out.commit()

    n_chess = len(chess_rows)
    n_shogi = len(shogi_rows)
    print(f"openings: {n_chess} chess + {n_shogi} shogi = {n_chess + n_shogi} -> {args.out_db}")
    out.close()


if __name__ == "__main__":
    main()
