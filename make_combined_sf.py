"""Build processed_sf_combined/ from chess SF + shogi SF shards for joint training.

Per the locked transfer design: one model meta-trained on chess (Stockfish) and
shogi (Fairy-Stockfish) openings, both labeled with engine centipawns squashed to
[-1,1]. Openings are game-prefixed ("chess_"+ECO already vs "shogi_"+fingerprint),
shogi game_ids are offset by +SHOGI_GID_OFFSET to avoid colliding with chess ids.

Chess opening DB (sf_openings.sqlite) stores raw ECO -> we prefix "chess_" here.
Shogi opening DB (sf_shogi_openings.sqlite) already stores "shogi_"+fingerprint.

Usage:
    python make_combined_sf.py \
        --chess-shards processed_sf_chess_small --chess-openings sf_openings.sqlite \
        --shogi-shards processed_sf_shogi      --shogi-openings sf_shogi_openings.sqlite \
        --out processed_sf_combined --out-openings sf_combined_openings.sqlite
"""

import argparse
import glob
import os
import shutil
import sqlite3

import numpy as np

SHOGI_GID_OFFSET = 10_000_000


def copy_chess_shards(src, dst):
    n = 0
    for p in sorted(glob.glob(os.path.join(src, "*.npz"))):
        name = os.path.basename(p)
        shutil.copy2(p, os.path.join(dst, name))  # chess game_ids untouched (< offset)
        n += 1
    print(f"  copied {n} chess shards")


def copy_shogi_shards(src, dst):
    n = 0
    for p in sorted(glob.glob(os.path.join(src, "*.npz"))):
        d = np.load(p)
        gid = d["game_id"].astype(np.int64) + SHOGI_GID_OFFSET
        out = os.path.join(dst, os.path.basename(p))
        save = {k: d[k] for k in d.files}
        save["game_id"] = gid.astype(np.int32)
        np.savez_compressed(out, **save)
        n += 1
    print(f"  copied {n} shogi shards (game_id += {SHOGI_GID_OFFSET})")


def build_combined_openings(chess_db, shogi_db, out_db):
    out = sqlite3.connect(out_db)
    out.execute("DROP TABLE IF EXISTS games")
    out.execute("CREATE TABLE games (id INTEGER PRIMARY KEY, eco TEXT NOT NULL)")

    cc = sqlite3.connect(chess_db)
    chess_rows = cc.execute("SELECT id, eco FROM games WHERE eco IS NOT NULL AND eco != ''").fetchall()
    cc.close()
    # prefix chess ECOs unless already prefixed
    chess_rows = [(int(i), e if e.startswith("chess_") else "chess_" + e) for i, e in chess_rows]

    sc = sqlite3.connect(shogi_db)
    shogi_rows = sc.execute("SELECT id, eco FROM games WHERE eco IS NOT NULL AND eco != ''").fetchall()
    sc.close()
    # offset shogi ids; ecos already "shogi_"+fp
    shogi_rows = [(int(i) + SHOGI_GID_OFFSET, e if e.startswith("shogi_") else "shogi_" + e)
                  for i, e in shogi_rows]

    out.executemany("INSERT INTO games (id, eco) VALUES (?, ?)", chess_rows + shogi_rows)
    out.commit()
    n_chess = sum(1 for _ in chess_rows)
    n_shogi = sum(1 for _ in shogi_rows)
    out.close()
    print(f"  combined openings: {n_chess} chess + {n_shogi} shogi games")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chess-shards", required=True)
    ap.add_argument("--chess-openings", required=True)
    ap.add_argument("--shogi-shards", required=True)
    ap.add_argument("--shogi-openings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-openings", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("Copying chess shards ...")
    copy_chess_shards(args.chess_shards, args.out)
    print("Copying shogi shards ...")
    copy_shogi_shards(args.shogi_shards, args.out)
    print("Building combined opening DB ...")
    build_combined_openings(args.chess_openings, args.shogi_openings, args.out_openings)
    print("Done.")


if __name__ == "__main__":
    main()
