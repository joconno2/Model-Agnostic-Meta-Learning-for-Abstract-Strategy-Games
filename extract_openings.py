"""Extract openings database from the full positions SQLite.

Creates a lightweight openings-only SQLite matching the schema
that train_disjoint.py expects: games(id, eco).
"""

import argparse
import sqlite3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Full positions SQLite")
    ap.add_argument("--out", required=True, help="Output openings SQLite")
    args = ap.parse_args()

    src = sqlite3.connect(args.db)
    dst = sqlite3.connect(args.out)

    dst.execute("CREATE TABLE IF NOT EXISTS games (id INTEGER PRIMARY KEY, eco TEXT NOT NULL)")

    rows = src.execute("SELECT id, eco FROM games WHERE eco IS NOT NULL").fetchall()
    dst.executemany("INSERT INTO games (id, eco) VALUES (?, ?)", rows)
    dst.commit()

    n = dst.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    ecos = dst.execute("SELECT COUNT(DISTINCT eco) FROM games").fetchone()[0]
    print(f"Wrote {n:,} games with {ecos} distinct ECO codes to {args.out}")

    src.close()
    dst.close()


if __name__ == "__main__":
    main()
