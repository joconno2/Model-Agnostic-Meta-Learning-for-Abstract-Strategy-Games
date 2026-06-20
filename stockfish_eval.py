"""Batch Stockfish evaluation of all FENs in the MAML-DASG SQLite database.

Reads positions table, evaluates each FEN with Stockfish, writes centipawn
evals back to the DB. Uses multiprocessing to saturate all cores.

Usage:
    python stockfish_eval.py --db chess_positions.sqlite \
        --stockfish ./stockfish/stockfish-ubuntu-x86-64-avx2 \
        --depth 15 --workers 60
"""

import argparse
import sqlite3
import multiprocessing as mp
import subprocess
import io
import time
import os


def init_engine(stockfish_path: str, depth: int):
    """Start a Stockfish process and return (proc, depth)."""
    proc = subprocess.Popen(
        [stockfish_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    proc.stdin.write("uci\n")
    proc.stdin.flush()
    # Wait for uciok
    for line in proc.stdout:
        if "uciok" in line:
            break
    proc.stdin.write("setoption name Threads value 1\n")
    proc.stdin.write("setoption name Hash value 16\n")
    proc.stdin.write("isready\n")
    proc.stdin.flush()
    for line in proc.stdout:
        if "readyok" in line:
            break
    return proc


def eval_fen(proc, fen: str, depth: int) -> int:
    """Evaluate a single FEN. Returns centipawns from side-to-move perspective."""
    proc.stdin.write(f"position fen {fen}\n")
    proc.stdin.write(f"go depth {depth}\n")
    proc.stdin.flush()

    cp = 0
    for line in proc.stdout:
        if line.startswith("bestmove"):
            break
        if "score cp" in line and "upperbound" not in line and "lowerbound" not in line:
            parts = line.split()
            idx = parts.index("cp")
            cp = int(parts[idx + 1])
        elif "score mate" in line and "upperbound" not in line and "lowerbound" not in line:
            parts = line.split()
            idx = parts.index("mate")
            mate_in = int(parts[idx + 1])
            cp = 30000 if mate_in > 0 else -30000
    return cp


# Global worker state
_worker_proc = None
_worker_depth = None


def worker_init(stockfish_path: str, depth: int):
    global _worker_proc, _worker_depth
    _worker_proc = init_engine(stockfish_path, depth)
    _worker_depth = depth


def worker_eval(batch):
    """Evaluate a batch of (row_id, fen) tuples. Returns list of (row_id, cp)."""
    global _worker_proc, _worker_depth
    results = []
    for row_id, fen in batch:
        cp = eval_fen(_worker_proc, fen, _worker_depth)
        results.append((row_id, cp))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="SQLite database with positions table")
    ap.add_argument("--stockfish", required=True, help="Path to Stockfish binary")
    ap.add_argument("--depth", type=int, default=15)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=200,
                    help="Positions per worker task")
    ap.add_argument("--resume", action="store_true",
                    help="Skip positions that already have stockfish_cp set")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)

    # Add column if missing
    cols = [r[1] for r in conn.execute("PRAGMA table_info(positions)").fetchall()]
    if "stockfish_cp" not in cols:
        conn.execute("ALTER TABLE positions ADD COLUMN stockfish_cp INTEGER")
        conn.commit()
        print("Added stockfish_cp column")

    # Count work
    if args.resume:
        total = conn.execute(
            "SELECT COUNT(*) FROM positions WHERE stockfish_cp IS NULL"
        ).fetchone()[0]
        query = "SELECT id, fen FROM positions WHERE stockfish_cp IS NULL ORDER BY id"
    else:
        total = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        query = "SELECT id, fen FROM positions ORDER BY id"

    print(f"Evaluating {total:,} positions at depth {args.depth} with {args.workers} workers")

    # Build batches
    cur = conn.execute(query)
    batches = []
    batch = []
    for row in cur:
        batch.append((row[0], row[1]))
        if len(batch) >= args.batch_size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)

    print(f"Created {len(batches)} batches of ~{args.batch_size}")

    conn.close()

    t0 = time.time()
    done = 0

    pool = mp.Pool(
        args.workers,
        initializer=worker_init,
        initargs=(args.stockfish, args.depth),
    )

    # Write results in main process
    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    try:
        for results in pool.imap_unordered(worker_eval, batches):
            conn.executemany(
                "UPDATE positions SET stockfish_cp = ? WHERE id = ?",
                [(cp, rid) for rid, cp in results],
            )
            done += len(results)

            if done % 10000 < args.batch_size:
                conn.commit()
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (total - done) / rate if rate > 0 else 0
                print(
                    f"  {done:,}/{total:,} ({100*done/total:.1f}%) "
                    f"| {rate:.0f} pos/s | ETA {eta/60:.1f}m"
                )

        conn.commit()
    finally:
        pool.terminate()
        pool.join()
        conn.close()

    elapsed = time.time() - t0
    print(f"Done. {done:,} positions in {elapsed:.0f}s ({done/elapsed:.0f} pos/s)")


if __name__ == "__main__":
    main()
