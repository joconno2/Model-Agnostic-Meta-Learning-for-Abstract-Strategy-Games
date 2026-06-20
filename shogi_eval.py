"""Batch Fairy-Stockfish evaluation of shogi positions in the board_states DB.

Mirror of stockfish_eval.py for shogi. Reads full_sfen from board_states,
evaluates each with Fairy-Stockfish (UCI_Variant=shogi), writes centipawns
(side-to-move perspective) to the engine_cp column. Multiprocessing.

Usage:
    python shogi_eval.py --db shogi_positions.sqlite \
        --engine engines/fairy-stockfish --depth 12 --workers 60
"""

import argparse
import sqlite3
import multiprocessing as mp
import subprocess
import time


def init_engine(engine_path):
    proc = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1,
    )
    proc.stdin.write("uci\n")
    proc.stdin.flush()
    for line in proc.stdout:
        if "uciok" in line:
            break
    proc.stdin.write("setoption name UCI_Variant value shogi\n")
    proc.stdin.write("setoption name Threads value 1\n")
    proc.stdin.write("setoption name Hash value 16\n")
    proc.stdin.write("isready\n")
    proc.stdin.flush()
    for line in proc.stdout:
        if "readyok" in line:
            break
    return proc


def eval_sfen(proc, full_sfen, depth):
    """Return centipawns from side-to-move perspective for a shogi SFEN."""
    proc.stdin.write(f"position sfen {full_sfen}\n")
    proc.stdin.write(f"go depth {depth}\n")
    proc.stdin.flush()
    cp = 0
    for line in proc.stdout:
        if line.startswith("bestmove"):
            break
        if "score cp" in line and "upperbound" not in line and "lowerbound" not in line:
            parts = line.split()
            cp = int(parts[parts.index("cp") + 1])
        elif "score mate" in line and "upperbound" not in line and "lowerbound" not in line:
            parts = line.split()
            mate_in = int(parts[parts.index("mate") + 1])
            cp = 30000 if mate_in > 0 else -30000
    return cp


_proc = None
_depth = None


def worker_init(engine_path, depth):
    global _proc, _depth
    _proc = init_engine(engine_path)
    _depth = depth


def worker_eval(batch):
    global _proc, _depth
    out = []
    for row_id, sfen in batch:
        out.append((row_id, eval_sfen(_proc, sfen, _depth)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    if args.resume:
        total = conn.execute("SELECT COUNT(*) FROM board_states WHERE engine_cp IS NULL").fetchone()[0]
        query = "SELECT id, full_sfen FROM board_states WHERE engine_cp IS NULL ORDER BY id"
    else:
        total = conn.execute("SELECT COUNT(*) FROM board_states").fetchone()[0]
        query = "SELECT id, full_sfen FROM board_states ORDER BY id"

    print(f"Evaluating {total:,} positions at depth {args.depth} with {args.workers} workers")

    batches, batch = [], []
    for row in conn.execute(query):
        batch.append((row[0], row[1]))
        if len(batch) >= args.batch_size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    conn.close()
    print(f"Created {len(batches)} batches of ~{args.batch_size}")

    t0 = time.time()
    done = 0
    pool = mp.Pool(args.workers, initializer=worker_init, initargs=(args.engine, args.depth))
    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        for results in pool.imap_unordered(worker_eval, batches):
            conn.executemany("UPDATE board_states SET engine_cp = ? WHERE id = ?",
                             [(cp, rid) for rid, cp in results])
            done += len(results)
            if done % 10000 < args.batch_size:
                conn.commit()
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (total - done) / rate if rate else 0
                print(f"  {done:,}/{total:,} ({100*done/total:.1f}%) | {rate:.0f} pos/s | ETA {eta/60:.1f}m")
        conn.commit()
    finally:
        pool.terminate()
        pool.join()
        conn.close()
    el = time.time() - t0
    print(f"Done. {done:,} positions in {el:.0f}s ({done/el:.0f} pos/s)")


if __name__ == "__main__":
    main()
