"""Scrape floodgate (wdoor) CSA games and build a shogi positions DB.

Mirrors parser.py (chess/lichess) for shogi. Downloads CSA game files from
http://wdoor.c.u-tokyo.ac.jp/shogi/x/<year>/<month>/<day>/, replays each game
with python-shogi, and records one row per position with the full SFEN (for
engine labeling) and the game outcome.

Output schema (board_states):
    id INTEGER PRIMARY KEY
    game_id INTEGER      -- sequential per game
    turn INTEGER         -- ply index (0-based); ply 4 used for opening fingerprint
    full_sfen TEXT       -- "board stm hands movenum" (feed to Fairy-Stockfish)
    winner TEXT          -- 'b' / 'w' / '-' (draw/unknown)
    engine_cp INTEGER    -- filled later by shogi_eval.py

Usage:
    python parse_shogi_floodgate.py --out shogi_positions.sqlite \
        --year 2024 --months 7,8 --max-games 4000 --workers 24
"""

import argparse
import os
import re
import sqlite3
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import shogi
import shogi.CSA

BASE = "http://wdoor.c.u-tokyo.ac.jp/shogi/x"
START_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"

HREF_RE = re.compile(r'href="([^"]+)"')


def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "aall-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_dir(url):
    """Return entry names (hrefs) under a floodgate directory URL."""
    try:
        html = fetch(url).decode("utf-8", "replace")
    except Exception:
        return []
    return [h for h in HREF_RE.findall(html) if not h.startswith("?") and not h.startswith("/")]


def collect_game_urls(year, months, max_games):
    """Walk year/month/day dirs and collect CSA game URLs up to max_games."""
    urls = []
    for month in months:
        m_url = f"{BASE}/{year}/{month:02d}/"
        days = sorted([d.strip("/") for d in list_dir(m_url) if d.strip("/").isdigit()], reverse=True)
        for day in days:
            d_url = f"{m_url}{day}/"
            for name in list_dir(d_url):
                if name.endswith(".csa"):
                    urls.append(d_url + name)
                    if len(urls) >= max_games:
                        return urls
    return urls


def download_one(url, dest_dir):
    name = url.rsplit("/", 1)[-1]
    path = os.path.join(dest_dir, name)
    if os.path.exists(path):
        return path
    try:
        data = fetch(url)
        with open(path, "wb") as f:
            f.write(data)
        return path
    except Exception:
        return None


def parse_game(path):
    """Parse one CSA file -> list of (turn, full_sfen, winner). Skip non-standard."""
    try:
        games = shogi.CSA.Parser.parse_file(path)
    except Exception:
        return None
    if not games:
        return None
    g = games[0]
    if g.get("sfen") != START_SFEN:
        return None  # only standard initial position
    win = g.get("win", "-")
    if win not in ("b", "w"):
        win = "-"
    moves = g.get("moves", [])
    if len(moves) < 8:
        return None

    rows = []
    board = shogi.Board()
    for ply, mv in enumerate(moves):
        rows.append((ply, board.sfen(), win))  # position BEFORE the move
        try:
            board.push_usi(mv)
        except Exception:
            break
    # final position after last move
    rows.append((len(moves), board.sfen(), win))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--months", default="7,8", help="comma-separated month numbers")
    ap.add_argument("--max-games", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--cache-dir", default="floodgate_csa")
    ap.add_argument("--max-pos-per-game", type=int, default=80)
    args = ap.parse_args()

    months = [int(m) for m in args.months.split(",")]
    os.makedirs(args.cache_dir, exist_ok=True)

    print(f"Collecting up to {args.max_games} game URLs from {args.year} months {months} ...")
    urls = collect_game_urls(args.year, months, args.max_games)
    print(f"  {len(urls)} game URLs")

    print(f"Downloading with {args.workers} workers ...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        paths = list(ex.map(lambda u: download_one(u, args.cache_dir), urls))
    paths = [p for p in paths if p]
    print(f"  {len(paths)} files downloaded")

    conn = sqlite3.connect(args.out)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("DROP TABLE IF EXISTS board_states")
    conn.execute("""
        CREATE TABLE board_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER NOT NULL,
            turn INTEGER NOT NULL,
            full_sfen TEXT NOT NULL,
            winner TEXT NOT NULL,
            engine_cp INTEGER
        )
    """)

    gid = 0
    total_pos = 0
    kept_games = 0
    buf = []
    for path in paths:
        rows = parse_game(path)
        if not rows:
            continue
        gid += 1
        kept_games += 1
        # cap positions per game: keep ply<=6 (openings) + an even sample of the rest
        if len(rows) > args.max_pos_per_game:
            head = [r for r in rows if r[0] <= 6]
            rest = [r for r in rows if r[0] > 6]
            step = max(1, len(rest) // (args.max_pos_per_game - len(head)))
            rows = head + rest[::step]
        for (turn, sfen, win) in rows:
            buf.append((gid, turn, sfen, win))
            total_pos += 1
        if len(buf) >= 20000:
            conn.executemany(
                "INSERT INTO board_states (game_id, turn, full_sfen, winner) VALUES (?,?,?,?)", buf)
            conn.commit()
            buf.clear()
            print(f"  games={kept_games} positions={total_pos}")

    if buf:
        conn.executemany(
            "INSERT INTO board_states (game_id, turn, full_sfen, winner) VALUES (?,?,?,?)", buf)
        conn.commit()

    conn.execute("CREATE INDEX idx_bs_game ON board_states(game_id)")
    conn.execute("CREATE INDEX idx_bs_cp ON board_states(engine_cp)")
    conn.commit()
    conn.close()
    print(f"Done. {kept_games} games, {total_pos} positions -> {args.out}")


if __name__ == "__main__":
    main()
