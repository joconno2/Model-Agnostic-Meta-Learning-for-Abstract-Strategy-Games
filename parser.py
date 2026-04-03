import argparse
import io
import os
import re
import sqlite3
from typing import Optional, Tuple

import chess
import chess.pgn
import zstandard as zstd


PROMO_MAP = {"n": 1, "b": 2, "r": 3, "q": 4}
N_ACTIONS = 64 * 64 * 5

def uci_to_action_id(uci: str) -> int:
    frm = chess.parse_square(uci[0:2])
    to = chess.parse_square(uci[2:4])
    promo = 0
    if len(uci) == 5:
        promo = PROMO_MAP.get(uci[4], 0)
    return (frm * 64 + to) * 5 + promo

def result_to_winner(result_str: str) -> Optional[bool]:
    # returns chess.WHITE / chess.BLACK as bool, or None for draw/unknown
    if result_str == "1-0":
        return chess.WHITE
    if result_str == "0-1":
        return chess.BLACK
    if result_str == "1/2-1/2":
        return None
    return None

def z_from_perspective(result_str: str, side_to_move: bool) -> int:
    # side_to_move: chess.WHITE(True) or chess.BLACK(False)
    if result_str == "1/2-1/2":
        return 0
    winner = result_to_winner(result_str)
    if winner is None:
        return 0
    return 1 if winner == side_to_move else -1

def parse_time_control(tc: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Lichess TimeControl examples:
      "600+5" (base seconds + increment)
      "300+0"
      "-" (unknown)
    Returns (base, inc) in seconds or (None, None)
    """
    if not tc or tc == "-" or tc == "?":
        return None, None
    m = re.match(r"^(\d+)\+(\d+)$", tc.strip())
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def open_pgn_zst_stream(path: str):
    f = open(path, "rb")
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(f)
    text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace", newline="")
    return f, reader, text

def init_db(conn: sqlite3.Connection):
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS games (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        site TEXT,
        event TEXT,
        utc_date TEXT,
        utc_time TEXT,
        result TEXT,
        termination TEXT,
        eco TEXT,
        opening TEXT,
        time_control TEXT,
        tc_base INTEGER,
        tc_inc INTEGER,
        white TEXT,
        black TEXT,
        white_elo INTEGER,
        black_elo INTEGER
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id INTEGER NOT NULL,
        ply INTEGER NOT NULL,
        turn INTEGER NOT NULL,              -- 0 white to move, 1 black to move
        fen TEXT NOT NULL,
        move_uci TEXT NOT NULL,
        action_id INTEGER NOT NULL,
        z INTEGER NOT NULL,                 -- outcome from side-to-move perspective
        FOREIGN KEY (game_id) REFERENCES games(id)
    );
    """)

    # Helpful indices for training queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_game ON positions(game_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_action ON positions(action_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_z ON positions(z);")
    conn.commit()

def header_int(headers, key: str) -> Optional[int]:
    v = headers.get(key)
    if v is None:
        return None
    try:
        return int(v)
    except:
        return None

def should_keep_game(headers, min_elo: Optional[int], min_base: Optional[int], only_rated: bool) -> bool:
    # Rated games usually have [Event "Rated ..."]
    event = headers.get("Event", "") or ""
    if only_rated and "Rated" not in event:
        return False

    we = header_int(headers, "WhiteElo")
    be = header_int(headers, "BlackElo")
    if min_elo is not None:
        if we is None or be is None:
            returnFalse
        if min(we, be) < min_elo:
            return False

    if min_base is not None:
        tc = headers.get("TimeControl", "")
        base, _inc = parse_time_control(tc)
        if base is None or base < min_base:
            return False

    # Keep only standard results
    res = headers.get("Result", "*")
    if res not in ("1-0", "0-1", "1/2-1/2"):
        return False

    return True

def main():
    ap = argparse.ArgumentParser(description="Stream parse Lichess .pgn.zst into SQLite for policy/value training.")
    ap.add_argument("--input", required=True, help="Path to .pgn.zst file")
    ap.add_argument("--output", required=True, help="Output .sqlite file path")
    ap.add_argument("--max-games", type=int, default=0, help="Max games to parse (0 = no limit)")
    ap.add_argument("--min-elo", type=int, default=0, help="Minimum of (WhiteElo, BlackElo). 0 disables.")
    ap.add_argument("--min-base", type=int, default=0, help="Minimum base time in seconds (e.g. 600). 0 disables.")
    ap.add_argument("--rated-only", action="store_true", help="Keep only games with Event containing 'Rated'")
    ap.add_argument("--batch", type=int, default=20000, help="How many position rows to batch-insert at once")
    args = ap.parse_args()

    in_path = args.input
    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    min_elo = args.min_elo if args.min_elo > 0 else None
    min_base = args.min_base if args.min_base > 0 else None

    conn = sqlite3.connect(out_path)
    init_db(conn)

    f, reader, text = open_pgn_zst_stream(in_path)

    game_count = 0
    kept_games = 0
    pos_buffer = []

    try:
        while True:
            game = chess.pgn.read_game(text)
            if game is None:
                break
            game_count += 1
            if args.max_games and game_count > args.max_games:
                break

            headers = game.headers
            if not should_keep_game(headers, min_elo=min_elo, min_base=min_base, only_rated=args.rated_only):
                continue

            result = headers.get("Result", "*")
            event = headers.get("Event")
            site = headers.get("Site")
            utc_date = headers.get("UTCDate")
            utc_time = headers.get("UTCTime")
            termination = headers.get("Termination")
            eco = headers.get("ECO")
            opening = headers.get("Opening")
            time_control = headers.get("TimeControl")
            tc_base, tc_inc = parse_time_control(time_control or "")

            white = headers.get("White")
            black = headers.get("Black")
            white_elo = header_int(headers, "WhiteElo")
            black_elo = header_int(headers, "BlackElo")

            cur = conn.execute("""
                INSERT INTO games(site,event,utc_date,utc_time,result,termination,eco,opening,time_control,tc_base,tc_inc,
                                  white,black,white_elo,black_elo)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (site, event, utc_date, utc_time, result, termination, eco, opening, time_control, tc_base, tc_inc,
                  white, black, white_elo, black_elo))
            game_id = cur.lastrowid
            kept_games += 1

            board = game.board()
            ply = 0

            for move in game.mainline_moves():
                fen = board.fen()
                turn = 0 if board.turn == chess.WHITE else 1
                uci = move.uci()
                action_id = uci_to_action_id(uci)
                if not (0 <= action_id < N_ACTIONS):
                    # Extremely rare, but safe-guard
                    board.push(move)
                    ply += 1
                    continue

                z = z_from_perspective(result, board.turn)

                pos_buffer.append((game_id, ply, turn, fen, uci, action_id, z))

                board.push(move)
                ply += 1

                # batch insert
                if len(pos_buffer) >= args.batch:
                    conn.executemany("""
                        INSERT INTO positions(game_id, ply, turn, fen, move_uci, action_id, z)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, pos_buffer)
                    conn.commit()
                    pos_buffer.clear()

            # periodic commit for games table row
            if kept_games % 2000 == 0:
                conn.commit()
                print(f"Parsed games: {game_count:,} | kept: {kept_games:,} | buffered positions: {len(pos_buffer):,}")

        # flush remaining
        if pos_buffer:
            conn.executemany("""
                INSERT INTO positions(game_id, ply, turn, fen, move_uci, action_id, z)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, pos_buffer)
            conn.commit()
            pos_buffer.clear()

        print(f"Done. Total games read: {game_count:,}. Games kept: {kept_games:,}. SQLite: {out_path}")

    finally:
        try:
            text.close()
        except:
            pass
        try:
            reader.close()
        except:
            pass
        try:
            f.close()
        except:
            pass
        conn.close()

if __name__ == "__main__":
    main()
 
