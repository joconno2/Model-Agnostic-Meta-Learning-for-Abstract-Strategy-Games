# encode_shogi.py
"""
Encode a Shogi SFEN board string into the unified tensor format.

SFEN board notation:
  - Uppercase = Black (sente), lowercase = White (gote)
  - Pieces: P(awn), L(ance), N(knight), S(ilver), G(old), B(ishop), R(ook), K(ing)
  - Promoted: +P(tokin), +L, +N, +S, +B(horse), +R(dragon)
  - Ranks separated by /, digits = empty squares
  - Rank 1 at top (Black's promotion zone), file 9 at left

Hand notation (separate field in DB):
  - e.g. "2P1B3p1b" = Black has 2P+1B, White has 3p+1b
  - "-" = no pieces in hand
"""

import numpy as np
from spec import UnifiedSpec, build_channel_index, HAND_MAX_COUNTS

# Map SFEN piece characters to unified spec piece types.
# Uppercase = Black pieces, lowercase = White pieces.
_SHOGI_PIECE_MAP = {
    "P": "P", "L": "L", "N": "N", "S": "S", "G": "G",
    "B": "B", "R": "R", "K": "K",
}

# Promoted pieces: +B = Horse (H), +R = Dragon (D), others keep +prefix
_SHOGI_PROMOTED_MAP = {
    "P": "+P", "L": "+L", "N": "+N", "S": "+S",
    "B": "H", "R": "D",
}

# Hand piece characters (only unpromoted pieces can be in hand)
_HAND_PIECES = "PLNSGBR"


def _parse_hand(hand_str: str):
    """
    Parse hand string like "2P1B3p1b" into per-color counts.

    Returns (black_hand, white_hand) where each is a dict {piece: count}.
    """
    black = {}
    white = {}
    if hand_str == "-" or not hand_str:
        return black, white

    i = 0
    while i < len(hand_str):
        # Read optional count
        count = 0
        while i < len(hand_str) and hand_str[i].isdigit():
            count = count * 10 + int(hand_str[i])
            i += 1
        if count == 0:
            count = 1

        if i >= len(hand_str):
            break

        ch = hand_str[i]
        i += 1

        if ch.upper() in _HAND_PIECES:
            if ch.isupper():
                black[ch.upper()] = black.get(ch.upper(), 0) + count
            else:
                white[ch.upper()] = white.get(ch.upper(), 0) + count

    return black, white


def encode_sfen_to_unified(board_str: str, player: str, hand_str: str, spec: UnifiedSpec):
    """
    Encode a Shogi position into the unified tensor format.

    Parameters
    ----------
    board_str : str
        SFEN board string (9 ranks separated by /).
    player : str
        Side to move: "b" (black/sente) or "w" (white/gote).
    hand_str : str
        Hand pieces string, e.g. "2P1b" or "-".
    spec : UnifiedSpec

    Returns
    -------
    x : np.ndarray, float32, shape (C, 9, 9)
    info : dict
    """
    idx = build_channel_index(spec)
    C = len(idx)
    x = np.zeros((C, spec.H, spec.W), dtype=np.float32)

    stm_is_black = (player == "b")

    # Parse board
    ranks = board_str.split("/")
    for rank_idx, rank_str in enumerate(ranks):
        col = 0
        i = 0
        while i < len(rank_str):
            ch = rank_str[i]

            if ch.isdigit():
                col += int(ch)
                i += 1
                continue

            # Check for promotion prefix
            promoted = False
            if ch == "+":
                promoted = True
                i += 1
                if i >= len(rank_str):
                    break
                ch = rank_str[i]

            piece_is_black = ch.isupper()
            base = ch.upper()

            if promoted and base in _SHOGI_PROMOTED_MAP:
                ptype = _SHOGI_PROMOTED_MAP[base]
            elif base in _SHOGI_PIECE_MAP:
                ptype = _SHOGI_PIECE_MAP[base]
            else:
                # Unknown piece, skip
                col += 1
                i += 1
                continue

            is_my_piece = (piece_is_black == stm_is_black)
            prefix = "my" if is_my_piece else "opp"
            ch_name = f"{prefix}_board_{ptype}"

            if ch_name in idx:
                # Shogi: rank_idx is row (0=top), col is column (0=left=file 9)
                x[idx[ch_name], rank_idx, col] = 1.0

            col += 1
            i += 1

    # Side to move plane
    if spec.include_side_to_move_plane:
        x[idx["side_to_move"], :, :] = 1.0

    # Hand pieces
    black_hand, white_hand = _parse_hand(hand_str)

    my_hand = black_hand if stm_is_black else white_hand
    opp_hand = white_hand if stm_is_black else black_hand

    for piece, count in my_hand.items():
        ch_name = f"my_hand_{piece}"
        if ch_name in idx and piece in HAND_MAX_COUNTS:
            x[idx[ch_name], :, :] = count / HAND_MAX_COUNTS[piece]

    for piece, count in opp_hand.items():
        ch_name = f"opp_hand_{piece}"
        if ch_name in idx and piece in HAND_MAX_COUNTS:
            x[idx[ch_name], :, :] = count / HAND_MAX_COUNTS[piece]

    info = {"sfen": board_str, "turn": "black" if stm_is_black else "white"}
    return x, info


if __name__ == "__main__":
    spec = UnifiedSpec()
    # Starting position
    board = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL"
    x, info = encode_sfen_to_unified(board, "b", "-", spec)
    print("OK")
    print("turn:", info["turn"])
    print("tensor shape:", x.shape)
    print("non-zero channels:", [i for i in range(x.shape[0]) if x[i].any()])

    # Test with hand pieces and promoted piece
    board2 = "lnsgk1snl/1r4g2/p1pppp1pp/6p2/1p5P1/2P6/PP+bPPPP1P/3S3R1/LN1GKGSNL"
    x2, info2 = encode_sfen_to_unified(board2, "b", "1B1b", spec)
    print("\nWith promoted bishop (horse) and hands:")
    print("turn:", info2["turn"])
    print("non-zero channels:", [i for i in range(x2.shape[0]) if x2[i].any()])
