#Channel layout

from dataclasses import dataclass
from typing import Dict, List

#Board piece tables (both Chess and Shogi)
#Shared planes where it best fits
#Shared: P, N, B, R, K 
#Chess only --> Q
#Shogi only: L, S, G, +P, +L, +N, + S, H

BOARD_PIECE_TYPES: list[str] = [
    "P", "N", "B", "R", "Q", "K",   # chess core (Q chess-only)
    "L", "S", "G",                  # shogi-only base pieces
    "+P", "+L", "+N", "+S",         # promoted pawn/lance/knight/silver
    "H", "D",                       # horse, dragon
]

HAND_PIECE_TYPES: List[str] = ["P", "L", "N", "S", "G", "B", "R"]  # shogi hand pieces

#Normalization purpose
HAND_MAX_COUNTS: Dict[str, int] = {
    "P": 18,
    "L": 4,
    "N": 4,
    "S": 4,
    "G": 4,
    "B": 2,
    "R": 2,
}

@dataclass(frozen = True)
class UnifiedSpec:
    #board is always padded for the 9x9 dimension 
    H: int = 9
    W: int = 9

    #Inclide a side to move plane. 
    include_side_to_move_plane: bool = True

def build_channel_index(spec: UnifiedSpec):
    """
    Returns mapping: channel_name -> channel_index
    """
    idx = {}
    c = 0

    # Board piece planes (my side)
    for t in BOARD_PIECE_TYPES:
        idx[f"my_board_{t}"] = c
        c += 1

    # Board piece planes (opponent side)
    for t in BOARD_PIECE_TYPES:
        idx[f"opp_board_{t}"] = c
        c += 1

    # Side to move plane
    if spec.include_side_to_move_plane:
        idx["side_to_move"] = c
        c += 1

    # Shogi hand planes (my side)
    for t in HAND_PIECE_TYPES:
        idx[f"my_hand_{t}"] = c
        c += 1

    # Shogi hand planes (opponent side)
    for t in HAND_PIECE_TYPES:
        idx[f"opp_hand_{t}"] = c
        c += 1

    return idx


def num_channels(spec: UnifiedSpec) -> int:
    return len(build_channel_index(spec))

