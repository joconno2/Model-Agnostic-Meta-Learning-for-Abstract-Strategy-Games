# encode_chess.py
import numpy as np
import chess

from spec import UnifiedSpec, build_channel_index

_CHESS_TYPE_MAP = {
    chess.PAWN: "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
    chess.KING: "K",
}

def encode_fen_to_unified(fen: str, spec: UnifiedSpec):
    """
    Returns:
      x: float32 [C, 9, 9]
      info: dict

    Relative encoding:
      - "my_*" = side-to-move
      - "opp_*" = opponent
    Chess board is 8x8 embedded in top-left of 9x9.
    """
    idx = build_channel_index(spec)
    C = len(idx)
    x = np.zeros((C, spec.H, spec.W), dtype=np.float32)

    board = chess.Board(fen)
    stm_is_white = (board.turn == chess.WHITE)

    for sq, piece in board.piece_map().items():
        # 8x8 coords
        r8 = 7 - chess.square_rank(sq)
        c8 = chess.square_file(sq)

        # embed into 9x9 (top-left)
        r, c = r8, c8

        ptype = _CHESS_TYPE_MAP[piece.piece_type]
        piece_is_white = (piece.color == chess.WHITE)
        is_my_piece = (piece_is_white == stm_is_white)

        ch = f"{'my' if is_my_piece else 'opp'}_board_{ptype}"
        x[idx[ch], r, c] = 1.0

    if spec.include_side_to_move_plane:
        x[idx["side_to_move"], :, :] = 1.0  # always 1 under relative encoding

    info = {"fen": fen, "turn": "white" if stm_is_white else "black"}
    return x, info

#Testing

if __name__ == "__main__":
    spec = UnifiedSpec()
    x, info = encode_fen_to_unified(chess.STARTING_FEN, spec)
    print("OK")
    print("turn:", info["turn"])
    print("tensor shape:", x.shape)
