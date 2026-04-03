#Chess encoder for input space to the Neural Network Classifier

import chess


N_ACTIONS_CHESS = 64*64*5 #20480
_PROMO_TO_CODE = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}

def uci_to_action_id (uci: str) -> int:
    """
   Should match parser dataset encoding for action_if
   action_id = ((from * 64 + to) * 5 + promo code)
   """
    move = chess.Move.from_uci(uci)
    frm = move.from_square
    to = move.to_square
    promo = _PROMO_TO_CODE.get(move.promotion, 0) #adds promo to uci encoding
    return ((frm * 64 + to) * 5 + promo)

#Legal masking; only taking into account legal moves
def legal_mask_fen(fen: str):
    """
    returns bool mask [20480], True = Legal
    """

    board = chess.Board(fen)
    mask = [False] * N_ACTIONS_CHESS
    for mv in board.legal_moves:
        a = uci_to_action_id(mv.uci())
        mask[a] = True

    return mask



