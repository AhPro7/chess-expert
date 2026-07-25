"""Encoding between python-chess objects and neural-network tensors.

Design choices (kept deliberately simple to avoid silent bugs):

* Board  -> tensor of shape (17, 8, 8), float32.
    planes 0-5   : white  P N B R Q K
    planes 6-11  : black  P N B R Q K
    plane  12    : side to move (all 1.0 if White to move, else all 0.0)
    planes 13-16 : castling rights  WK, WQ, BK, BQ (all 1.0 / all 0.0)

  We do NOT flip the board for the side to move. Instead the network is told
  whose turn it is via plane 12. This is slightly less sample-efficient than
  flipping, but it removes an entire class of "train fine / play nonsense"
  bugs caused by inconsistent flipping of the move labels.

* Move  <-> integer in [0, 4095].
    index = from_square * 64 + to_square
  Promotions collapse onto the same (from, to) index; at inference we default
  to promoting to a queen. Under-promotions (to N/B/R) are rare and ignored.

Square indexing follows python-chess: a1 = 0, b1 = 1, ..., h8 = 63.
row = square // 8  (rank), col = square % 8 (file). The same mapping is used
everywhere, so nothing ever needs to be un-flipped.
"""

from __future__ import annotations

import chess
import numpy as np

NUM_PLANES = 17
ACTION_SIZE = 64 * 64  # 4096

# Order matters: matches plane layout above.
_PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Convert a python-chess Board to a (17, 8, 8) float32 numpy array."""
    tensor = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row, col = square // 8, square % 8
        piece_index = _PIECE_TYPES.index(piece.piece_type)
        plane = piece_index + (0 if piece.color == chess.WHITE else 6)
        tensor[plane, row, col] = 1.0

    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    return tensor


def move_to_index(move: chess.Move) -> int:
    """Map a Move to an integer action in [0, 4095]."""
    return move.from_square * 64 + move.to_square


def index_to_move(index: int, board: chess.Board) -> chess.Move | None:
    """Resolve an action index back to a *legal* Move on the given board.

    Because promotions share a (from, to) index, we pick the queen promotion
    when the move is a promotion. Returns None if no legal move matches.
    """
    from_square = index // 64
    to_square = index % 64
    for move in board.legal_moves:
        if move.from_square == from_square and move.to_square == to_square:
            # Prefer the queen promotion among promotion candidates.
            if move.promotion in (None, chess.QUEEN):
                return move
    return None
