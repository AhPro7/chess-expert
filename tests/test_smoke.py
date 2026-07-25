"""Fast, training-free tests for the core pieces.

These use a randomly-initialized model — they check the *plumbing* (shapes,
encoding round-trips, legal-move guarantees, promotion handling), not strength.

Run with either:
    pytest -q
    python -m tests.test_smoke
"""

from __future__ import annotations

import os
import sys
import tempfile

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encoding import (  # noqa: E402
    ACTION_SIZE,
    NUM_PLANES,
    board_to_tensor,
    index_to_move,
    move_to_index,
)
from src.model import ChessPolicyNet  # noqa: E402
from src.play import ChessEngine  # noqa: E402


def _random_engine() -> ChessEngine:
    net = ChessPolicyNet()
    net.eval()
    tmp = os.path.join(tempfile.gettempdir(), "chess_expert_test.pt")
    torch.save({"state_dict": net.state_dict(), "config": net.config}, tmp)
    return ChessEngine(tmp)


def test_encoding_shape_and_counts():
    t = board_to_tensor(chess.Board())
    assert t.shape == (NUM_PLANES, 8, 8)
    assert t.dtype == np.float32
    # start position: 32 pieces + 64 (white to move) + 4*64 castling = 352
    assert int(t.sum()) == 352


def test_move_index_round_trip():
    board = chess.Board()
    for move in board.legal_moves:
        assert index_to_move(move_to_index(move), board) == move


def test_model_forward_shape():
    net = ChessPolicyNet()
    out = net(torch.zeros(2, NUM_PLANES, 8, 8))
    assert out.shape == (2, ACTION_SIZE)


def test_full_game_is_always_legal():
    engine = _random_engine()
    board = chess.Board()
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < 200:
        move = engine.select_move(board, temperature=0.5)
        if move is None:
            break
        assert move in board.legal_moves
        board.push(move)
        plies += 1
    assert plies > 0


def test_promotion_is_always_queen():
    """A pawn about to promote must never under-promote (Q only, matching labels)."""
    engine = _random_engine()
    board = chess.Board("k7/4P3/8/8/8/8/8/4K3 w - - 0 1")  # e7 pawn free to promote
    candidates = engine.move_probabilities(board)
    promotions = [m for m in candidates if m.promotion is not None]
    assert any(m.promotion == chess.QUEEN for m in candidates)
    assert all(m.promotion == chess.QUEEN for m in promotions)
    assert all(m in board.legal_moves for m in candidates)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nAll smoke tests passed.")
