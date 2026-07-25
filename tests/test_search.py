"""Tests for the look-ahead search — independent of the neural net.

The one thing that silently ruins a value/search feature is the sign convention
(value is from the side-to-move's perspective; negamax negates on recursion). We
pin it down here with a *material* leaf evaluation, so these tests verify the
search + sign logic without needing a trained value head. When the trained value
head drops into the same slot, the convention is already proven correct.

Run: pytest -q  /  python -m tests.test_search
"""

from __future__ import annotations

import os
import sys

import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.play import material_eval, negamax, search_move  # noqa: E402


def test_material_eval_is_side_to_move_pov():
    # White is up a queen. From White's POV (White to move) that's positive...
    fen = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
    assert material_eval(chess.Board(fen)) > 0
    # ...and from Black's POV (same position, Black to move) it's negative.
    fen_black = "4k3/8/8/8/8/8/8/3QK3 b - - 0 1"
    assert material_eval(chess.Board(fen_black)) < 0


def test_search_avoids_hanging_the_queen():
    """Qxd5 wins a pawn but the e6 pawn recaptures the queen — search must avoid it."""
    board = chess.Board("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1")
    qxd5 = chess.Move.from_uci("d1d5")
    assert qxd5 in board.legal_moves

    best, scores = search_move(board, depth=2, leaf_eval=material_eval)
    assert scores[qxd5] < 0, "hanging the queen should score as losing material"
    assert best != qxd5, "search must not choose the move that hangs the queen"


def test_search_takes_free_material():
    """An undefended enemy queen on d4 should be captured."""
    board = chess.Board("4k3/8/8/8/3q4/8/8/3QK3 w - - 0 1")
    qxd4 = chess.Move.from_uci("d1d4")
    assert qxd4 in board.legal_moves

    best, scores = search_move(board, depth=2, leaf_eval=material_eval)
    assert best == qxd4, "search should grab the free queen"
    assert scores[qxd4] > 0


def test_negamax_scores_checkmate():
    """A side-to-move that is checkmated is the worst possible score."""
    # Fool's-mate position: White is checkmated, White to move.
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert board.is_checkmate()
    assert negamax(board, depth=1, leaf_eval=material_eval) < -1000


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nAll search tests passed.")
