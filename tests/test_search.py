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

from src.play import (  # noqa: E402
    material_eval,
    negamax,
    ordered_candidates,
    search_move,
)


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


def test_captures_are_always_candidates_even_with_bad_policy():
    """A capture must be searched even if the policy ranks it dead last.

    This is the tactical blind-spot fix: refuting captures live outside the
    policy's top moves, so a top-K-only candidate set never sees them.
    """
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
    exd5 = chess.Move.from_uci("e4d5")
    assert board.is_capture(exd5)
    # Policy that hates captures (scores them 0, everything else 1) + branch=1.
    policy = {m: (0.0 if board.is_capture(m) else 1.0) for m in board.legal_moves}
    cands = ordered_candidates(board, policy, branch=1)
    assert exd5 in cands, "capture must be force-included as a candidate"


def test_search_sees_refutation_outside_policy():
    """Depth-2 search must avoid hanging the queen even when the refuting capture
    (…exd5) is outside the (mock) policy's preferred moves."""
    board = chess.Board("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1")
    qxd5 = chess.Move.from_uci("d1d5")

    def bad_policy(b):
        return {m: (0.0 if b.is_capture(m) else 1.0) for m in b.legal_moves}

    good = lambda b: ordered_candidates(b, bad_policy(b), branch=1)   # noqa: E731
    best, scores = search_move(board, depth=2, leaf_eval=material_eval,
                               candidate_moves=good)
    assert best != qxd5, "search must not hang the queen"
    assert scores[qxd5] < 0

    # Contrast: a top-1-policy-only candidate set (no forced captures) is blind to
    # the refutation and would happily hang the queen.
    blind = lambda b: sorted(  # noqa: E731
        (m for m in b.legal_moves if m.promotion in (None, chess.QUEEN)),
        key=lambda m: bad_policy(b)[m], reverse=True)[:1]
    _, blind_scores = search_move(board, depth=2, leaf_eval=material_eval,
                                  candidate_moves=blind)
    assert blind_scores.get(qxd5, 0) >= 0, "sanity: blind search fails to see the loss"


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
