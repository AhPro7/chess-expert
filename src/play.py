"""Inference: load a trained checkpoint and pick moves on CPU.

Two rules matter most here:

1. The engine may only ever return a LEGAL move. We compute logits over all 4096
   (from, to) actions, mask to the legal moves, and choose among those.

2. `value` and search are always from the SIDE-TO-MOVE's perspective. A position
   scores +1 if the side to move is winning, -1 if losing. `negamax` negates the
   child's score on the way up, so both players "want" a high score for
   themselves. Getting this sign wrong makes the engine steer toward *losing* —
   so it is pinned down by a material-stub test (see tests/test_search.py).

The optional search (a shallow, policy-guided negamax that evaluates leaves with
the value head) is what lets the engine look a couple of moves ahead and stop
hanging pieces — i.e. what makes it "look smart".
"""

from __future__ import annotations

import math

import chess
import numpy as np
import torch

from .encoding import board_to_tensor, move_to_index
from .model import ChessPolicyNet

# Scores for terminal nodes, well outside the value head's [-1, 1] range.
MATE_SCORE = 1e6

_PIECE_VALUE = {
    chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
    chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0,
}


# ---- Pure search (no neural net — testable with a stub leaf eval) ---------

def material_eval(board: chess.Board) -> float:
    """Material balance from the SIDE-TO-MOVE's perspective, in pawns.

    Used only to unit-test the search/sign logic independently of the network.
    """
    score = 0.0
    for piece_type, value in _PIECE_VALUE.items():
        score += value * len(board.pieces(piece_type, board.turn))
        score -= value * len(board.pieces(piece_type, not board.turn))
    return score


def _terminal_score(board: chess.Board) -> float | None:
    """Score for a finished game (side-to-move POV), or None if not over."""
    if board.is_checkmate():
        return -MATE_SCORE  # side to move has been mated → worst possible
    if board.is_game_over(claim_draw=True):
        return 0.0          # stalemate / draw
    return None


def negamax(board: chess.Board, depth: int, leaf_eval, candidate_moves=None) -> float:
    """Negamax: best achievable score for the side to move, looking `depth` ahead.

    leaf_eval(board) -> float from the side-to-move POV.
    candidate_moves(board) -> iterable of moves to search (default: all legal).
    """
    terminal = _terminal_score(board)
    if terminal is not None:
        return terminal
    if depth <= 0:
        return leaf_eval(board)

    moves = candidate_moves(board) if candidate_moves else list(board.legal_moves)
    best = -math.inf
    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, leaf_eval, candidate_moves)
        board.pop()
        if score > best:
            best = score
    return best


def search_move(board: chess.Board, depth: int, leaf_eval, candidate_moves=None,
                margin: float = 1e-4):
    """Return (best_move, {move: score}) via depth-`depth` negamax at the root."""
    roots = list(candidate_moves(board)) if candidate_moves else list(board.legal_moves)
    scores: dict[chess.Move, float] = {}
    for move in roots:
        board.push(move)
        scores[move] = -negamax(board, depth - 1, leaf_eval, candidate_moves)
        board.pop()
    if not scores:
        return None, {}
    best = max(scores.values())
    # Among near-best moves (within margin), keep any one — the caller decides
    # whether to pick deterministically or sample for variety.
    best_move = max(scores, key=scores.get)
    return best_move, scores


# ---- Neural engine --------------------------------------------------------

class ChessEngine:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        config = ckpt.get("config", {"channels": 128, "num_blocks": 10})
        self.model = ChessPolicyNet(**config).to(self.device)
        # strict=False so an older policy-only checkpoint still loads (value head
        # stays random); we only enable search when the value head was trained.
        self.model.load_state_dict(ckpt["state_dict"], strict=False)
        self.model.eval()
        self.has_value = bool(ckpt.get("value_trained", False))

    @torch.no_grad()
    def _forward(self, board: chess.Board):
        tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(self.device)
        policy_logits, value = self.model(tensor)
        return policy_logits.squeeze(0).cpu().numpy(), float(value.item())

    def move_probabilities(self, board: chess.Board) -> dict[chess.Move, float]:
        """Softmax probabilities over the *legal* moves for this position."""
        logits, _ = self._forward(board)

        # Promotions: python-chess yields four moves (Q/R/B/N) sharing one (from,
        # to) index/logit. Keep only the queen promotion so each index maps to a
        # unique move (matches training) and we never under-promote on a tie.
        legal = [m for m in board.legal_moves if m.promotion in (None, chess.QUEEN)]
        if not legal:
            return {}

        legal_logits = np.array([logits[move_to_index(m)] for m in legal])
        legal_logits -= legal_logits.max()  # numerical stability
        probs = np.exp(legal_logits)
        probs /= probs.sum()
        return dict(zip(legal, probs))

    @torch.no_grad()
    def evaluate(self, board: chess.Board) -> float:
        """Value-head score for this position, side-to-move POV, in [-1, 1]."""
        terminal = _terminal_score(board)
        if terminal is not None:
            return terminal
        _, value = self._forward(board)
        return value

    def _top_policy_moves(self, board: chess.Board, n: int) -> list[chess.Move]:
        """The n most likely legal moves (search only these, to stay fast)."""
        probs = self.move_probabilities(board)
        ordered = sorted(probs, key=probs.get, reverse=True)
        return ordered[:n] if n and n > 0 else ordered

    def select_move(
        self,
        board: chess.Board,
        temperature: float = 0.0,
        top_k: int | None = None,
        depth: int = 0,
        branch: int = 4,
    ) -> chess.Move | None:
        """Choose a legal move.

        depth == 0 (or no trained value head) -> pure policy:
            temperature 0 = greedy; >0 = sample (with optional top_k) for variety.
        depth  > 0 (needs a value-trained checkpoint) -> look `depth` moves ahead
            with a policy-guided negamax, evaluating leaves with the value head.
            This is what avoids hanging pieces / makes it look smart.
        """
        # ---- Search path ----
        if depth > 0 and self.has_value:
            leaf = self.evaluate
            candidates = lambda b: self._top_policy_moves(b, branch)  # noqa: E731
            best_move, scores = search_move(board, depth, leaf, candidates)
            if not scores:
                return None
            if temperature <= 0:
                return best_move
            # Non-determinism: sample among moves within a small score margin.
            best = max(scores.values())
            near = [m for m, s in scores.items() if best - s <= 0.05]
            return near[int(np.random.randint(len(near)))] if near else best_move

        # ---- Policy path ----
        probs = self.move_probabilities(board)
        if not probs:
            return None
        moves = list(probs.keys())
        weights = np.array(list(probs.values()), dtype=np.float64)
        if temperature <= 0:
            return moves[int(weights.argmax())]
        if top_k is not None and 0 < top_k < len(moves):
            keep = np.argsort(weights)[-top_k:]
            mask = np.zeros_like(weights)
            mask[keep] = 1.0
            weights = weights * mask
        scaled = weights ** (1.0 / temperature)
        total = scaled.sum()
        if total <= 0 or not np.isfinite(total):
            return moves[int(weights.argmax())]
        scaled /= total
        return moves[int(np.random.choice(len(moves), p=scaled))]


def _demo_self_play(checkpoint: str, max_plies: int = 40, depth: int = 0) -> None:
    """Quick sanity check: play a full game against itself, printing SAN moves."""
    engine = ChessEngine(checkpoint)
    board = chess.Board()
    plies = 0
    while not board.is_game_over() and plies < max_plies:
        move = engine.select_move(board, temperature=0.5, top_k=5, depth=depth)
        if move is None:
            break
        assert move in board.legal_moves, "engine produced an ILLEGAL move!"
        print(f"{plies + 1:2d}. {board.san(move)}")
        board.push(move)
        plies += 1
    print("Result:", board.result(claim_draw=True), "| reason:",
          "game over" if board.is_game_over() else "ply limit")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Self-play sanity check")
    parser.add_argument("--checkpoint", default="models/chess_expert.pt")
    parser.add_argument("--plies", type=int, default=40)
    parser.add_argument("--depth", type=int, default=0,
                        help="Search depth (needs a value-trained checkpoint)")
    args = parser.parse_args()
    _demo_self_play(args.checkpoint, args.plies, args.depth)
