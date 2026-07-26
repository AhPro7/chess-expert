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


def _legalq(board: chess.Board) -> list[chess.Move]:
    """Legal moves, collapsing promotions to queen (matches training/encoding)."""
    return [m for m in board.legal_moves if m.promotion in (None, chess.QUEEN)]


def ordered_candidates(board: chess.Board, policy_scores: dict | None = None,
                       branch: int = 5, max_cand: int = 16) -> list[chess.Move]:
    """Moves to search at a node.

    The set is  top-`branch` by policy  ∪  ALL captures  ∪  ALL checks.
    Including every capture/check is what fixes the tactical blind spot: the
    opponent's refuting capture is usually *outside* the policy's top moves, so a
    top-K-only search never sees the piece being taken. Ordered captures
    (MVV-LVA) → checks → policy so alpha-beta prunes well.

    policy_scores maps move -> score (higher = more likely). None means "no policy"
    (use all legal moves) — used by the material-stub tests.
    """
    legal = _legalq(board)
    if policy_scores:
        by_policy = sorted(legal, key=lambda m: policy_scores.get(m, -1.0), reverse=True)
        keep = set(by_policy[:branch])
    else:
        keep = set(legal)
    for m in legal:                       # force-include every capture and check
        if board.is_capture(m) or board.gives_check(m):
            keep.add(m)

    def key(m: chess.Move):
        if board.is_capture(m):
            victim = (chess.PAWN if board.is_en_passant(m)
                      else board.piece_at(m.to_square).piece_type)
            attacker = board.piece_at(m.from_square).piece_type
            # MVV-LVA: prefer taking a big piece with a small one.
            return (3, _PIECE_VALUE[victim] * 10 - _PIECE_VALUE[attacker])
        if board.gives_check(m):
            return (2, 0.0)
        return (1, policy_scores.get(m, 0.0) if policy_scores else 0.0)

    ordered = sorted(keep, key=key, reverse=True)
    return ordered[:max_cand] if max_cand else ordered


def negamax(board: chess.Board, depth: int, leaf_eval, candidate_moves=None,
            alpha: float = -math.inf, beta: float = math.inf) -> float:
    """Negamax with alpha-beta: best score for the side to move, `depth` ahead.

    leaf_eval(board) -> float from the side-to-move POV.
    candidate_moves(board) -> iterable of moves to search (default: all legal).
    """
    terminal = _terminal_score(board)
    if terminal is not None:
        return terminal
    if depth <= 0:
        return leaf_eval(board)

    moves = candidate_moves(board) if candidate_moves else _legalq(board)
    best = -math.inf
    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, leaf_eval, candidate_moves, -beta, -alpha)
        board.pop()
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break  # prune
    return best


def search_move(board: chess.Board, depth: int, leaf_eval, candidate_moves=None):
    """Return (best_move, {move: score}) via depth-`depth` negamax at the root.

    Every root candidate is searched with a full window so all scores are exact
    (the caller may sample among near-best moves for variety).
    """
    roots = list(candidate_moves(board)) if candidate_moves else _legalq(board)
    scores: dict[chess.Move, float] = {}
    for move in roots:
        board.push(move)
        scores[move] = -negamax(board, depth - 1, leaf_eval, candidate_moves)
        board.pop()
    if not scores:
        return None, {}
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

    def _candidates(self, board: chess.Board, branch: int) -> list[chess.Move]:
        """Search candidates: top-`branch` policy moves ∪ all captures ∪ all checks."""
        return ordered_candidates(board, self.move_probabilities(board), branch)

    def _leaf_eval(self, mode: str):
        """Leaf evaluator for the search (side-to-move POV).

        'value'    — the neural value head (weak/OOD; search on it can hurt).
        'material' — plain material count (cheap, honest tactical safety; no engine).
        'blend'    — material (dominant, for tactics) + a small value-head nudge.
        """
        if mode == "material":
            return material_eval
        if mode == "blend":
            return lambda b: material_eval(b) + 0.5 * self.evaluate(b)
        return self.evaluate

    def select_move(
        self,
        board: chess.Board,
        temperature: float = 0.0,
        top_k: int | None = None,
        depth: int = 0,
        branch: int = 4,
        eval_mode: str = "material",
    ) -> chess.Move | None:
        """Choose a legal move.

        depth == 0 (or no trained value head) -> pure policy:
            temperature 0 = greedy; >0 = sample (with optional top_k) for variety.
        depth  > 0 (needs a value-trained checkpoint) -> look `depth` moves ahead
            with a policy-guided negamax, evaluating leaves with the value head.
            This is what avoids hanging pieces / makes it look smart.
        """
        # ---- Search path ----
        # 'material' needs no value head, so search works even for policy-only nets.
        if depth > 0 and (self.has_value or eval_mode == "material"):
            leaf = self._leaf_eval(eval_mode)
            candidates = lambda b: self._candidates(b, branch)  # noqa: E731
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
