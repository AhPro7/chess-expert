"""Inference: load a trained checkpoint and pick moves on CPU.

The single most important rule here: the model may only ever return a LEGAL
move. We compute logits over all 4096 (from, to) actions, then mask to the
legal moves for the current position and choose among those. An untrained or
weak model still plays legally — it just plays badly.
"""

from __future__ import annotations

import chess
import numpy as np
import torch

from .encoding import board_to_tensor, index_to_move, move_to_index
from .model import ChessPolicyNet


class ChessEngine:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        config = ckpt.get("config", {"channels": 128, "num_blocks": 10})
        self.model = ChessPolicyNet(**config).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def move_probabilities(self, board: chess.Board) -> dict[chess.Move, float]:
        """Softmax probabilities over the *legal* moves for this position."""
        tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(self.device)
        logits = self.model(tensor).squeeze(0).cpu().numpy()

        # For promotions, python-chess yields four moves (Q/R/B/N) that all map
        # to the SAME (from, to) index and therefore share one logit. Keep only
        # the queen promotion so each index maps to a unique move — this matches
        # how training labels collapse promotions, and avoids silently
        # under-promoting on argmax ties. Queen promotion is always legal
        # whenever any promotion is, so this can never empty the list.
        legal = [m for m in board.legal_moves if m.promotion in (None, chess.QUEEN)]
        if not legal:
            return {}

        # Mask: gather the logit for each legal move, softmax over just those.
        legal_logits = np.array([logits[move_to_index(m)] for m in legal])
        legal_logits -= legal_logits.max()  # numerical stability
        probs = np.exp(legal_logits)
        probs /= probs.sum()
        return dict(zip(legal, probs))

    def select_move(
        self, board: chess.Board, temperature: float = 0.0
    ) -> chess.Move | None:
        """Choose a legal move.

        temperature == 0 -> greedy (argmax). > 0 -> sample, higher = more random
        (adds variety to demos so games aren't identical every time).
        """
        probs = self.move_probabilities(board)
        if not probs:
            return None

        moves = list(probs.keys())
        weights = np.array(list(probs.values()))

        if temperature <= 0:
            return moves[int(weights.argmax())]

        scaled = weights ** (1.0 / temperature)
        scaled /= scaled.sum()
        return moves[int(np.random.choice(len(moves), p=scaled))]


def _demo_self_play(checkpoint: str, max_plies: int = 40) -> None:
    """Quick sanity check: play a full game against itself, printing SAN moves."""
    engine = ChessEngine(checkpoint)
    board = chess.Board()
    plies = 0
    while not board.is_game_over() and plies < max_plies:
        move = engine.select_move(board, temperature=0.5)
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
    args = parser.parse_args()
    _demo_self_play(args.checkpoint, args.plies)
