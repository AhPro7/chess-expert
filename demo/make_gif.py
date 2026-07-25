"""Generate an animated GIF of the model playing a full game (self-play).

This is the primary artifact for sharing: a short clip of the neural network
picking legal moves on a rendered board.

Usage:
    python -m demo.make_gif --checkpoint models/chess_expert.pt \
        --out demo/self_play.gif --plies 60 --temperature 0.6
"""

from __future__ import annotations

import argparse
import os
import sys

import chess

# Allow running as `python demo/make_gif.py` as well as `-m demo.make_gif`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.render import render_board  # noqa: E402
from src.play import ChessEngine  # noqa: E402


def make_gif(checkpoint: str, out: str, plies: int, temperature: float,
             frame_ms: int) -> None:
    engine = ChessEngine(checkpoint)
    board = chess.Board()

    frames = [render_board(board)]  # opening position
    for ply in range(plies):
        move = engine.select_move(board, temperature=temperature)
        if move is None or board.is_game_over():
            break
        assert move in board.legal_moves, "engine produced an ILLEGAL move!"
        board.push(move)
        frames.append(render_board(board, last_move=move))
        if (ply + 1) % 10 == 0:
            print(f"  {ply + 1} plies rendered")

    # Hold the final position a little longer.
    tail = [frames[-1]] * 6

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:] + tail,
        duration=frame_ms,
        loop=0,
    )
    print(f"Saved {len(frames)} frames -> {out}")
    print("Final result:", board.result(claim_draw=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a self-play GIF")
    parser.add_argument("--checkpoint", default="models/chess_expert.pt")
    parser.add_argument("--out", default="demo/self_play.gif")
    parser.add_argument("--plies", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--frame-ms", type=int, default=550)
    args = parser.parse_args()
    make_gif(args.checkpoint, args.out, args.plies, args.temperature, args.frame_ms)


if __name__ == "__main__":
    main()
