"""Play a game against the Chess Expert in your terminal.

You are White by default. Enter moves in UCI (e.g. e2e4, g1f3) or SAN (e.g. Nf3).
Type 'quit' to exit, 'board' to reprint.

Usage:
    python -m demo.play_cli --checkpoint models/chess_expert.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.play import ChessEngine  # noqa: E402


def parse_user_move(text: str, board: chess.Board) -> chess.Move | None:
    text = text.strip()
    try:
        return board.parse_uci(text)
    except ValueError:
        pass
    try:
        return board.parse_san(text)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Play vs Chess Expert")
    parser.add_argument("--checkpoint", default="models/chess_expert.pt")
    parser.add_argument("--engine-white", action="store_true",
                        help="Let the engine play White (you play Black)")
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    engine = ChessEngine(args.checkpoint)
    board = chess.Board()
    engine_color = chess.WHITE if args.engine_white else chess.BLACK
    print("You are", "Black" if args.engine_white else "White",
          "— enter moves as UCI (e2e4) or SAN (Nf3). 'quit' to exit.\n")
    print(board, "\n")

    while not board.is_game_over():
        if board.turn == engine_color:
            move = engine.select_move(board, temperature=args.temperature)
            if move is None:
                break
            print(f"Engine plays: {board.san(move)}\n")
            board.push(move)
        else:
            text = input("Your move: ")
            if text.strip().lower() in {"quit", "exit"}:
                print("Goodbye.")
                return
            if text.strip().lower() == "board":
                print(board, "\n")
                continue
            move = parse_user_move(text, board)
            if move is None or move not in board.legal_moves:
                print("Illegal or unparseable move, try again.")
                continue
            board.push(move)
        print(board, "\n")

    print("Game over:", board.result(claim_draw=True))


if __name__ == "__main__":
    main()
