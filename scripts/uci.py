"""A minimal UCI engine wrapper around ChessExpert.

This makes the model usable by any UCI GUI (Arena, Cute Chess, BanksiaGUI) and,
importantly, by **lichess-bot** — so it can play rated games on Lichess and earn a
real bot rating.

    # try it in a terminal:
    python scripts/uci.py
    > uci
    > position startpos moves e2e4
    > go
    < bestmove ...

Options (via `setoption`):
    Depth   search depth for the material tactical-safety search (default 2)
    Temp    move-sampling temperature ×100 (0 = always best; default 0)
"""

from __future__ import annotations

import os
import sys

import chess

from src.play import ChessEngine

CHECKPOINT = os.environ.get("CE_CHECKPOINT", "models/chess_expert.pt")


def main() -> None:
    engine = ChessEngine(CHECKPOINT)
    board = chess.Board()
    depth = int(os.environ.get("CE_DEPTH", "2"))
    temp = 0.0

    def out(line: str) -> None:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    for raw in sys.stdin:
        cmd = raw.strip()
        if cmd == "uci":
            out("id name ChessExpert")
            out("id author AhPro7 (github.com/AhPro7/chess-expert)")
            out("option name Depth type spin default 2 min 0 max 4")
            out("option name Temp type spin default 0 min 0 max 200")
            out("uciok")
        elif cmd == "isready":
            out("readyok")
        elif cmd == "ucinewgame":
            board = chess.Board()
        elif cmd.startswith("setoption"):
            parts = cmd.split()
            if "Depth" in parts:
                depth = int(parts[parts.index("value") + 1])
            elif "Temp" in parts:
                temp = int(parts[parts.index("value") + 1]) / 100.0
        elif cmd.startswith("position"):
            tokens = cmd.split()
            if "startpos" in tokens:
                board = chess.Board()
                idx = tokens.index("startpos") + 1
            elif "fen" in tokens:
                fen = " ".join(tokens[tokens.index("fen") + 1: tokens.index("fen") + 7])
                board = chess.Board(fen)
                idx = tokens.index("fen") + 7
            else:
                idx = len(tokens)
            if idx < len(tokens) and tokens[idx] == "moves":
                for uci in tokens[idx + 1:]:
                    board.push(chess.Move.from_uci(uci))
        elif cmd.startswith("go"):
            move = engine.select_move(board, depth=depth, eval_mode="material",
                                      temperature=temp, top_k=5)
            if move is None:  # no legal move
                out("bestmove 0000")
            else:
                out(f"bestmove {move.uci()}")
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
