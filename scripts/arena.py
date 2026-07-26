"""Self-play arena: measure whether one setting/model beats another.

Plays N games between two engine configs (alternating colors) and reports the
score from A's perspective. Use it to (a) confirm search actually helps
(Master vs Club), and (b) quantify "it got stronger" after retraining
(new-model vs old-model).

    # search vs greedy, same model
    python scripts/arena.py --a-depth 2 --b-depth 0 --games 6

    # new model vs old model, both searching
    python scripts/arena.py --a models/new.pt --b models/old.pt \
        --a-depth 2 --b-depth 2 --games 20
"""

from __future__ import annotations

import argparse
import os
import sys

import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.play import ChessEngine, material_eval  # noqa: E402


def play_game(ea, cfg_a, eb, cfg_b, a_is_white, max_plies, adj):
    """Return +1 if A wins, -1 if B wins, 0 draw.

    If the game hits the ply limit, adjudicate by material: the side up at least
    `adj` pawns of material is credited the win (a strength proxy).
    """
    board = chess.Board()
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        if (board.turn == chess.WHITE) == a_is_white:
            move = ea.select_move(board, **cfg_a)
        else:
            move = eb.select_move(board, **cfg_b)
        if move is None:
            break
        board.push(move)
        plies += 1

    if board.is_checkmate():
        white_won = board.turn == chess.BLACK
        return 1 if (white_won == a_is_white) else -1

    # Adjudicate unfinished games by material (from White's POV).
    mat = material_eval(board) if board.turn == chess.WHITE else -material_eval(board)
    a_material = mat if a_is_white else -mat
    if a_material >= adj:
        return 1
    if a_material <= -adj:
        return -1
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Self-play arena")
    p.add_argument("--a", default="models/chess_expert.pt")
    p.add_argument("--b", default="models/chess_expert.pt")
    p.add_argument("--a-depth", type=int, default=2)
    p.add_argument("--b-depth", type=int, default=0)
    p.add_argument("--a-temp", type=float, default=0.3)
    p.add_argument("--b-temp", type=float, default=0.3)
    p.add_argument("--a-eval", default="material", choices=["value", "material", "blend"])
    p.add_argument("--b-eval", default="material", choices=["value", "material", "blend"])
    p.add_argument("--branch", type=int, default=5)
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--max-plies", type=int, default=120)
    p.add_argument("--adjudicate", type=float, default=3.0,
                   help="Material (pawns) advantage to decide a ply-limited game")
    args = p.parse_args()

    ea = ChessEngine(args.a)
    eb = ChessEngine(args.b) if args.b != args.a else ea
    cfg_a = {"depth": args.a_depth, "temperature": args.a_temp, "branch": args.branch,
             "eval_mode": args.a_eval}
    cfg_b = {"depth": args.b_depth, "temperature": args.b_temp, "branch": args.branch,
             "eval_mode": args.b_eval}

    wins = draws = losses = 0
    for g in range(args.games):
        r = play_game(ea, cfg_a, eb, cfg_b, a_is_white=(g % 2 == 0),
                      max_plies=args.max_plies, adj=args.adjudicate)
        wins += r == 1
        draws += r == 0
        losses += r == -1
        tag = {1: "A win", 0: "draw", -1: "B win"}[r]
        print(f"  game {g+1}/{args.games}: {tag}")

    score = wins + 0.5 * draws
    print(f"\nA (depth {args.a_depth}) vs B (depth {args.b_depth}) over {args.games} games")
    print(f"  A: {wins}W {draws}D {losses}L  →  score {score}/{args.games} "
          f"({100*score/args.games:.0f}%)")


if __name__ == "__main__":
    main()
