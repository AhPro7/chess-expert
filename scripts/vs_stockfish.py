"""Estimate the model's Elo by playing it against Stockfish capped to fixed Elos.

Stockfish is only a *measuring stick* here — the model is still trained purely on
grandmaster games. We set Stockfish's `UCI_Elo` to a target and see whether our
engine wins/draws/loses; the crossover (~50%) is our estimated rating.

    python scripts/vs_stockfish.py --games 4 --depth 2 --elos 1350 1500 1700 2000

Requires a `stockfish` binary on PATH (brew install stockfish).
"""

from __future__ import annotations

import argparse
import os
import sys

import chess
import chess.engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.play import ChessEngine  # noqa: E402


def play_game(our, sf, our_white, depth, temp, sf_time, max_plies):
    board = chess.Board()
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        if (board.turn == chess.WHITE) == our_white:
            move = our.select_move(board, depth=depth, eval_mode="material",
                                   temperature=temp, top_k=5)
        else:
            move = sf.play(board, chess.engine.Limit(time=sf_time)).move
        if move is None:
            break
        board.push(move)
        plies += 1
    if board.is_checkmate():
        our_won = (board.turn == chess.BLACK) == our_white
        return 1 if our_won else -1
    return 0  # draw / ply-limit


def main() -> None:
    p = argparse.ArgumentParser(description="Estimate Elo vs Stockfish")
    p.add_argument("--checkpoint", default="models/chess_expert.pt")
    p.add_argument("--elos", type=int, nargs="+", default=[1350, 1500, 1700, 2000])
    p.add_argument("--games", type=int, default=4, help="games per Elo level")
    p.add_argument("--depth", type=int, default=2, help="our search depth")
    p.add_argument("--temp", type=float, default=0.15)
    p.add_argument("--sf-time", type=float, default=0.1, help="Stockfish sec/move")
    p.add_argument("--max-plies", type=int, default=160)
    p.add_argument("--stockfish", default="stockfish")
    args = p.parse_args()

    our = ChessEngine(args.checkpoint)
    sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    print(f"Our engine: depth {args.depth} material search vs Stockfish (capped Elo)\n")

    results = []
    try:
        for elo in args.elos:
            sf.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            w = d = l = 0
            for g in range(args.games):
                r = play_game(our, sf, our_white=(g % 2 == 0), depth=args.depth,
                              temp=args.temp, sf_time=args.sf_time,
                              max_plies=args.max_plies)
                w += r == 1; d += r == 0; l += r == -1
            score = w + 0.5 * d
            pct = 100 * score / args.games
            results.append((elo, w, d, l, pct))
            print(f"  vs Stockfish {elo:>4} Elo: {w}W {d}D {l}L  →  {pct:.0f}%")
    finally:
        sf.quit()

    # Estimate: highest Elo where we still score >= 50%.
    beat = [elo for elo, *_ , pct in results if pct >= 50]
    print("\nEstimated strength: "
          + (f"~{max(beat)} Elo or a bit higher (scored ≥50% there)"
             if beat else f"below {min(e for e, *_ in results)} Elo"))


if __name__ == "__main__":
    main()
