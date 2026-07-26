"""Play the model vs Stockfish across several Elo caps and save every WIN as a PGN.

Designed for a Colab GPU run. For each Elo, plays N games (model alternates
colours); games that the model wins are saved for the highlight reel.

    python scripts/tournament.py --elos 1300 1400 1500 1600 1700 \
        --games 3 --depth 5 --out demo/wins

NOTE on depth: the search calls the network at every node and branches wide, so
cost grows exponentially. depth 4–6 is the realistic range; depth ~12 will not
finish. Use a GPU (`--device cuda`) to speed the network up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import chess
import chess.engine
import chess.pgn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from src.play import ChessEngine  # noqa: E402


def play_game(engine, sf, model_white, depth, temp, sf_time, max_plies, headers):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers.update(headers)
    node = game
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        if (board.turn == chess.WHITE) == model_white:
            mv = engine.select_move(board, depth=depth, eval_mode="material",
                                    temperature=temp, top_k=5)
        else:
            mv = sf.play(board, chess.engine.Limit(time=sf_time)).move
        if mv is None:
            break
        board.push(mv)
        node = node.add_variation(mv)
        plies += 1
    result = board.result(claim_draw=True)
    game.headers["Result"] = result
    model_won = board.is_checkmate() and ((board.turn == chess.BLACK) == model_white)
    return result, game, model_won, plies


def main() -> None:
    p = argparse.ArgumentParser(description="Model vs Stockfish tournament → save wins")
    p.add_argument("--checkpoint", default="models/chess_expert.pt")
    p.add_argument("--elos", type=int, nargs="+", default=[1300, 1400, 1500, 1600, 1700])
    p.add_argument("--games", type=int, default=3, help="games per Elo")
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--temp", type=float, default=0.15)
    p.add_argument("--sf-time", type=float, default=0.1)
    p.add_argument("--max-plies", type=int, default=220)
    p.add_argument("--out", default="demo/wins")
    p.add_argument("--device", default="", help="cuda / cpu (auto if empty)")
    p.add_argument("--stockfish", default="stockfish")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | depth {args.depth}")
    if args.depth >= 8:
        print("⚠️  depth >= 8 may take a very long time per move — consider 4–6.")

    engine = ChessEngine(args.checkpoint, device=device)
    sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    os.makedirs(args.out, exist_ok=True)

    wins, tally = [], []
    try:
        for elo in args.elos:
            sf.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            w = d = l = 0
            for g in range(args.games):
                model_white = (g % 2 == 0)
                names = ("Chess Expert", f"Stockfish {elo}")
                headers = {
                    "Event": "Chess Expert vs Stockfish",
                    "Site": "github.com/AhPro7/chess-expert",
                    "White": names[0] if model_white else names[1],
                    "Black": names[1] if model_white else names[0],
                }
                result, game, won, plies = play_game(
                    engine, sf, model_white, args.depth, args.temp,
                    args.sf_time, args.max_plies, headers)
                w += won
                d += result == "1/2-1/2" or (result == "*" and not won)
                l += not won and result not in ("1/2-1/2", "*")
                tag = "WIN ✅" if won else ("loss" if result in ("1-0", "0-1") else "draw")
                side = "White" if model_white else "Black"
                print(f"  SF {elo} · game {g+1}/{args.games} · model {side} "
                      f"→ {result} [{tag}]  ({plies} plies)", flush=True)
                if won:
                    path = os.path.join(args.out, f"win_{elo}_g{g+1}.pgn")
                    with open(path, "w") as fh:
                        print(game, file=fh)
                    wins.append({"pgn": os.path.basename(path), "elo": elo,
                                 "model_white": model_white})
            tally.append((elo, w, d, l))
    finally:
        sf.quit()

    with open(os.path.join(args.out, "wins.json"), "w") as fh:
        json.dump(wins, fh, indent=2)

    print("\n=== Results (model score per Elo) ===")
    for elo, w, d, l in tally:
        print(f"  vs Stockfish {elo}: {w}W {d}D {l}L")
    print(f"\n{len(wins)} winning game(s) saved to {args.out}/  "
          f"→ build the reel with scripts/win_reel.py")


if __name__ == "__main__":
    main()
