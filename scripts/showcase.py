"""Play a full game and save a shareable PGN + a captioned GIF.

Great for LinkedIn: a clean clip of the model playing, plus the PGN so people can
replay/analyse it.

    # model vs itself (clean, honest showcase)
    python scripts/showcase.py --out demo/showcase --depth 2

    # model (White) vs Stockfish capped to an Elo it can handle
    python scripts/showcase.py --out demo/showcase --vs-stockfish 1350

Outputs <out>.pgn and <out>.gif.
"""

from __future__ import annotations

import argparse
import os
import sys

import chess
import chess.engine
import chess.pgn
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.render import render_board  # noqa: E402
from src.play import ChessEngine  # noqa: E402


def _font(size: int):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def caption(board_img: Image.Image, title: str, sub: str) -> Image.Image:
    """Add a header strip above the board with a title + subtitle."""
    w = board_img.width
    strip_h = 52
    out = Image.new("RGB", (w, board_img.height + strip_h), (26, 24, 22))
    d = ImageDraw.Draw(out)
    d.text((12, 8), title, fill=(232, 230, 227), font=_font(22))
    d.text((12, 32), sub, fill=(150, 190, 90), font=_font(14))
    out.paste(board_img, (0, strip_h))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Save a showcase game (PGN + GIF)")
    p.add_argument("--checkpoint", default="models/chess_expert.pt")
    p.add_argument("--out", default="demo/showcase")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--temp", type=float, default=0.35)
    p.add_argument("--max-plies", type=int, default=80)
    p.add_argument("--frame-ms", type=int, default=650)
    p.add_argument("--vs-stockfish", type=int, default=0,
                   help="If >0, play White vs Stockfish capped to this Elo")
    p.add_argument("--stockfish", default="stockfish")
    args = p.parse_args()

    engine = ChessEngine(args.checkpoint)
    sf = None
    white_name, black_name = "Chess Expert", "Chess Expert"
    if args.vs_stockfish:
        sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        sf.configure({"UCI_LimitStrength": True, "UCI_Elo": args.vs_stockfish})
        black_name = f"Stockfish {args.vs_stockfish}"

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers.update({"Event": "Chess Expert showcase", "Site": "github.com/AhPro7/chess-expert",
                         "White": white_name, "Black": black_name})
    node = game
    frames = [caption(render_board(board), "Chess Expert", "grandmaster-trained neural net")]

    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < args.max_plies:
        if sf is not None and board.turn == chess.BLACK:
            move = sf.play(board, chess.engine.Limit(time=0.1)).move
        else:
            move = engine.select_move(board, depth=args.depth, eval_mode="material",
                                      temperature=args.temp, top_k=5)
        if move is None:
            break
        san = board.san(move)
        num = board.fullmove_number
        label = f"{num}.{'' if board.turn == chess.WHITE else '..'} {san}"
        board.push(move)
        node = node.add_variation(move)
        frames.append(caption(render_board(board, last_move=move),
                              f"Chess Expert   ·   move {label}",
                              f"{white_name} vs {black_name}"))
        plies += 1

    result = board.result(claim_draw=True)
    game.headers["Result"] = result
    if sf is not None:
        sf.quit()

    # Hold the final position with the result.
    frames += [caption(render_board(board), f"Result: {result}",
                       f"{white_name} vs {black_name}")] * 6

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    gif_path, pgn_path = args.out + ".gif", args.out + ".pgn"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=args.frame_ms, loop=0)
    with open(pgn_path, "w") as fh:
        print(game, file=fh)

    print(f"Result: {result}  ({plies} plies)")
    print(f"Saved GIF  -> {gif_path}")
    print(f"Saved PGN  -> {pgn_path}")
    print(f"\nTo make an MP4 (nicer on LinkedIn), if you have ffmpeg:\n"
          f"  ffmpeg -i {gif_path} -movflags faststart -pix_fmt yuv420p "
          f"-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' {args.out}.mp4")


if __name__ == "__main__":
    main()
