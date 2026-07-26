"""Turn the tournament's winning PGNs into one cool highlight reel (MP4 or GIF).

    python scripts/win_reel.py --wins demo/wins --out demo/wins_reel

Intro card → for each win: a matchup title card, the game playing out (board with
player names + last move), then the final position with a green "WIN" banner →
outro card. Encodes MP4 with ffmpeg when available (much smaller), else a GIF.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import chess
import chess.pgn
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.render import render_board  # noqa: E402

BG = (22, 21, 19)
FG = (232, 230, 227)
GREEN = (140, 190, 80)


def _font(size: int):
    for path in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _centered(d, y, text, font, fill, w):
    tw = d.textlength(text, font=font)
    d.text(((w - tw) / 2, y), text, fill=fill, font=font)


def title_card(size, big, small, accent=FG):
    w, h = size
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    _centered(d, h // 2 - 46, big, _font(34), accent, w)
    _centered(d, h // 2 + 6, small, _font(20), (150, 148, 144), w)
    return img


def board_frame(board, white, black, last_move=None, last_label="", banner=""):
    bimg = render_board(board, last_move=last_move)
    w = bimg.width
    bar = 46
    out = Image.new("RGB", (w, bimg.height + 2 * bar), BG)
    d = ImageDraw.Draw(out)

    def player(y, name, is_white, right):
        cy = y + bar // 2
        d.ellipse([14, cy - 9, 32, cy + 9],
                  fill=(240, 240, 240) if is_white else (20, 20, 20),
                  outline=(150, 150, 150))
        d.text((42, cy - 11), name, fill=FG, font=_font(19))
        if right:
            d.text((w - 14 - d.textlength(right, font=_font(14)), cy - 8),
                   right, fill=GREEN, font=_font(14))

    player(0, black, False, last_label if ".." in last_label else "")
    out.paste(bimg, (0, bar))
    player(bar + bimg.height, white, True,
           last_label if ".." not in last_label else "")

    if banner:
        bw, bh = 160, 48
        bx, by = (w - bw) // 2, bar + (bimg.height - bh) // 2
        d.rectangle([bx, by, bx + bw, by + bh], fill=(30, 110, 40), outline=GREEN, width=3)
        f = _font(26)
        tw = d.textlength(banner, font=f)
        d.text((bx + (bw - tw) / 2, by + (bh - 30) / 2), banner,
               fill=(255, 255, 255), font=f)
    return out


def game_frames(pgn_path, hold_start=4, hold_end=10):
    game = chess.pgn.read_game(open(pgn_path))
    white = game.headers.get("White", "White")
    black = game.headers.get("Black", "Black")
    result = game.headers.get("Result", "*")
    board = game.board()
    frames = [board_frame(board, white, black)] * hold_start
    for mv in game.mainline_moves():
        san = board.san(mv)
        label = f"{board.fullmove_number}.{'' if board.turn == chess.WHITE else '..'} {san}"
        board.push(mv)
        frames.append(board_frame(board, white, black, last_move=mv, last_label=label))
    winner = "Chess Expert" if (
        (result == "1-0" and "Chess Expert" in white) or
        (result == "0-1" and "Chess Expert" in black)) else ""
    frames += [board_frame(board, white, black,
                           banner="WIN" if winner else result)] * hold_end
    return frames, white, black, result


def main() -> None:
    p = argparse.ArgumentParser(description="Build a highlight reel from winning PGNs")
    p.add_argument("--wins", default="demo/wins", help="dir of winning PGNs (+wins.json)")
    p.add_argument("--out", default="demo/wins_reel")
    p.add_argument("--fps", type=int, default=3)
    args = p.parse_args()

    manifest = os.path.join(args.wins, "wins.json")
    if os.path.exists(manifest):
        entries = json.load(open(manifest))
        pgns = [os.path.join(args.wins, e["pgn"]) for e in entries]
    else:
        pgns = sorted(glob.glob(os.path.join(args.wins, "*.pgn")))
    if not pgns:
        raise SystemExit(f"No winning PGNs found in {args.wins}/")
    print(f"Building reel from {len(pgns)} win(s)")

    size = board_frame(chess.Board(), "a", "b").size
    frames = [title_card(size, "♟  Chess Expert", "beats Stockfish — highlight reel")] * 8
    for i, pgn in enumerate(pgns, 1):
        gframes, white, black, result = game_frames(pgn)
        opp = black if "Chess Expert" in white else white
        frames += [title_card(size, f"Game {i}", f"Chess Expert  vs  {opp}", GREEN)] * 6
        frames += gframes
        print(f"  + game {i}: {white} vs {black} ({result}) — {len(gframes)} frames")
    frames += [title_card(size, f"{len(pgns)} wins", "github.com/AhPro7/chess-expert")] * 10

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if shutil.which("ffmpeg"):
        tmp = tempfile.mkdtemp()
        for i, fr in enumerate(frames):
            fr.save(os.path.join(tmp, f"f{i:05d}.png"))
        out_mp4 = args.out + ".mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(args.fps), "-i",
             os.path.join(tmp, "f%05d.png"), "-pix_fmt", "yuv420p",
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out_mp4],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"\nSaved reel -> {out_mp4}  ({len(frames)} frames)")
    else:
        out_gif = args.out + ".gif"
        frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                       duration=int(1000 / args.fps), loop=0)
        print(f"\nffmpeg not found — saved GIF -> {out_gif} (may be large). "
              f"Install ffmpeg for a compact MP4.")


if __name__ == "__main__":
    main()
