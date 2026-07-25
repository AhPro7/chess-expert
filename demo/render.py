"""Render a chess board to a PNG image with matplotlib (no cairo needed).

Pieces are drawn with the filled Unicode glyphs and coloured white/black with a
thin outline so they're readable on both light and dark squares.
"""

from __future__ import annotations

import chess
import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from PIL import Image

# Filled glyphs used for BOTH colours; colour is applied via text colour.
_GLYPH = {
    chess.PAWN: "♟",
    chess.KNIGHT: "♞",
    chess.BISHOP: "♝",
    chess.ROOK: "♜",
    chess.QUEEN: "♛",
    chess.KING: "♚",
}

_LIGHT = "#EEEED2"
_DARK = "#769656"
_LAST = "#F6F669"  # highlight for the last move's squares


def render_board(board: chess.Board, last_move: chess.Move | None = None,
                 size: int = 480) -> Image.Image:
    """Return a PIL Image of the current position (White at the bottom)."""
    dpi = 80
    inches = size / dpi
    fig, ax = plt.subplots(figsize=(inches, inches), dpi=dpi)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    highlight = set()
    if last_move is not None:
        highlight = {last_move.from_square, last_move.to_square}

    for square in chess.SQUARES:
        col = square % 8
        row = square // 8
        light = (row + col) % 2 == 1
        color = _LIGHT if light else _DARK
        if square in highlight:
            color = _LAST
        ax.add_patch(plt.Rectangle((col, row), 1, 1, facecolor=color, edgecolor="none"))

        piece = board.piece_at(square)
        if piece is not None:
            glyph = _GLYPH[piece.piece_type]
            txt_color = "white" if piece.color == chess.WHITE else "#202020"
            stroke = "#202020" if piece.color == chess.WHITE else "#f0f0f0"
            ax.text(
                col + 0.5, row + 0.5, glyph,
                fontsize=size / 16, ha="center", va="center",
                color=txt_color,
                path_effects=[path_effects.withStroke(linewidth=1.2, foreground=stroke)],
            )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    img = Image.frombytes(
        "RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()
    ).convert("RGB")
    plt.close(fig)
    return img
