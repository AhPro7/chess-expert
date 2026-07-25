"""Play against Chess Expert with a clickable board — in Colab or Jupyter.

Uses ipywidgets (built into Colab), so there's no external site and no JS glue.
Click one of your pieces, then click where it should go. The engine replies.

In a Colab/Jupyter cell:

    from demo.colab_gui import play
    play("models/chess_expert.pt")          # you are White
    play("models/chess_expert.pt", human_white=False)  # you are Black

The game logic (resolve_move / legal_targets) is plain python-chess so it can be
unit-tested without a display; the widget layer on top is thin.
"""

from __future__ import annotations

import chess

# Use the FILLED glyphs for every piece and colour them white/black via CSS.
# (The outline glyphs render nearly invisible on light squares.)
_GLYPH = {
    chess.PAWN: "♟", chess.KNIGHT: "♞", chess.BISHOP: "♝",
    chess.ROOK: "♜", chess.QUEEN: "♛", chess.KING: "♚",
}
_LIGHT = "#EBECD0"
_DARK = "#779556"
_SEL = "#F6F169"      # selected square
_TARGET = "#F7EC74"   # legal destination
_LAST = "#F6F169"     # last move

# CSS injected once so pieces are big, solid, and clearly two-tone.
_CSS = """
<style>
.ce-sq {
  margin: 0 !important;
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  font-family: 'Segoe UI Symbol','Noto Sans Symbols2','DejaVu Sans',sans-serif !important;
  font-weight: 400 !important;
  line-height: 1 !important;
  transition: none !important;
}
.ce-sq:hover { filter: brightness(1.05); }
.ce-white { color: #FFFFFF !important;
  text-shadow: 0 0 1px #000, 0 0 2px #000, 0 1px 1px rgba(0,0,0,.55) !important; }
.ce-black { color: #111111 !important;
  text-shadow: 0 0 1px rgba(255,255,255,.35) !important; }
.ce-board { border: 3px solid #3a3a3a; display: inline-block; line-height: 0; }
.ce-coord { color:#bbb; font-size:12px; font-family:monospace; }
</style>
"""


# ---- Pure game logic (no widgets — testable) -----------------------------

def legal_targets(board: chess.Board, from_square: int) -> set[int]:
    """Squares this piece can legally move to."""
    return {m.to_square for m in board.legal_moves if m.from_square == from_square}


def resolve_move(board: chess.Board, from_square: int, to_square: int) -> chess.Move | None:
    """Return the legal Move for from->to, auto-promoting to queen. None if illegal."""
    move = chess.Move(from_square, to_square)
    if move in board.legal_moves:
        return move
    promo = chess.Move(from_square, to_square, promotion=chess.QUEEN)
    if promo in board.legal_moves:
        return promo
    return None


# ---- Widget layer --------------------------------------------------------

def play(checkpoint: str = "models/chess_expert.pt", human_white: bool = True,
         temperature: float = 0.0, square_size: int = 72):
    """Launch the clickable board.

    temperature>0 adds variety to engine moves; square_size sets the board size
    in pixels (pieces scale with it).
    """
    import ipywidgets as widgets
    from IPython.display import HTML, display

    from src.play import ChessEngine

    SQ = int(square_size)
    LABEL = max(18, SQ // 3)
    display(HTML(_CSS))  # base styling
    # Force the piece glyph size (Jupyter's default button CSS otherwise wins).
    display(HTML(
        f"<style>.ce-sq{{font-size:{int(SQ * 0.86)}px !important;"
        f"line-height:1 !important;padding:0 !important;}}</style>"
    ))

    engine = ChessEngine(checkpoint)
    board = chess.Board()
    human_color = chess.WHITE if human_white else chess.BLACK
    state = {"selected": None, "last_move": None}

    status = widgets.HTML()
    buttons: dict[int, widgets.Button] = {}

    # Rows are displayed from the human's back rank at the bottom.
    ranks = range(7, -1, -1) if human_white else range(8)
    files = range(8) if human_white else range(7, -1, -1)

    def square_color(sq: int) -> str:
        return _LIGHT if (chess.square_rank(sq) + chess.square_file(sq)) % 2 else _DARK

    def render():
        targets = (
            legal_targets(board, state["selected"])
            if state["selected"] is not None else set()
        )
        last = state["last_move"]
        for sq, btn in buttons.items():
            piece = board.piece_at(sq)
            btn.description = _GLYPH[piece.piece_type] if piece else ""
            # Piece colour via CSS class (buttons can't set text colour directly).
            btn.remove_class("ce-white")
            btn.remove_class("ce-black")
            if piece is not None:
                btn.add_class("ce-white" if piece.color == chess.WHITE else "ce-black")

            if sq == state["selected"]:
                color = _SEL
            elif sq in targets:
                color = _TARGET
            elif last and sq in (last.from_square, last.to_square):
                color = _LAST
            else:
                color = square_color(sq)
            btn.style.button_color = color

        if board.is_game_over(claim_draw=True):
            msg = f"🏁 Game over — {board.result(claim_draw=True)}"
        else:
            turn = "Your move" if board.turn == human_color else "Engine thinking…"
            chk = "  ⚠️ CHECK" if board.is_check() else ""
            msg = f"{turn}{chk}"
        status.value = (
            f"<div style='font-size:17px;font-weight:600;margin:2px 0 8px 2px'>"
            f"{msg}</div>"
        )

    def engine_reply():
        if board.turn != human_color and not board.is_game_over(claim_draw=True):
            move = engine.select_move(board, temperature=temperature)
            if move is not None:
                board.push(move)
                state["last_move"] = move

    def on_click(sq: int):
        if board.turn != human_color or board.is_game_over(claim_draw=True):
            return
        piece = board.piece_at(sq)
        if state["selected"] is None:
            # Select one of your own pieces.
            if piece is not None and piece.color == human_color:
                state["selected"] = sq
        else:
            move = resolve_move(board, state["selected"], sq)
            if move is not None:
                board.push(move)
                state["last_move"] = move
                state["selected"] = None
                render()
                engine_reply()
            elif piece is not None and piece.color == human_color:
                state["selected"] = sq  # reselect
            else:
                state["selected"] = None  # deselect
        render()

    def coord(text, w, h):
        return widgets.HTML(
            f"<div class='ce-coord' style='width:{w}px;height:{h}px;display:flex;"
            f"align-items:center;justify-content:center'>{text}</div>"
        )

    grid_rows = []
    for r in ranks:
        row = [coord(str(r + 1), LABEL, SQ)]  # rank number on the left
        for f in files:
            sq = chess.square(f, r)
            btn = widgets.Button(
                description="",
                layout=widgets.Layout(
                    width=f"{SQ}px", height=f"{SQ}px", padding="0px", margin="0px"
                ),
            )
            btn.add_class("ce-sq")
            btn.on_click(lambda b, s=sq: on_click(s))
            buttons[sq] = btn
            row.append(btn)
        grid_rows.append(widgets.HBox(row, layout=widgets.Layout(margin="0px")))

    # File letters (a–h) under the board, aligned with the columns (width = SQ).
    file_letters = "abcdefgh" if human_white else "hgfedcba"
    file_row = widgets.HBox(
        [coord("", LABEL, LABEL)] + [coord(c, SQ, LABEL) for c in file_letters],
        layout=widgets.Layout(margin="0px"),
    )
    board_box = widgets.VBox(grid_rows + [file_row], layout=widgets.Layout(margin="0px"))
    board_box.add_class("ce-board")

    def new_game(_):
        board.reset()
        state["selected"] = None
        state["last_move"] = None
        engine_reply()  # engine moves first if human is Black
        render()

    new_btn = widgets.Button(
        description="New game", button_style="success", icon="refresh",
        layout=widgets.Layout(width="140px", margin="10px 0 0 0"),
    )
    new_btn.on_click(new_game)

    display(widgets.VBox([status, board_box, new_btn]))
    engine_reply()  # if the human is Black, the engine (White) opens
    render()
