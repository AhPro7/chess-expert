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

# Filled Unicode glyphs; coloured via the button's text/background.
_GLYPH = {
    (chess.PAWN, chess.WHITE): "♙", (chess.PAWN, chess.BLACK): "♟",
    (chess.KNIGHT, chess.WHITE): "♘", (chess.KNIGHT, chess.BLACK): "♞",
    (chess.BISHOP, chess.WHITE): "♗", (chess.BISHOP, chess.BLACK): "♝",
    (chess.ROOK, chess.WHITE): "♖", (chess.ROOK, chess.BLACK): "♜",
    (chess.QUEEN, chess.WHITE): "♕", (chess.QUEEN, chess.BLACK): "♛",
    (chess.KING, chess.WHITE): "♔", (chess.KING, chess.BLACK): "♚",
}
_LIGHT = "#EEEED2"
_DARK = "#769656"
_SEL = "#F6F669"      # selected square
_TARGET = "#BBCB2B"   # legal destination


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
         temperature: float = 0.0):
    """Launch the clickable board. temperature>0 adds variety to engine moves."""
    import ipywidgets as widgets
    from IPython.display import display

    from src.play import ChessEngine

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
        for sq, btn in buttons.items():
            piece = board.piece_at(sq)
            btn.description = _GLYPH[(piece.piece_type, piece.color)] if piece else ""
            if sq == state["selected"]:
                color = _SEL
            elif sq in targets:
                color = _TARGET
            elif state["last_move"] and sq in (
                state["last_move"].from_square, state["last_move"].to_square):
                color = _SEL
            else:
                color = square_color(sq)
            btn.style.button_color = color

        if board.is_game_over(claim_draw=True):
            msg = f"Game over — {board.result(claim_draw=True)}"
        else:
            turn = "Your" if board.turn == human_color else "Engine's"
            chk = " — CHECK!" if board.is_check() else ""
            msg = f"{turn} move{chk}"
        status.value = f"<b style='font-size:16px'>{msg}</b>"

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

    # Build the 8x8 grid of buttons.
    grid_rows = []
    for r in ranks:
        row = []
        for f in files:
            sq = chess.square(f, r)
            btn = widgets.Button(
                description="",
                layout=widgets.Layout(width="52px", height="52px", padding="0px"),
            )
            btn.style.font_size = "30px"
            btn.on_click(lambda b, s=sq: on_click(s))
            buttons[sq] = btn
            row.append(btn)
        grid_rows.append(widgets.HBox(row))

    def new_game(_):
        board.reset()
        state["selected"] = None
        state["last_move"] = None
        engine_reply()  # engine moves first if human is Black
        render()

    new_btn = widgets.Button(description="New game", button_style="info")
    new_btn.on_click(new_game)

    display(widgets.VBox([status, widgets.VBox(grid_rows), new_btn]))
    engine_reply()  # if the human is Black, the engine (White) opens
    render()
