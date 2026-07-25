"""Gradio Space: play against Chess Expert in the browser.

Self-contained Gradio SDK Space (no Docker): the engine code (`src/`, `demo/`) is
bundled in the Space repo next to this file, and the trained model is downloaded
from the Hugging Face Hub at startup.
"""

import os
import sys

MODEL_REPO = os.environ.get("MODEL_REPO", "Ahmed007/chess-expert")
MODEL_FILE = os.environ.get("MODEL_FILE", "chess_expert.pt")

# The bundled src/ and demo/ packages sit next to this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess  # noqa: E402
import gradio as gr  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from src.play import ChessEngine  # noqa: E402
from demo.render import render_board  # noqa: E402

# Load the trained model from the HF model repo.
_ckpt = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
engine = ChessEngine(_ckpt)
SEARCH_DEPTH = 2 if engine.has_value else 0  # look-ahead if value head is trained


def _legal_san(board: chess.Board) -> list[str]:
    return [board.san(m) for m in board.legal_moves]


def _status(board: chess.Board, human_color: str) -> str:
    if board.is_game_over(claim_draw=True):
        return f"### 🏁 Game over — {board.result(claim_draw=True)}"
    human_turn = (board.turn == chess.WHITE) == (human_color == "White")
    who = "**Your move**" if human_turn else "Engine thinking…"
    check = "  ⚠️ CHECK" if board.is_check() else ""
    return f"### {who}{check}"


def _engine_reply(board: chess.Board):
    if not board.is_game_over(claim_draw=True):
        move = engine.select_move(board, temperature=0.3, top_k=5, depth=SEARCH_DEPTH)
        if move is not None:
            board.push(move)
            return move
    return None


def new_game(human_color):
    board = chess.Board()
    last = None
    if human_color == "Black":  # engine (White) opens
        last = _engine_reply(board)
    return (
        render_board(board, last_move=last),
        gr.update(choices=_legal_san(board), value=None),
        _status(board, human_color),
        board.fen(),
    )


def play_move(fen, san, human_color):
    board = chess.Board(fen)
    if not san:
        return (render_board(board), gr.update(choices=_legal_san(board)),
                "Pick a move from the dropdown, then press **Play**.", fen)
    try:
        move = board.parse_san(san)
    except ValueError:
        return (render_board(board), gr.update(choices=_legal_san(board)),
                "That move isn't legal here — pick another.", fen)

    board.push(move)
    last = move
    reply = _engine_reply(board)
    if reply is not None:
        last = reply
    return (
        render_board(board, last_move=last),
        gr.update(choices=_legal_san(board), value=None),
        _status(board, human_color),
        board.fen(),
    )


with gr.Blocks(title="Chess Expert") as demo:
    gr.Markdown(
        "# ♟️ Chess Expert\n"
        "A neural network that learned chess from **grandmaster games** "
        "(behavioral cloning + a value head for shallow look-ahead). "
        "Pick a move from the dropdown and press **Play**. "
        "It plays solid, human-like chess — and, having only shallow search, "
        "the occasional blunder. [Code](https://github.com/AhPro7/chess-expert)"
    )
    color = gr.Radio(["White", "Black"], value="White", label="You play")
    board_img = gr.Image(type="pil", label="Board", height=460)
    move_dd = gr.Dropdown(choices=[], label="Your move")
    with gr.Row():
        play_btn = gr.Button("▶ Play move", variant="primary")
        new_btn = gr.Button("↻ New game")
    status = gr.Markdown()
    fen = gr.State(chess.Board().fen())

    outputs = [board_img, move_dd, status, fen]
    new_btn.click(new_game, [color], outputs)
    color.change(new_game, [color], outputs)
    play_btn.click(play_move, [fen, move_dd, color], outputs)
    demo.load(new_game, [color], outputs)

if __name__ == "__main__":
    demo.launch()
