---
title: Chess Expert
emoji: ♟️
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Chess Expert — play a neural net that learned from grandmaster games

Behavioral-cloning chess model (policy + value heads) with shallow look-ahead
search. Self-contained Gradio Space (no Docker): the engine code is bundled here
and the model is downloaded from the
[`Ahmed007/chess-expert`](https://huggingface.co/Ahmed007/chess-expert) model repo.

Pick a move from the dropdown and press **Play**. Configure via Space variables:
`MODEL_REPO`, `MODEL_FILE`.

Source: https://github.com/AhPro7/chess-expert
