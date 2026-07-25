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
search. The Space downloads the model from the [`Ahmed007/chess-expert`](https://huggingface.co/Ahmed007/chess-expert)
model repo and the engine code from [GitHub](https://github.com/AhPro7/chess-expert).

Pick a move from the dropdown and press **Play**. Configure via Space secrets/vars:
`MODEL_REPO`, `MODEL_FILE`, `CODE_REPO`.
