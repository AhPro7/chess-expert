# ♟️ Chess Expert

[![Play online](https://img.shields.io/badge/🤗%20Hugging%20Face-Play%20online-yellow)](https://huggingface.co/spaces/Ahmed007/chess-expert)
[![Model](https://img.shields.io/badge/🤗%20Model-Ahmed007%2Fchess--expert-blue)](https://huggingface.co/Ahmed007/chess-expert)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep-learning model that plays chess. It learns purely from **grandmaster games** —
no hand-written rules, no opening book, no search engine. Given a position, a
convolutional neural network predicts the move a strong player would make.

This is **behavioral cloning**, the same idea behind the published
[Maia](https://maiachess.com/) engine: the network imitates human grandmasters. It
plays solid, human-like chess — and, because it has no search, it will occasionally
make a tactical blunder. That honesty is the point: this is an **ML-engineering
project**, not a claim to beat Stockfish.

<!-- Add your GIF here once trained: -->
<!-- ![Chess Expert self-play](demo/self_play.gif) -->

---

## How it works

| Step | What happens |
|------|--------------|
| **Encode** | Each board → a `17 × 8 × 8` tensor: 12 piece planes (6 types × 2 colors), 1 side-to-move plane, 4 castling-rights planes. No board-flipping. |
| **Move encoding** | Every move → an integer `from×64 + to` (4096 classes). Promotions default to queen. |
| **Model** | A residual CNN — **128 channels × 10 residual blocks** — with two heads: a **policy** (4096 move logits) and a **value** (how good the position is for the side to move, in −1…1). |
| **Train** | Policy: cross-entropy vs. the grandmaster's move. Value: MSE vs. the game outcome. Metrics: *move-match accuracy* + *value MAE*. |
| **Play** | **Mask to legal moves** (never illegal). Optionally **look ahead** with a shallow, policy-guided negamax that scores leaf positions with the value head — so it avoids hanging pieces. |

Training happens on a **Colab GPU**. Inference runs **on CPU** — one forward pass per
move (~0.1–0.3s), no GPU needed to play.

---

## Quickstart (play on CPU — no GPU needed)

First get a trained checkpoint into `models/chess_expert.pt` — either train one on
Colab (see below) or, if you've committed the checkpoint to this repo, it's already
there after cloning.

```bash
git clone https://github.com/YOUR_USERNAME/chess-expert.git
cd chess-expert
pip install -r requirements.txt

# Watch the model play itself and save an animated GIF
python -m demo.make_gif --checkpoint models/chess_expert.pt --out demo/self_play.gif

# Or play against it yourself in the terminal (you are White)
python -m demo.play_cli --checkpoint models/chess_expert.pt
```

Enter moves in UCI (`e2e4`) or SAN (`Nf3`). Type `quit` to exit.

**Prefer a board you can click?** In Colab or Jupyter:

```python
from demo.colab_gui import play
play("models/chess_expert.pt")          # you are White — click a piece, then its target
play("models/chess_expert.pt", human_white=False)  # play Black
```

> **Tip:** commit your trained `models/chess_expert.pt` to the repo so anyone who
> clones can play instantly (the `.gitignore` is already set up to keep that one file).

---

## Train your own (Colab)

Open **`notebooks/train_colab.ipynb`** in Google Colab and run it top-to-bottom:

1. Set the runtime to **GPU** (and High-RAM if available).
2. Point `REPO_URL` at your fork.
3. It downloads grandmaster archives, parses them, trains, and lets you download
   `chess_expert.pt`.

Want a stronger model? Train on **more games** and for **more epochs** — that's the
lever, not a bigger network.

### Or run the pipeline manually

```bash
# 1. Download grandmaster PGNs → data/gm_games.pgn
bash scripts/download_data.sh

# 2. Parse into memory-mapped training samples (uint8, RAM-safe)
python -m src.data --pgn data/gm_games.pgn --out data/samples --min-elo 0

# 3. Train (GPU strongly recommended)
python -m src.train --data data/samples --out models/chess_expert.pt \
    --epochs 20 --batch-size 4096
```

**Speed:** TF32 + cuDNN autotune are always on for CUDA; on Ampere+ (L4/A100) add
`--amp on` for bf16 mixed precision. **Resume** a stopped run with
`--resume models/chess_expert.resume.pt` (a sidecar written every epoch);
`--epochs` is the total target.

**Monitoring:** training logs to `runs/` for TensorBoard — loss (total/policy/value),
move-match, value MAE, throughput, LR, and a **self-play board filmstrip each epoch**
(Images tab). View with `tensorboard --logdir runs`.

**Share:** push the checkpoint + logs to the Hugging Face Hub:

```bash
huggingface-cli login
python scripts/upload_hf.py --repo-id YOUR_USERNAME/chess-expert
```

**Play online (free, runs in the browser):** the model is exported to **ONNX** and
runs client-side with onnxruntime-web — no server, no GPU, no paid Space. The site
lives in [`docs/`](docs/) (`index.html` + vendored `chess.js` + the `.onnx`).

```bash
# 1. Export the trained model to ONNX
python scripts/export_onnx.py --checkpoint models/chess_expert.pt --out docs/chess_expert.onnx
```

Then host it either way (both free):

- **GitHub Pages:** commit `docs/chess_expert.onnx`, push, then enable
  *Settings → Pages → Source: `main` / `docs`*. Live at
  `https://YOUR_USERNAME.github.io/chess-expert/`.
- **Hugging Face Static Space** (free, unlike Gradio which now needs PRO):
  ```bash
  huggingface-cli login
  python scripts/create_static_space.py --repo-id YOUR_USERNAME/chess-expert
  ```

> A Gradio version also exists in [`space/`](space/), but hosting Gradio Spaces on
> HF now requires a PRO subscription — the static ONNX site above is the free path.

---

## Project layout

```
src/
  encoding.py   board ↔ tensor, move ↔ index (legal-aware)
  model.py      ChessPolicyNet (residual CNN)
  data.py       PGN → uint8 samples (memory-mapped)
  train.py      training loop + move-match accuracy
  play.py       ChessEngine — legal-masked move selection (CPU)
demo/
  render.py     matplotlib board renderer (no external deps)
  make_gif.py   self-play → animated GIF  ← the shareable artifact
  play_cli.py   play against the model in your terminal
notebooks/
  train_colab.ipynb   the GPU training driver
scripts/
  download_data.sh    fetch grandmaster PGN archives
```

---

## Not hanging pieces (policy + a material safety net)

The policy alone just imitates grandmaster moves — no notion of the future — so it
sometimes hangs a piece. The stronger levels fix this with a **shallow negamax
search**:

- **Candidates = the policy's top moves ∪ every capture ∪ every check.** Including
  all captures is the key: the opponent's refuting capture is usually *outside* the
  policy's favourites, so a top-K-only search never sees the piece being taken.
- **Leaves are scored by material** (with a capture *quiescence*), i.e. "did that
  lose or win a piece?" — not by the neural value head. Measured in self-play, the
  value head is too weak/noisy and searching on it actually made the model *weaker*
  (31% vs a plain greedy policy); the **material** search beats greedy (56%) and,
  more importantly, stops the free hangs a human punishes.

This is honest: the *ideas* still come 100% from the grandmaster-trained model —
material is just basic piece values (a "don't give away pieces" rule), **not** an
engine evaluation.

```python
engine.select_move(board, depth=2, eval_mode="material")
```

Quantify any change with the self-play arena: `python -m scripts.arena --a-depth 2 --b-depth 0`.

## Honest limitations

- **Shallow tactics only** → catches hangs and short combinations, not deep plans.
- **Style first** → it imitates the *distribution* of grandmaster moves; the search
  is a thin tactical safety net on top.
- **The value head is weak** (game-outcome labels are noisy); it ships in the model
  but the search deliberately ignores it in favour of material.
- **v2 idea:** a sharper policy (more GM data), deeper search / MCTS.
  <img width="480" height="480" alt="self_play" src="https://github.com/user-attachments/assets/b4578c68-ef25-4c60-8cfb-2bba18ea8536" />


## Data & credit

Grandmaster games from [pgnmentor.com](https://www.pgnmentor.com/). Board handling by
[python-chess](https://python-chess.readthedocs.io/). Inspired by the
[Maia Chess](https://maiachess.com/) research on human-like chess models.

## License

MIT.
