# ♟️ Chess Expert

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
| **Model** | A residual CNN — **128 channels × 10 residual blocks** (~4–5M params) → 4096 move logits. Policy-only. |
| **Train** | Cross-entropy against the move the grandmaster actually played. Metric: *move-match accuracy*. |
| **Play** | Run the net, **mask to legal moves**, pick the best. The model can *never* output an illegal move. |

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

## Honest limitations

- **No search** → occasional tactical blunders. It knows what a GM *usually* plays,
  not how to calculate a 5-move combination.
- **Style, not strength** → it imitates the *distribution* of grandmaster moves.
- **v2 idea:** add a value head + a shallow search (alpha-beta / MCTS) to turn the
  imitator into an engine that actively *outplays* opponents.

## Data & credit

Grandmaster games from [pgnmentor.com](https://www.pgnmentor.com/). Board handling by
[python-chess](https://python-chess.readthedocs.io/). Inspired by the
[Maia Chess](https://maiachess.com/) research on human-like chess models.

## License

MIT.
