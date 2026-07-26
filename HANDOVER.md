# Chess Expert — Session Handover

_A deep-learning chess model that learns from grandmaster games. Built as a
portfolio/PoC to support a master's-degree application. This doc lets a fresh
session continue seamlessly._

---

## TL;DR — what this is
A **behavioral-cloning** chess engine (à la Maia): a residual CNN (128ch × 10
blocks) with a **policy head** (predict the GM's move) and a **value head**
(position score). It plays in the browser via **ONNX**, trains on **Colab GPU**,
and is deployed as free static web demos. It's a **PoC** — honest framing: "a
grandmaster-imitation neural net," NOT an engine-beater.

**Working dir:** `/Users/ahmedhaytham/Desktop/chess-expert` · **Python:** `python3`
(no `python` alias on this Mac).

## Live things (URLs)
- **GitHub repo:** https://github.com/AhPro7/chess-expert  (owner `AhPro7`)
- **GitHub Pages demo:** https://ahpro7.github.io/chess-expert/  (serves `docs/`, auto-updates on push — runs the **v2** model)
- **HF Static Space v1:** https://huggingface.co/spaces/Ahmed007/chess-expert
- **HF Static Space v2:** https://huggingface.co/spaces/Ahmed007/chess-expert-v2  (runs the v2 model)
- **HF Model repos:** `Ahmed007/chess-expert` (v1), `Ahmed007/chess-expert-v2` (v2, the current strong one)

## Auth / credentials
- **GitHub:** pushes use a **Personal Access Token** the user pastes in chat — **exposed, should be rotated.** Pushes are done inline: `git push https://<TOKEN>@github.com/AhPro7/chess-expert.git main` (token is NOT stored in `.git/config`). The repo remote is the clean HTTPS URL.
  - ⚠️ **The Claude Code auto-mode classifier intermittently BLOCKS `git push` when the token is in the URL** (flags the secret). When blocked, **the user must run the push themselves** (`! git push https://<TOKEN>@…`). As of 2026-07-26 commit `0395694` (Colab stockfish fix) **is pushed**; commit `9b38fd0` (docs Grandmaster/Super Master levels) is **NOT** — it needs a user push → GitHub Pages won't show the new levels until then.
- **Hugging Face:** this machine is **already logged in** as `Ahmed007` (cached token at `~/.cache/huggingface/token`). `HfApi()` works without a token arg.
- No SSH keys, no `gh` CLI, no `gh` on the machine.

---

## Current state (DONE)
- ✅ Full pipeline: `src/` (encoding, model, data, train, play) + `demo/` (render, GIF, colab GUI, CLI) + `scripts/` + `docs/` (static web app) + `tests/`.
- ✅ **v2 model** trained on Colab and deployed **everywhere** (`models/chess_expert.pt` locally = v2, `docs/chess_expert.onnx` = v2, both HF v2 repos = v2).
- ✅ Tactical strength fix shipped (see findings). Browser Master no longer hangs pieces.
- ✅ Strength tooling: Stockfish benchmark, self-play arena, tournament + win-reel, UCI wrapper.
- ✅ Colab notebooks: `notebooks/train_colab.ipynb` (train/log/play/upload), `notebooks/beat_stockfish_colab.ipynb` (tournament → win reel).
- ✅ **Web difficulty levels (2026-07-26):** added **Grandmaster** (depth 3, ~10–30 s/move) and **Super Master** (depth 4, ~1–3 min/move) above Master. In `docs/index.html` (GitHub Pages) AND on the **HF v2 Static Space** (surgically edited its custom-styled `index.html`, not overwritten with `docs/`). Browser search is JS+wasm 1-thread, so depth ≥4 is minutes/move — depth 6 is Colab-GPU-only, NOT a browser button.
- ✅ **HF v2 Space renamed (display only, 2026-07-26):** title/`<h1>`/card now **"Chess Expert AI"** (URL unchanged: `Ahmed007/chess-expert-v2`). `docs/`/GitHub Pages still titled "Chess Expert".
- ✅ **Colab Stockfish PATH fix (2026-07-26):** `beat_stockfish_colab.ipynb` setup cell now symlinks `/usr/games/stockfish` → `/usr/local/bin/stockfish` (apt installs off-PATH → `FileNotFoundError: 'stockfish'`). Same fix applies to `vs_stockfish.py`/`tournament.py` (they take `--stockfish`).

## KEY FINDINGS & DECISIONS (do not re-litigate these)
1. **Value-head search HURTS.** Self-play arena: depth-2 **value-head** search scored **31%** vs a plain greedy policy — the value head (game-outcome labels) is too weak/noisy to guide search. **Material** search scored **56%** and stops piece hangs. → The engine's smart levels use **`eval_mode="material"`**, NOT the value head. The value head still ships in the model but search ignores it. **Do not switch search back to the value head.**
2. **The hanging-pieces bug = search blind spot.** Old search only expanded the top-K *policy* moves, so the opponent's refuting capture (usually outside top-K) was invisible. **Fix: candidates = top-K policy ∪ ALL captures ∪ ALL checks** + MVV-LVA ordering + alpha-beta. Pinned by `tests/test_search.py::test_search_sees_refutation_outside_policy`.
3. **Deep search is exponential.** NN is called per node; depth **4–6 is the ceiling**, **depth ~12 will NOT finish**. depth 6 beat Stockfish-1500 once; depth 8 beat 1500 in a one-off match.
   - **Speed levers (added 2026-07-26):** the real lever for depth ≥6 is **narrowing the tree**, not caching. `tournament.py`/`select_move` take **`--branch`** (quiet-move width) and **`--max-cand`** (hard cap per node, truncates AFTER MVV-LVA so refuting captures stay). Local depth-8 bench (quiet pos, CPU): `--branch 4 --max-cand 16` blows up; **`--branch 3 --max-cand 8` = 24 s/move, `--branch 2 --max-cand 6` = 7 s/move, same move.** For depth-8 tournaments use `--branch 3 --max-cand 8`.
   - A **position-keyed NN eval cache** (`board._transposition_key()`, bounded 200k) was added but hit-rate is only **~7%** in quiet positions → minor constant-factor win, NOT the answer. Kept because it's free/safe and helps more in capture-heavy tactics. Real GPU win would be *batching children per node* — NOT built (would need re-pinning the alpha-beta against tests).
4. **Strength:** ~beginner/club, ~1350 vs Stockfish at depth 2 (Stockfish's Elo cap plays stronger than the label, so likely a bit higher on Lichess). Deeper search → stronger.
5. **Encoding parity is sacred.** Board→tensor is 17×8×8; **start-position plane sum must equal 352** (32 pieces + 64 side-to-move + 256 castling). The **JS in `docs/index.html` must mirror `src/encoding.py` exactly** — square index `(rank-1)*8 + file`, flat index `plane*64 + row*8 + col`, plane order white P N B R Q K / black P N B R Q K / side-to-move / WK WQ BK BQ. A console self-test logs the 352 check.
6. **Value sign convention:** value is from the **side-to-move's** POV (+1 = side to move winning). Negamax negates on recursion. Pinned by `tests/test_search.py`.
7. **Data honesty:** trained ONLY on real GM PGN archives (pgnmentor.com, ~40 players). The material safety-net is basic piece values, **NOT an engine** — the user is firm: learn from real experts, **NO Stockfish/engine distillation.** Stockfish is used ONLY as a *measuring stick* for strength, never for training.
8. **Hosting:** **Gradio Spaces now need HF PRO** (402 error). Free path = **Static Spaces** (in-browser ONNX) + **GitHub Pages**. The old Gradio app in `space/` is kept but unused.

## Repo map (key files)
```
src/encoding.py   board↔tensor (17×8×8), move↔index (from*64+to); plane-sum 352
src/model.py      ChessPolicyNet: policy(4096) + value(tanh) heads
src/data.py       PGN→uint8 samples + values.npy; --max-per-pos opening cap
src/train.py      train loop, cosine LR+warmup, TensorBoard, --resume, AMP(off by default)
src/play.py       ChessEngine.select_move(depth, eval_mode, branch); negamax + material
                  eval + candidate-union (captures/checks) + alpha-beta. MATERIAL is default.
demo/render.py    matplotlib board→PIL image (Agg, no cairo)
demo/colab_gui.py ipywidgets clickable board (Colab); uses material search
docs/index.html   SELF-CONTAINED static web app: onnxruntime-web + vendored chess.js.
                  Material search + capture quiescence in JS. Difficulty presets.
                  Loads ./chess_expert.onnx (v2). THE thing users actually play.
docs/chess.js     vendored chess.js 0.13.4 (ES module)
docs/chess_expert.onnx  the deployed model (v2)
scripts/export_onnx.py       checkpoint→ONNX (for docs/ and Spaces)
scripts/create_static_space.py  create/upload a free HF Static Space
scripts/upload_hf.py         upload checkpoint+logs to an HF model repo
scripts/vs_stockfish.py      estimate Elo vs Stockfish (measuring stick)
scripts/arena.py             self-play A vs B (material-adjudicated)
scripts/tournament.py        model vs Stockfish across Elos → save wins
scripts/win_reel.py          stitch winning PGNs → highlight-reel MP4 (ffmpeg) with WIN banner
scripts/showcase.py          one game → PGN + captioned GIF (player-name bars)
scripts/uci.py               UCI wrapper (for lichess-bot / any UCI GUI)
tests/test_search.py         search sign convention + blind-spot fix (material stub)
tests/test_smoke.py          encoding/model/legality/promotion
notebooks/train_colab.ipynb        train → TensorBoard → play → upload → redeploy
notebooks/beat_stockfish_colab.ipynb  tournament → win reel → download MP4
```

## How to run (local, uses v2 model by default)
```bash
pip install -r requirements.txt          # deps
brew install stockfish                   # only for vs-stockfish / tournament
python3 -m tests.test_search && python3 -m tests.test_smoke   # tests

# play locally
python3 -m http.server 8000 -d docs      # → http://localhost:8000  (nice board UI)
python3 -m demo.play_cli                 # terminal

# content
python3 scripts/showcase.py --out demo/mygame --depth 2        # PGN + captioned GIF
python3 scripts/tournament.py --elos 1300 1400 1500 1600 1700 --games 3 --depth 5 --out demo/wins
python3 scripts/win_reel.py --wins demo/wins --out demo/wins_reel   # → MP4

# strength
python3 scripts/vs_stockfish.py --games 6 --depth 2
```
NOTE: `scripts/*.py` add repo root to `sys.path`, so `python3 scripts/X.py` works
(don't need `-m`). Generated `demo/*.mp4|gif|pgn` and `demo/wins/` are gitignored.

## Deploying an update
- **GitHub Pages:** edit `docs/`, commit, push → auto-updates. To ship a new model: `python3 scripts/export_onnx.py --checkpoint models/chess_expert.pt --out docs/chess_expert.onnx`, commit, push. (CDN cache ~minutes; hard-refresh.)
- **HF Static Space:** `python3 scripts/create_static_space.py --repo-id Ahmed007/chess-expert[-v2]` (uploads `docs/index.html`+`chess.js`+`chess_expert.onnx`). The v2 space's `index.html` has a custom `style.css` variant — only its ONNX was replaced to match v2 exactly.
- **Retrain (Colab):** `notebooks/train_colab.ipynb` → produces `chess_expert.pt` → export ONNX → redeploy.

## User preferences / constraints
- Goal: LinkedIn reach + master's-degree credibility. Wants **honest** framing.
- **Learn from real GM games only — NO engine distillation.** (Material eval as a tactical safety net is fine and agreed; it's not "an engine.")
- Prefers CPU inference; pays for Colab GPU for training/tournaments.
- Wants it stronger but accepts it's a PoC, not an engine-beater.
- English is not first language — keep guidance clear and concrete.

## Open TODOs / where we were headed
- **In progress:** user is running the **beat-Stockfish tournament** on Colab GPU (depth 5, Elos 1300–1700, 3 games each) to produce a **win-reel MP4** for LinkedIn. (Reminded them: depth 4–6, not 12; start with 1 game to gauge time.)
- **Next likely asks:** put the confirmed win-rate + reel into `LINKEDIN_POST.md`; possibly add a live Stockfish **eval bar** to the showcase/reel; raise default `--max-plies` so games finish.
- **Later (user said "later"):** **Lichess bot** for a real public Elo — `scripts/uci.py` is ready; needs a BOT account (irreversible upgrade, 0 rated games) + running `lichess-bot` pointed at `python3 scripts/uci.py` on a persistent host. Write the full setup guide when they're ready.
- Optionally point the README "Play online" badge at v2 / GitHub Pages (currently v1 HF space).

## Gotchas (don't repeat these)
- Colab **caches imported modules** — after `git pull`, tell users to restart runtime or `del sys.modules[...]` before re-importing `demo`/`src`.
- ipywidgets **can't set text color** on buttons → the Colab GUI colors pieces via injected CSS classes; Jupyter's default button CSS overrode inline font-size (fixed with `!important`).
- chess.js 0.13.4 is an **ES module** (no global `Chess`) → loaded via `<script type="module">import {Chess}`.
- `import chess.engine` inside a function shadows module-level `chess` → import it at top.
- onnxruntime-web needs single-thread on static hosts: `ort.env.wasm.numThreads = 1` + `wasmPaths` to the CDN.
- Two HF repos can share a name across types (model vs space): `Ahmed007/chess-expert` exists as both.

---
_Advisor tool available — call it before substantive work and before declaring done.
Memory dir has project/user/feedback notes (auto-loaded)._
