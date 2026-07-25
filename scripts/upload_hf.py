"""Upload the trained model + TensorBoard logs to the Hugging Face Hub.

First authenticate (once per Colab session):
    from huggingface_hub import notebook_login; notebook_login()
    # or in a terminal:  huggingface-cli login

Then:
    python scripts/upload_hf.py --repo-id YOUR_USERNAME/chess-expert

Uploads:
    chess_expert.pt              the deployable checkpoint (policy + value)
    logs/                        the TensorBoard runs (so the curves live on the Hub)
    README.md                    a short model card
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import HfApi

_CARD = """\
---
license: mit
tags:
  - chess
  - pytorch
  - reinforcement-learning
  - behavioral-cloning
---

# Chess Expert

A deep-learning chess model that learns from **grandmaster games** (behavioral
cloning, à la Maia). A residual CNN with a **policy head** (which move) and a
**value head** (how good the position is), plus an optional shallow look-ahead
search so it avoids hanging pieces.

- Code: https://github.com/AhPro7/chess-expert
- `chess_expert.pt` — load with the repo's `src.play.ChessEngine`.
- `logs/` — TensorBoard training curves (`tensorboard --logdir logs`).

Trained on CPU-friendly inference; see the repo README for how to play (terminal
or a clickable Colab board).
"""


def push(repo_id: str, checkpoint: str, logdir: str, token: str | None,
         private: bool) -> None:
    api = HfApi(token=token or None)
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
    print(f"Repo ready: https://huggingface.co/{repo_id}")

    # Model card
    api.upload_file(
        path_or_fileobj=_CARD.encode(), path_in_repo="README.md",
        repo_id=repo_id, repo_type="model",
    )

    # Checkpoint
    if os.path.isfile(checkpoint):
        api.upload_file(
            path_or_fileobj=checkpoint,
            path_in_repo=os.path.basename(checkpoint),
            repo_id=repo_id, repo_type="model",
        )
        print(f"Uploaded {checkpoint}")
    else:
        print(f"WARNING: checkpoint {checkpoint} not found — skipped")

    # TensorBoard logs
    if os.path.isdir(logdir):
        api.upload_folder(
            folder_path=logdir, path_in_repo="logs",
            repo_id=repo_id, repo_type="model",
        )
        print(f"Uploaded logs from {logdir}/")
    else:
        print(f"No logdir {logdir}/ — skipped")

    print(f"\nDone → https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload model + logs to HF Hub")
    parser.add_argument("--repo-id", required=True, help="e.g. AhPro7/chess-expert")
    parser.add_argument("--checkpoint", default="models/chess_expert.pt")
    parser.add_argument("--logdir", default="runs")
    parser.add_argument("--token", default="", help="HF token (else uses cached login)")
    parser.add_argument("--private", action="store_true", help="Make the repo private")
    args = parser.parse_args()
    push(args.repo_id, args.checkpoint, args.logdir, args.token, args.private)


if __name__ == "__main__":
    main()
