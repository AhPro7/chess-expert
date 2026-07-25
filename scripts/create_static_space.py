"""Create (or update) a FREE *Static* Hugging Face Space that runs the model in
the browser (ONNX) — no PRO subscription needed (Gradio/Docker Spaces now do).

First export the model to ONNX and authenticate, then run this:

    python scripts/export_onnx.py --checkpoint models/chess_expert.pt \
        --out docs/chess_expert.onnx
    huggingface-cli login
    python scripts/create_static_space.py --repo-id Ahmed007/chess-expert

Uploads the static site (index.html + chess.js + chess_expert.onnx) from docs/.
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import HfApi

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOCS = os.path.join(_HERE, "docs")

_README = """\
---
title: Chess Expert
emoji: ♟️
colorFrom: green
colorTo: gray
sdk: static
pinned: false
license: mit
---

# Chess Expert — play a neural net that learned from grandmaster games

Runs entirely in your browser via ONNX (onnxruntime-web) — no server, no GPU.
Model: [Ahmed007/chess-expert](https://huggingface.co/Ahmed007/chess-expert) ·
Code: [github.com/AhPro7/chess-expert](https://github.com/AhPro7/chess-expert)
"""


def create(repo_id: str, token: str, private: bool) -> None:
    onnx = os.path.join(_DOCS, "chess_expert.onnx")
    if not os.path.isfile(onnx):
        raise SystemExit(
            "docs/chess_expert.onnx not found — export it first:\n"
            "  python scripts/export_onnx.py --checkpoint models/chess_expert.pt "
            "--out docs/chess_expert.onnx"
        )

    api = HfApi(token=token or None)
    api.create_repo(
        repo_id, repo_type="space", space_sdk="static",
        exist_ok=True, private=private,
    )
    print(f"Static Space ready: https://huggingface.co/spaces/{repo_id}")

    api.upload_file(path_or_fileobj=_README.encode(), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="space")
    for name in ("index.html", "chess.js", "chess_expert.onnx"):
        api.upload_file(
            path_or_fileobj=os.path.join(_DOCS, name), path_in_repo=name,
            repo_id=repo_id, repo_type="space",
        )
        print(f"  uploaded {name}")

    print(f"\nLive shortly → https://huggingface.co/spaces/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a free Static HF Space")
    parser.add_argument("--repo-id", required=True, help="e.g. Ahmed007/chess-expert")
    parser.add_argument("--token", default="")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    create(args.repo_id, args.token, args.private)


if __name__ == "__main__":
    main()
