"""Create (or update) a Gradio Space that lets people play the model online.

Authenticate first (`huggingface-cli login` or `notebook_login()`), then:

    python scripts/create_space.py --repo-id Ahmed007/chess-expert

Uploads space/app.py, space/requirements.txt, space/README.md to the Space. The
app downloads the model from the model repo of the same name at runtime.
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import HfApi

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPACE_DIR = os.path.join(_HERE, "space")


def create(repo_id: str, token: str, private: bool) -> None:
    api = HfApi(token=token or None)
    api.create_repo(
        repo_id, repo_type="space", space_sdk="gradio",
        exist_ok=True, private=private,
    )
    print(f"Space ready: https://huggingface.co/spaces/{repo_id}")

    for name in ("app.py", "requirements.txt", "README.md"):
        api.upload_file(
            path_or_fileobj=os.path.join(_SPACE_DIR, name),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="space",
        )
        print(f"  uploaded {name}")

    print(f"\nBuilding now → https://huggingface.co/spaces/{repo_id}")
    print("First build takes a few minutes (installs torch). Then just play!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Gradio Space for the model")
    parser.add_argument("--repo-id", required=True, help="e.g. Ahmed007/chess-expert")
    parser.add_argument("--token", default="", help="HF token (else uses cached login)")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    create(args.repo_id, args.token, args.private)


if __name__ == "__main__":
    main()
