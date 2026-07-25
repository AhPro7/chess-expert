"""Download the .pt from HuggingFace, export to ONNX, place in docs/ for GitHub Pages
and the HF Static Space.

Usage (no local checkpoint needed):
    python scripts/export_and_deploy.py

Or specify a local checkpoint already on disk:
    python scripts/export_and_deploy.py --checkpoint models/chess_expert.pt

The script:
  1. Downloads chess_expert.pt from Ahmed007/chess-expert (if not already present).
  2. Exports it to docs/chess_expert.onnx (GitHub Pages).
  3. Verifies the ONNX model with onnxruntime (if installed).
  4. Prints next steps for GitHub Pages and HF Static Space.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

# ── project root on path so `from src.model import …` works ──────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import torch

from src.model import ChessPolicyNet

MODEL_REPO = "Ahmed007/chess-expert"
MODEL_FILE = "chess_expert.pt"
LOCAL_CKPT = os.path.join(_ROOT, "models", "chess_expert.pt")
ONNX_OUT  = os.path.join(_ROOT, "docs", "chess_expert.onnx")


def _download_checkpoint(local_path: str) -> str:
    """Download from HF Hub if not already present locally."""
    if os.path.isfile(local_path):
        print(f"[1/3] Using existing checkpoint: {local_path}")
        return local_path

    print(f"[1/3] Downloading {MODEL_FILE} from {MODEL_REPO} …")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub not installed — run: pip install huggingface_hub")

    # Download to the HF cache but also copy into models/
    cached = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    import shutil
    shutil.copy2(cached, local_path)
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"    → saved to {local_path}  ({size_mb:.1f} MB)")
    return local_path


def _export_onnx(checkpoint: str, out: str) -> None:
    """Load the PyTorch checkpoint and export to ONNX."""
    print(f"[2/3] Exporting to ONNX: {out}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("config", {"channels": 128, "num_blocks": 10})
    print(f"      model config: {cfg}")
    print(f"      value_trained: {ckpt.get('value_trained', False)}")

    net = ChessPolicyNet(**cfg)
    net.load_state_dict(ckpt["state_dict"], strict=False)
    net.eval()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    dummy = torch.zeros(1, 17, 8, 8)

    # Suppress the TracerWarning about slice-indexing — it's safe here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            net, dummy, out,
            input_names=["board"],
            output_names=["policy", "value"],
            dynamic_axes={
                "board":  {0: "batch"},
                "policy": {0: "batch"},
                "value":  {0: "batch"},
            },
            opset_version=17,
        )

    size_mb = os.path.getsize(out) / 1e6
    print(f"      → {out}  ({size_mb:.1f} MB)")


def _verify_onnx(out: str) -> None:
    """Quick sanity-check: run a forward pass with onnxruntime."""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("[3/3] onnxruntime not installed — skipping ONNX verification")
        print("      (install with: pip install onnxruntime)")
        return

    print("[3/3] Verifying ONNX model with onnxruntime …")
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    dummy = np.zeros((1, 17, 8, 8), dtype=np.float32)
    policy, value = sess.run(None, {"board": dummy})
    assert policy.shape == (1, 4096), f"Unexpected policy shape: {policy.shape}"
    assert value.shape  == (1,),      f"Unexpected value shape:  {value.shape}"
    print(f"      policy shape: {policy.shape}  value shape: {value.shape}  ✓")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download chess_expert.pt from HF and export to ONNX for browser play"
    )
    parser.add_argument(
        "--checkpoint", default=LOCAL_CKPT,
        help=f"Local .pt path (downloaded from HF if missing). Default: {LOCAL_CKPT}"
    )
    parser.add_argument(
        "--out", default=ONNX_OUT,
        help=f"Output ONNX path. Default: {ONNX_OUT}"
    )
    args = parser.parse_args()

    ckpt = _download_checkpoint(args.checkpoint)
    _export_onnx(ckpt, args.out)
    _verify_onnx(args.out)

    print()
    print("=" * 60)
    print("DONE!  Next steps:")
    print()
    print("  GitHub Pages (free):")
    print("    git add docs/chess_expert.onnx")
    print("    git commit -m 'add ONNX model for in-browser play'")
    print("    git push")
    print("    → enable Settings → Pages → Source: main / docs")
    print("    → live at https://AhPro7.github.io/chess-expert/")
    print()
    print("  HF Static Space (free, no PRO needed):")
    print("    huggingface-cli login")
    print("    python scripts/create_static_space.py --repo-id Ahmed007/chess-expert-space")
    print("=" * 60)


if __name__ == "__main__":
    main()
