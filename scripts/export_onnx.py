"""Export a trained checkpoint to ONNX for in-browser inference.

    python scripts/export_onnx.py --checkpoint models/chess_expert.pt \
        --out docs/chess_expert.onnx

The ONNX model takes a (1,17,8,8) float32 board tensor and returns
(policy[1,4096], value[1]). The static web demo (docs/index.html) runs it with
onnxruntime-web — no server, no GPU.
"""

from __future__ import annotations

import argparse
import os

import torch

from src.model import ChessPolicyNet


def export(checkpoint: str, out: str) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {"channels": 128, "num_blocks": 10})
    net = ChessPolicyNet(**cfg)
    net.load_state_dict(ckpt["state_dict"], strict=False)
    net.eval()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    dummy = torch.zeros(1, 17, 8, 8)
    torch.onnx.export(
        net, dummy, out,
        input_names=["board"], output_names=["policy", "value"],
        dynamic_axes={"board": {0: "batch"}, "policy": {0: "batch"},
                      "value": {0: "batch"}},
        opset_version=17,
    )
    size_mb = os.path.getsize(out) / 1e6
    print(f"Exported {out} ({size_mb:.1f} MB) — value_trained={ckpt.get('value_trained', False)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export checkpoint to ONNX")
    parser.add_argument("--checkpoint", default="models/chess_expert.pt")
    parser.add_argument("--out", default="docs/chess_expert.onnx")
    args = parser.parse_args()
    export(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
