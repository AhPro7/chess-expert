"""Train the ChessPolicyNet on parsed (position, move) samples.

Fast path for GPUs (falls back cleanly to CPU):
  * The whole uint8 dataset is loaded into RAM and batched with a manual loop —
    no per-sample Python overhead, and uint8->float casting happens vectorized
    on the GPU.
  * TF32 matmuls/convs + cuDNN autotune (`benchmark`) are always on for CUDA —
    a solid, rock-stable speedup on Ampere+ (A100/L4) with no numerical risk.
  * Mixed-precision autocast (AMP) is OPT-IN via `--amp on`. It can add another
    speedup on some GPUs, but it hangs/errors on others, so it defaults to off.

Usage:
    python -m src.train --data data/samples --out models/chess_expert.pt \
        --epochs 20 --batch-size 4096
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .model import ChessPolicyNet


def train(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_cuda = device.type == "cuda"
    print(f"Device: {device}")

    # ---- Speed switches (CUDA only) ---------------------------------------
    if use_cuda:
        torch.backends.cudnn.benchmark = True          # autotune conv kernels
        torch.backends.cuda.matmul.allow_tf32 = True   # TF32 matmuls
        torch.backends.cudnn.allow_tf32 = True

    use_amp = use_cuda and args.amp == "on"
    # bf16 needs no loss scaling; fp16 does. Keep this contract dead simple.
    amp_dtype = (
        torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if use_amp:
        print(f"AMP: on ({str(amp_dtype).split('.')[-1]}), scaler={use_scaler}")
    else:
        print("AMP: off (fp32)")

    # ---- Load all data into RAM as tensors --------------------------------
    positions = np.load(os.path.join(args.data, "positions.npy"))  # (N,17,8,8) uint8
    moves = np.load(os.path.join(args.data, "moves.npy"))          # (N,) int16
    pos_t = torch.from_numpy(np.ascontiguousarray(positions))       # uint8, CPU
    mv_t = torch.from_numpy(moves.astype(np.int64))                 # long,  CPU
    N = mv_t.shape[0]

    # Fixed, seeded train/val split by index (positions of the SAME game may land
    # in both — fine for a portfolio model; val accuracy reads slightly optimistic).
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(42))
    val_n = max(1, int(N * args.val_split))
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]
    train_n = train_idx.shape[0]
    print(f"Train: {train_n}  Val: {val_n}  ({pos_t.nbytes / 1e9:.1f} GB in RAM)")

    model = ChessPolicyNet(channels=args.channels, num_blocks=args.blocks).to(device)
    if use_cuda and args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.channels}ch x {args.blocks} blocks — {n_params:,} params")

    bs = args.batch_size
    total_batches = (train_n + bs - 1) // bs

    def fetch(idx: torch.Tensor):
        """Gather a batch: uint8 CPU -> device -> float; labels -> device."""
        xb = pos_t.index_select(0, idx).to(device, non_blocking=True).float()
        yb = mv_t.index_select(0, idx).to(device, non_blocking=True)
        if use_cuda and args.channels_last:
            xb = xb.to(memory_format=torch.channels_last)
        return xb, yb

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        last_log = start
        running_loss = 0.0
        win_loss, win_correct, win_seen = 0.0, 0, 0

        epoch_order = train_idx[torch.randperm(train_n)]  # reshuffle each epoch
        for i in range(1, total_batches + 1):
            batch_idx = epoch_order[(i - 1) * bs : i * bs]
            xb, yb = fetch(batch_idx)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            n = yb.size(0)
            running_loss += loss.item() * n
            win_loss += loss.item() * n
            win_correct += (logits.argmax(dim=1) == yb).sum().item()
            win_seen += n

            if i % args.log_interval == 0 or i == total_batches:
                now = time.time()
                pos_per_s = win_seen / max(now - last_log, 1e-9)
                pct = i / total_batches
                eta = (now - start) / pct - (now - start)
                print(
                    f"  epoch {epoch:2d} | batch {i:>5d}/{total_batches} "
                    f"({pct:4.0%}) | loss {win_loss / win_seen:.3f} | "
                    f"move-match {win_correct / win_seen:5.1%} | "
                    f"{pos_per_s:.0f} pos/s | epoch ETA {eta / 60:4.1f}m",
                    flush=True,
                )
                last_log = now
                win_loss, win_correct, win_seen = 0.0, 0, 0

        train_loss = running_loss / train_n

        # ---- Validation ----
        model.eval()
        val_correct, val_seen = 0, 0
        with torch.no_grad():
            for start_j in range(0, val_n, bs):
                batch_idx = val_idx[start_j : start_j + bs]
                xb, yb = fetch(batch_idx)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model(xb)
                val_correct += (logits.argmax(dim=1) == yb).sum().item()
                val_seen += yb.size(0)
        val_acc = val_correct / val_seen

        print(
            f"Epoch {epoch:2d}/{args.epochs} DONE | "
            f"train loss {train_loss:.3f} | val move-match {val_acc:.1%} | "
            f"{time.time() - start:.0f}s\n",
            flush=True,
        )

        # Checkpoint every epoch so a disconnect never loses a full run.
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": model.config}, args.out)

    print(f"Saved checkpoint -> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ChessPolicyNet")
    parser.add_argument("--data", required=True, help="samples dir from src.data")
    parser.add_argument("--out", default="models/chess_expert.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument(
        "--amp", choices=["on", "off"], default="off",
        help="Mixed-precision autocast. Off by default (it hangs/errs on some "
             "GPUs). TF32 + cuDNN autotune are always on regardless. Try 'on' "
             "for a possible extra speedup if your GPU likes it.",
    )
    parser.add_argument(
        "--channels-last", action="store_true",
        help="channels_last memory format (optional; drop it if anything misbehaves)",
    )
    parser.add_argument(
        "--log-interval", type=int, default=50,
        help="Print live train loss/accuracy every N batches",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="(unused — kept for backward compatibility; data is now in RAM)",
    )
    parser.add_argument("--device", default="", help="cuda / cpu (auto if empty)")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
