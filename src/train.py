"""Train the ChessPolicyNet on parsed (position, move) samples.

Designed to run on a Colab / Kaggle GPU, but works on CPU for small data.

Usage:
    python -m src.train --data data/samples --out models/chess_expert.pt \
        --epochs 20 --batch-size 1024
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .data import ChessDataset
from .model import ChessPolicyNet


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 move-match accuracy: how often argmax == the expert's move."""
    return (logits.argmax(dim=1) == targets).float().mean().item()


def train(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    dataset = ChessDataset.from_dir(args.data)  # memory-mapped uint8
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {train_size}  Val: {val_size}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers)

    model = ChessPolicyNet(channels=args.channels, num_blocks=args.blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.channels}ch x {args.blocks} blocks — {n_params:,} params")

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        running_loss = 0.0
        for positions, moves in train_loader:
            positions = positions.to(device)
            moves = moves.to(device, dtype=torch.long)

            optimizer.zero_grad()
            logits = model(positions)
            loss = criterion(logits, moves)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * positions.size(0)

        train_loss = running_loss / train_size

        # Validation
        model.eval()
        val_acc, val_seen = 0.0, 0
        with torch.no_grad():
            for positions, moves in val_loader:
                positions = positions.to(device)
                moves = moves.to(device, dtype=torch.long)
                logits = model(positions)
                val_acc += accuracy(logits, moves) * positions.size(0)
                val_seen += positions.size(0)
        val_acc /= val_seen

        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"loss {train_loss:.3f} | val move-match {val_acc:.1%} | "
            f"{time.time() - start:.0f}s"
        )

    torch.save(
        {"state_dict": model.state_dict(), "config": model.config},
        args.out,
    )
    print(f"Saved checkpoint -> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ChessPolicyNet")
    parser.add_argument("--data", required=True, help="samples dir from src.data")
    parser.add_argument("--out", default="models/chess_expert.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="", help="cuda / cpu (auto if empty)")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
