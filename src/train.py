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
import datetime
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .model import ChessPolicyNet


def _make_writer(logdir: str):
    """Create a TensorBoard SummaryWriter, or None if tensorboard is missing."""
    if not logdir:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("tensorboard not installed — skipping TB logging (pip install tensorboard)")
        return None
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(logdir, stamp)
    print(f"TensorBoard logs -> {run_dir}   (view: tensorboard --logdir {logdir})")
    return SummaryWriter(run_dir)


def _selfplay_frames(checkpoint: str, n_plies: int = 16, depth: int = 0):
    """Play a short game with the current model and return frames (K,H,W,3) uint8."""
    import chess

    from demo.render import render_board
    from .play import ChessEngine

    engine = ChessEngine(checkpoint)
    board = chess.Board()
    frames = [np.asarray(render_board(board), dtype=np.uint8)]
    for _ in range(n_plies):
        move = engine.select_move(board, temperature=0.4, top_k=5, depth=depth)
        if move is None or board.is_game_over(claim_draw=True):
            break
        board.push(move)
        frames.append(np.asarray(render_board(board, last_move=move), dtype=np.uint8))
    return np.stack(frames)


def train(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_cuda = device.type == "cuda"
    print(f"Device: {device}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    use_amp = use_cuda and args.amp == "on"

    # ---- Speed switches (CUDA only) ---------------------------------------
    if use_cuda:
        # cuDNN autotune helps in fp32, but under fp16 it can spend minutes
        # searching (or hang) on some cards — so turn it off when AMP is on.
        torch.backends.cudnn.benchmark = not use_amp
        torch.backends.cuda.matmul.allow_tf32 = True   # TF32 matmuls (Ampere+)
        torch.backends.cudnn.allow_tf32 = True
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

    # Value targets (side-to-move POV) enable the value head + search. Older
    # sample dirs may not have them — then we train policy-only.
    values_path = os.path.join(args.data, "values.npy")
    has_values = os.path.exists(values_path)
    if has_values:
        val_t = torch.from_numpy(np.load(values_path).astype(np.float32))
        print("Value targets found — training policy + value head.")
    else:
        val_t = torch.zeros(N, dtype=torch.float32)
        print("No values.npy — training policy only (re-run src.data for value "
              "targets to enable search).")

    # Fixed, seeded train/val split by index (positions of the SAME game may land
    # in both — fine for a portfolio model; val accuracy reads slightly optimistic).
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(42))
    val_n = max(1, int(N * args.val_split))
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]
    train_n = train_idx.shape[0]
    print(f"Train: {train_n}  Val: {val_n}  ({pos_t.nbytes / 1e9:.1f} GB in RAM)")

    # If resuming, build the model from the checkpoint's own config so the
    # architecture always matches the saved weights.
    resume_ck = torch.load(args.resume, map_location=device) if args.resume else None
    cfg = (
        resume_ck["config"] if resume_ck
        else {"channels": args.channels, "num_blocks": args.blocks}
    )
    model = ChessPolicyNet(**cfg).to(device)
    if use_cuda and args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    value_weight = args.value_weight if has_values else 0.0
    writer = _make_writer(args.logdir)
    global_step = 0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['channels']}ch x {cfg['num_blocks']} blocks — {n_params:,} params")

    # Resume: restore weights, optimizer state, and the epoch counter.
    start_epoch = 1
    if resume_ck is not None:
        model.load_state_dict(resume_ck["state_dict"])
        if "optimizer" in resume_ck:
            optimizer.load_state_dict(resume_ck["optimizer"])
            start_epoch = resume_ck.get("epoch", 0) + 1
            print(f"Resumed from {args.resume}: continuing at epoch {start_epoch}")
        else:
            print(f"Resumed WEIGHTS from {args.resume} (no optimizer/epoch in file — "
                  "fresh optimizer, starting at epoch 1)")
    if start_epoch > args.epochs:
        print(f"Nothing to do: start epoch {start_epoch} > --epochs {args.epochs}. "
              "Raise --epochs to train further.")
        return

    # Sidecar file holding optimizer + epoch for --resume (ends in .pt so it is
    # git-ignored; the small deployable checkpoint stays at --out).
    resume_path = os.path.splitext(args.out)[0] + ".resume.pt"

    bs = args.batch_size
    total_batches = (train_n + bs - 1) // bs

    def fetch(idx: torch.Tensor):
        """Gather a batch: uint8 CPU -> device -> float; move + value labels."""
        xb = pos_t.index_select(0, idx).to(device, non_blocking=True).float()
        yb = mv_t.index_select(0, idx).to(device, non_blocking=True)
        vb = val_t.index_select(0, idx).to(device, non_blocking=True)
        if use_cuda and args.channels_last:
            xb = xb.to(memory_format=torch.channels_last)
        return xb, yb, vb

    # Accumulate stats as GPU tensors and only .item() them at log time. Calling
    # .item() every batch forces a GPU->CPU sync that stalls the whole pipeline;
    # doing it once per log interval (not every batch) removes ~50 stalls per log.
    def zeros():
        return (
            torch.zeros((), device=device),  # loss sum
            torch.zeros((), device=device, dtype=torch.long),  # correct
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        start = time.time()
        last_log = start
        epoch_loss = torch.zeros((), device=device)
        win_loss, win_correct = zeros()
        win_ploss = torch.zeros((), device=device)  # policy loss sum (window)
        win_vloss = torch.zeros((), device=device)  # value  loss sum (window)
        win_seen = 0

        epoch_order = train_idx[torch.randperm(train_n)]  # reshuffle each epoch
        for i in range(1, total_batches + 1):
            batch_idx = epoch_order[(i - 1) * bs : i * bs]
            xb, yb, vb = fetch(batch_idx)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits, value = model(xb)
                p_loss = policy_criterion(logits, yb)
                v_loss = value_criterion(value, vb) if value_weight else \
                    torch.zeros((), device=device)
                loss = p_loss + value_weight * v_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            n = yb.size(0)
            batch_loss = loss.detach() * n
            epoch_loss += batch_loss
            win_loss += batch_loss
            win_ploss += p_loss.detach() * n
            win_vloss += v_loss.detach() * n
            win_correct += (logits.detach().argmax(dim=1) == yb).sum()
            win_seen += n
            global_step += 1

            if i % args.log_interval == 0 or i == total_batches:
                now = time.time()
                pos_per_s = win_seen / max(now - last_log, 1e-9)
                pct = i / total_batches
                eta = (now - start) / pct - (now - start)
                # A few .item() syncs per log interval (not per batch).
                avg_loss = (win_loss / win_seen).item()
                avg_ploss = (win_ploss / win_seen).item()
                avg_vloss = (win_vloss / win_seen).item()
                acc = (win_correct.float() / win_seen).item()
                if writer:
                    writer.add_scalar("train/loss_total", avg_loss, global_step)
                    writer.add_scalar("train/loss_policy", avg_ploss, global_step)
                    writer.add_scalar("train/loss_value", avg_vloss, global_step)
                    writer.add_scalar("train/move_match", acc, global_step)
                    writer.add_scalar("train/pos_per_sec", pos_per_s, global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"],
                                      global_step)
                print(
                    f"  epoch {epoch:2d} | batch {i:>5d}/{total_batches} "
                    f"({pct:4.0%}) | loss {avg_loss:.3f} | "
                    f"move-match {acc:5.1%} | "
                    f"{pos_per_s:.0f} pos/s | epoch ETA {eta / 60:4.1f}m",
                    flush=True,
                )
                last_log = now
                win_loss, win_correct = zeros()
                win_ploss = torch.zeros((), device=device)
                win_vloss = torch.zeros((), device=device)
                win_seen = 0

        train_loss = (epoch_loss / train_n).item()

        # ---- Validation ----
        model.eval()
        val_correct = torch.zeros((), device=device, dtype=torch.long)
        val_verr = torch.zeros((), device=device)  # value abs error sum
        val_seen = 0
        with torch.no_grad():
            for start_j in range(0, val_n, bs):
                batch_idx = val_idx[start_j : start_j + bs]
                xb, yb, vb = fetch(batch_idx)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    logits, value = model(xb)
                val_correct += (logits.argmax(dim=1) == yb).sum()
                val_verr += (value.float() - vb).abs().sum()
                val_seen += yb.size(0)
        val_acc = (val_correct.float() / val_seen).item()
        val_mae = (val_verr / val_seen).item()

        value_note = f" | val value-MAE {val_mae:.3f}" if value_weight else ""
        print(
            f"Epoch {epoch:2d}/{args.epochs} DONE | "
            f"train loss {train_loss:.3f} | val move-match {val_acc:.1%}"
            f"{value_note} | {time.time() - start:.0f}s\n",
            flush=True,
        )

        # Checkpoint every epoch so a disconnect never loses a full run.
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        # Small deployable checkpoint (what play.py / the GUI load). The
        # value_trained flag tells the engine whether search can be trusted.
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": model.config,
                "value_trained": bool(value_weight),
            },
            args.out,
        )
        # Larger sidecar with optimizer + epoch, for `--resume`.
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": model.config,
                "value_trained": bool(value_weight),
            },
            resume_path,
        )

        # ---- TensorBoard: per-epoch scalars + a self-play board filmstrip ----
        if writer:
            writer.add_scalar("val/move_match", val_acc, epoch)
            writer.add_scalar("val/value_mae", val_mae, epoch)
            writer.add_scalar("epoch/train_loss", train_loss, epoch)
            writer.add_scalar("epoch/seconds", time.time() - start, epoch)
            if args.tb_images:
                try:
                    frames = _selfplay_frames(
                        args.out, n_plies=args.tb_image_plies,
                        depth=2 if value_weight else 0,
                    )
                    writer.add_images("selfplay/game", frames, epoch,
                                      dataformats="NHWC")
                except Exception as exc:  # never let logging kill training
                    print(f"  (self-play image logging skipped: {exc})")

    if writer:
        writer.close()
    print(f"Saved checkpoint -> {args.out}  (resume state -> {resume_path})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ChessPolicyNet")
    parser.add_argument("--data", required=True, help="samples dir from src.data")
    parser.add_argument("--out", default="models/chess_expert.pt")
    parser.add_argument(
        "--resume", default="",
        help="Path to a *.resume.pt file to continue training from "
             "(e.g. models/chess_expert.resume.pt). --epochs is the TOTAL target.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument(
        "--logdir", default="runs",
        help="TensorBoard log dir (empty string disables TB logging)",
    )
    parser.add_argument(
        "--tb-images", action=argparse.BooleanOptionalAction, default=True,
        help="Log a self-play board filmstrip to TensorBoard each epoch",
    )
    parser.add_argument("--tb-image-plies", type=int, default=16)
    parser.add_argument(
        "--value-weight", type=float, default=1.0,
        help="Weight of the value-head MSE loss (0 = policy only)",
    )
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
