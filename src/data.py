"""Turn PGN files of expert games into (position, move) training samples.

Each ply in each game becomes one sample: the board *before* the move (encoded
as a 17x8x8 tensor) paired with the move index the expert actually played.

Memory strategy (this is what lets big GM runs fit on Colab):
  * Positions are stored as **uint8** (planes are just 0/1), which is 1/4 the
    size of float32 — ~1.1 KB per position instead of ~4.3 KB.
  * They are saved as an uncompressed **positions.npy** that training can
    **memory-map** (`np.load(..., mmap_mode="r")`), so only the current batch is
    ever materialized in RAM. RAM stays flat no matter how many positions there
    are.

Output is a directory containing:
    positions.npy  (N, 17, 8, 8)  uint8
    moves.npy      (N,)           int16   (move index in [0, 4095])

Usage:
    python -m src.data --pgn data/gm_games.pgn --out data/samples \
        --min-elo 2400 --max-games 50000
"""

from __future__ import annotations

import argparse
import glob
import os

import chess
import chess.pgn
import numpy as np
from torch.utils.data import Dataset

from .encoding import board_to_tensor, move_to_index


def _game_passes_filter(headers: chess.pgn.Headers, min_elo: int) -> bool:
    """Keep the game only if both players are at/above min_elo (when known)."""
    if min_elo <= 0:
        return True
    try:
        white = int(headers.get("WhiteElo", "0") or "0")
        black = int(headers.get("BlackElo", "0") or "0")
    except ValueError:
        return False
    return white >= min_elo and black >= min_elo


def pgn_to_samples(
    pgn_paths: list[str],
    min_elo: int = 0,
    max_games: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse PGN file(s) into (positions, moves) numpy arrays.

    Returns:
        positions: (N, 17, 8, 8) uint8
        moves:     (N,) int16   (move index in [0, 4095])
    """
    positions: list[np.ndarray] = []
    moves: list[int] = []
    games_used = 0

    for path in pgn_paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            while True:
                if max_games is not None and games_used >= max_games:
                    break
                game = chess.pgn.read_game(handle)
                if game is None:
                    break  # end of this file
                if not _game_passes_filter(game.headers, min_elo):
                    continue

                board = game.board()
                for move in game.mainline_moves():
                    # Store as uint8 right away to keep peak memory low.
                    positions.append(board_to_tensor(board).astype(np.uint8))
                    moves.append(move_to_index(move))
                    board.push(move)

                games_used += 1
                if games_used % 500 == 0:
                    print(f"  parsed {games_used} games, {len(moves):,} positions")

        if max_games is not None and games_used >= max_games:
            break

    print(f"Done: {games_used} games -> {len(moves):,} positions")
    if not moves:
        raise RuntimeError(
            "No positions parsed. Check the PGN path / --min-elo filter."
        )
    return (
        np.asarray(positions, dtype=np.uint8),
        np.asarray(moves, dtype=np.int16),
    )


class ChessDataset(Dataset):
    """PyTorch Dataset over memory-mapped uint8 positions.

    __getitem__ returns a float32 tensor-ready array for a single sample, so the
    full dataset never has to be float32 in RAM at once.
    """

    def __init__(self, positions: np.ndarray, moves: np.ndarray):
        self.positions = positions  # uint8, possibly a memmap
        self.moves = moves

    def __len__(self) -> int:
        return len(self.moves)

    def __getitem__(self, idx: int):
        # Cast just this one position to float32 for the network.
        return self.positions[idx].astype(np.float32), int(self.moves[idx])

    @classmethod
    def from_dir(cls, path: str, mmap: bool = True) -> "ChessDataset":
        """Load positions.npy (memory-mapped by default) and moves.npy."""
        mode = "r" if mmap else None
        positions = np.load(os.path.join(path, "positions.npy"), mmap_mode=mode)
        moves = np.load(os.path.join(path, "moves.npy"))
        return cls(positions, moves)


def main() -> None:
    parser = argparse.ArgumentParser(description="PGN -> training samples (dir of .npy)")
    parser.add_argument(
        "--pgn",
        required=True,
        help="PGN file, directory of PGNs, or glob (e.g. 'data/*.pgn')",
    )
    parser.add_argument(
        "--out", required=True, help="Output directory (positions.npy + moves.npy)"
    )
    parser.add_argument(
        "--min-elo",
        type=int,
        default=0,
        help="Keep games where both players are >= this Elo (0 = keep all)",
    )
    parser.add_argument("--max-games", type=int, default=None)
    args = parser.parse_args()

    if os.path.isdir(args.pgn):
        paths = sorted(glob.glob(os.path.join(args.pgn, "*.pgn")))
    else:
        paths = sorted(glob.glob(args.pgn)) or [args.pgn]
    print(f"Reading {len(paths)} PGN file(s): {paths}")

    positions, moves = pgn_to_samples(paths, args.min_elo, args.max_games)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "positions.npy"), positions)
    np.save(os.path.join(args.out, "moves.npy"), moves)
    size_gb = positions.nbytes / 1e9
    print(f"Saved {len(moves):,} samples -> {args.out}/ ({size_gb:.2f} GB uint8)")


if __name__ == "__main__":
    main()
