#!/usr/bin/env bash
# Download grandmaster game collections (PGN) for training.
#
# Source: pgnmentor.com — per-player archives of titled grandmasters.
# These are genuine GM games (unlike Lichess "elite", which is 2400+ online
# players). Edit the PLAYERS list to add more.
#
# Usage: bash scripts/download_data.sh

set -euo pipefail

mkdir -p data
cd data

# ~40 titled grandmasters (world champions + top modern players + classics).
# Missing archives are skipped automatically, so it's safe to list generously.
PLAYERS=(
  Carlsen Kasparov Fischer Karpov Anand Kramnik Caruana Nakamura Aronian
  Nepomniachtchi Giri Grischuk Mamedyarov Radjabov Topalov Ivanchuk Gelfand
  Leko Svidler Morozevich Shirov Adams Short Ding So Firouzja Rapport Duda
  Tal Botvinnik Petrosian Spassky Smyslov Korchnoi Bronstein Keres Capablanca
  Alekhine Rubinstein Euwe Larsen Portisch Polgar Beliavsky
)
BASE="https://www.pgnmentor.com/players"

for player in "${PLAYERS[@]}"; do
  echo "Downloading ${player}..."
  curl -fsSL -o "${player}.zip" "${BASE}/${player}.zip" || {
    echo "  (skipped ${player} — not found)"; continue; }
  unzip -o -q "${player}.zip" && rm -f "${player}.zip"
done

echo "Merging into data/gm_games.pgn ..."
cat ./*.pgn > gm_games.pgn
echo "Done. Wrote data/gm_games.pgn ($(grep -c '\[Event ' gm_games.pgn) games)."
