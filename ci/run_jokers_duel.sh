#!/usr/bin/env bash
set -euo pipefail
# Measurement only (no state saved, no chaining): the Duel's habitues with elite decks
# (whist-4-it 07ac) -- gain of Dette de Jeu x2 and of the admin deck against them.
python -u scripts/duel_jokers.py --checkpoint checkpoints_jokers/jokers.pt --duels 500 --workers 4
