#!/usr/bin/env bash
set -euo pipefail
# Measurement only (no state saved, no chaining): bot joker gains by role x
# contract type on the jokers champion, for whist-4-it's EliteDecks.DEFENCE.
python -u scripts/joker_defence_gains.py --checkpoint checkpoints_jokers/jokers.pt --hands 2500 --workers 4
