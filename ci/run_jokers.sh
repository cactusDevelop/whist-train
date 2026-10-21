#!/usr/bin/env bash
set -euo pipefail
# An iteration lasts ~18 min past --until: keep the save step inside the 6 h job cap.
UNTIL=$(date -u -d '+310 minutes' +%Y-%m-%dT%H:%M)
if [ "${INIT:-false}" = "true" ]; then
  # The network shipped in the game (043f, iteration 36), kept apart before
  # jokers.pt is overwritten; never copied twice, a second init reuses it.
  BEST=checkpoints_jokers/archive_043f_iter36_exported.pt
  [ -f "$BEST" ] || cp checkpoints_jokers/jokers.pt "$BEST"
  python -u scripts/train_jokers.py --init "$BEST" --workers 4 --until "$UNTIL"
else
  python -u scripts/train_jokers.py --resume --workers 4 --until "$UNTIL"
fi
