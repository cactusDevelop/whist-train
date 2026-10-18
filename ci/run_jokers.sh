#!/usr/bin/env bash
set -euo pipefail
# An iteration lasts ~18 min past --until: keep the save step inside the 6 h job cap.
UNTIL=$(date -u -d '+310 minutes' +%Y-%m-%dT%H:%M)
if [ "${INIT:-false}" = "true" ]; then
  python -u scripts/train_jokers.py --init checkpoints_jokers/archive_3e5d_iter29_exported.pt --workers 4 --until "$UNTIL"
else
  python -u scripts/train_jokers.py --resume --workers 4 --until "$UNTIL"
fi
