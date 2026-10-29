#!/usr/bin/env bash
set -euo pipefail
# Measurement only: hunts the simulated player's empty hand (8f41 run 40783453624 crash), champion weights.
python -u scripts/diag_empty_hand.py --workers 4 --replay-it 19
