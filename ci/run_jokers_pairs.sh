#!/usr/bin/env bash
set -euo pipefail
# Measurement only (Synergies 05, 6483): combo pairs, 5 minutes each.
python -u scripts/pair_jokers.py --minutes 5 --workers 4
