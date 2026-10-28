#!/usr/bin/env bash
set -euo pipefail
# Measurement only: profit test of the 4 misere jokers on forced miseres (champion weights).
python -u scripts/misere_jokers.py --matches 600 --workers 4
