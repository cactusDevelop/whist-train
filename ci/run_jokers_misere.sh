#!/usr/bin/env bash
set -euo pipefail
# Measurement only: profit test of the misere jokers on forced miseres (+ Pot-de-Vin, 8f41) and of
# Main de Fer / Brouillard (--set atout), each with and without the elites' Eye (champion weights).
for s in misere atout; do
  for e in "" "--no-eye"; do
    echo "=== set $s $e ==="
    python -u scripts/misere_jokers.py --matches 1200 --workers 4 --set $s $e
  done
done
