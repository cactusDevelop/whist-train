#!/usr/bin/env bash
set -euo pipefail
# Measurement only (Synergies 31/10): the 5 negative jokers with and without the Eye, then targeted pairs.
for e in "" "--no-eye"; do
  echo "=== set oeil $e ==="
  python -u scripts/misere_jokers.py --matches 800 --workers 4 --set oeil $e
done
echo "=== paires ==="
python -u scripts/pair_jokers.py --minutes 13 --workers 4
