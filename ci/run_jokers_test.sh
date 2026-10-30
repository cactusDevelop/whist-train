#!/usr/bin/env bash
set -euo pipefail
# Measurement only (no state saved, no chaining): rule tests of the jokers port, then a
# smoke iteration of the joker training on the champion, in a throwaway dir.
python -u scripts/test_jokers.py
rm -rf /tmp/smoke && mkdir -p /tmp/smoke
python -u scripts/train_jokers.py --checkpoint-dir /tmp/smoke --init checkpoints_jokers/jokers.pt --iters 1 --hands 80 \
  --gate-matches 16 --profit-matches 40 --workers 4
grep -h "main_de_fer" /tmp/smoke/*.md joker_profit_summary.md 2>/dev/null | head -5 || true
