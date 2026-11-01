#!/usr/bin/env bash
set -euo pipefail
# IA libre (scripts/train_free.py). FREE_MINUTES court = verification rapide : cycles et mesures reduits.
MIN=${FREE_MINUTES:-}
if [ -n "$MIN" ] && [ "$MIN" -lt 90 ]; then
  EXTRA="--cycle-seconds $(( MIN * 60 / 3 )) --select-deals 400 --gate-deals 2000"
else
  EXTRA=""
fi
python -u scripts/train_free.py --until "$(date -u -d "+${MIN:-320} minutes" +%Y-%m-%dT%H:%M)" $EXTRA
