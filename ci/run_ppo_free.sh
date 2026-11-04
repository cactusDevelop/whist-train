#!/usr/bin/env bash
set -euo pipefail
# IA libre (scripts/train_free.py). FREE_MINUTES court = verification rapide : cycles et mesures reduits.
MIN=${FREE_MINUTES:-}
if [ -n "$MIN" ] && [ "$MIN" -lt 90 ]; then
  EXTRA="--cycle-seconds $(( MIN * 60 / 3 )) --select-deals 400 --gate-deals 2000"
else
  EXTRA="--cycle-seconds 1800"
fi
# defense ciblee (05/11) : dds sur 10 % des decisions de defense hors 1er pli, <= 50 % du temps
python -u scripts/train_free.py --until "$(date -u -d "+${MIN:-320} minutes" +%Y-%m-%dT%H:%M)" $EXTRA \n  --defense-rollout-prob 0.1 --defense-rollout-samples 8 --defense-rollout-margin 0.5 --defense-rollout-share 0.5
