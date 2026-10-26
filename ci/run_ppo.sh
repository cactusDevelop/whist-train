#!/usr/bin/env bash
set -euo pipefail
UNTIL=$(date -u -d '+330 minutes' +%Y-%m-%dT%H:%M)
python -u scripts/train.py --no-godot-sync --no-git-push --no-record-samples --export-every 0 --eval-every 4000 --champion-deals 10000 --rollout-policy dds --rollout-per-deal 2 --rollout-samples 8 --rollout-surcoupe-samples 8 --rollout-coupe-samples 8 --rollout-atout-samples 0 --rollout-workers 4 --rollout-margin-tricks 0.5 --oracle-critic \
  --stats-file training_stats_phase1_v3.jsonl --stats-file-phase2 training_stats_phase2_v3.jsonl \
  --until "$UNTIL"
