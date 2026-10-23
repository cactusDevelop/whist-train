#!/usr/bin/env bash
set -euo pipefail
UNTIL=$(date -u -d '+330 minutes' +%Y-%m-%dT%H:%M)
python -u scripts/train.py --no-godot-sync --no-git-push --no-record-samples --export-every 0 --rollout-policy dds --rollout-per-deal 6 --rollout-samples 12 --rollout-surcoupe-samples 12 --rollout-coupe-samples 12 --rollout-workers 4 \
  --stats-file training_stats_phase1_v3.jsonl --stats-file-phase2 training_stats_phase2_v3.jsonl \
  --until "$UNTIL"
