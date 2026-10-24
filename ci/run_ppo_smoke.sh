#!/usr/bin/env bash
set -euo pipefail
# Smoke test (measure only, never saved): the ppo run for 12 minutes on a throwaway copy of the state.
UNTIL=$(date -u -d '+12 minutes' +%Y-%m-%dT%H:%M)
python -u scripts/train.py --no-godot-sync --no-git-push --no-record-samples --export-every 0 --rollout-policy dds --rollout-per-deal 2 --rollout-samples 8 --rollout-surcoupe-samples 8 --rollout-coupe-samples 8 --rollout-atout-samples 8 --rollout-workers 4 \
  --stats-file training_stats_phase1_v3.jsonl --stats-file-phase2 training_stats_phase2_v3.jsonl \
  --until "$UNTIL"
