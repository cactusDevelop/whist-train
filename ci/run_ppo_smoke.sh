#!/usr/bin/env bash
set -euo pipefail
# Smoke test (measure only, never saved): the ppo run for 12 minutes on a throwaway copy of the state.
UNTIL=$(date -u -d '+12 minutes' +%Y-%m-%dT%H:%M)
python -u scripts/train.py --no-godot-sync --no-git-push --no-record-samples --export-every 0 --eval-every 150 --eval-games 50 --team-errors-deals 50 --champion-deals 40 --duel-deals 0 --expert-match-contracts 0 --rollout-policy dds --rollout-per-deal 2 --rollout-samples 8 --rollout-surcoupe-samples 8 --rollout-coupe-samples 8 --rollout-atout-samples 0 --rollout-workers 4 --rollout-margin-tricks 0.5 --declarer-distill-coef 0.2 --oracle-critic --oracle-warm-lots 3 \
  --stats-file training_stats_phase1_v3.jsonl --stats-file-phase2 training_stats_phase2_v3.jsonl \
  --until "$UNTIL"
