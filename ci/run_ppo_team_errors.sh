#!/usr/bin/env bash
set -euo pipefail
# Measurement only (no state saved, no chaining): team errors (surcoupe +
# renonce non coupee) of the current card-play champion on the ppo state.
python -u scripts/team_errors.py --deals 3000 --out-file /tmp/team_errors_measure.jsonl
