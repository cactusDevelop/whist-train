#!/usr/bin/env bash
set -euo pipefail
# Measurement only: ExpertBot coordinate tuning in duplicate (scripts/expert_tune.py).
python -u scripts/expert_tune.py --minutes 4 --deals 200 --validate-deals 400 --workers 4
