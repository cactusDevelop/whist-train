#!/usr/bin/env bash
set -euo pipefail
# Measurement only: ExpertBot coordinate tuning in duplicate (scripts/expert_tune.py), one night.
python -u scripts/expert_tune.py --minutes 320 --deals 20000 --validate-deals 60000 --workers 4
