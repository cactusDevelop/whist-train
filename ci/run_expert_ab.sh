#!/usr/bin/env bash
set -euo pipefail
# Measurement only: ExpertBot coordinate tuning in duplicate (scripts/expert_tune.py).
TUNE_MINUTES=4 TUNE_DEALS=200 TUNE_VALIDATE=400 python -u scripts/expert_tune.py --minutes ${TUNE_MINUTES:-320} --deals ${TUNE_DEALS:-20000} --validate-deals ${TUNE_VALIDATE:-60000} --workers 4
