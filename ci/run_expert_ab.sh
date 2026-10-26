#!/usr/bin/env bash
set -euo pipefail
# Measurement only: ExpertBot card-play audit (PIMC + DDS loss per situation) -- scripts/expert_play_audit.py.
python -u scripts/expert_play_audit.py --deals 300 --samples 6 --workers 4
