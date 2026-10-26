#!/usr/bin/env bash
set -euo pipefail
# Measurement only: ExpertBot card-play audit (PIMC + DDS loss per situation) -- scripts/expert_play_audit.py.
python -u scripts/expert_play_audit.py --deals 80 --samples 4 --workers 4
