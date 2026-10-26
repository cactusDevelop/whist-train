#!/usr/bin/env bash
set -euo pipefail
# Measurement only: ExpertBot play-rule A/B in duplicate (scripts/expert_bid_ab.py).
python -u scripts/expert_bid_ab.py --deals 40000 --workers 4
