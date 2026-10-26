#!/usr/bin/env bash
set -euo pipefail
# Measurement only: ExpertBot bidding audit (impossible contracts via DDS, fatal raises) -- scripts/expert_bid_audit.py.
python -u scripts/expert_bid_audit.py --deals 20000
