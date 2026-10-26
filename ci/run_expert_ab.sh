#!/usr/bin/env bash
set -euo pipefail
# Measurement only: ExpertBot bidding prudence A/B (scripts/expert_bid_ab.py) + ExpertBot team errors.
python -u scripts/expert_bid_ab.py --deals 6000 --workers 4
python -u scripts/team_errors.py --deals 3000 --players expert --out-file /tmp/team_errors_expert.jsonl
