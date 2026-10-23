#!/usr/bin/env bash
set -uo pipefail
# Measurement only: ExpertBot rule tests, then exactness + timing of the double-dummy teacher (engine/dds.c).
python -u scripts/test_expert_bot.py || exit 1
# Hard cap so the log is always uploaded even if one solve hangs.
timeout 1500 python -u scripts/test_dds.py --positions 400 --full 30 --budget 120
