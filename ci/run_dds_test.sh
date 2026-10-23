#!/usr/bin/env bash
set -uo pipefail
# Measurement only: exactness + timing of the double-dummy teacher (engine/dds.c).
# Hard cap so the log is always uploaded even if one solve hangs.
timeout 1500 python -u scripts/test_dds.py --positions 400 --full 30 --budget 120
