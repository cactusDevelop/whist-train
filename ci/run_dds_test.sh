#!/usr/bin/env bash
set -euo pipefail
# Measurement only: exactness + timing of the double-dummy teacher (engine/dds.c).
python -u scripts/test_dds.py --positions 400 --full 30
