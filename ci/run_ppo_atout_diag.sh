#!/usr/bin/env bash
set -euo pipefail
# Measurement only: does the DDS teacher teach NOT drawing trumps? (scripts/atout_teacher_diag.py)
python -u scripts/atout_teacher_diag.py --positions 400 --samples 8 --workers 4
