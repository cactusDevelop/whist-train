#!/usr/bin/env bash
set -euo pipefail
# Measurement only: current AI vs card-play champion, gain by contract AND camp (scripts/champion_by_camp.py).
python -u scripts/champion_by_camp.py --deals 20000
