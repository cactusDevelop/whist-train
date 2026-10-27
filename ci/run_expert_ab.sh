#!/usr/bin/env bash
set -euo pipefail
# Tests ExpertBot puis reglage coordonne en duplicate, borne par les regles utilisateur (scripts/expert_tune.py).
python -u scripts/test_expert_bot.py
python -u scripts/expert_tune.py --minutes 320 --deals 20000 --validate-deals 60000 --workers 4
