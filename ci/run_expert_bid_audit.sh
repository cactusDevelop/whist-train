#!/usr/bin/env bash
set -euo pipefail
# Promotion du renom d'atout : tests et fixtures de parite Godot.
python -u scripts/test_expert_bot.py
python -u tests/scripts/dump_expert_bot_fixtures.py
gh release upload checkpoint-latest tests/fixtures/expert_bot_parity.json --clobber
