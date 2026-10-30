#!/usr/bin/env bash
set -euo pipefail
# Promotion expert_emb1027 (signalement emballage a 1 atout) : tests, ligue, maillon faible, fixtures de parite Godot.
python -u scripts/test_expert_bot.py
python -u tests/scripts/dump_expert_bot_fixtures.py
gh release upload checkpoint-latest tests/fixtures/expert_bot_parity.json --clobber
python -u scripts/bot_league.py --deals 40000 --workers 4
python -u scripts/emballage_blame.py --deals 20000 --workers 4
