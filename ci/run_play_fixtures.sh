#!/usr/bin/env bash
set -euo pipefail
# BRANCHE dominion-a76f (regles jokers 0e9a + encheres 1.2.18) : cas ExpertBot sous jokers (vue miroir, atout garde).
mkdir -p godot_export
python -u scripts/dump_expert_seen_fixtures.py --out godot_export/expert_seen_cases.json
gh release upload checkpoint-latest godot_export/expert_seen_cases.json --clobber
