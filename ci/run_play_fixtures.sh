#!/usr/bin/env bash
set -euo pipefail
# Cas de parite des ports GDScript (jeu v6, enchere v2, regles d'enchere, ExpertBot sous jokers), sur les vrais reseaux. Mesure seule.
# 08/11 : regeneres apres les encheres 1.2.18 (whist-4-it 87c39f4) et 0e9a.
mkdir -p godot_export
python -u scripts/dump_play_parity_fixtures.py --out godot_export/play_parity_cases.json
python -u scripts/dump_bid_parity_fixtures.py --out godot_export/bid_parity_cases.json
python -u scripts/dump_bid_rules_fixtures.py --out godot_export/bid_rules_cases.json
python -u scripts/dump_expert_seen_fixtures.py --out godot_export/expert_seen_cases.json
for f in play_parity_cases bid_parity_cases bid_rules_cases expert_seen_cases; do
  gh release upload checkpoint-latest "godot_export/$f.json" --clobber
done
