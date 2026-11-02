#!/usr/bin/env bash
set -euo pipefail
# Cas de parite du port GDScript de l'observation de jeu (v6 : bloc P allies), sur les vrais reseaux. Mesure seule.
mkdir -p godot_export
python -u scripts/dump_play_parity_fixtures.py --out godot_export/play_parity_cases.json
gh release upload checkpoint-latest godot_export/play_parity_cases.json --clobber
