#!/usr/bin/env bash
set -euo pipefail
# Stats IA pour le joueur (whist-4-it Parametres > Stats IA) : IA du jeu aux 4 sieges, poids publies
# (godot_weights_ingame = copie de whist-4-it/game). Mesure seule. Input minutes = nombre de donnes.
python -u scripts/ai_insights.py --deals "${FREE_MINUTES:-100000}" --workers 4 --weights godot_weights_ingame $(cat ci/ai_insights_args 2>/dev/null || true) --out ai_insights.json
gh release upload checkpoint-latest ai_insights.json --clobber
