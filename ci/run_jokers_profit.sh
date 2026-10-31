#!/usr/bin/env bash
set -euo pipefail
# Mesure seule (31/10) : rentabilite de tous les jokers sous l'empreinte courante, champion courant.
python -u scripts/measure_profit.py --minutes "${PROFIT_MINUTES:-35}" --workers 4
gh release upload checkpoint-latest godot_export/joker_profit_stats.json joker_profit_summary.md --clobber
