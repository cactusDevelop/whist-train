#!/usr/bin/env bash
set -euo pipefail
# A/B des regles de misere de l'ExpertBot (signalement 24/11 Petite Misere). Mesure seule. Input minutes = duree.
python -u scripts/petite_misere_ab.py --minutes "${FREE_MINUTES:-300}" --workers 4 --weights godot_weights_ingame
