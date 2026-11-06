#!/usr/bin/env bash
set -euo pipefail
# Mesure seule (G1 06/11) : moments utiles des jokers du joueur (jouer / garder / recycler,
# fin de match) + valeur d'un energie et d'une pioche. Input minutes = duree (defaut 150).
python -u scripts/joker_moments.py --minutes "${FREE_MINUTES:-150}" --workers 4 --k 2
gh release upload checkpoint-latest joker_moments.json joker_moments_summary.md --clobber
