#!/usr/bin/env bash
set -euo pipefail
# Mesure seule (G1 06/11) : moments utiles des jokers du joueur (jouer / garder / recycler,
# fin de match) + valeur d'un energie et d'une pioche. Input minutes = duree (defaut 150).
# 8ebd : minutes suffixees par « v » (ex. 90v) = valeur du ⚡ / de la pioche seule (joker_values*).
M="${FREE_MINUTES:-150}"
if [[ "$M" == *v ]]; then
  python -u scripts/joker_moments.py --minutes "${M%v}" --workers 4 --k 2 --value-only
  gh release upload checkpoint-latest joker_values.json joker_values_summary.md --clobber
else
  python -u scripts/joker_moments.py --minutes "$M" --workers 4 --k 2
  gh release upload checkpoint-latest joker_moments.json joker_moments_summary.md --clobber
fi
