#!/usr/bin/env bash
set -euo pipefail
# Mesure seule (G1 06/11) : moments utiles des jokers du joueur (jouer / garder / recycler,
# fin de match) + valeur d'un energie et d'une pioche. Input minutes = duree (defaut 150).
# 8ebd : minutes suffixees par « v » (ex. 90v) = valeur du ⚡ / de la pioche seule (joker_values*).
M="${FREE_MINUTES:-150}"
# 10/11 (conseil) : minutes suffixees par « s » (ex. 150s) = graines NEUVES (run id) pour confirmer un ecart.
SEED_ARG=()
if [[ "$M" == *s ]]; then M="${M%s}"; SEED_ARG=(--seed "$(( ${GITHUB_RUN_ID:-1} % 100000000 + 7000000 ))"); fi
if [[ "$M" == *v ]]; then
  python -u scripts/joker_moments.py --minutes "${M%v}" --workers 4 --k 2 --value-only
  gh release upload checkpoint-latest joker_values.json joker_values_summary.md --clobber
else
  # 09/11 : rapport d'equilibrage par lot (scripts/joker_report.py) contre la mesure precedente.
  gh release download checkpoint-latest -p joker_moments.json -O prev_joker_moments.json --clobber || true
  python -u scripts/joker_moments.py --minutes "$M" --workers 4 --k 2 "${SEED_ARG[@]}"
  gh release upload checkpoint-latest joker_moments.json joker_moments_summary.md --clobber
  python -u scripts/joker_report.py --cur joker_moments.json --prev prev_joker_moments.json || true
  [ -f prev_joker_moments.json ] && cp prev_joker_moments.json joker_moments_prev.json || true
  gh release upload checkpoint-latest joker_report.md --clobber || true
  [ -f joker_moments_prev.json ] && gh release upload checkpoint-latest joker_moments_prev.json --clobber || true
fi
