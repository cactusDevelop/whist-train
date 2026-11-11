#!/usr/bin/env bash
set -euo pipefail
# 10e conseil 11/11 : PIMC en direct en DEFENSE D'EMBALLAGE, 4 profils adverses, variantes def (partout) / trig
# (seulement quand l'adversaire gagne le pli). Mesure seule. Input minutes = duree totale (partagee entre profils).
python -u scripts/pimc_vs_expert.py --oracle --minutes "${FREE_MINUTES:-240}" --workers 4 --samples 16 \
  --others expert,joueur_1027,expert_tune1027,joueur_0927 --seed "$(( ${GITHUB_RUN_ID:-20261111} % 1000000007 ))"
