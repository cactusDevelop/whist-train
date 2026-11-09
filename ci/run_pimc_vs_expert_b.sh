#!/usr/bin/env bash
set -euo pipefail
# PIMC (dds sur mondes) contre ExpertBot en duplicate, defense et declarant (mesure seule). Input minutes = duree.
# Graine = numero du run : deux runs = donnes differentes (resultats cumulables).
python -u scripts/pimc_vs_expert.py --minutes "${FREE_MINUTES:-60}" --workers 4 --samples 8 --seed "$(( ${GITHUB_RUN_ID:-20261109} % 1000000007 ))"
