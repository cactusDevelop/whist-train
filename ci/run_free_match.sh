#!/usr/bin/env bash
set -euo pipefail
# IA libre contre ExpertBot, championne et hasard (mesure seule). Input minutes = nombre de donnes par ligne.
# ci/free_match_args (optionnel) : options en plus, ex. "--pairs hybride:expert,best:expert --hybrid-roles piccolo,grande_misere --seed 20261109 --bids expert"
EXTRA=$(cat ci/free_match_args 2>/dev/null || true)
python -u scripts/free_match.py --deals "${FREE_MINUTES:-10000}" $EXTRA
