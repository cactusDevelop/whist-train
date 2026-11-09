#!/usr/bin/env bash
set -euo pipefail
# A/B duplicate des reglages de jeu de l'ExpertBot (variantes : scripts/expert_bid_ab.py VARIANTS). Mesure seule.
python -u scripts/expert_bid_ab.py --deals "${FREE_MINUTES:-30000}" --workers 4 --seed "$(( ${GITHUB_RUN_ID:-20261109} % 1000000007 ))"
