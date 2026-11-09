#!/usr/bin/env bash
set -euo pipefail
# A/B duplicate des reglages de jeu de l'ExpertBot (variantes : scripts/expert_bid_ab.py VARIANTS). Mesure seule.
# ci/expert_ab_others (optionnel) : profils PROFILES des adversaires, un passage par profil.
for o in $(cat ci/expert_ab_others 2>/dev/null || echo expert); do
  echo "### adversaires : $o"
  python -u scripts/expert_bid_ab.py --deals "${FREE_MINUTES:-30000}" --workers 4 --others "$o" --seed "$(( ${GITHUB_RUN_ID:-20261109} % 1000000007 ))"
done
