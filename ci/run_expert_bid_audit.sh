#!/usr/bin/env bash
set -euo pipefail
# A/B duplicate des reglages de jeu de l'ExpertBot (variantes : scripts/expert_bid_ab.py VARIANT_SETS). Mesure seule.
# ci/expert_ab_others (optionnel) : profils PROFILES des adversaires, un passage par profil.
# ci/expert_ab_sets (optionnel, 12/11) : lignes "<jeu> <decalage de graine>" enchainees dans le meme run
# (meme jeu + autre decalage = confirmation sur graines neuves). Defaut : "a27 0".
SETS=$(cat ci/expert_ab_sets 2>/dev/null || echo "a27 0")
while read -r set off; do
  [ -z "${set:-}" ] && continue
  for o in $(cat ci/expert_ab_others 2>/dev/null || echo expert); do
    echo "### jeu $set decalage $off adversaires : $o"
    AB_SET="$set" python -u scripts/expert_bid_ab.py --deals "${FREE_MINUTES:-30000}" --workers 4 --others "$o" --seed "$(( (${GITHUB_RUN_ID:-20261109} + off) % 1000000007 ))" $([ -f ci/expert_ab_trace ] && echo --trace)
  done
done <<< "$SETS"
