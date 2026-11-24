#!/usr/bin/env bash
set -euo pipefail
# A/B des regles de misere de l'ExpertBot (signalement 24/11 Petite Misere). Mesure seule.
# Input minutes = duree ; input part = graine (vide = principale). Chaine (meme graine + 7) tant que
# ci/misere_ab_stop_at (UTC) laisse >= 60 min ; duree = min(330, temps restant - 15).
SEED="${AB_PART:-20261124}"
python -u scripts/petite_misere_ab.py --minutes "${FREE_MINUTES:-300}" --workers 4 --weights godot_weights_ingame --seed "$SEED"
if [ -f ci/misere_ab_stop_at ]; then
  left=$(( ( $(date -u -d "$(cat ci/misere_ab_stop_at)" +%s) - $(date -u +%s) ) / 60 - 15 ))
  [ "$left" -gt 330 ] && left=330
  if [ "$left" -ge 60 ]; then
    gh workflow run train.yml -f mode=misere_ab -f minutes="$left" -f part="$((SEED + 7))"
  else
    echo "arret de la chaine misere_ab : $(cat ci/misere_ab_stop_at)"
  fi
fi
