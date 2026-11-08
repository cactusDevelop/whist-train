#!/usr/bin/env bash
set -euo pipefail
# 08/11 (conseil IA joker) : mesure seule, hors echantillon, regles actuelles (f165) :
# champion iter 56 en reseau / hybride contre regles seules, + heuristique et « jamais » (baselines de eval_worker).
# Question : le reseau apporte-t-il quelque chose par-dessus les regles ecrites ?
CK=checkpoints_jokers
python -u scripts/compare_jokers_oos.py --any-rules --matches 1200 --workers 4 --modes "rules;hybrid;net" "$CK/jokers.pt"
