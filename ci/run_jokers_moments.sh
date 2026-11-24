#!/usr/bin/env bash
set -euo pipefail
# Mesure seule (G1 06/11) : moments utiles des jokers du joueur (jouer / garder / recycler,
# fin de match) + valeur d'un energie et d'une pioche. Input minutes = duree (defaut 150).
# 8ebd : minutes suffixees par « v » (ex. 90v) = valeur du ⚡ / de la pioche seule (joker_values*).
M="${FREE_MINUTES:-150}"
# 23/11 (iter 17 / 1 acceptation sur 22) : input part = variante.
#   part=confirm : contre-mesure iter 17 (jokers.pt) contre iter 56 (asset chiffre jokers_alt.pt.enc), graines neuves
#   part=bots_alt : moments sous les regles actuelles avec les bots iter 56, sorties joker_moments_alt* (ne touche
#                   pas joker_moments.json, celui du jeu) ; memes graines que la mesure precedente (apparie)
#   part=boss_check (conseil 24/11) : mesure pre-enregistree hybride champion vs heuristique sur les matchs
#                   fossoyeur seuls, graines neuves (scripts/joker_boss_check.py), sorties joker_boss_check*
if [ "${AB_PART:-}" = boss_check ]; then
  python -u scripts/joker_boss_check.py checkpoints_jokers/jokers.pt --boss fossoyeur --minutes "$M" --workers 4 --seed "${GITHUB_RUN_ID:-1}"
  gh release upload checkpoint-latest joker_boss_check.json joker_boss_check.md --clobber
  exit 0
fi
if [ -n "${AB_PART:-}" ]; then
  gh release download checkpoint-latest -p jokers_alt.pt.enc -D /tmp --clobber
  openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -pass env:STATE_KEY -in /tmp/jokers_alt.pt.enc -out /tmp/jokers_alt.pt
  if [ "$AB_PART" = confirm ]; then
    python -u scripts/joker_confirm.py checkpoints_jokers/jokers.pt /tmp/jokers_alt.pt --minutes "$M" --workers 4       --seed "${GITHUB_RUN_ID:-1}"
    gh release upload checkpoint-latest joker_confirm.json joker_confirm.md --clobber
  elif [ "$AB_PART" = confirm_night ]; then
    # 24/11 : chaine en fin de nuit jokers : champion de la nuit (jokers.pt) contre jokers_alt (iter 17), graines neuves
    python -u scripts/joker_confirm.py checkpoints_jokers/jokers.pt /tmp/jokers_alt.pt --minutes "$M" --workers 4       --seed "$(( ${GITHUB_RUN_ID:-1} + 1 ))" --out joker_confirm_night
    gh release upload checkpoint-latest joker_confirm_night.json joker_confirm_night.md --clobber
  elif [ "$AB_PART" = bots_alt ]; then
    python -u scripts/joker_moments.py --minutes "$M" --workers 4 --k 2 --checkpoint /tmp/jokers_alt.pt --matches 5440 --out-prefix joker_moments_alt
    gh release upload checkpoint-latest joker_moments_alt.json joker_moments_alt_summary.md --clobber
    gh release download checkpoint-latest -p joker_moments.json --clobber || true
    python -u scripts/joker_report.py --cur joker_moments_alt.json --prev joker_moments.json --out joker_report_alt.md || true
    [ -f joker_report_alt.md ] && gh release upload checkpoint-latest joker_report_alt.md --clobber || true
  else
    echo "part inconnu : $AB_PART"; exit 1
  fi
  exit 0
fi
# 10/11 (conseil) : minutes suffixees par « s » (ex. 150s) = graines NEUVES (run id) pour confirmer un ecart.
SEED_ARG=()
if [[ "$M" == *s ]]; then M="${M%s}"; SEED_ARG=(--seed "$(( ${GITHUB_RUN_ID:-1} % 100000000 + 7000000 ))"); fi
if [[ "$M" == *e ]]; then
  # 16/11 (whist-4-it) : minutes suffixees par « e » = decomposition des moteurs (scripts/joker_engines.py)
  python -u scripts/joker_engines.py --minutes "${M%e}" --workers 4 --k 2
  gh release upload checkpoint-latest joker_engines.json joker_engines_summary.md --clobber
elif [[ "$M" == *v ]]; then
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
