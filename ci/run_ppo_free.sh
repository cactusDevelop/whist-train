#!/usr/bin/env bash
set -euo pipefail
# IA libre (scripts/train_free.py). FREE_MINUTES court = verification rapide : cycles et mesures reduits.
MIN=${FREE_MINUTES:-}
if [ -n "$MIN" ] && [ "$MIN" -lt 90 ]; then
  EXTRA="--cycle-seconds $(( MIN * 60 / 3 )) --select-deals 400 --gate-deals 2000 --gate-extra-deals 2000"
else
  EXTRA="--cycle-seconds 1800 --gate-extra-deals 30000"
fi
# defense ciblee (05/11) : dds sur 10 % des decisions de defense hors 1er pli ; 05/11 soir : 4 mondes (x2 debit), <= 75 % du temps (17 portes refusees avec 8 mondes / 50 %, ~1000 cibles/cycle)
# 06/11 : 4 runs sans progres en defense (BILAN -1.05) -> dds limite a emballage+solitaire (pires contrats), prob 0.3
# 06/11 soir (diag free_emb_diag) : emballage = FOCUS (entames + prises a plusieurs cartes gagnantes, les deux camps, prob 0.5) ; solitaire garde la defense dds 0.3
# 08/11 (diag misere 42465520905, defense -7.46) : misere etalee = entames+defausses de defense notees par suites ExpertBot (toutes, 6 mondes, marge 4 pts)
# 07/11 : FOCUS retire (35 portes refusees, emballage -3.67 inchange) ; porte a deux temps (+30000 donnes si positif sous le bruit)
python -u scripts/train_free.py --until "$(date -u -d "+${MIN:-320} minutes" +%Y-%m-%dT%H:%M)" $EXTRA \
  --defense-rollout-prob 0.3 --defense-rollout-samples 4 --defense-rollout-margin 0.5 --defense-rollout-share 0.75 \
  --defense-rollout-roles solitaire   --misere-rollout-prob 1.0 --misere-rollout-samples 6 --misere-rollout-margin 4
