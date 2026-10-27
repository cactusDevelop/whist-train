#!/usr/bin/env bash
set -euo pipefail
# Measurement only: ExpertBot bidding audit (impossible contracts via DDS, fatal raises) -- scripts/expert_bid_audit.py.
# 27/10 soir : regles utilisateur (defauts) contre le tune du matin (PROPOSE_MIN 3...), memes donnes.
TUNE0='{"PROPOSE_MIN": 3, "PROPOSE_PRACTICE": 3, "EMBALLER_MIN": 1, "EMBALLER_PRACTICE": 1, "ABONDANCE_MIN": 10, "PROPOSER_ASSUMED": 7, "EMBALLER_ASSUMED": 2, "SOLO_MIN_TOP_HONOURS": 1, "RAISE_CAP": 9, "PROPOSE_MIN_HONOURS": 0, "WAIT_BALANCED": false, "EMBALLER_LONG_TRUMPS": 0, "EMBALLER_SHORT_TRUMPS": 0}'
python -u scripts/expert_bid_audit.py --deals 20000 > audit_regles.txt 2>&1 &
python -u scripts/expert_bid_audit.py --deals 20000 --params "$TUNE0" > audit_tune0.txt 2>&1 &
wait
echo "########## REGLES UTILISATEUR"; cat audit_regles.txt
echo "########## TUNE DU MATIN"; cat audit_tune0.txt
