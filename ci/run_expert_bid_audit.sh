#!/usr/bin/env bash
set -euo pipefail
# Measurement only: ExpertBot bidding audit (impossible contracts via DDS, fatal raises) -- scripts/expert_bid_audit.py.
# 27/10 : reglages actuels (tune 40411623914) contre anciens (PROPOSE_MIN 5 regle utilisateur...), memes donnes.
OLD='{"PROPOSE_MIN": 5, "EMBALLER_MIN": 3, "ABONDANCE_MIN": 9, "PROPOSER_ASSUMED": 5, "EMBALLER_ASSUMED": 3, "SOLO_MIN_TOP_HONOURS": 2, "RAISE_CAP": 11}'
python -u scripts/expert_bid_audit.py --deals 20000 > audit_new.txt 2>&1 &
python -u scripts/expert_bid_audit.py --deals 20000 --params "$OLD" > audit_old.txt 2>&1 &
wait
echo "########## NOUVEAUX REGLAGES"; cat audit_new.txt
echo "########## ANCIENS REGLAGES"; cat audit_old.txt
