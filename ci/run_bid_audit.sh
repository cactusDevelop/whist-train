#!/usr/bin/env bash
set -euo pipefail
# 11/11 pivot encheres (10e conseil) : audit des encheres ExpertBot par profil (4 en parallele, 1 coeur chacun).
# Input minutes = donnes par profil (defaut 20000). Points reels par contrat = ou les encheres coutent.
D="${FREE_MINUTES:-20000}"; SEED="$(( ${GITHUB_RUN_ID:-20261111} % 1000000007 ))"
for p in expert joueur_1027 expert_tune1027 joueur_0927; do
  python -u scripts/expert_bid_audit.py --deals "$D" --seed "$SEED" --profile "$p" > "bid_audit_$p.txt" 2>&1 &
done
wait
for p in expert joueur_1027 expert_tune1027 joueur_0927; do echo "### profil $p"; grep -v "donnes, [0-9]*s$" "bid_audit_$p.txt"; done
