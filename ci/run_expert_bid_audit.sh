#!/usr/bin/env bash
set -euo pipefail
# GATES 30/10 : renom d'atout (audit + ligue de profils concurrents) et encheres de l'Acte I. Mesure seule.
python -u scripts/test_expert_bot.py
python -u scripts/trump_renom_audit.py --deals 20000 --workers 4
python -u scripts/bot_league.py --deals 40000 --workers 4 --only expert,expert_renom1030,expert_renom1030b
python -u scripts/act1_bid_audit.py --deals 20000 --workers 4
