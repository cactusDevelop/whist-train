#!/usr/bin/env bash
set -euo pipefail
# Ligue : expert_emb1027 (ExpertBot + bloc emballer / attente de joueur_1027) contre les autres profils.
python -u scripts/test_expert_bot.py
python -u scripts/bot_league.py --deals 40000 --workers 4
