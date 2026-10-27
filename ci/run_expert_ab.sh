#!/usr/bin/env bash
set -euo pipefail
# Ligue des profils de bot (agents/expert_bot.PROFILES : ExpertBot et regles utilisateur EN CONCURRENTS),
# puis reglage du profil joueur_1027 borne par ses regles, puis ligue finale avec ce reglage.
python -u scripts/test_expert_bot.py
python -u scripts/bot_league.py --deals 40000 --workers 4
python -u scripts/expert_tune.py --profile joueur_1027 --minutes 240 --deals 20000 --validate-deals 40000 --workers 4 | tee tune.txt
FINAL=$(grep '^FINAL ' tune.txt | sed 's/^FINAL //')
python -u scripts/bot_league.py --deals 40000 --workers 4 --extra "joueur_1027+tune=$FINAL"
