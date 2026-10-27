#!/usr/bin/env bash
set -euo pipefail
# Cout de chaque regle du profil joueur_1027 (retrait contre le profil complet, ajout seul a l'ExpertBot).
python -u scripts/test_expert_bot.py
python -u scripts/profile_ablation.py --profile joueur_1027 --deals 40000 --workers 4
