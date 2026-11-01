#!/usr/bin/env bash
set -euo pipefail
# IA libre contre ExpertBot, championne et hasard (mesure seule). Input minutes = nombre de donnes par ligne.
python -u scripts/free_match.py --deals "${FREE_MINUTES:-10000}"
