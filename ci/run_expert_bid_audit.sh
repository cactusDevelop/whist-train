#!/usr/bin/env bash
set -euo pipefail
# Renom d'atout : taux apres promotion de KEEP_TRUMPS_VS_VOID. Mesure seule.
python -u scripts/trump_renom_audit.py --deals 20000 --workers 4
