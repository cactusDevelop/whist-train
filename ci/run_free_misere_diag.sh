#!/usr/bin/env bash
set -euo pipefail
# Ou best perd ses points en misere etalee vs ExpertBot (mesure seule). Input minutes = misere etalees visees.
python -u scripts/misere_free_diag.py --deals "${FREE_MINUTES:-2000}" --minutes 45
