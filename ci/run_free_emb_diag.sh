#!/usr/bin/env bash
set -euo pipefail
# Ou best perd ses points en emballage vs ExpertBot (mesure seule). Input minutes = emballages vises.
python -u scripts/emballage_free_diag.py --deals "${FREE_MINUTES:-4000}" --minutes 150
