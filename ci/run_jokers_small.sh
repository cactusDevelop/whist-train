#!/usr/bin/env bash
set -euo pipefail
# Petit bloc de verification (pas de chainage) : 80 minutes d'entrainement jokers.
JOKER_MINUTES=80 bash ci/run_jokers.sh
