#!/usr/bin/env bash
set -euo pipefail
# Measurement only (03/11): rule search on match score, champion untouched.
# FREE_MINUTES (input minutes) = smoke: tiny sample sizes.
if [ -n "${FREE_MINUTES:-}" ]; then
  python -u scripts/search_joker_rules.py --until "$(date -u -d "+${FREE_MINUTES} minutes" +%Y-%m-%dT%H:%M)" \
    --workers 4 --discover 8 --screen 8 --confirm 8 --min-decisions 1
else
  python -u scripts/search_joker_rules.py --until "${RULES_UNTIL:-$(date -u -d "+290 minutes" +%Y-%m-%dT%H:%M)}" --workers 4 --discover 500 --screen 1000 --confirm 2000
fi
