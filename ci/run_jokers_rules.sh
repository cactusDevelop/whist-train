#!/usr/bin/env bash
set -euo pipefail
# Measurement only. 09/11 (conseil) : recherche des regles JOUEUR des jokers sur le score de match du
# joueur, en duplicate (scripts/search_player_rules.py) ; resultat player_rules_search.json (release).
# 03/11 : l'ancienne recherche des regles des bots = scripts/search_joker_rules.py.
# FREE_MINUTES (input minutes) = smoke : petits echantillons.
python -u scripts/test_jokers.py
if [ -n "${FREE_MINUTES:-}" ]; then
  python -u scripts/search_player_rules.py --until "$(date -u -d "+${FREE_MINUTES} minutes" +%Y-%m-%dT%H:%M)" \
    --workers 4 --discover 16 --screen 8 --confirm 8 --min-seen 1 --passes 2 --screen-z -9
else
  python -u scripts/search_player_rules.py --until "${RULES_UNTIL:-$(date -u -d "+325 minutes" +%Y-%m-%dT%H:%M)}" --workers 4 --discover 2000 --screen 4000 --confirm 8000 --passes 3 --min-seen 100
fi
