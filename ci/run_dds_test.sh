#!/usr/bin/env bash
set -uo pipefail
# BRANCHE bids-1218 : tests du port des encheres 1.2.18 (moteur, ExpertBot, jokers), pas de dds.
python -u scripts/test_engine.py || exit 1
python -u scripts/test_expert_bot.py || exit 1
python -u scripts/test_jokers.py || exit 1
