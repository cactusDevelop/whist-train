#!/usr/bin/env bash
set -euo pipefail
# Cas de parite des ports GDScript. 08/11 : poids = ceux PUBLIES dans le jeu (godot_weights_ingame/ = whist-4-it/game,
# export du 22/10), encheres 1.2.18 ; + hybride (best.pt) ; + ExpertBot complet et sequences d'enchere.
python -u scripts/test_engine.py
mkdir -p godot_export
G=godot_weights_ingame
python -u scripts/dump_play_parity_fixtures.py --godot-dir $G --out godot_export/play_parity_cases.json
python -u scripts/dump_bid_parity_fixtures.py --godot-dir $G --out godot_export/bid_parity_cases.json
python -u scripts/dump_bid_rules_fixtures.py --out godot_export/bid_rules_cases.json
python -u scripts/dump_expert_bot_fixtures.py --out godot_export/expert_bot_parity.json
python -u scripts/dump_bid_cross_check_sequences.py
python -u scripts/dump_expert_seen_fixtures.py --out godot_export/expert_seen_cases.json  # 10/11 : avant seulement dans jokers_test
mkdir -p /tmp/ppofree
gh release download checkpoint-latest -p "state-ppo_free.tar.gz.enc" -D /tmp/ppofree --clobber
openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -pass env:STATE_KEY -in /tmp/ppofree/state-ppo_free.tar.gz.enc | tar xz -C /tmp/ppofree
python -u scripts/dump_play_parity_fixtures.py --free-best /tmp/ppofree/checkpoints_free/best.pt --out godot_export/play_parity_cases_libre.json
for f in play_parity_cases bid_parity_cases bid_rules_cases expert_bot_parity bid_cross_check_sequences play_parity_cases_libre expert_seen_cases; do
  gh release upload checkpoint-latest "godot_export/$f.json" --clobber
done
