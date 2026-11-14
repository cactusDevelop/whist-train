#!/usr/bin/env bash
set -euo pipefail
# Routage d'enchere (15e conseil etape 2) : encherisseur mesure (etat ppo_free, comme free_match --bid-cells)
# compare au jeu, exporte, + cas de parite du routage et de l'enchere reseau pour whist-4-it. Mesure seule.
python -u scripts/export_bid_route.py --ingame godot_weights_ingame --out godot_export_bid
python -u scripts/dump_bid_parity_fixtures.py --godot-dir godot_export_bid --out godot_export_bid/bid_parity_cases.json
tar cz -C godot_export_bid . | openssl enc -aes-256-cbc -pbkdf2 -iter 200000 -pass env:STATE_KEY -out bid_route_export.tar.gz.enc
gh release upload checkpoint-latest bid_route_export.tar.gz.enc --clobber
