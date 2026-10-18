#!/usr/bin/env bash
# Usage : bash ci/fetch_state.sh <ppo|jokers> <fichier de clé> <dossier cible>
set -euo pipefail
MODE=$1; KEY=$2; DEST=$3
TMP=$(mktemp -d)
gh release download checkpoint-latest -R cactusDevelop/whist-train -p "state-$MODE.tar.gz.enc" -D "$TMP"
openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -pass file:"$KEY" -in "$TMP/state-$MODE.tar.gz.enc" | tar xz -C "$DEST"
rm -rf "$TMP"
