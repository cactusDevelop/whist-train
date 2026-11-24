#!/usr/bin/env bash
set -euo pipefail
# Measurement only (24/11, conseil palier jokers) : copies of the jokers state in /tmp, nothing saved.
#   part=sanity : candidate = champion (--epochs 0) -> vs_champion must be exactly 0 (duplicate) ;
#                 by_kind_vs_heuristic = champion vs heuristic per opponent (grand_maitre), 3000 matches.
#   part=lr1e4  : 3 iterations like production but --lr 1e-4 --epochs 1 (prod : 3e-4 x 2).
# (before 24/11 : scripts/diag_empty_hand.py, 8f41 empty-hand hunt, closed)
CK=checkpoints_jokers
D=/tmp/diag; rm -rf "$D"; mkdir -p "$D"
cp "$CK/jokers.pt" "$D/jokers.pt"
if [ -f "$CK/jokers_last.pt" ]; then cp "$CK/jokers_last.pt" "$D/jokers_last.pt"; else cp "$CK/jokers.pt" "$D/jokers_last.pt"; fi
case "${AB_PART:-}" in
  sanity)
    cp "$CK/jokers.pt" "$D/jokers_last.pt"  # candidate strictly = champion
    python -u scripts/train_jokers.py --checkpoint-dir "$D" --resume --iters 1 --hands 40 --epochs 0 \
      --gate-matches 3000 --profit-matches 0 --workers 4 ;;
  lr1e4)
    python -u scripts/train_jokers.py --checkpoint-dir "$D" --resume --iters 3 --lr 1e-4 --epochs 1 \
      --profit-matches 0 --workers 4 ;;
  *) echo "part inconnu : '${AB_PART:-}' (sanity | lr1e4)"; exit 1 ;;
esac
