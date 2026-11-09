#!/usr/bin/env bash
set -euo pipefail
# Pause: while ci/jokers_paused exists, fail before training so the run does not chain.
if [ -f ci/jokers_paused ] && [ -z "${JOKER_MINUTES:-}" ]; then echo "jokers en pause : $(cat ci/jokers_paused)"; exit 1; fi
# An iteration lasts ~18 min past --until: keep the save step inside the 6 h job cap.
UNTIL=$(date -u -d "+${JOKER_MINUTES:-310} minutes" +%Y-%m-%dT%H:%M)
# Stop date (UTC): chained runs end here, ci/jokers_stop_at holds it. Failing stops the chain.
if [ -f ci/jokers_stop_at ] && [ -z "${JOKER_MINUTES:-}" ]; then
  STOP=$(cat ci/jokers_stop_at)
  if [[ "$(date -u +%Y-%m-%dT%H:%M)" > "$STOP" ]]; then echo "chainage jokers fini : $STOP"; exit 1; fi
  if [[ "$UNTIL" > "$STOP" ]]; then UNTIL=$STOP; fi
fi
# 02/11 : --lr 3e-4 --epochs 2 (avant 1e-3 x 6 sur 3 iterations rejouees = 18 passes) : la perte baissait
# pendant que le candidat s'eloignait du champion (iter 4..7 : -2.6 -> -19.6), surapprentissage des cibles bruitees.
# 09/11 (conseil) : lr 1e-4, porte a deux temps (+1600 matchs neufs, decision sur eux seuls), candidat = moyenne exp. 0.7 des poids.
# Rules changed since jokers.pt (spec hash not rule-equivalent): warm start from
# that champion, kept apart once as archive_<old hash>_champion.pt.
OLD=$(python - <<'PY'
import os, torch
from engine import jokers as J
p = "checkpoints_jokers/jokers.pt"
h = torch.load(p, map_location="cpu", weights_only=False).get("spec_hash") if os.path.exists(p) else None
print("" if h is None or h in J.RULE_EQUIVALENT_HASHES else h)
PY
)
if [ -n "$OLD" ] || [ "${INIT:-false}" = "true" ]; then
  BEST=checkpoints_jokers/archive_${OLD:-init}_champion.pt
  [ -f "$BEST" ] || cp checkpoints_jokers/jokers.pt "$BEST"
  rm -f checkpoints_jokers/jokers_last.pt
  echo "demarrage a chaud depuis $BEST"
  python -u scripts/train_jokers.py --init "$BEST" --workers 4 --lr 1e-4 --epochs 2 --gate-extra-matches 1600 --avg 0.7 --until "$UNTIL"
else
  python -u scripts/train_jokers.py --resume --workers 4 --lr 1e-4 --epochs 2 --gate-extra-matches 1600 --avg 0.7 --until "$UNTIL"
fi
