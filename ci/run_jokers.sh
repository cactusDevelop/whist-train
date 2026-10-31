#!/usr/bin/env bash
set -euo pipefail
# Pause: while ci/jokers_paused exists, fail before training so the run does not chain.
if [ -f ci/jokers_paused ] && [ -z "${JOKER_MINUTES:-}" ]; then echo "jokers en pause : $(cat ci/jokers_paused)"; exit 1; fi
# An iteration lasts ~18 min past --until: keep the save step inside the 6 h job cap.
UNTIL=$(date -u -d "+${JOKER_MINUTES:-310} minutes" +%Y-%m-%dT%H:%M)
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
  python -u scripts/train_jokers.py --init "$BEST" --workers 4 --until "$UNTIL"
else
  python -u scripts/train_jokers.py --resume --workers 4 --until "$UNTIL"
fi
