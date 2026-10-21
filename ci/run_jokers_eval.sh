#!/usr/bin/env bash
set -euo pipefail
# Measurement only (no state saved, no chaining): restores the jokers state, then
# 1) smoke tests of the champion gate (resume and --init) in throwaway dirs,
# 2) out-of-sample comparison under the current rules (hybrid = what ships).
CK=checkpoints_jokers
for d in smoke_resume smoke_init; do rm -rf "/tmp/$d"; mkdir -p "/tmp/$d"; done
cp "$CK/jokers.pt" "$CK/jokers_last.pt" /tmp/smoke_resume/
echo "=== smoke resume ==="
python -u scripts/train_jokers.py --checkpoint-dir /tmp/smoke_resume --resume --iters 2 --hands 80 \
  --gate-matches 16 --profit-matches 0 --workers 4
echo "=== smoke init ==="
python -u scripts/train_jokers.py --checkpoint-dir /tmp/smoke_init --init "$CK/archive_043f_iter36_exported.pt" \
  --iters 2 --hands 80 --gate-matches 16 --profit-matches 0 --workers 4
python - <<'PY'
import torch
for d in ("/tmp/smoke_resume", "/tmp/smoke_init"):
    for f in ("jokers.pt", "jokers_last.pt"):
        ck = torch.load(d + "/" + f, map_location="cpu", weights_only=False)
        print(d, f, "iter", ck["iter"], "champion", ck.get("champion"), "gate", (ck["eval"] or {}).get("gate"))
PY
echo "=== hors echantillon (regles 79ac) ==="
python -u scripts/compare_jokers_oos.py --any-rules --matches 800 --workers 4 --modes "rules;hybrid;net" \
  "$CK/archive_043f_iter36_exported.pt" "$CK/jokers.pt" "$CK/jokers_last.pt"
