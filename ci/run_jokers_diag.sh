#!/usr/bin/env bash
set -euo pipefail
# Measurement only (no state saved, no chaining): why the joker gate accepts nothing
# since iter 93. Each variant warm-starts from the champion in a throwaway dir.
run() {
  name=$1; shift
  echo "##### variante $name : $*"
  rm -rf "/tmp/diag_$name" && mkdir -p "/tmp/diag_$name"
  python -u scripts/train_jokers.py --checkpoint-dir "/tmp/diag_$name" --init checkpoints_jokers/jokers.pt \
    --profit-matches 0 --workers 4 "$@" | grep -E '^\{' | python -c '
import json, sys
for l in sys.stdin:
    r = json.loads(l)
    print("  iter %d loss %.4f vs_champion %s vs_heuristic %s champion_vs_heuristic %s accepted %s
    poses %s" % (
        r["iter"], r["loss"], r["vs_champion"], r["vs_heuristic"], r["champion_vs_heuristic"], r["gate"]["accepted"], r.get("casts_per_match")), flush=True)
'
}
run base --iters 1
run doux_lr1e-4_1ep --iters 1 --lr 1e-4 --epochs 1
run horizon_match --iters 2 --horizon match
