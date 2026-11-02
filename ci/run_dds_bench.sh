#!/usr/bin/env bash
set -euo pipefail
# Mesure seule : A/B de vitesse du solveur dds (compilation x premiere sonde) dans un meme job.
python -u scripts/bench_dds.py
