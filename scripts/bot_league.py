"""Ligue des profils de bot (agents/expert_bot.PROFILES) -- utilisateur
2026-10-27 : « mets les regles que je viens de te donner en concurrent a
l'ExpertBot, pas en remplacement [...] Le but n'est pas de creer
artificiellement le meilleur bot mais de trouver la meilleure IA. »
Mesure seule, cloud.

Chaque paire de profils joue en duplicate sur les memes donnes neuves
(scripts/expert_tune.duel : une paire de sieges B contre A, deux placements,
vrais points) ; un profil peut aussi recevoir des reglages en plus
(--extra profil='{json}', ex. le FINAL d'un tune). Classement = somme des
gains contre tous les autres.

Usage : python -u scripts/bot_league.py --deals 40000 --workers 4
"""

import argparse
import itertools
import json
import multiprocessing as mp
import sys

sys.path.insert(0, ".")

from agents.expert_bot import PROFILES
from scripts.expert_tune import duel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=40000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--extra", action="append", default=[], help="profil='{json}' (reglages en plus)")
    args = parser.parse_args()
    profiles = {k: dict(v) for k, v in PROFILES.items()}
    for spec in args.extra:
        name, js = spec.split("=", 1)
        profiles[name] = dict(profiles.get(name.split("+")[0], {}), **json.loads(js))
    for name, params in profiles.items():
        print("profil %-16s %s" % (name, json.dumps(params, sort_keys=True)), flush=True)
    score = {k: 0.0 for k in profiles}
    with mp.Pool(args.workers) as pool:
        for i, (a, b) in enumerate(itertools.combinations(profiles, 2)):
            m, se = duel(pool, args.workers, args.seed * 1000 + 1000 * i, args.deals, profiles[a], profiles[b])
            score[b] += m
            score[a] -= m
            print("%-16s vs %-16s : %+.3f +- %.3f pt/donne pour %s (z %.1f)" % (
                b, a, m, se, b, m / se if se else 0), flush=True)
    print("\n=== Classement (somme des gains en duplicate) ===")
    for k in sorted(score, key=score.get, reverse=True):
        print("%-16s %+.3f" % (k, score[k]))
    print("LEAGUE_DONE")


if __name__ == "__main__":
    main()
