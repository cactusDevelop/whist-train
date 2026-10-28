"""Cout de chaque regle d'un profil (agents/expert_bot.PROFILES) -- mesure
seule, cloud. Utilisateur 2026-10-27 : joueur_1027 perd -2.4 pt/donne contre
joueur_0927 ; on cherche quelle regle coute.

Deux sens, en duplicate sur des donnes neuves (scripts/expert_tune.duel) :
  - retrait : profil SANS la regle (reglages par defaut) contre le profil
    complet ; positif = la regle coute ce nombre de points par donne ;
  - ajout : ExpertBot AVEC la seule regle contre l'ExpertBot ;
    negatif = la regle fait perdre des points a elle seule.
Une regle = un groupe de reglages (proposer, emballer) ou un reglage seul.

Usage : python -u scripts/profile_ablation.py --profile joueur_1027 --deals 40000 --workers 4
"""

import argparse
import json
import multiprocessing as mp
import sys

sys.path.insert(0, ".")

from agents.expert_bot import PROFILES
from scripts.expert_tune import duel


GROUPS = {
    "proposer": ("PROPOSE_MIN", "PROPOSE_TRUMP_HONOUR_TRICKS"),
    "emballer": ("EMBALLER_MIN", "EMBALLER_LONG_TRUMPS", "EMBALLER_SHORT_TRUMPS"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="joueur_1027")
    parser.add_argument("--deals", type=int, default=40000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    full = dict(PROFILES[args.profile])
    rules = {}
    for name, keys in GROUPS.items():
        if len([k for k in keys if k in full]) > 1:
            rules[name] = tuple(k for k in keys if k in full)
    for k in sorted(full):
        rules[k] = (k,)
    print("profil %s %s" % (args.profile, json.dumps(full, sort_keys=True)), flush=True)
    rows = []
    with mp.Pool(args.workers) as pool:
        for i, (name, keys) in enumerate(rules.items()):
            without = {k: v for k, v in full.items() if k not in keys}
            only = {k: full[k] for k in keys}
            m_out, se_out = duel(pool, args.workers, args.seed * 1000 + 2000 * i, args.deals, full, without)
            m_add, se_add = duel(pool, args.workers, args.seed * 1000 + 2000 * i + 1000, args.deals, {}, only)
            rows.append((name, m_out, se_out, m_add, se_add))
            print("%-28s retrait %+.3f +- %.3f   ajout a expert %+.3f +- %.3f" % (
                name, m_out, se_out, m_add, se_add), flush=True)
    print("\n=== Cout des regles (retrait positif = la regle coute) ===")
    for name, m_out, se_out, m_add, se_add in sorted(rows, key=lambda r: -r[1]):
        print("%-28s retrait %+.3f (z %.1f)   ajout %+.3f (z %.1f)" % (
            name, m_out, m_out / se_out if se_out else 0, m_add, m_add / se_add if se_add else 0))
    print("ABLATION_DONE")


if __name__ == "__main__":
    main()
