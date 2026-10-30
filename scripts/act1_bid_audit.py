"""Encheres de l'Acte I (utilisateur 30/10 : « des encherisseurs moins agressifs
[...] les nouveaux joueurs se font eclater »). Mesure seule, cloud.

Memes donnes pour chaque variante : un siege « joueur » (ExpertBot, tournant a
chaque donne) contre 3 adversaires a la variante. Le derapage du jeu de la
carte et les jokers du roguelike ne sont pas simules ; le joueur simule est
plus fort qu'un debutant : seules les differences entre variantes comptent.
Sortie : points du joueur, contrats adverses pour 1000 donnes, niveau moyen,
reussite, points du joueur sur ces donnes.

Usage : python -u scripts/act1_bid_audit.py --deals 20000 --workers 4
"""

import argparse
import multiprocessing as mp
import random
import sys

sys.path.insert(0, ".")

from agents.expert_bot import ACT1_BID_PARAMS, PROFILES
from engine.whisted import DeckChain
from scripts.eval_vs_random import declarer_successes
from scripts.expert_bid_ab import play_deal, stat
from scripts.train import MEASURE_WHISTED_RATIO, declarer_role_and_seats

VARIANTS = {
    "avant_1030": PROFILES["expert_tune1027"],
    "expert": {},
    "marge1": {"SOLO_MARGIN": 1, "RAISE_MARGIN": 1},
    "cap8": {"RAISE_CAP": 8},
    "emb5": {"EMBALLER_MIN": 5},
    "solo7": {"SOLO_MIN": 7},
    "doux": ACT1_BID_PARAMS,
}


def worker(args):
    seed, n = args
    rng = random.Random(seed)
    chain = DeckChain(MEASURE_WHISTED_RATIO, rng)
    rows = []  # (variante, points joueur, type adverse ou None, niveau, reussi)
    for i in range(n):
        hands, dealer, whisted = chain.deal(rng)
        hands = [list(h) for h in hands]
        hero = i % 4
        ref = None
        for name, params in VARIANTS.items():
            pts, c, game = play_deal(hands, dealer, whisted, [{} if s == hero else params for s in range(4)])
            if name == "expert":
                ref = c, game
            kind, level, ok = None, 0, False
            if c is not None:
                _r, decl = declarer_role_and_seats(c)
                if hero not in decl:
                    kind, level, ok = c["type"], c.get("level") or 0, all(declarer_successes(c, game, decl))
            rows.append((name, pts[hero], kind, level, ok))
        c, game = ref
        if game is None:
            chain.finish_unplayed(hands, dealer)
        else:
            chain.finish_played(game, dealer, c["type"])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261032)
    args = parser.parse_args()
    per = args.deals // args.workers
    with mp.Pool(args.workers) as pool:
        parts = pool.map(worker, [(args.seed * 1000 + w, per) for w in range(args.workers)])
    rows = [r for p in parts for r in p]
    print("variante : pts joueur/donne ; puis par type : contrats/1000, niveau moyen, reussite, pts joueur")
    for name in VARIANTS:
        rs = [r for r in rows if r[0] == name]
        m, se = stat([r[1] for r in rs])
        print("%-11s %+.3f +- %.3f" % (name, m, se))
        for kind in sorted({r[2] for r in rs if r[2]}):
            ks = [r for r in rs if r[2] == kind]
            print("    %-10s %7.1f %5.2f %5.1f%% %+7.2f" % (
                kind, 1000.0 * len(ks) / len(rs), sum(r[3] for r in ks) / len(ks),
                100.0 * sum(r[4] for r in ks) / len(ks), sum(r[1] for r in ks) / len(ks)))
    print("ACT1_DONE")


if __name__ == "__main__":
    main()
