"""Maillon faible des emballages perdus (utilisateur 30/10 : « si un contrat
est perdu et qu'une des IAs du contrat a 5 plis et l'autre 2, c'est celle qui
en a deux qu'il faut isoler et tuer »). Mesure seule, cloud.

4 ExpertBot par profil, memes donnes. Pour chaque emballage perdu, le maillon
faible = le siege du duo qui a fait le moins de plis (egalite : aucun).
Tableau par role et par nombre d'atouts en main : emballages, reussite, points
du duo, plis du siege, part de maillon faible.

Usage : python -u scripts/emballage_blame.py --deals 20000 --workers 4
"""

import argparse
import multiprocessing as mp
import random
import sys

sys.path.insert(0, ".")

from agents.expert_bot import PROFILES
from engine.cards import suit_of
from engine.whisted import DeckChain
from scripts.expert_bid_ab import play_deal
from scripts.train import MEASURE_WHISTED_RATIO

SHOWN = ("expert", "expert_tune1027")


def worker(args):
    seed, n = args
    rng = random.Random(seed)
    chain = DeckChain(MEASURE_WHISTED_RATIO, rng)
    rows = []  # (profil, role, atouts, reussi, points du duo, plis, maillon faible)
    for _ in range(n):
        hands, dealer, whisted = chain.deal(rng)
        hands = [list(h) for h in hands]
        last = None
        for name in SHOWN:
            pts, c, game = play_deal(hands, dealer, whisted, [PROFILES[name]] * 4)
            if name == SHOWN[0]:
                last = c, game
            if c is None or c["type"] != "emballage":
                continue
            emballeur, proposeur = c["spokesperson"], c["declarer"] if c["declarer"] != c["spokesperson"] else c["partner"]
            duo = game.tricks_won[emballeur] + game.tricks_won[proposeur]
            won = duo >= c["level"]
            gap = game.tricks_won[proposeur] - game.tricks_won[emballeur]
            for seat, role, weak in ((emballeur, "emballeur", gap > 0), (proposeur, "proposeur", gap < 0)):
                trumps = sum(1 for x in hands[seat] if suit_of(x) == c["suit"])
                rows.append((name, role, min(trumps, 7), won, pts[emballeur] + pts[proposeur],
                             game.tricks_won[seat], not won and weak))
        c, game = last
        if game is None:
            chain.finish_unplayed(hands, dealer)
        else:
            chain.finish_played(game, dealer, c["type"])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    per = args.deals // args.workers
    with mp.Pool(args.workers) as pool:
        parts = pool.map(worker, [(args.seed * 1000 + w, per) for w in range(args.workers)])
    rows = [r for p in parts for r in p]
    for name in SHOWN:
        for role in ("emballeur", "proposeur"):
            print("\n=== %s, %s : atouts, emballages/1000 donnes, reussite, pts duo, plis, maillon faible ===" % (name, role))
            for t in range(0, 8):
                rs = [r for r in rows if r[0] == name and r[1] == role and r[2] == t]
                if rs:
                    n = len(rs)
                    print("  %d%s %7.1f %5.1f%% %+7.2f %5.2f %5.1f%%" % (
                        t, "+" if t == 7 else " ", 1000.0 * n / (per * args.workers), 100.0 * sum(r[3] for r in rs) / n,
                        sum(r[4] for r in rs) / n, sum(r[5] for r in rs) / n, 100.0 * sum(r[6] for r in rs) / n))
    print("BLAME_DONE")


if __name__ == "__main__":
    main()
