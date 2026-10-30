"""Reglage automatique de l'ExpertBot (IA de tous les sieges en jeu) par
recherche coordonnee en duplicate -- nuit du 2026-10-26 (demande utilisateur :
« pourquoi laisser cette nuit sans entrainement ? »). Mesure seule, cloud.

Configuration courante `best` (dict de constantes agents/expert_bot.py,
{} = defauts). Pour chaque reglage et chaque valeur voisine : paires
candidat (best + changement) contre best sur des donnes NEUVES
(scripts/expert_bid_ab.play_deal, deux placements, vrais points). Adopte si
gain > Z_ACCEPT erreurs standard. Tours repetes jusqu'a un tour sans gain ou
la fin du budget ; puis validation de best contre les defauts sur des donnes
neuves. Chaque adoption est imprimee (ligne BEST {...}).

Usage : python -u scripts/expert_tune.py --profile joueur_1027 --minutes 320 --deals 20000 --workers 4
"""

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
import time

sys.path.insert(0, ".")

from engine.whisted import DeckChain
from scripts.expert_bid_ab import play_deal
from scripts.train import MEASURE_WHISTED_RATIO

# profil (agents/expert_bot.PROFILES) -> reglage -> valeurs essayees (entiers : voisins +-1
# dans ces bornes ; booleens). Chaque profil garde SES regles : un profil joueur_* reste borne
# par les regles de l'utilisateur (27/10 : le tune 40411623914 avait franchi PROPOSE_MIN 5 /
# EMBALLER_MIN 3 ; elles vivent maintenant en profils concurrents de l'ExpertBot).
_PLAY = {"SECOND_HAND_LOW": bool, "INFER_OPP_TRUMPS": bool, "TRUMP_LEAD_SEQUENCE": bool,
         "SIDE_LEAD_SEQUENCE": bool, "LEAD_PARTNER_SUIT": bool}
SPACES = {
    "expert": dict({
        "PROPOSE_MIN": (1, 7), "EMBALLER_MIN": (3, 5), "SOLO_MIN": (5, 8), "ABONDANCE_MIN": (8, 11),
        "PROPOSER_ASSUMED": (3, 9), "EMBALLER_ASSUMED": (1, 5), "PROPOSER_TRUMPS_ASSUMED": (3, 7),
        "PROPOSE_MIN_LENGTH": (3, 6), "SOLO_MIN_LENGTH": (4, 7), "SOLO_MIN_TOP_HONOURS": (0, 3),
        "ABONDANCE_MIN_HONOURS": (1, 4), "SOLO_MARGIN": (-1, 1), "RAISE_MARGIN": (-1, 1), "RAISE_CAP": (8, 13),
    }, **_PLAY),
    # regles 27/09 : proposer >= 5, emballer >= 3 ; « etre large », pas de bluff : marges >= 0
    "joueur_0927": dict({
        "PROPOSE_MIN": (5, 7), "EMBALLER_MIN": (3, 5), "SOLO_MIN": (6, 8), "ABONDANCE_MIN": (9, 11),
        "PROPOSER_ASSUMED": (5, 6), "EMBALLER_ASSUMED": (3, 5), "SOLO_MIN_TOP_HONOURS": (1, 3),
        "SOLO_MARGIN": (0, 1), "RAISE_MARGIN": (0, 1), "RAISE_CAP": (9, 13),
    }, **_PLAY),
    # regles 27/10 : proposer >= 6 avec >= 2 plis d'honneurs d'atout, emballer >= 4 (5+ atouts ou
    # 4 dont une figure), attente de la main moyenne (non reglable)
    "joueur_1027": dict({
        "PROPOSE_MIN": (6, 7), "PROPOSE_TRUMP_HONOUR_TRICKS": (2, 3), "EMBALLER_MIN": (4, 5),
        "SOLO_MIN": (6, 8), "ABONDANCE_MIN": (9, 11), "PROPOSER_ASSUMED": (5, 6), "EMBALLER_ASSUMED": (3, 5),
        "SOLO_MIN_TOP_HONOURS": (1, 3), "SOLO_MARGIN": (0, 1), "RAISE_MARGIN": (0, 1), "RAISE_CAP": (9, 13),
    }, **_PLAY),
}
Z_ACCEPT = 2.5
VALIDATE_DEALS = 60000


def worker(args):
    seed, n, best, cand = args
    rng = random.Random(seed)
    chain = DeckChain(MEASURE_WHISTED_RATIO, rng)
    gains = []
    for _ in range(n):
        hands, dealer, whisted = chain.deal(rng)
        hands = [list(h) for h in hands]
        base, bc, bgame = play_deal(hands, dealer, whisted, [best] * 4)
        g = 0.0
        for seats in ((0, 2), (1, 3)):
            pts, _c, _g = play_deal(hands, dealer, whisted, [cand if s in seats else best for s in range(4)])
            g += sum(pts[s] - base[s] for s in seats)
        gains.append(g)
        if bgame is None:
            chain.finish_unplayed(hands, dealer)
        else:
            chain.finish_played(bgame, dealer, bc["type"])
    return gains


def duel(pool, workers, seed, deals, best, cand):
    per = deals // workers
    gains = [g for part in pool.map(worker, [(seed + w, per, best, cand) for w in range(workers)]) for g in part]
    m = sum(gains) / len(gains)
    sd = math.sqrt(sum((x - m) ** 2 for x in gains) / (len(gains) - 1))
    return m, sd / math.sqrt(len(gains))


def neighbours(space, key, current):
    spec = space[key]
    if spec is bool:
        return [not current]
    lo, hi = spec
    return [v for v in (current - 1, current + 1) if lo <= v <= hi]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=320)
    parser.add_argument("--deals", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261100)
    parser.add_argument("--validate-deals", type=int, default=VALIDATE_DEALS)
    parser.add_argument("--profile", default="expert", choices=sorted(SPACES))
    args = parser.parse_args()
    from agents import expert_bot
    space = SPACES[args.profile]
    defaults = {k: getattr(expert_bot, k) for k in space}
    start = dict(expert_bot.PROFILES[args.profile])
    best = dict(start)
    print("profil %s, depart %s" % (args.profile, json.dumps(start, sort_keys=True)), flush=True)
    t0 = time.time()
    deadline = t0 + args.minutes * 60 - min(20 * 60, args.minutes * 12)  # garde ~20 min pour la validation
    seed = args.seed * 1000
    total_gain = 0.0
    with mp.Pool(args.workers) as pool:
        rnd = 0
        while time.time() < deadline:
            rnd += 1
            improved = False
            for key in space:
                cur = best.get(key, defaults[key])
                for val in neighbours(space, key, cur):
                    if time.time() >= deadline:
                        break
                    cand = dict(best)
                    cand[key] = val
                    seed += 1000
                    m, se = duel(pool, args.workers, seed, args.deals, best, cand)
                    ok = m > Z_ACCEPT * se
                    print("tour %d  %-24s %s -> %s : %+.3f +- %.3f (z %.1f)%s  [%.0f min]" % (
                        rnd, key, cur, val, m, se, m / se if se else 0, "  ADOPTE" if ok else "",
                        (time.time() - t0) / 60), flush=True)
                    if ok:
                        best = cand
                        total_gain += m
                        improved = True
                        print("BEST %s" % json.dumps(best, sort_keys=True), flush=True)
                        break
            if not improved:
                print("tour %d sans gain : arret" % rnd, flush=True)
                break
        print("\n=== Validation best vs profil de depart (%d donnes neuves) ===" % args.validate_deals, flush=True)
        m, se = duel(pool, args.workers, args.seed * 1000 + 7_777_777, args.validate_deals, start, best)
        print("VALIDATION %+.3f +- %.3f pts/donne (z %.1f) ; somme des gains adoptes %+.3f" % (
            m, se, m / se if se else 0, total_gain), flush=True)
        print("FINAL %s" % json.dumps(best, sort_keys=True))
    print("TUNE_DONE %.0f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
