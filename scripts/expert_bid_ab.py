"""A/B duplicate des reglages de prudence des encheres de l'ExpertBot
(agents/expert_bot.set_bid_params ; signalement 2026-10-26 « contrats
impossibles, surencheres abusives »). Mesure seule, cloud.

Chaque donne est jouee 1 + 2 x V fois, jeu de la carte ExpertBot partout :
  - reference : 4 sieges aux reglages par defaut ;
  - pour chaque variante : sieges 0/2 variante (1/3 defaut), puis 1/3.
Gain de la variante = points reels des sieges variante - points des memes
sieges dans la reference, somme des deux placements (pts/donne). Par
variante aussi : reussite et frequence des contrats DECLARES par un siege
variante, par type -- un bot trop timide se voit (passes, contrats perdus
par la defense).

Usage : python -u scripts/expert_bid_ab.py --deals 20000 --workers 4
"""

import argparse
import math
import multiprocessing as mp
import os
import random
import sys
import time

sys.path.insert(0, ".")

from agents import expert_bot
from engine.bidding import Bidding
from engine.trou import detect_trou
from engine.whisted import DeckChain
from scripts.eval_vs_random import declarer_successes
from scripts.expert_bot_match import bot_player, play_out
from scripts.train import (MEASURE_WHISTED_RATIO, declarer_role_and_seats, describe_contract,
                           dispatch_play_phase2, real_points_by_seat)

VARIANTS = {  # 10/11 A/B 24 : piste PIMC vs nouvel ExpertBot -- defense sans atout, entame tardive : garder ses maitresses
    "DEF_NT_KEEP_all": {"DEF_NT_KEEP_MASTERS": "all"},
    "DEF_NT_KEEP_late": {"DEF_NT_KEEP_MASTERS": "late"},
    "DEF_NT_KEEP_late_team": {"DEF_NT_KEEP_MASTERS": "late_team"},
}

def play_deal(hands, dealer, whisted, params_by_seat):
    """-> (points des 4 sieges, contrat ou None, jeu ou None)."""
    bidding = Bidding(dealer)
    trou = detect_trou(hands)
    if trou is not None:
        bidding.seed_trou(trou)
    while not bidding.finished:
        seat = bidding.turn
        expert_bot.set_bid_params(params_by_seat[seat])
        expert_bot.apply_bid(bidding, seat, hands[seat])
    expert_bot.set_bid_params({})
    if bidding.general_pass:
        return [0.0] * 4, None, None
    contract = bidding.locked_contract
    role, _decl = declarer_role_and_seats(contract)
    game = dispatch_play_phase2(contract, [list(h) for h in hands], dealer, bidding.bid_summary, whisted)
    while not game.done:
        seat = game.turn
        expert_bot.set_bid_params(params_by_seat[seat])
        game.step(seat, expert_bot.choose_card(game, seat))
    expert_bot.set_bid_params({})
    return real_points_by_seat(role, game, contract), contract, game


def worker(args):
    seed, n, others = args
    from agents.expert_bot import PROFILES
    other = dict(PROFILES[others])
    if os.environ.get("FOLLOW_TRACE"):
        expert_bot.FOLLOW_TRACE = {}
    rng = random.Random(seed)
    chain = DeckChain(MEASURE_WHISTED_RATIO, rng)
    out = []  # (variante, gain, [(cle, ok, essais)] des contrats declares par la variante)
    base_rows = []
    for _ in range(n):
        hands, dealer, whisted = chain.deal(rng)
        hands = [list(h) for h in hands]
        base, bc, bgame = play_deal(hands, dealer, whisted, [other] * 4)
        if bc is not None:
            _r, decl = declarer_role_and_seats(bc)
            oks = declarer_successes(bc, bgame, decl)
            base_rows.append((describe_contract(bc), bc["type"], sum(oks), len(oks)))
        for name, params in VARIANTS.items():
            gain = 0.0
            rows = []
            cells = []  # 10/11 : (contrat de reference|camp des sieges variante, gain du placement)
            # 10/11 : placements = VRAIS camps du contrat de reference (avant : (0,2)/(1,3) -> declarants voisins
            # = 1 declarant + 1 defenseur par placement, gain nul par construction ; A1/A2/A3 "0 %" = artefact)
            if bc is not None:
                _r, bdecl0 = declarer_role_and_seats(bc)
                placements = (tuple(sorted(bdecl0)), tuple(s for s in range(4) if s not in bdecl0))
            else:
                placements = ((0, 2), (1, 3))
            for seats in placements:
                pbs = [{**other, **params} if s in seats else other for s in range(4)]
                pts, c, game = play_deal(hands, dealer, whisted, pbs)
                g = sum(pts[s] - base[s] for s in seats)
                gain += g
                if bc is not None:
                    _r, bdecl = declarer_role_and_seats(bc)
                    cells.append(("%s|%s" % (bc["type"], "decl" if bdecl & set(seats) else "def"), g))
                if c is not None:
                    _r, decl = declarer_role_and_seats(c)
                    if decl & set(seats):
                        oks = declarer_successes(c, game, decl)
                        rows.append((describe_contract(c), c["type"], sum(oks), len(oks)))
            out.append((name, gain, rows, cells))
        if bgame is None:
            chain.finish_unplayed(hands, dealer)
        else:
            chain.finish_played(bgame, dealer, bc["type"])
    if expert_bot.FOLLOW_TRACE is not None:  # diagnostic 10/11 : branches "coequipier gagnant, on fournit"
        for k, c in sorted(expert_bot.FOLLOW_TRACE.items(), key=lambda kv: -kv[1])[:60]:
            print("TRACE %d %s %d" % (seed, " ".join(map(str, k)), c), flush=True)
    return out, base_rows


def stat(xs):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1))
    return m, sd / math.sqrt(len(xs))


def table(rows, n_deals, title):
    agg = {}
    for key, typ, ok, tries in rows:
        for k in (key, "_" + typ, "_TOUT"):
            a = agg.setdefault(k, [0, 0, 0])
            a[0] += 1
            a[1] += ok
            a[2] += tries
    print("  %s : contrats declares / donne, reussite" % title)
    for k in sorted(agg, key=lambda k: (not k.startswith("_"), -agg[k][0])):
        n, ok, tries = agg[k]
        print("    %-16s %6d %6.3f %6.1f%%" % (k.lstrip("_"), n, n / n_deals, 100 * ok / max(1, tries)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--others", default="expert", help="profil (PROFILES) de la reference et des sieges adverses")
    parser.add_argument("--trace", action="store_true", help="diagnostic : compte les branches coequipier gagnant (TRACE)")
    args = parser.parse_args()
    if args.trace:
        os.environ["FOLLOW_TRACE"] = "1"
    t0 = time.time()
    per = args.deals // args.workers
    with mp.Pool(args.workers) as pool:
        parts = pool.map(worker, [(args.seed * 1000 + w, per, args.others) for w in range(args.workers)])
    n = per * args.workers
    print("=== A/B encheres ExpertBot : %d donnes, %d variantes, %.0fs ===" % (n, len(VARIANTS), time.time() - t0))
    table([r for _o, b in parts for r in b], n, "reference (4 sieges)")
    for name, params in VARIANTS.items():
        gains = [g for o, _b in parts for v, g, _r, _c in o if v == name]
        rows = [r for o, _b in parts for v, _g, rs, _c in o if v == name for r in rs]
        cells = {}
        for o, _b in parts:
            for v, _g, _r, cs in o:
                if v == name:
                    for k, g in cs:
                        cells.setdefault(k, []).append(g)
        changed = sum(1 for g in gains if g != 0)
        m, se = stat(gains)
        print("\n%s %s : gain %+.3f +- %.3f pts/donne (z %.1f), donnes changees %.1f%%"
              % (name, params, m, se, m / se if se else 0.0, 100 * changed / len(gains)))
        table(rows, n, "variante (sieges variante, 2 placements)")
        print("  gain par contrat de reference | camp des sieges variante (pts par placement, z ; contribution/donne)")
        for k in sorted(cells, key=lambda k: -len(cells[k])):
            xs = cells[k]
            if len(xs) >= 2:
                cm, cse = stat(xs)
                print("    %-22s n %6d  %+.3f +- %.3f  z %+5.1f  %+.4f/donne" % (
                    k, len(xs), cm, cse, cm / cse if cse else 0.0, sum(xs) / n))
    names = list(VARIANTS)
    by = {name: [g for o, _b in parts for v, g, _r, _c in o if v == name] for name in names}
    print("\n=== ecarts apparies (memes donnes) ===")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m, se = stat([x - y for x, y in zip(by[a], by[b])])
            print("%s - %s : %+.3f +- %.3f (z %.1f)" % (a, b, m, se, m / se if se else 0.0))
    print("AB_DONE")


if __name__ == "__main__":
    main()
