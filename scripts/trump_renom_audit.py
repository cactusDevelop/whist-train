"""Adversaires en renom d'atout (utilisateur 30/10 : « il faut les garder pour
couper plutot que de burn ceux de son coequipier »). Mesure seule, cloud.

4 ExpertBot, contrats a l'atout hors misere. A chaque entame d'un siege qui
garde une carte hors atout, quand les adversaires n'ont VRAIMENT plus d'atout
(mains lues par l'audit, pas par le bot) :
  - tire : entame atout ;
  - force : entame la couleur ou un coequipier est en renom connu alors qu'il
    garde une maitresse hors atout dans une autre couleur.
Tableau par camp (declarant equipe, declarant solo, defense) : positions et
fautes pour 1000 donnes.

Usage : python -u scripts/trump_renom_audit.py --deals 20000 --workers 4
"""

import argparse
import multiprocessing as mp
import random
import sys

sys.path.insert(0, ".")

from agents import expert_bot
from agents.expert_bot import PublicView
from engine.bidding import Bidding
from engine.cards import suit_of
from engine.trou import detect_trou
from engine.whisted import DeckChain
from scripts.train import MEASURE_WHISTED_RATIO, dispatch_play_phase2


def bid(hands, dealer):
    bidding = Bidding(dealer)
    trou = detect_trou(hands)
    if trou is not None:
        bidding.seed_trou(trou)
    while not bidding.finished:
        expert_bot.apply_bid(bidding, bidding.turn, hands[bidding.turn])
    return bidding


def audit_lead(game, seat, card, rows):
    v = PublicView(game, seat)
    side = [c for c in v.hand if suit_of(c) != v.trump]
    if not side or any(suit_of(c) == v.trump for o in v.opponents() for c in game.hands[o]):
        return
    camp = "defense" if seat not in v.declarer_side else ("equipe" if v.team_contract else "solo")
    led = suit_of(card)
    tire = led == v.trump
    masters = [c for c in side if v.is_master(c) and suit_of(c) != led]
    force = not tire and bool(masters) and any(v.may_ruff(m, led) for m in v.teammates(seat))
    rows.append((camp, tire, force))


def worker(args):
    seed, n = args
    rng = random.Random(seed)
    chain = DeckChain(MEASURE_WHISTED_RATIO, rng)
    rows = []
    for _ in range(n):
        hands, dealer, whisted = chain.deal(rng)
        hands = [list(h) for h in hands]
        bidding = bid(hands, dealer)
        if bidding.general_pass:
            chain.finish_unplayed(hands, dealer)
            continue
        contract = bidding.locked_contract
        game = dispatch_play_phase2(contract, [list(h) for h in hands], dealer, bidding.bid_summary, whisted)
        while not game.done:
            seat = game.turn
            card = expert_bot.choose_card(game, seat)
            if (game.trump_suit is not None and contract["type"] != "misere" and not game.current_trick
                    and not getattr(game, "pending_discards", None)
                    and not getattr(game, "trou_pending", lambda: False)()):
                audit_lead(game, seat, card, rows)
            game.step(seat, card)
        chain.finish_played(game, dealer, contract["type"])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    per = args.deals // args.workers
    with mp.Pool(args.workers) as pool:
        parts = pool.map(worker, [(args.seed * 1000 + w, per) for w in range(args.workers)])
    rows = [r for p in parts for r in p]
    k = 1000.0 / (per * args.workers)
    print("=== entames, adversaires sans atout : camp, positions, tire, force (pour 1000 donnes) ===")
    for camp in ("equipe", "solo", "defense"):
        rs = [r for r in rows if r[0] == camp]
        print("%-8s %8.1f %8.1f %8.1f" % (camp, k * len(rs), k * sum(r[1] for r in rs), k * sum(r[2] for r in rs)))
    print("RENOM_AUDIT_DONE")


if __name__ == "__main__":
    main()
