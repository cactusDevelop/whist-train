"""Equilibrage du Duel de whist-4-it (demande whist-4-it du 2026-10-19,
DEMANDE-WHIST-4-IT-DUEL.md) : avance de la Belle = round(P * tanh(ecart / E)).

On simule des Belles de 4 donnes jouees (une passe generale redonne, meme
donneur, et ne compte pas -- whist_table.gd), 4 ExpertBot, rivaux aux
sieges 0 et 2, paquet whiste d'une donne a l'autre (belle_match.gd : "the
deck whiste stays"). Ecart de Belle = points du siege 0 - points du siege 2.
Sans jokers ni Trahison (l'ExpertBot ne trahit pas ; deux joueurs a jokers
ne sont pas simules).

comeback_rate[A] = part des Belles gagnees par le rival parti a -A
(ecart de Belle > A en sa faveur), les deux orientations comptees ; les
egalites (= A) sont a part dans tie_rate.

Duplicate entre deux ExpertBot identiques : ecart toujours 0 (bot
deterministe, memes cartes) -- pas simule.

Usage : venv/Scripts/python.exe scripts/duel_balance.py [--belles 20000] [--seed 1019]
-> godot_export/duel_balance.json
"""

import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, ".")

from agents import expert_bot
from engine.bidding import Bidding
from engine.trou import detect_trou
from engine.whisted import DeckChain
from scripts.train import declarer_role_and_seats, dispatch_play_phase2, real_points_by_seat

BELLE_HANDS = 4  # DuelRules.BELLE_HANDS
RIVALS = (0, 2)
P, E = 20, 15  # CHANTIER-DUEL-PLAN.md
ADVANCES = (6, 12, 15, 20)
QUANTILES = (10, 25, 50, 75, 90)
OUT = os.path.join("godot_export", "duel_balance.json")


def play_hand(chain, rng):
    """Une donne jouee (les passes generales redonnent) -> points des 4 sieges."""
    while True:
        hands, dealer, whisted = chain.deal(rng)
        bidding = Bidding(dealer)
        trou = detect_trou(hands)
        if trou is not None:
            bidding.seed_trou(trou)
        while not bidding.finished:
            expert_bot.apply_bid(bidding, bidding.turn, hands[bidding.turn])
        if bidding.general_pass:
            chain.finish_unplayed(hands, dealer)
            continue
        contract = bidding.locked_contract
        role, _declarers = declarer_role_and_seats(contract)
        game = dispatch_play_phase2(contract, [list(h) for h in hands], dealer, bidding.bid_summary, whisted)
        while not game.done:
            seat = game.turn
            game.step(seat, expert_bot.choose_card(game, seat))
        chain.finish_played(game, dealer, contract["type"])
        return real_points_by_seat(role, game, contract)


def quantile(sorted_xs, q):
    i = (len(sorted_xs) - 1) * q / 100
    lo = math.floor(i)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (i - lo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--belles", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1019)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    gaps = []
    for b in range(args.belles):
        chain = DeckChain(1.0, rng)
        chain.pack = list(range(52))
        rng.shuffle(chain.pack)  # le paquet arrive melange, puis reste whiste
        total = [0.0] * 4
        for _ in range(BELLE_HANDS):
            pts = play_hand(chain, rng)
            total = [t + p for t, p in zip(total, pts)]
        gaps.append(total[RIVALS[0]] - total[RIVALS[1]])
        if (b + 1) % 2000 == 0:
            print("%d belles" % (b + 1), flush=True)
    n = len(gaps)
    abs_sorted = sorted(abs(g) for g in gaps)
    # Les deux orientations : le rival en retard est le siege 0 puis le siege 2.
    both = gaps + [-g for g in gaps]
    comeback, ties = {}, {}
    for a in ADVANCES:
        comeback[str(a)] = round(sum(g > a for g in both) / len(both), 4)
        ties[str(a)] = round(sum(g == a for g in both) / len(both), 4)
    mean = sum(gaps) / n
    sd = math.sqrt(sum((g - mean) ** 2 for g in gaps) / (n - 1))
    out = {
        "belle_gap_quantiles": {str(q): round(quantile(abs_sorted, q), 1) for q in QUANTILES},
        "comeback_rate": comeback,
        "tie_rate": ties,
        "n_belles": n,
        "gap_mean_seat0_minus_seat2": round(mean, 2),
        "gap_sd": round(sd, 2),
        "advance_formula": {"P": P, "E": E},
        "notes": "4 ExpertBot, rivaux sieges 0 et 2, 4 donnes jouees (passe generale redonne), paquet whiste ; "
                 "sans jokers ni Trahison ; comeback = victoire stricte du rival parti a -avance (egalites dans tie_rate) ; "
                 "duplicate ExpertBot contre ExpertBot = 0 exactement (bot deterministe). whist-ai scripts/duel_balance.py",
    }
    os.makedirs("godot_export", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
