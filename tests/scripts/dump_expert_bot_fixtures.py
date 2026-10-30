"""Fixtures de parite de game/whist_expert_bot.gd : donnes completes jouees
par 4 ExpertBot Python (whist-ai/agents/expert_bot.py @ 9a3041e), chaque
decision enregistree. Rejouees par tests/expert_bot_parity_runner.gd.

Usage (depuis whist-ai, son venv) :
  venv/Scripts/python.exe ../whist-4-it/tests/scripts/dump_expert_bot_fixtures.py
"""

import json
import os
import random
import sys

sys.path.insert(0, ".")

from agents import expert_bot
from engine.bidding import Bidding
from engine.cards import deal
from engine.trou import detect_trou
from scripts.train import describe_contract, dispatch_play_phase2

SEED = 20260927
CANDIDATES = 40000
PER_CONTRACT = 25
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "expert_bot_parity.json")


def play_deal(rng):
    hands = deal(rng)
    while detect_trou(hands) is not None:
        # Trou : regle du 2026-10-01 (seed sans choix d'atout, fixe par l'entame
        # du bouche-trou), que le runner ne rejoue pas encore (LegacyTrouSeed =
        # ancienne regle) -> donnes Trou exclues depuis le 2026-10-07 (regle
        # Passe Parole « une seule fois »). Couvertes par expert_full_table.
        hands = deal(rng)
    dealer = rng.randrange(4)
    record = {"dealer": dealer, "hands": [list(h) for h in hands], "trou": None, "bids": []}
    bidding = Bidding(dealer)
    trou = detect_trou(hands)
    if trou is not None:
        bidding.seed_trou(trou)
        record["trou"] = trou
    tags = set()
    while not bidding.finished:
        seat = bidding.turn
        choice = expert_bot.apply_bid(bidding, seat, hands[seat])
        if isinstance(choice, tuple):
            record["bids"].append([seat, "raise_to", choice[1]])
            tags.add("raise_to")
        else:
            record["bids"].append([seat, choice])
            if choice == 2:
                tags.add("pass_parole")
    if bidding.general_pass:
        record["contract"] = None
        return record, "passe", tags
    contract = bidding.locked_contract
    game = dispatch_play_phase2(contract, hands, dealer, bidding.bid_summary)
    record["plays"] = []
    while not game.done:
        seat = game.turn
        card = expert_bot.choose_card(game, seat)
        record["plays"].append([seat, card])
        game.step(seat, card)
    record["contract"] = describe_contract(contract)
    if trou is not None:
        tags.add("trou_detected")
    return record, record["contract"], tags


def main():
    rng = random.Random(SEED)
    kept, per_key, tag_counts = [], {}, {}
    for _ in range(CANDIDATES):
        record, key, tags = play_deal(rng)
        rare = [t for t in tags if tag_counts.get(t, 0) < PER_CONTRACT]
        if per_key.get(key, 0) < PER_CONTRACT or rare:
            kept.append(record)
            per_key[key] = per_key.get(key, 0) + 1
            for t in tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"source": "whist-ai agents/expert_bot.py @ 332ea89 (Passe Parole une seule fois, sans Trou)", "seed": SEED, "deals": kept}, f,
                  separators=(",", ":"))
    print("%d donnes gardees -> %s" % (len(kept), OUT))
    print("contrats:", dict(sorted(per_key.items())))
    print("tags:", tag_counts)


if __name__ == "__main__":
    main()
