"""Fixtures de parite de l'ExpertBot sensible aux regles (lot 1, 31/10) pour le
portage Godot de whist-4-it : positions tirees de matchs simules ou
joker_match.rules_seen n'est pas None, avec la carte attendue.
Cas : seat, hand, play_log, contract, bid_summary, seen, card.
Cloud seulement (ci/run_jokers_test.sh).

Usage : python scripts/dump_expert_seen_fixtures.py --out expert_seen_cases.json
"""

import argparse
import json
import random
import sys

sys.path.insert(0, ".")

from agents import expert_bot
from engine import joker_match as jm

PER_KIND = 150


def kind_of(seen, game):
    if seen["inverted"]:
        return "inversion"
    return "atout" if seen["trump"] != game.trump_suit else "legalite"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="expert_seen_cases.json")
    args = parser.parse_args()
    cases, counts = [], {}
    original = jm.rules_seen

    def spy(hand, seat):
        seen = original(hand, seat)
        if seen is not None:
            kind = kind_of(seen, hand.game)
            if counts.get(kind, 0) < PER_KIND:
                g = hand.game
                counts[kind] = counts.get(kind, 0) + 1
                cases.append({"kind": kind, "seat": seat, "hand": sorted(g.hands[seat]), "play_log": [list(p) for p in g.play_log],
                              "contract": g.public_contract(), "bid_summary": g.bid_summary, "seen": seen,
                              "card": expert_bot.choose_card(g, seat, seen)})
        return seen

    jm.rules_seen = spy
    policy = lambda hand, seat: 0
    seed = 0
    while seed < 4000 and (len(counts) < 3 or min(counts.values()) < PER_KIND):
        rng = random.Random(90000 + seed)
        seed += 1
        cfg = jm.random_config(rng)
        cfg["style"] = "expert"
        match = jm.Match(cfg, rng.randrange(1 << 30))
        for _ in range(jm.HANDS_PER_MATCH):
            hand = match.next_hand()
            jm.play_out(hand, policy)
            match.finish(hand)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"cases": cases}, f, ensure_ascii=False)
    print("fixtures ExpertBot sensibles aux regles :", counts, "matchs", seed)


if __name__ == "__main__":
    main()
