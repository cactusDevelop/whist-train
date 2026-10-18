"""Cas de parite des REGLES d'enchere de l'IA exportee (agents/abondance_rule.py :
regle d'abondance + garde-fou des solos 2026-10-04 + garde-fou de l'emballeur
2026-10-05 + garde-fou de la grande misere 2026-10-08 + garde-fou des emballages
relances trop haut 2026-10-09) pour whist-4-it
(tests/bid_rules_parity_runner.gd -> WhistAIBridgeV2.abondance_rule_action /
solo_cap_legal / bid_guards_legal).

Etats d'enchere reels (journal d'actions que le runner Godot REJOUE dans un vrai
WhistBidding) : mains aleatoires et mains fortes (redonne jusqu'a >= 8 plis surs
quelque part), actions aleatoires orientees solo/abondance/relance pour atteindre
les cas ou le garde-fou intervient. Par cas : ids legaux, abondance imposee
attendue (-1 = aucune), ids legaux apres garde-fou des solos, puis apres tous
les garde-fous (solos + emballeur).

Usage : venv/Scripts/python.exe scripts/dump_bid_rules_fixtures.py [--out PATH]
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, ".")

from agents.abondance_rule import (abondance_rule, apply_bid_guards, apply_solo_cap, emballage_raise_too_high,
                                   emballer_too_short,
                                   grande_misere_too_loose, solo_too_high)
from agents.expert_bot import estimate_tricks
from engine.bidding import ACTION_RAISE, Bidding, bid_action_to_dict, legal_action_ids_with_hand, legal_action_mask_with_hand
from engine.cards import deal
from engine.trou import detect_trou

CASES = 600
MIN_CAPPED, MIN_RAISE_CAPPED, MIN_FORCED, MIN_EMB_GUARDED = 150, 30, 40, 60
# garde-fou grande misere (2026-10-08) : annonce directe et relance d'un piccolo
MIN_GM_GUARDED, MIN_GM_RAISE_GUARDED = 60, 20
# garde-fou des emballages au-dessus de 10 : relance (2026-10-09) ou entree directe (2026-10-10)
MIN_EMB_RAISE_GUARDED = 40


def strong_hands(rng):
    for _ in range(2000):
        hands = deal(rng)
        if max(estimate_tricks(h, t) for h in hands for t in range(4)) >= 8:
            return hands
    return hands


def biased(rng, legal):
    for kind in ("emballer", "solo", "abondance", "misere", "raise"):
        pref = [a for a in legal if bid_action_to_dict(a)["action"] == kind]
        if pref and rng.random() < 0.6:
            return rng.choice(pref)
    return rng.choice(legal)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join("godot_export", "bid_rules_cases.json"))
    args = parser.parse_args()
    rng = random.Random(20261004)
    cases = []
    n = {"capped": 0, "raise_capped": 0, "forced": 0, "emb_guarded": 0, "gm_guarded": 0, "gm_raise_guarded": 0,
         "emb_raise_guarded": 0}
    mins = {"capped": MIN_CAPPED, "raise_capped": MIN_RAISE_CAPPED, "forced": MIN_FORCED, "emb_guarded": MIN_EMB_GUARDED,
            "gm_guarded": MIN_GM_GUARDED, "gm_raise_guarded": MIN_GM_RAISE_GUARDED, "emb_raise_guarded": MIN_EMB_RAISE_GUARDED}
    deals = 0
    while (len(cases) < CASES or any(n[k] < mins[k] for k in mins)) and deals < 200000:
        deals += 1
        hands = strong_hands(rng) if rng.random() < 0.7 else deal(rng)
        dealer = rng.randrange(4)
        b = Bidding(dealer)
        trou = detect_trou(hands)
        if trou is not None:
            b.seed_trou(trou)
        log = []
        while not b.finished:
            seat = b.turn
            hand = hands[seat]
            legal = legal_action_ids_with_hand(b, seat, hand)
            mask = legal_action_mask_with_hand(b, seat, hand)
            forced = abondance_rule(b, seat, hand)
            capped = [a for a, ok in enumerate(apply_solo_cap(b, seat, hand, mask)) if ok]
            is_capped = capped != legal
            guarded = [a for a, ok in enumerate(apply_bid_guards(b, seat, hand, mask)) if ok]
            emb_guarded = any(emballer_too_short(hand, a) for a in legal)
            raise_capped = ACTION_RAISE in legal and solo_too_high(b, seat, hand, ACTION_RAISE)
            gm_raise_guarded = ACTION_RAISE in legal and grande_misere_too_loose(b, seat, hand, ACTION_RAISE)
            gm_guarded = gm_raise_guarded or any(grande_misere_too_loose(b, seat, hand, a) for a in legal)
            emb_raise_guarded = any(emballage_raise_too_high(b, seat, a) for a in legal)
            keep = (is_capped and n["capped"] < MIN_CAPPED) or (raise_capped and n["raise_capped"] < MIN_RAISE_CAPPED) \
                or (forced is not None and n["forced"] < MIN_FORCED) \
                or (emb_guarded and n["emb_guarded"] < MIN_EMB_GUARDED) \
                or (gm_guarded and n["gm_guarded"] < MIN_GM_GUARDED) \
                or (gm_raise_guarded and n["gm_raise_guarded"] < MIN_GM_RAISE_GUARDED) \
                or (emb_raise_guarded and n["emb_raise_guarded"] < MIN_EMB_RAISE_GUARDED) \
                or (len(cases) < CASES and rng.random() < 0.05)
            if keep:
                cases.append({"dealer": dealer, "deal_hands": [sorted(h) for h in hands], "trou": trou,
                              "bid_log": [list(x) for x in log], "seat": seat, "hand": sorted(hand),
                              "expected_legal": legal, "expected_forced": -1 if forced is None else forced,
                              "expected_capped": capped, "expected_guarded": guarded})
                n["capped"] += is_capped
                n["raise_capped"] += raise_capped
                n["forced"] += forced is not None
                n["emb_guarded"] += emb_guarded
                n["gm_guarded"] += gm_guarded
                n["gm_raise_guarded"] += gm_raise_guarded
                n["emb_raise_guarded"] += emb_raise_guarded
            action = biased(rng, legal)
            assert b.apply(seat, action), (seat, action)
            log.append((seat, action))
    print("%d cas (%d donnes) : %s" % (len(cases), deals, n))
    if any(n[k] < mins[k] for k in mins):
        raise SystemExit("pas assez de cas cibles : %s" % n)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"cases": cases}, f)
    print("OK : %s" % args.out)


if __name__ == "__main__":
    main()
