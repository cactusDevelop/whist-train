"""Tests de agents/expert_bot.py : legalite (jamais une enchere ni une carte
illegale, sur toutes les familles de contrat y compris misere et ecart de
petite misere), absence de triche (la decision ne depend pas des mains
cachees), et regles d'evaluation de main de l'utilisateur.

Lancer : venv/Scripts/python.exe scripts/test_expert_bot.py
"""

import random
import sys

sys.path.insert(0, ".")

from agents import expert_bot
from agents.expert_bot import estimate_tricks, side_suit_tricks, trump_tricks
from engine.bidding import Bidding, legal_action_ids_with_hand
from engine.cards import deal
from engine.trou import detect_trou

CHECKS = [0]


def check(label, cond):
    CHECKS[0] += 1
    if not cond:
        raise AssertionError("ECHEC : %s" % label)


def card(suit, rank):
    return suit * 13 + rank


def test_hand_evaluation():
    A, K, Q, J = 12, 11, 10, 9
    check("A-R-D annexe = 3 plis (regle utilisateur)", side_suit_tricks([A, K, Q]) == 3)
    check("as sec = 1", side_suit_tricks([A]) == 1)
    check("R-D-V sans as = 2", side_suit_tricks([K, Q, J]) == 2)
    check("R sec = 0", side_suit_tricks([K]) == 0)
    check("atout A-R-D + 3 petits = 6", trump_tricks([A, K, Q, 4, 3, 2]) == 6)
    check("atout A + 4 petits = 3", trump_tricks([A, 5, 4, 3, 2]) == 3)
    check("atout R sec = 0 (non garde)", trump_tricks([K]) == 0)
    hand = [card(0, r) for r in (A, K, Q, 4, 3)] + [card(1, A), card(2, A), card(2, K)] + \
        [card(3, r) for r in (2, 3, 4, 5, 6)]
    check("main de proposition >= 5", estimate_tricks(hand, 0) >= 5)


def random_bidding(rng, hands, dealer):
    bidding = Bidding(dealer)
    trou = detect_trou(hands)
    if trou is not None:
        bidding.seed_trou(trou)
    while not bidding.finished:
        seat = bidding.turn
        bidding.apply(seat, rng.choice(legal_action_ids_with_hand(bidding, seat, hands[seat])))
    return bidding


def test_bot_bidding_always_legal():
    from scripts.expert_bot_match import run_bot_bidding  # leve si illegal
    rng = random.Random(7)
    for _ in range(3000):
        hands = deal(rng)
        run_bot_bidding(hands, rng.randrange(4))
        CHECKS[0] += 1


def test_bot_play_legal_all_families_and_no_cheating():
    from scripts.expert_bot_match import run_bot_bidding
    from scripts.train import build_game, dispatch_play_phase2
    rng = random.Random(11)
    families = set()
    forced = ["petite_misere", "piccolo", "grande_misere", "misere_etalee", "solitaire", "emballage"]
    n = 0
    while n < 1500:
        hands = deal(rng)
        dealer = rng.randrange(4)
        if n % 3 == 2:  # contrat impose, sans enchere (Phase 1) : toutes les miseres
            game, _seats = build_game(forced[(n // 3) % len(forced)], hands, rng, dealer)
            families.add(getattr(game, "kind", None) or game.contract_type)
        else:  # enchere du bot (emballage, solo, trou) ou aleatoire (tout)
            bidding = run_bot_bidding(hands, dealer) if n % 3 == 0 else random_bidding(rng, hands, dealer)
            if bidding.general_pass:
                continue
            contract = bidding.locked_contract
            families.add(contract.get("kind") or contract["type"])
            game = dispatch_play_phase2(contract, hands, dealer, bidding.bid_summary, False)
        while not game.done:
            seat = game.turn
            c = expert_bot.choose_card(game, seat)
            check("carte legale", game.legal_action_mask(seat)[c])
            if rng.random() < 0.1:
                # triche ? redistribuer les cartes cachees des AUTRES sieges
                saved = [list(h) for h in game.hands]
                # (en misere etalee, la main des declarants est PUBLIQUE : pas melangee)
                public = set(game.declarers) if getattr(game, "reveal_declarer_hands", False) else set()
                others = [s for s in range(4) if s != seat and s not in public]
                pool = [x for s in others for x in game.hands[s]]
                rng.shuffle(pool)
                for s in others:
                    k = len(game.hands[s])
                    game.hands[s], pool = pool[:k], pool[k:]
                check("decision independante des mains cachees", expert_bot.choose_card(game, seat) == c)
                game.hands = saved
            game.step(seat, c)
        n += 1
    for fam in ("solo", "emballage", "trou", "petite", "piccolo", "grande", "etalee"):
        check("famille couverte : %s" % fam, fam in families)


def test_hybrid_bidder_legal_and_guarded():
    """Hybride (regles + reseau quelconque) : toujours legal, jamais un solo
    de niveau L avec moins de L-1 plis surs, decisions claires imposees."""
    import torch
    from agents import evolution
    from agents.hybrid_bidder import HybridBidder, bot_choice_as_id, rule_layer
    from engine.bidding import bid_action_to_dict
    torch.manual_seed(3)
    rng = random.Random(5)
    hybrids = [HybridBidder(evolution.new_net()) for _ in range(4)]
    for _ in range(1500):
        hands = deal(rng)
        b = Bidding(rng.randrange(4))
        trou = detect_trou(hands)
        if trou is not None:
            b.seed_trou(trou)
        while not b.finished:
            seat = b.turn
            a = hybrids[seat].decide(b, seat, hands[seat])
            check("hybride legal", a in legal_action_ids_with_hand(b, seat, hands[seat]))
            info = bid_action_to_dict(a)
            if info["action"] in ("solo", "abondance", "solo_chelem"):
                check("garde-fou solo", estimate_tricks(hands[seat], info["suit"]) >= info.get("level", 13) - 1)
            force, permitted = rule_layer(b, seat, hands[seat])
            check("jamais vide", len(permitted) > 0)
            if force is not None:
                check("decision imposee respectee", a == force)
            check("hybride applique", b.apply(seat, a))
    # le coup du bot est toujours representable en id reseau et legal
    rng = random.Random(9)
    for _ in range(1500):
        hands = deal(rng)
        b = Bidding(rng.randrange(4))
        while not b.finished:
            seat = b.turn
            a = bot_choice_as_id(b, seat, expert_bot.choose_bid(b, seat, hands[seat]))
            check("coup du bot legal en id", a in legal_action_ids_with_hand(b, seat, hands[seat]))
            b.apply(seat, a)


def test_team_trump_rules_20261023():
    """Signalement 2026-10-23 : couper en renom quand le partenaire n'est pas
    sur de tenir ; tirer l'atout tant qu'UN adversaire peut couper."""
    from engine.team_trump import TeamTrumpDeal
    spades = 3
    # 1. partenaire (2) entame un petit coeur, siege 3 fournit plus petit, siege 1
    #    (plus tard) peut avoir mieux ; le siege 0 n'a plus de coeur -> il coupe
    hands = [list(range(39, 46)) + list(range(26, 32)), list(range(4, 13)) + list(range(13, 17)),
             [1, 2, 3] + list(range(17, 26)) + [32], [0] + list(range(33, 39)) + list(range(46, 52))]
    g = TeamTrumpDeal(hands, 0, 2, spades, 8, 2, "d", "p", "x")
    g.step(2, 3)
    g.step(3, 0)
    check("renom : coupe si le partenaire n'est pas sur", expert_bot.choose_card(g, 0) // 13 == spades)
    # 2. le siege 1 a montre qu'il n'a plus d'atout, le siege 3 en a encore -> on tire
    hands = [[49, 50, 51] + list(range(0, 10)), [10, 11, 12] + list(range(13, 23)),
             list(range(42, 49)) + [23, 24, 25, 26, 27, 28], [39, 40, 41] + list(range(29, 39))]
    g = TeamTrumpDeal(hands, 0, 2, spades, 8, 0, "d", "p", "x")
    for seat, c in ((0, 51), (1, 13), (2, 42), (3, 39)):
        g.step(seat, c)
    check("atout : tirer tant qu'un adversaire peut couper", expert_bot.choose_card(g, 0) // 13 == spades)


def main():
    for t in (test_hand_evaluation, test_bot_bidding_always_legal, test_hybrid_bidder_legal_and_guarded,
              test_bot_play_legal_all_families_and_no_cheating, test_team_trump_rules_20261023):
        t()
        print("ok  %s" % t.__name__, flush=True)
    print("%d verifications, toutes OK" % CHECKS[0])


if __name__ == "__main__":
    main()
