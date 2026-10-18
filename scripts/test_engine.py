"""Suite de tests permanente et exhaustive du moteur (voir DESIGN.md,
"aucune marge d'erreur permise"). Contrairement a scripts/smoke_test.py
(plomberie, un run ponctuel), ce fichier verifie la CORRECTION reelle :
chaque cellule du bareme de points transcrite directement depuis
WHIST-RULES-REFERENCE.txt (projet Godot whist-4-it), plus des invariants
structurels (somme nulle, legalite, terminaison) sur l'ensemble de leur
domaine. Zero dependance externe (pas de pytest) -- assert simples,
meme convention que l'ancien scripts/test_cobaye.py (archive).

Lancer : venv/Scripts/python.exe scripts/test_engine.py
"""

import itertools
import random
import sys

sys.path.insert(0, ".")

from engine.bidding import (
    NUM_BID_ACTIONS, bid_action_to_dict,
    ABONDANCE_MIN_LEVEL, Bidding, EMBALLAGE_MIN_LEVEL, first_leader, legal_action_ids_with_hand,
)
from engine.cards import deal, trick_winner
from engine.scoring import (
    abondance_result, apply_solo_result, emballage_failure, emballage_success,
    emballage_or_trou_team_value, misere_result, solo_base_value, solo_failure, solo_success,
    trou_result,
)
from engine.trou import detect_trou

CHECKS_RUN = [0]


def check(label, condition):
    CHECKS_RUN[0] += 1
    if not condition:
        raise AssertionError("ECHEC : %s" % label)


# --- engine/scoring.py : chaque cellule du tableau de WHIST-RULES-REFERENCE.txt ---

def test_emballage_and_trou_table():
    # ligne = plis pris par le camp declarant (0..13)
    # colonnes : Emba8, Emba9, Emba10, Emba11, Emba12, Emba13, Trou8, Trou9 -- valeur "Ctrt"
    table = [
        [-21, -23, -25, -27, -29, -31, -16, -16],
        [-19, -21, -23, -25, -27, -29, -16, -16],
        [-17, -19, -21, -23, -25, -27, -16, -16],
        [-15, -17, -19, -21, -23, -25, -16, -16],
        [-13, -15, -17, -19, -21, -23, -16, -16],
        [-11, -13, -15, -17, -19, -21, -16, -16],
        [-9, -11, -13, -15, -17, -19, -16, -16],
        [-7, -9, -11, -13, -15, -17, -16, -16],
        [5, -7, -9, -11, -13, -15, 16, -16],
        [7, 5, -7, -9, -11, -13, 16, 16],
        [9, 7, 5, -7, -9, -11, 16, 16],
        [11, 9, 7, 5, -7, -9, 16, 16],
        [13, 11, 9, 7, 5, -7, 16, 16],
        [20, 18, 16, 14, 12, 5, 30, 30],  # Emba13 a -5 cote Adv (voir note ci-dessous), Ctrt = +5 correct
    ]
    emballage_levels = [8, 9, 10, 11, 12, 13]
    trou_levels = [8, 9]
    for tricks, row in enumerate(table):
        for col, level in enumerate(emballage_levels):
            expected = row[col]
            got = emballage_or_trou_team_value(level, tricks)
            check("Emballage %d, %d plis -> %+d (attendu %+d)" % (level, tricks, got, expected), got == expected)
        for col, level in enumerate(trou_levels):
            expected = row[len(emballage_levels) + col]
            got = trou_result(level, tricks)
            check("Trou %d, %d plis -> %+d (attendu %+d)" % (level, tricks, got, expected), got == expected)
    # La seule cellule asymetrique du site (Emba13/13 plis, Adv affiche -10)
    # est une coquille de saisie confirmee par l'utilisateur (voir
    # WHIST-RULES-REFERENCE.txt) : le vrai jeu (game/scoring.gd) implemente
    # -5, symetrique. On verifie donc la symetrie, PAS le -10 du site.
    check("Emballage 13 a 13 plis reste symetrique (+5/-5), pas le -10 du site (coquille connue)",
          emballage_success(13, 13) == 5)


def test_solo_abondance_chelem_table():
    # ligne = plis pris par le/la soliste (0..13)
    # colonnes : Solo6,7,8, Abon9,10,11,12, SChelem -- valeur "Ctrt" (soloist)
    table = [
        [-33, -36, -39, -30, -45, -60, -75, -90],
        [-30, -33, -36, -30, -45, -60, -75, -90],
        [-27, -30, -33, -30, -45, -60, -75, -90],
        [-24, -27, -30, -30, -45, -60, -75, -90],
        [-21, -24, -27, -30, -45, -60, -75, -90],
        [-18, -21, -24, -30, -45, -60, -75, -90],
        [15, -18, -21, -30, -45, -60, -75, -90],
        [21, 15, -18, -30, -45, -60, -75, -90],
        [27, 21, 15, -30, -45, -60, -75, -90],
        [27, 21, 15, 30, -45, -60, -75, -90],
        [27, 21, 15, 30, 45, -60, -75, -90],
        [27, 21, 15, 30, 45, 60, -75, -90],
        [27, 21, 15, 30, 45, 60, 75, -90],
        [27, 21, 15, 30, 45, 60, 75, 90],
    ]
    solo_levels = [6, 7, 8]
    abondance_levels = [9, 10, 11, 12, 13]  # 13 = Solo Chelem, meme formule
    for tricks, row in enumerate(table):
        for col, level in enumerate(solo_levels):
            expected_ctrt = row[col]
            got = apply_solo_result(solo_base_value(level, tricks))
            check("Solo %d, %d plis -> soloist %+d (attendu %+d)" % (level, tricks, got["soloist"], expected_ctrt),
                  got["soloist"] == expected_ctrt)
        for col, level in enumerate(abondance_levels):
            expected_ctrt = row[len(solo_levels) + col]
            got = abondance_result(level, tricks)
            check("Abondance/Chelem %d, %d plis -> soloist %+d (attendu %+d)" % (
                level, tricks, got["soloist"], expected_ctrt), got["soloist"] == expected_ctrt)


def test_solo_opponent_values_match_table_adv_column():
    # Colonne "Adv." du meme tableau (verifiee separement pour lisibilite)
    adv_table = [
        [11, 12, 13, 10, 15, 20, 25, 30],
        [10, 11, 12, 10, 15, 20, 25, 30],
        [9, 10, 11, 10, 15, 20, 25, 30],
        [8, 9, 10, 10, 15, 20, 25, 30],
        [7, 8, 9, 10, 15, 20, 25, 30],
        [6, 7, 8, 10, 15, 20, 25, 30],
        [-5, 6, 7, 10, 15, 20, 25, 30],
        [-7, -5, 6, 10, 15, 20, 25, 30],
        [-9, -7, -5, 10, 15, 20, 25, 30],
        [-9, -7, -5, -10, 15, 20, 25, 30],
        [-9, -7, -5, -10, -15, 20, 25, 30],
        [-9, -7, -5, -10, -15, -20, 25, 30],
        [-9, -7, -5, -10, -15, -20, -25, 30],
        [-9, -7, -5, -10, -15, -20, -25, -30],
    ]
    levels = [6, 7, 8, 9, 10, 11, 12, 13]
    for tricks, row in enumerate(adv_table):
        for col, level in enumerate(levels):
            expected = row[col]
            if level <= 8:
                got = apply_solo_result(solo_base_value(level, tricks))
            else:
                got = abondance_result(level, tricks)
            check("niveau %d, %d plis -> opponent %+d (attendu %+d)" % (level, tricks, got["opponent"], expected),
                  got["opponent"] == expected)


def test_misere_table():
    # (n, k) -> {kind: (CtrtG_ou_None, CtrtP_ou_None, Adv)}
    cases = {
        (1, 0): {"petite": (None, -18, 6), "piccolo": (None, -24, 8), "grande": (None, -36, 12), "etalee": (None, -48, 16)},
        (1, 1): {"petite": (18, None, -6), "piccolo": (24, None, -8), "grande": (36, None, -12), "etalee": (48, None, -16)},
        (2, 0): {"petite": (None, -12, 12), "piccolo": (None, -16, 16), "grande": (None, -24, 24), "etalee": (None, -32, 32)},
        (2, 1): {"petite": (24, -24, 0), "piccolo": (32, -32, 0), "grande": (48, -48, 0), "etalee": (64, -64, 0)},
        (2, 2): {"petite": (12, None, -12), "piccolo": (16, None, -16), "grande": (24, None, -24), "etalee": (32, None, -32)},
        (3, 0): {"petite": (None, -6, 18), "piccolo": (None, -8, 24), "grande": (None, -12, 36), "etalee": (None, -16, 48)},
        (3, 1): {"petite": (30, -18, 6), "piccolo": (40, -24, 8), "grande": (60, -36, 12), "etalee": (80, -48, 16)},
        (3, 2): {"petite": (18, -30, -6), "piccolo": (24, -40, -8), "grande": (36, -60, -12), "etalee": (48, -80, -16)},
        (3, 3): {"petite": (6, None, -18), "piccolo": (8, None, -24), "grande": (12, None, -36), "etalee": (16, None, -48)},
    }
    for (n, k), by_kind in cases.items():
        for kind, (ctrt_g, ctrt_p, adv) in by_kind.items():
            got = misere_result(kind, n, k)
            if ctrt_g is not None:
                check("misere %s n=%d k=%d -> winner %+d (attendu %+d)" % (kind, n, k, got["winner"], ctrt_g),
                      got["winner"] == ctrt_g)
            if ctrt_p is not None:
                check("misere %s n=%d k=%d -> loser %+d (attendu %+d)" % (kind, n, k, got["loser"], ctrt_p),
                      got["loser"] == ctrt_p)
            check("misere %s n=%d k=%d -> opponent %+d (attendu %+d)" % (kind, n, k, got["opponent"], adv),
                  got["opponent"] == adv)


# --- invariants structurels sur tout le domaine (pas seulement les cellules du site) ---

def test_emballage_trou_zero_sum_everywhere():
    for level in range(8, 14):
        for tricks in range(0, 14):
            value = emballage_or_trou_team_value(level, tricks)
            check("Emballage %d/%d plis somme nulle" % (level, tricks), 2 * value + 2 * (-value) == 0)
    for level in (8, 9):
        for tricks in range(0, 14):
            value = trou_result(level, tricks)
            check("Trou %d/%d plis somme nulle" % (level, tricks), 2 * value + 2 * (-value) == 0)


def test_solo_abondance_zero_sum_everywhere():
    for level in range(6, 14):
        for tricks in range(0, 14):
            if level <= 8:
                result = apply_solo_result(solo_base_value(level, tricks))
            else:
                result = abondance_result(level, tricks)
            total = result["soloist"] + 3 * result["opponent"]
            check("niveau %d/%d plis somme nulle (soloist + 3*opponent == 0)" % (level, tricks), total == 0)


def test_misere_zero_sum_everywhere():
    for kind in ("petite", "piccolo", "grande", "etalee"):
        for n in range(1, 5):
            for k in range(0, n + 1):
                result = misere_result(kind, n, k)
                winner = result["winner"] or 0
                loser = result["loser"] or 0
                total = k * winner + (n - k) * loser + (4 - n) * result["opponent"]
                check("misere %s n=%d k=%d somme nulle" % (kind, n, k), total == 0)


def test_solo_success_ceiling_and_asymmetric_failure_step():
    # Solo ne recompense jamais au-dela de 8 plis (Abondance existe pour ca)
    check("Solo 6 a 8 plis == Solo 6 a 13 plis (plafonne)",
          solo_success(6, 8) == solo_success(6, 13))
    # Echec asymetrique : step 1 (pas 2) cote echec pour Solo
    check("Solo 8, echec de 1 pli (7 pris) -> -6", solo_failure(8, 7) == -6)
    check("Solo 8, echec de 2 plis (6 pris) -> -7", solo_failure(8, 6) == -7)
    # Emballage : step 2 symetrique succes/echec
    check("Emballage 8, echec de 1 pli (7 pris) -> -7", emballage_failure(8, 7) == -7)


# --- engine/bidding.py : invariants structurels ---

def test_first_leader_rule():
    check("Emballage entame a gauche du donneur",
          first_leader(3, {"type": "emballage", "level": 8}) == 0)
    check("Solo simple (6-8) entame a gauche du donneur",
          first_leader(3, {"type": "solo", "level": 8, "declarer": 2}) == 0)
    check("Abondance (>=9) entame par le declarant",
          first_leader(3, {"type": "solo", "level": 9, "declarer": 2}) == 2)
    check("Solo Chelem (13) entame par le declarant",
          first_leader(1, {"type": "solo", "level": 13, "declarer": 3}) == 3)
    check("Misere entame a gauche du donneur",
          first_leader(3, {"type": "misere"}) == 0)
    check("Trou entame par le bouche-trou (regle du 2026-10-01)",
          first_leader(3, {"type": "trou", "declarer": 1, "partner": 2}) == 2)


def _trou_hands():
    """Premiere donne (graine fixe) avec un Trou simple ou le bouche-trou a
    une autre carte de la couleur d'ancrage ET une carte d'une autre couleur."""
    rng = random.Random(7)
    while True:
        hands = deal(rng)
        info = detect_trou(hands)
        if info is None or info["is_royal"]:
            continue
        ph = hands[info["partner"]]
        if any(x // 13 == info["anchor_suit"] and x != info["anchor_card"] for x in ph) and                 any(x // 13 != info["anchor_suit"] for x in ph):
            return hands, info


def test_trou_royal_anchor_is_partners_best_heart():
    rng = random.Random(11)
    found = 0
    for _ in range(200000):
        hands = deal(rng)
        info = detect_trou(hands)
        if info is None or not info["is_royal"]:
            continue
        found += 1
        best = max((x for s in range(4) if s != info["declarer"] for x in hands[s] if x // 13 == 0))
        check("Trou Royal : ancre = plus haut Coeur hors declarant, chez le bouche-trou",
              info["anchor_card"] == best and best in hands[info["partner"]] and info["anchor_suit"] == 0)
        if found >= 5:
            break
    check("au moins un Trou Royal trouve", found > 0)


def test_trou_seeded_without_atout_then_fixed_by_lead():
    from engine.trou import new_trou_deal, final_scores as trou_final_scores
    hands, info = _trou_hands()
    d, p, anchor, asuit = info["declarer"], info["partner"], info["anchor_card"], info["anchor_suit"]
    check("ancre = l'As manquant, chez le bouche-trou", anchor == asuit * 13 + 12 and anchor in hands[p])
    b = Bidding(dealer_index=0)
    b.seed_trou(info)
    c = b.contracts[0]
    check("Trou seede sans atout ni niveau", c["suit"] is None and c["level"] == 0 and c["type"] == "trou")
    check("rang 13 sur l'echelle", b.ladder_rank(0) == 13)
    check("plus aucune action trou_choice legale",
          all(not (48 <= a < 52) for s in range(4) for a in b.legal_action_ids(s)))
    for anchor_lead, expected_level, lead_card in ((True, 8, info["anchor_card"]), (False, 9, None)):
        contract = dict(c)
        leader = first_leader(0, contract)
        check("le bouche-trou entame", leader == p)
        if lead_card is None:  # petite carte de la couleur de l'As -> 9 plis quand meme
            lead_card = min(x for x in hands[p] if x // 13 == asuit and x != anchor)
        game = new_trou_deal(hands, d, p, anchor, leader, contract=contract)
        obs_before = game.public_contract()
        check("contrat public sans atout/niveau avant l'entame",
              obs_before["trump"] is None and obs_before["level"] is None)
        game.step(p, lead_card)
        check("atout = couleur de l'entame", game.trump_suit == asuit and contract["suit"] == asuit)
        check("niveau %d (%s)" % (expected_level, "ancre" if anchor_lead else "autre carte"),
              game.level == expected_level and contract["level"] == expected_level)
        while not game.done:
            seat = game.turn
            game.step(seat, next(x for x, ok in enumerate(game.legal_action_mask(seat)) if ok))
        check("score via le niveau fixe", trou_final_scores(game)[d] == float(trou_result(expected_level,
              game.declarer_side_tricks())))
    other_card = min(x for x in hands[p] if x // 13 != asuit)
    other = new_trou_deal(hands, d, p, anchor, p)
    other.step(p, other_card)
    check("autre couleur -> 9 plis dans cette couleur", other.trump_suit == other_card // 13 and other.level == 9)
    wrong = new_trou_deal(hands, d, p, anchor, p)
    try:
        wrong.step(d, hands[d][0])
        check("seul le bouche-trou peut ouvrir un Trou", False)
    except ValueError:
        pass


def test_bidding_always_terminates_all_pass():
    """Personne ne declare jamais rien -> passe generale, jamais de blocage."""
    for dealer in range(4):
        b = Bidding(dealer)
        # premier siege attend, les 3 suivants passent, puis le 1er passe aussi
        seat = b.turn
        assert b.can_wait(seat)
        b.wait(seat)
        guard = 0
        while not b.finished:
            guard += 1
            check("pas de blocage infini (all-pass)", guard < 20)
            assert b.pass_bid(b.turn)
        check("all-pass -> redonne generale", b.general_pass)
        check("all-pass -> aucun contrat verrouille", b.locked_contract is None)


def test_bidding_random_playouts_never_crash_and_terminate():
    """Simule des enchere aleatoires (parmi les actions LEGALES uniquement)
    sur des mains reelles, avec Trou seede quand il apparait -- doit
    toujours terminer (locked_contract xor general_pass), jamais boucler
    ni planter, sur un grand nombre de donnes."""
    rng = random.Random(12345)
    for i in range(500):
        hands = deal(rng)
        b = Bidding(dealer_index=i % 4)
        trou_info = detect_trou(hands)
        if trou_info is not None:
            b.seed_trou(trou_info)
        guard = 0
        while not b.finished:
            guard += 1
            check("pas de blocage infini (donne %d)" % i, guard < 200)
            seat = b.turn
            legal = b.legal_action_ids(seat)
            check("au moins une action legale a chaque tour (donne %d)" % i, len(legal) > 0)
            action = rng.choice(legal)
            ok = b.apply(seat, action)
            check("apply() reussit toujours sur une action annoncee legale (donne %d)" % i, ok)
        check("termine par redonne XOR contrat verrouille (donne %d)" % i,
              b.general_pass != (b.locked_contract is not None))
        if b.locked_contract is not None:
            c = b.locked_contract
            score = b._score(c)
            for other in b._live_contracts():
                if other is not c:
                    check("le contrat verrouille domine tout le reste (donne %d)" % i, score > b._score(other))


def test_ladder_strictly_increasing_within_each_family():
    from engine.bidding import LADDER
    solo_ranks = [LADDER["solo-%d" % lvl] for lvl in (6, 7, 8)]
    check("Solo 6<7<8 sur l'echelle", solo_ranks == sorted(solo_ranks))
    emballage_ranks = [LADDER["emballage-%d" % lvl] for lvl in range(8, 14)]
    check("Emballage 8..13 croissant sur l'echelle", emballage_ranks == sorted(emballage_ranks))
    misere_ranks = [LADDER["misere-%s" % k] for k in ("petite", "piccolo", "grande", "etalee")]
    check("Misere petite<piccolo<grande<etalee sur l'echelle", misere_ranks == sorted(misere_ranks))


def test_hand_aware_legality_filters_propose_and_emballer_by_suit_held():
    """Bidding elle-meme reste hand-agnostic (comme le vrai game/bidding.gd
    -- "Card-holding ... is enforced one layer up"). legal_action_ids_with_
    hand() est CETTE couche du dessus, cote entrainement -- doit filtrer
    exactement comme game/whist_ai_features_v2_bid.gd:_hand_has_suit()
    cote Godot, sinon le reseau apprendrait une surface d'action plus
    large que celle qu'il verra jamais en jeu reel."""
    b = Bidding(dealer_index=3)
    seat = b.turn  # premier siege a parler
    hand_no_hearts = [c for c in range(52) if c // 13 != 0]  # aucune carte de Coeur (suit 0)

    hand_agnostic = set(b.legal_action_ids(seat))
    check("sans le filtre de main, proposer Coeur est repute legal (hand-agnostic)",
          4 + 0 in hand_agnostic)  # _PROPOSE_BASE=4, suit 0 = Coeur

    hand_aware = set(legal_action_ids_with_hand(b, seat, hand_no_hearts))
    check("avec le filtre de main, proposer une couleur non tenue (Coeur) est exclu",
          4 + 0 not in hand_aware)
    check("avec le filtre de main, proposer une couleur tenue reste legal",
          any(a in hand_aware for a in (4 + 1, 4 + 2, 4 + 3)))
    check("le filtre de main ne retire jamais pass/wait (jamais lies a une couleur)",
          1 in hand_aware and 0 in hand_aware)

    # emballer : seat 0 (premier a parler) propose Pique (tient forcement une carte,
    # main non testee ici), puis seat suivant tente d'emballer sans tenir Pique.
    b2 = Bidding(dealer_index=3)
    proposer = b2.turn
    assert b2.propose(proposer, 3)  # Pique
    emballer_seat = b2.turn
    hand_no_spades = [c for c in range(52) if c // 13 != 3]
    hand_aware_2 = set(legal_action_ids_with_hand(b2, emballer_seat, hand_no_spades))
    check("emballer une couleur non tenue (Pique) est exclu meme si l'enchere l'autoriserait",
          8 + 3 not in hand_aware_2)  # _EMBALLER_BASE=8


# --- engine/cards.py : legalite de pli ---

def test_trick_winner_trump_beats_led_suit():
    # led = coeur (suit 0), quelqu'un coupe a trefle (suit 2, atout)
    plays = [(0, 0 * 13 + 5), (1, 0 * 13 + 12), (2, 2 * 13 + 0), (3, 0 * 13 + 3)]
    check("l'atout bat la couleur demandee meme avec un petit atout",
          trick_winner(plays, trump_suit=2) == 2)


def test_trick_winner_no_trump_highest_led_suit_wins():
    plays = [(0, 0 * 13 + 5), (1, 0 * 13 + 12), (2, 2 * 13 + 0), (3, 0 * 13 + 3)]
    check("sans atout, la plus haute carte de la couleur demandee gagne (l'As de coeur ici)",
          trick_winner(plays, trump_suit=None) == 1)


def test_pass_parole_declarer_jump():
    """Passe parole : le PROPOSEUR peut sauter a n'importe quel niveau
    level+1..13, une seule fois ; l'emballeur n'a que +1 (regle confirmee
    par l'utilisateur le 2026-09-27, miroir de whist-4-it 9c0dfb3
    game/bidding.gd::raise_contract_to)."""
    b = Bidding(3)  # siege 0 parle en premier
    check("propose", b.propose(0, 0))
    check("emballer", b.emballer(1, 0))
    check("abondance depasse l'emballage", b.abondance(2, 3, 9))
    check("passe", b.pass_bid(3))
    check("emballeur a la main", b.turn == 1)
    check("emballeur ne peut pas sauter", not b.can_raise_to(1, 10))
    check("passe parole", b.pass_parole(1))
    check("proposeur a la main", b.turn == 0)
    check("pas de saut au niveau actuel", not b.can_raise_to(0, 8))
    check("pas au-dela de 13", not b.can_raise_to(0, 14))
    check("saut legal 9..13", all(b.can_raise_to(0, lv) for lv in range(9, 14)))
    check("saut a 12", b.apply_raise_to(0, 12))
    c = b.contracts[b.lie_contract_index[0]]
    check("niveau 12", c["level"] == 12)
    check("parole consommee", not c["parole_pending"])
    check("bid_summary : relance tracee", b.bid_summary[0]["raised"])
    check("plus de second saut", not b.can_raise_to(0, 13))
    # ids reseau du saut (52..55 = niveaux 10..13), +1 toujours via ACTION_RAISE
    b = Bidding(3)
    b.propose(0, 0), b.emballer(1, 0), b.abondance(2, 3, 9), b.pass_bid(3), b.pass_parole(1)
    ids = b.legal_action_ids(0)
    check("ids de saut 52..55 legaux en parole (niveau 8)", all(52 + i in ids for i in range(4)))
    check("+1 par ACTION_RAISE", 3 in ids)
    check("id 52 = saut au niveau 10", bid_action_to_dict(52) == {"action": "raise", "level": 10})
    check("application par id", b.apply(0, 53) and b.contracts[b.lie_contract_index[0]]["level"] == 11)
    check("plus aucun saut ensuite", not any(i >= 52 for i in b.legal_action_ids(b.turn)))
    check("NUM_BID_ACTIONS = 56", NUM_BID_ACTIONS == 56)


def test_emballer_starts_above_8_when_needed():
    """Bug report 2026-09-29 (miroir de whist-4-it game/bidding.gd) :
    emballer cree l'emballage au plus petit niveau 8..13 qui depasse tous
    les contrats vivants, au lieu d'etre refuse parce que 8 ne suffit pas."""
    b = Bidding(0)  # siege 1 parle en premier
    check("propose carreau", b.propose(1, 1))
    check("emballe carreau (8)", b.emballer(2, 1))
    check("niveau 8", b.contracts[b.lie_contract_index[2]]["level"] == 8)
    check("propose trefle", b.propose(3, 2))
    check("niveau d'entree trefle = 9", b.emballage_entry_level(2) == 9)
    check("emballer trefle legal", b.can_emballer(0, 2))
    check("emballe trefle", b.emballer(0, 2))
    c = b.contracts[b.lie_contract_index[0]]
    check("emballage trefle cree a 9", c["type"] == "emballage" and c["level"] == 9)
    check("emballage carreau domine", b._is_dominated(b.lie_contract_index[2]))
    # au-dessus de 13 rien ne depasse -> refuse (solo 13 vivant)
    b = Bidding(0)
    b.propose(1, 1)
    check("solo chelem", b.solo_chelem(2, 0))
    check("aucun niveau d'emballage ne depasse", b.emballage_entry_level(1) is None)
    check("emballer refuse", not b.can_emballer(3, 1))


def main():
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    for t in tests:
        t()
    print("OK : %d checks passes sur %d fonctions de test (bareme + invariants structurels)." % (
        CHECKS_RUN[0], len(tests)))


if __name__ == "__main__":
    main()
