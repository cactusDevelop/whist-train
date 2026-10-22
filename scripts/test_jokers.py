"""Tests du port Python des jokers (engine/jokers.py) contre les regles de
whist-4-it/roguelike/joker_effects.gd + joker_rules.gd.

Usage : venv/Scripts/python.exe scripts/test_jokers.py
"""

import random
import sys

sys.path.insert(0, ".")

from engine import jokers as J
from engine.team_trump import TeamTrumpDeal

H, D, C, S = 0, 1, 2, 3


def card(suit, rank):  # rank : 2..14 (As = 14)
    return suit * 13 + rank - 2


def setup(trump=S, partner_of=(2, 3, 0, 1)):
    state = J.MatchState([[], [], [], []], random.Random(0))
    rules = J.JokerRules(state)
    hands = [[card(s, r) for s in range(4) for r in range(2 + i * 3, 2 + i * 3 + 3)] + [card(H, 14 - i)]
             for i in range(4)]
    game = TeamTrumpDeal([sorted(h) for h in hands], 0, 2, trump, 8, 0, "emballage", "emballage", "defenseur")
    contract = {"type": "emballage", "declarer": 0, "partner": 2, "suit": trump, "level": 8}
    rules.new_hand(game, contract)
    return state, rules, game


def check(name, got, want):
    ok = got == want
    print("%s %s : %r%s" % ("OK " if ok else "KO ", name, got, "" if ok else " (attendu %r)" % (want,)))
    return ok


def main():
    oks = []
    # Embuscade (siege 1) : la coupe d'un adversaire (siege 0) ne gagne pas.
    state, rules, game = setup()
    rules.effects.append(J.make_effect("embuscade", 1))
    plays = [(1, card(H, 5)), (2, card(H, 9)), (3, card(H, 3)), (0, card(S, 2))]
    oks.append(check("embuscade bloque la coupe adverse", rules.trick_winner(plays, H, S), 2))
    rules.on_trick_resolved(2, plays)
    oks.append(check("embuscade depensee apres usage", rules.effects, []))
    # ... mais pas la coupe du partenaire de contrat (siege 2 = partner_of[0]).
    state, rules, game = setup()
    rules.effects.append(J.make_effect("embuscade", 0))
    plays = [(1, card(H, 5)), (3, card(H, 9)), (0, card(H, 3)), (2, card(S, 2))]
    oks.append(check("embuscade laisse couper le partenaire", rules.trick_winner(plays, H, S), 2))
    # Camps tires du contrat (whist-4-it a3d70cf) : les defenseurs sont allies.
    state, rules, game = setup()
    rules.effects.append(J.make_effect("embuscade", 1))
    plays = [(0, card(H, 5)), (1, card(H, 9)), (2, card(H, 3)), (3, card(S, 2))]
    oks.append(check("defenseurs : embuscade laisse couper l'autre defenseur", rules.trick_winner(plays, H, S), 3))
    rules.bind(game, {"type": "misere", "kind": "grande", "declarers": [1, 3]})
    oks.append(check("misere : deux declarants adversaires", rules.is_opponent(1, 3), True))
    oks.append(check("misere : defenseurs allies", rules.is_opponent(0, 2), False))
    rules.bind(game, {"type": "solo", "suit": S, "level": 6, "declarer": 1})
    oks.append(check("solo : defenseurs allies", (rules.is_opponent(0, 2), rules.is_opponent(0, 1)), (False, True)))
    rules.bind(game, None)
    oks.append(check("sans contrat : tous adversaires", rules.is_opponent(0, 2), True))
    # Coup de Vent : pas d'atout pour ce pli.
    state, rules, game = setup()
    rules.effects.append(J.make_effect("coup_de_vent", 1))
    oks.append(check("coup de vent : atout du pli", rules.trick_trump(), None))
    plays = [(0, card(H, 5)), (1, card(H, 9)), (2, card(H, 3)), (3, card(S, 14))]
    oks.append(check("coup de vent : la coupe ne gagne plus", rules.trick_winner(plays, H, None), 1))
    rules.on_trick_resolved(1, plays)
    oks.append(check("coup de vent : portee du pli", rules.effects, []))
    # Kermesse : pli gagne par une figure = 2, par un As = 1.
    state, rules, game = setup()
    rules.effects.append(J.make_effect("kermesse", 1))
    oks.append(check("kermesse figure", rules.trick_value(1, [(1, card(H, 13))]), 2))
    oks.append(check("kermesse as", rules.trick_value(1, [(1, card(H, 14))]), 1))
    rules.effects.append(J.make_effect("kermesse", 3))
    oks.append(check("deux kermesses s'empilent", rules.trick_value(1, [(1, card(H, 12))]), 3))
    # Voeu de pauvrete (043f) : un joker pioche tous les 3 plis perdus.
    state, rules, game = setup()
    state.draw_pile[1] = ["main_lourde", "embuscade"]
    rules.effects.append(J.make_effect("voeu_de_pauvrete", 1))
    for w in (0, 1, 2):
        rules.on_trick_resolved(w, [])
    oks.append(check("voeu : 2 plis perdus, rien", state.joker_hand[1], []))
    rules.on_trick_resolved(3, [])
    oks.append(check("voeu : 3e pli perdu, une pioche", (len(state.joker_hand[1]), state.energy[1]), (1, 0)))
    # Kermesse (043f) : plis du camp du proprietaire seulement.
    state, rules, game = setup()
    rules.effects.append(J.make_effect("kermesse", 1))
    oks.append(check("kermesse : figure adverse, rien", rules.trick_value(0, [(0, card(H, 13))]), 1))
    # Economie : pioche, cout, passif arme a la pioche, defausse recyclee.
    state = J.MatchState([[], ["kermesse", "voeu_de_pauvrete", "main_lourde"], [], []], random.Random(1))
    rules = J.JokerRules(state)
    state.start_hand()
    oks.append(check("pioche 3 / energie 3", (len(state.joker_hand[1]), state.energy[1]), (3, 3)))
    oks.append(check("passif arme a la pioche", [e.id for e in rules.effects], ["voeu_de_pauvrete"]))
    oks.append(check("passif non jouable", state.play_error(1, "voeu_de_pauvrete"), "passive"))
    state.energy[1] = 1
    oks.append(check("energie insuffisante", state.play_error(1, "kermesse"), "energy"))
    state.end_hand()
    state.draw_jokers(1, 2)
    oks.append(check("defausse remelangee dans la pioche", len(state.joker_hand[1]) + len(state.draw_pile[1]), 3))
    # JokerHand : la valeur du pli et le meneur suivant passent dans le moteur.
    state, rules, game = setup(trump=S)
    hand = J.JokerHand(game, {"type": "emballage", "declarer": 0, "partner": 2, "suit": S, "level": 8}, state, rules)
    rules.effects.append(J.make_effect("kermesse", 1))
    hand.step(0, card(H, 2))
    hand.step(1, card(H, 13) if card(H, 13) in game.hands[1] else game.hands[1][0])
    hand.step(2, game.hands[2][0])
    hand.step(3, game.hands[3][0])
    oks.append(check("tricks_won total = valeur du pli", sum(game.tricks_won) in (1, 2), True))
    oks.append(check("le gagnant mene", game.turn, [s for s in range(4) if game.tricks_won[s]][0]))
    oks += v2_tests()
    print("%d/%d" % (sum(oks), len(oks)))
    sys.exit(0 if all(oks) else 1)


def v2_tests():
    """Catalogue complet (v2, 2026-10-13) contre joker_effects.gd."""
    oks = []
    contract = {"type": "emballage", "declarer": 0, "partner": 2, "suit": S, "level": 8}

    def trick_with(jid, owner, plays, trump=S):
        state, rules, game = setup(trump=trump)
        e = J.make_effect(jid, owner)
        rules.effects.append(e)
        if isinstance(e, J.TrickJoker):
            rules.joker_cards[owner] = jid
        return rules.trick_winner(plays, suit_of_card(plays[0][1]), rules.trick_trump()), rules

    def suit_of_card(c):
        return c // 13

    # Cartes de pli (le joueur 0 et son partenaire 2 contre 1 et 3).
    # 6c54 : le Cameleon vaut un Valet (une Dame avant)
    oks.append(check("cameleon : vaut un Valet", J.make_effect("cameleon", 0).make_card(H, None), card(H, 11)))
    w, _ = trick_with("cameleon", 0, [(1, card(H, 9)), (2, card(H, 3)), (3, card(H, 10)), (0, card(H, 11))])
    oks.append(check("cameleon : le Valet bat le 10", w, 0))
    # 592e (whist-4-it 8a6f635) : egalite joker / vraie carte -> la vraie carte gagne, entre deux jokers le premier.
    w, _ = trick_with("cameleon", 0, [(1, card(H, 11)), (2, card(H, 3)), (3, card(H, 10)), (0, card(H, 11))])
    oks.append(check("cameleon : perd l'egalite face au vrai Valet pose avant", w, 1))
    w, _ = trick_with("cameleon", 0, [(0, card(H, 11)), (1, card(H, 11)), (2, card(H, 3)), (3, card(H, 10))])
    oks.append(check("cameleon : perd l'egalite face au vrai Valet pose apres", w, 1))
    w, rules = trick_with("cameleon", 0, [(0, card(H, 11)), (1, card(H, 9)), (2, card(H, 11)), (3, card(H, 10))])
    oks.append(check("cameleon : vrai Valet du partenaire", w, 2))
    state, rules, game = setup(trump=S)
    rules.effects += [J.make_effect("cameleon", 0), J.make_effect("cameleon", 1)]
    rules.joker_cards.update({0: "cameleon", 1: "cameleon"})
    w = rules.trick_winner([(0, card(H, 11)), (1, card(H, 11)), (2, card(H, 3)), (3, card(H, 10))], H, rules.trick_trump())
    oks.append(check("deux jokers a egalite : le premier pose garde le pli", w, 0))
    w, _ = trick_with("cameleon", 0, [(1, card(H, 11)), (2, card(H, 3)), (3, card(D, 11)), (0, card(H, 11))])
    oks.append(check("cameleon : pas d'egalite avec une autre couleur", w, 1))
    w, _ = trick_with("zero", 0, [(1, card(H, 9)), (2, card(H, 3)), (3, card(H, 4)), (0, card(H, 2))])
    oks.append(check("zero : ne gagne jamais", w, 1))
    w, _ = trick_with("traitre", 0, [(1, card(H, 9)), (2, card(H, 3)), (3, card(H, 4)), (0, card(H, 2))])
    oks.append(check("traitre : la plus faible (hors traitre) gagne", w, 2))
    w, _ = trick_with("borgne", 0, [(1, card(H, 9)), (2, card(H, 3)), (3, card(S, 4)), (0, card(H, 14))])
    oks.append(check("borgne : perd si quelqu'un coupe", w, 3))
    w, _ = trick_with("borgne", 0, [(1, card(H, 9)), (2, card(H, 3)), (3, card(H, 4)), (0, card(H, 14))])
    oks.append(check("borgne : As sans coupe", w, 0))
    w, rules = trick_with("collet", 0, [(1, card(H, 9)), (2, card(H, 3)), (3, card(H, 4)), (0, card(H, 14))])
    oks.append(check("collet (6c54) : ma prochaine carte ne gagne pas", w, 1))
    rules.on_trick_resolved(1, [(1, card(H, 9)), (2, card(H, 3)), (3, card(H, 4)), (0, card(H, 14))])
    oks.append(check("collet : une seule carte", rules.effects, []))
    _, rules = trick_with("carte_fantome", 0, [(1, card(H, 9)), (2, card(H, 3)), (3, card(H, 4)), (0, card(H, 2))])
    oks.append(check("carte fantome : pli a 0", rules.trick_value(1, [(1, card(H, 9))]), 0))
    _, rules = trick_with("double_valet", 0, [(1, card(H, 9)), (0, card(H, 11))])
    oks.append(check("double valet : pli a 2 pour son proprietaire", rules.trick_value(0, [(0, card(H, 11))]), 2))
    w, _ = trick_with("inversion", 1, [(1, card(H, 9)), (2, card(H, 3)), (3, card(S, 4)), (0, card(H, 2))])
    oks.append(check("inversion : la plus faible de la couleur demandee", w, 0))
    # JokerHand.step_joker_card : la main ne diminue pas, le pli se resout.
    state, rules, game = setup(trump=S)
    hand = J.JokerHand(game, contract, state, rules)
    game.turn = 1
    hand.step(1, card(H, 5))
    hand.step(2, card(H, 8))
    hand.step(3, card(H, 11))
    before = list(game.hands[0])
    rules.effects.append(J.make_effect("cameleon", 0))
    hand.step_joker_card(0, card(H, 11), "cameleon")
    oks.append(check("carte de joker : main intacte", game.hands[0], before))
    oks.append(check("carte de joker : pli resolu, le vrai Valet garde l'egalite",
                     (game.trick_count, game.tricks_won[3], game.turn), (1, 1, 3)))
    # Sorts et pieges.
    state, rules, game = setup()
    rules.effects.append(J.make_effect("fausse_piste", 0))
    oks.append(check("fausse piste : sort adverse annule", rules.play(1, "coup_de_vent"), {"cancelled": True}))
    oks.append(check("fausse piste : une seule fois", rules.play(3, "coup_de_vent"), {}))
    state, rules, game = setup()
    rules.effects.append(J.make_effect("fausse_piste", 0))
    oks.append(check("fausse piste (6c54) : une aura adverse aussi", rules.play(1, "kermesse"), {"cancelled": True}))
    state, rules, game = setup()
    rules.effects.append(J.make_effect("silence_du_cercle", 0))
    oks.append(check("silence : aucun joker adverse, aura comprise",
                     [rules.play(1, "coup_de_vent"), rules.play(1, "kermesse")], [{"error": "silence"}] * 2))
    oks.append(check("silence : le proprietaire et son partenaire jouent",
                     [rules.blocked("main_lourde", 0), rules.blocked("main_lourde", 2)], [False, False]))
    for _ in range(5):
        rules.on_trick_resolved(0, [])
    oks.append(check("silence : toute la donne (043f)", rules.blocked("coup_de_vent", 1), True))
    state, rules, game = setup(trump=H)
    rules.effects.append(J.make_effect("lune_noire", 0))
    oks.append(check("lune noire : atout rouge annule", rules.trick_trump(), None))
    oks.append(check("lune noire : injouable sans atout rouge", state.play_error(0, "lune_noire"), "not_in_hand"))
    state, rules, game = setup(trump=S)
    state.joker_hand[0] = ["lune_noire"]
    state.energy[0] = 3
    oks.append(check("lune noire : pointless a pique", state.play_error(0, "lune_noire"), "pointless"))
    state, rules, game = setup(trump=S)
    oks.append(check("brouillard (6c54) : cible une couleur", rules.play(0, "brouillard_de_la_senne"), {"error": "target"}))
    rules.play(0, "brouillard_de_la_senne", C)
    game.turn = 0
    oks.append(check("brouillard : atout choisi quand le proprietaire entame", rules.trick_trump(), C))
    game.turn = 1
    oks.append(check("brouillard : atout du contrat quand un autre entame", rules.trick_trump(), S))
    state, rules, game = setup()
    rules.effects.append(J.make_effect("gueule_de_bois", 0))
    oks.append(check("gueule de bois (6c54) : l'adversaire joue sa plus forte",
                     rules.bot_card(1, card(H, 5), [card(H, 5), card(H, 14), card(H, 9)]), card(H, 14)))
    oks.append(check("gueule de bois : pas le partenaire",
                     rules.bot_card(2, card(H, 14), [card(H, 14), card(H, 5)]), card(H, 14)))
    state, rules, game = setup()
    oks.append(check("dette : cible adverse seulement", rules.play(0, "dette_de_jeu", 2), {"error": "target"}))
    rules.play(0, "dette_de_jeu", 1)
    rules.on_trick_resolved(1, [])
    oks.append(check("dette : 5 points cedes", rules.score_bonus, [5, -5, 0, 0]))
    state, rules, game = setup()
    rules.trick_winners = [0, 1, 3]
    game.tricks_won[3] = 1
    rules.play(0, "remonte_pli")
    oks.append(check("remonte-pli : dernier pli adverse", (game.tricks_won[0], game.tricks_won[3]), (1, 0)))
    state, rules, game = setup()
    rules.play(0, "horloge_arretee")
    oks.append(check("horloge : la donne s'arrete apres ce pli", rules.hand_tricks(), 1))
    game.trick_count = 1
    game.tricks_won[:] = [1, 0, 0, 0]
    rules.on_trick_resolved(0, [])
    oks.append(check("horloge (6c54) : les 12 plis restants au camp en tete", game.tricks_won, [13, 0, 0, 0]))
    state, rules, game = setup()
    game.trick_count = 1
    game.tricks_won[:] = [1, 0, 0, 0]
    rules.play(1, "horloge_arretee")
    game.trick_count = 2
    game.tricks_won[:] = [1, 1, 0, 0]
    rules.on_trick_resolved(1, [])
    oks.append(check("horloge : egalite, personne", game.tricks_won, [1, 1, 0, 0]))
    # Enchere et ecarts.
    state, rules, game = setup()
    oks.append(check("main de fer (6c54) : cible une couleur", rules.play(0, "main_de_fer"), {"error": "target"}))
    rules.play(0, "main_de_fer", H)
    c = {"type": "solo", "suit": S, "level": 6, "declarer": 0}
    oks.append(check("main de fer : son solo dans la couleur choisie", (rules.on_contract(c), c["suit"]), (True, H)))
    c = {"type": "emballage", "suit": S, "level": 8, "declarer": 1, "partner": 3}
    oks.append(check("main de fer : rien sur le contrat d'un autre", (rules.on_contract(c), c["suit"]), (False, S)))
    state, rules, game = setup()
    rules.bind(game, {"type": "misere", "kind": "petite", "declarers": [0]})
    rules.play(0, "pot_de_vin")
    oks.append(check("pot-de-vin (6c54) : 1er pli de sa misere a 0, puis 1",
                     [rules.trick_value(0, [(0, card(H, 2))]), rules.trick_value(0, [(0, card(H, 3))])], [0, 1]))
    state, rules, game = setup()
    rules.trick_winners = [0]
    game.tricks_won[0] = 1
    oks.append(check("pli refile (6c54) : vers un adversaire seulement", rules.play(0, "pli_refile", 2), {"error": "target"}))
    oks.append(check("pli refile : mon dernier pli passe", (rules.play(0, "pli_refile", 1), game.tricks_won[:2]),
                     ({"to": 1}, [0, 1])))
    state, rules, game = setup()
    game.trick_count = 1
    oks.append(check("courant d'air (6c54) : la plus forte", rules.play(0, "courant_d_air"), {"discarded": card(H, 14)}))
    rules.effects = [J.make_effect("valet_borgne", 0)]
    oks.append(check("valet borgne (6c54) : ses points perdus a 0", rules.adjust_deltas([-7, 3, -3, 1], {}), [0, 3, -3, 1]))
    oks.append(check("6c54 : zinneke +3, treizieme 4 plis, couts inversion / chapeau",
                     (J.CoeurDeZinneke.BONUS, J.TreiziemeCarte.LAST, J.jcost("inversion"), J.jcost("coup_du_chapeau")),
                     (3, 4, 2, 1)))
    # Main Lourde (6c54) : pioche 2, puis defausse 1 (bot : un joker qu'il ne lance jamais d'abord).
    state = J.MatchState([[], [], [], []], random.Random(0))
    rules = J.JokerRules(state)
    state.joker_hand[1] = ["main_lourde"]
    state.draw_pile[1] = ["kermesse", "courant_d_air"]
    state.energy[1] = 3
    J.spend(state, 1, "main_lourde", rules.play(1, "main_lourde"))
    oks.append(check("main lourde : pioche 2, le bot defausse le non lancable",
                     (state.joker_hand[1], state.discard[1], state.to_discard[1]),
                     (["kermesse"], ["main_lourde", "courant_d_air"], 0)))
    state.joker_hand[0] = ["main_lourde"]
    state.draw_pile[0] = ["zero", "miroir"]
    J.spend(state, 0, "main_lourde", rules.play(0, "main_lourde"))
    oks.append(check("main lourde : le joueur defausse le plus recent", state.joker_hand[0], ["miroir"]))
    c = {"type": "solo", "suit": S, "level": 6, "declarer": 0}
    rules.effects = [J.make_effect("bluff", 0)]
    oks.append(check("bluff : demi-perte (vers 0)", rules.adjust_deltas([-7, 3, 3, 1], c), [-3, 3, 3, 1]))
    rules.effects = []
    rules.curses = [J.make_effect("pacte_du_tenancier", 0)]
    oks.append(check("pacte : -5 par contrat rate (043f)", rules.adjust_deltas([-7, 3, 3, 1], c), [-12, 3, 3, 1]))
    # Economie du joueur : amelioration, maudits, Soupcon, Oeil.
    oks.append(check("amelioration : cout -1", (J.jcost("inversion+"), J.known("zero+")), (1, False)))
    state = J.MatchState([["cartes_marquees", "regard_furtif", "livre_de_comptes"], [], [], []], random.Random(2), 9)
    rules = J.JokerRules(state)
    state.arm_curses()
    oks.append(check("maudit arme, jamais pioche", (state.curses[0], len(state.draw_pile[0])), (["cartes_marquees"], 2)))
    oks.append(check("cartes marquees : sorts a 0", rules.cost_of(0, "regard_furtif"), 0))
    state.joker_hand[0] = ["regard_furtif"]
    state.spend(0, "regard_furtif")
    oks.append(check("soupcon : +2 (arriere-salle), plafonne", state.suspicion, 10))
    J.watch_eye(state, rules, "points", "regard_furtif")
    oks.append(check("oeil : sanction points et jauge videe", (rules.score_bonus[0], state.suspicion), (-20, 0)))
    oks += review_20261014_tests()
    return oks


def review_20261014_tests():
    """whist-4-it 2c63f60 + 6bf4a7c (tests/roguelike_catalog_runner.gd)."""
    from engine import joker_match as jm
    from engine.solitaire import new_solitaire_deal
    oks = []
    # Inversion : aura, chaque pli de la donne a la plus faible.
    state, rules, game = setup()
    rules.play(0, "inversion")
    plays = [(0, card(S, 13)), (1, card(S, 3)), (2, card(S, 14)), (3, card(S, 5))]
    oks.append(check("inversion : le 3 prend", rules.trick_winner(plays, S, S), 1))
    rules.on_trick_resolved(1, plays)
    plays = [(0, card(C, 13)), (1, card(C, 3)), (2, card(C, 14)), (3, card(C, 5))]
    oks.append(check("inversion : aura, le pli suivant aussi", rules.trick_winner(plays, C, S), 1))
    oks.append(check("inversion : lancee par les bots en misere", ("inversion" in J.CASTABLE, J.jtype("inversion")), (True, "aura")))
    # Rachat : un pli de plus au decompte.
    state, rules, game = setup()
    hands = [[card(S, r) for r in range(2, 15)], [card(H, r) for r in range(2, 15)],
             [card(D, r) for r in range(2, 15)], [card(C, r) for r in range(2, 15)]]
    g = new_solitaire_deal(hands, 0, S, 6, 0)
    c = {"type": "solo", "suit": S, "level": 6, "declarer": 0}
    rules.bind(g, c)
    rules.scorer = jm.base_score
    rules.effects = [J.make_effect("rachat", 0)]
    g.tricks_won[:] = [5, 3, 3, 2]
    failed = jm.base_score(g, c)
    g.tricks_won[:] = [6, 3, 3, 2]
    made = jm.base_score(g, c)
    g.tricks_won[:] = [5, 3, 3, 2]
    oks.append(check("rachat : solo a un pli rachete = fait", rules.adjust_deltas(failed, c), made))
    oks.append(check("rachat : plis de la table intacts", g.tricks_won, [5, 3, 3, 2]))
    c = {"type": "solo", "suit": S, "level": 6, "declarer": 1}
    g2 = new_solitaire_deal(hands, 1, S, 6, 0)
    rules.bind(g2, c)
    g2.tricks_won[:] = [3, 6, 2, 2]
    their_made = jm.base_score(g2, c)
    g2.tricks_won[:] = [4, 5, 2, 2]
    their_failed = jm.base_score(g2, c)
    g2.tricks_won[:] = [3, 6, 2, 2]
    oks.append(check("rachat : en defense, un pli pris au declarant", rules.adjust_deltas(their_made, c), their_failed))
    oks.append(check("rachat : rien en misere", rules.adjust_deltas([1, 2, 3, 4], {"type": "misere", "kind": "petite", "declarers": [0]}), [1, 2, 3, 4]))
    oks.append(check("rachat : salle sans soupcon (043f), pot-de-vin a 1", (J.jsuspicion("rachat") > 0, J.jcost("pot_de_vin")), (False, 1)))
    # Cartes sur Table : le contrat porte la main exposee.
    state, rules, game = setup()
    rules.bind(None, None)
    rules.play(0, "cartes_sur_table")
    c = {"type": "solo", "suit": S, "level": 6, "declarer": 1}
    oks.append(check("cartes sur table : contrat expose", (rules.on_contract(c), c.get("exposed"), state.energy[0]), (True, [0], 1)))
    from agents.expert_bot import PublicView
    g3 = new_solitaire_deal([list(h) for h in hands], 1, S, 6, 0)
    g3.set_public_context(dealer=0, bid_summary=None, whisted=False, level_known=True)
    g3.exposed = [0]
    oks.append(check("cartes sur table : l'ExpertBot voit la main", sorted(PublicView(g3, 1).revealed), [0]))
    # Fossoyeur : emballage aussi.
    state, rules, game = setup()
    rules.effects = [J.make_effect("fossoyeur_de_plis", 0)]
    rules.on_trick_resolved(2, [])
    oks.append(check("fossoyeur : pli du partenaire d'emballage", rules.score_bonus[0], 1))
    # Brim Dore : agit depuis le deck.
    st = J.MatchState([["brimborion_dore", "zero"], [], [], []], random.Random(1))
    oks.append(check("brim dore : jamais pioche", (st.curses[0], st.draw_pile[0]), (["brimborion_dore"], ["zero"])))
    # Manche Truquee.
    state, rules, game = setup(trump=H)
    rules.play(0, "manche_truquee")
    plays = [(1, card(S, 14)), (2, card(S, 3)), (3, card(S, 5)), (0, card(S, 2))]
    oks.append(check("manche truquee : le 2 compte comme atout", rules.trick_winner(plays, S, H), 0))
    plays2 = [(1, card(S, 14)), (2, card(H, 14)), (3, card(S, 5)), (0, card(S, 2))]
    oks.append(check("manche truquee (6c54) : l'atout le plus fort, d'office", rules.trick_winner(plays2, S, H), 0))
    rules.on_trick_resolved(0, plays)
    oks.append(check("manche truquee : portee du pli", rules.trick_winner(plays, S, H), 1))
    rules.bind(game, {"type": "misere", "kind": "grande", "declarers": [1]})
    game.trump_suit = None
    oks.append(check("manche truquee : pointless sans atout", J.make_effect("manche_truquee", 0).pointless(rules), True))
    # Coup du Chapeau.
    state, rules, game = setup()
    rules.play(0, "coup_du_chapeau")
    for w in (0, 0):
        rules.on_trick_resolved(w, [])
    oks.append(check("coup du chapeau : deux de suite, rien", rules.score_bonus[0], 0))
    rules.on_trick_resolved(0, [])
    oks.append(check("coup du chapeau : le troisieme, +6", rules.score_bonus[0], 6))
    # Pourboire.
    state, rules, game = setup()
    rules.play(0, "pourboire")
    rules.on_trick_resolved(0, [])
    rules.play(0, "pourboire")
    rules.on_trick_resolved(1, [])
    oks.append(check("pourboire : +1 sou si pli pris", state.sous_won, 1))
    # Chope Vide.
    state, rules, game = setup()
    rules.play(0, "chope_vide")
    rules.adjust_deltas([0, 0, 0, 0], {"type": "emballage", "declarer": 0, "partner": 2})
    oks.append(check("chope vide : un joker de plus a la suivante", state.next_hand_draws, [1, 0, 0, 0]))
    state, rules, game = setup()
    rules.play(0, "chope_vide")
    state.played_this_hand[0] = 1
    rules.adjust_deltas([0, 0, 0, 0], {"type": "emballage", "declarer": 0, "partner": 2})
    oks.append(check("chope vide : un joker joue, rien", state.next_hand_draws, [0, 0, 0, 0]))
    # Faux Pas.
    state, rules, game = setup()
    rules.play(0, "faux_pas")
    plays = [(0, card(S, 14)), (1, card(H, 2)), (2, card(S, 3)), (3, card(S, 5))]
    rules.trick_winner(plays, S, S)
    rules.on_trick_resolved(1, plays)
    plays = [(0, card(C, 14)), (1, card(D, 2)), (2, card(C, 3)), (3, card(C, 5))]
    rules.trick_winner(plays, C, S)
    rules.on_trick_resolved(0, plays)
    oks.append(check("faux pas : 3 cedes au proprietaire, une fois (043f)", rules.score_bonus, [3, -3, 0, 0]))
    oks.extend(spammer_tests())
    oks.extend(elite_tests())
    return oks


def spammer_tests():
    """79ac : elites/boss piochent tout leur deck (+ Pacte du Tenancier),
    3 + SPAMMER_ENERGY[mise] + 2 ⚡ ; Mises Rouge/Noire/Doree."""
    from engine.joker_match import Match
    oks = []
    for mise in range(4):
        m = Match({"kind": "elite", "act": 3, "eye": "points", "deck": ["pourboire"], "mise": mise}, 7)
        m.rules.new_hand(None, None)
        m.state.start_hand()
        deck1 = len(m.state.draw_pile[1]) + len(m.state.discard[1]) + len(m.state.joker_hand[1])
        oks.append(check("spammer mise %d : allumettes elites / joueur" % mise,
                         (m.state.energy[1], m.state.energy[3], m.state.energy[0]),
                         (6 + mise, 6 + mise, 3 - (1 if mise >= 1 else 0))))
        oks.append(check("spammer mise %d : tout le deck pioche" % mise,
                         (len(m.state.joker_hand[1]), m.state.draw_pile[1]), (deck1, [])))
        oks.append(check("spammer mise %d : pacte dans le deck, seuil de l'Oeil" % mise,
                         ("pacte_du_tenancier" in m.state.curses[1], m.state.eye_threshold),
                         (True, 8 if mise >= 2 else 10)))
    for boss, mise, want in (("moine", 2, ("moine",)), ("moine", 3, ("moine", "baronne")),
                             ("baronne", 3, ("baronne", "moine")), ("fossoyeur", 3, ("fossoyeur", "baronne"))):
        m = Match({"kind": "boss", "boss": boss, "act": 2, "eye": "points", "mise": mise}, 3)
        oks.append(check("mise %d %s : regles de boss" % (mise, boss),
                         (m.bosses, any(isinstance(e, J.BaronneSees) for e in m.rules.curses), m.state.spammers),
                         (want, "baronne" in want, [2])))
    m = Match({"kind": "boss", "boss": "moine", "act": 1, "eye": "points", "mise": 0}, 5)
    m.rules.new_hand(None, None)
    m.state.start_hand()
    oks.append(check("boss mise 0 : 6 allumettes, joueur 3", (m.state.energy[2], m.state.energy[0]), (6, 3)))
    return oks



def elite_tests():
    """3d47 (elite_decks.gd) : contrat attitre (2 familles differentes, deck
    fixe), biais d'enchere d'un pli, adaptation en defense a l'acte III
    (jokers empruntes jamais defausses)."""
    from agents import expert_bot
    from engine.bidding import Bidding, bid_action_to_dict
    from engine.joker_match import Match, specialty_bid
    oks = []
    seen = set()
    for seed in range(40):
        m = Match({"kind": "elite", "act": 1 + seed % 3, "eye": "points", "deck": [], "mise": 0}, seed)
        f1, f3 = m.specialties[1], m.specialties[3]
        seen.add((f1, f3))
        if f1 == f3 or m.adapts != ([1, 3] if m.config["act"] >= 3 else []):
            break
        deck = J.SPECIALTY_DECKS[f1]  # dc78 : les maudits (cavalier_seul) vont dans les curses
        if (sorted(m.state.draw_pile[1]) != sorted(j for j in deck if not J.acts_from_deck(j))
                or ("cavalier_seul" in m.state.curses[1]) != (f1 == "solo")):
            break
    oks.append(check("elites : 2 familles differentes, deck fixe, adaptees a l'acte III", seed, 39))
    oks.append(check("elites : les 6 paires de familles sortent", len(seen), 6))
    m = Match({"kind": "boss", "boss": "grand_maitre", "act": 3, "eye": "points", "mise": 0}, 1)
    oks.append(check("boss : deck du Grand Maitre (dc78), sceau dans les curses",
                     (sorted(m.state.draw_pile[2]), "sceau_du_grand_maitre" in m.state.curses[2]),
                     (sorted(J.BOSS_DECKS["grand_maitre"][:-1]), True)))

    state = J.MatchState([[], ["main_lourde", "kermesse"], [], []], random.Random(0))
    state.joker_hand[1] = ["main_lourde", "kermesse"]
    state.draw_pile[1] = []
    state.energy[1] = 9
    state.adapt(1, J.DEFENCE["trou"])
    state.spend(1, "inversion")
    after_spend = (sorted(state.discard[1]), sorted(state.joker_hand[1]))
    state.end_hand()
    oks.append(check("adaptation : main mise de cote, emprunt joue hors defausse", after_spend,
                     (["kermesse", "main_lourde"], ["coup_du_chapeau", "inversion", "kermesse"])))
    oks.append(check("adaptation : empruntes retires en fin de donne", (sorted(state.discard[1]), state.joker_hand[1],
                     state.borrowed[1]), (["kermesse", "main_lourde"], [], [])))

    rng = random.Random(5)
    kinds = {"emballage": set(), "solo": set(), "misere": set()}
    bad = 0
    for _ in range(400):
        deck = list(range(52))
        rng.shuffle(deck)
        hands = [sorted(deck[i * 13:(i + 1) * 13]) for i in range(4)]
        for fam in kinds:
            b = Bidding(0)
            seat = b.turn
            hand = hands[seat]
            if expert_bot.choose_bid(b, seat, hand) != expert_bot.ACTION_PASS:
                continue
            a = specialty_bid(fam, hand, b, seat)
            if a is None:
                continue
            info = bid_action_to_dict(a)
            kinds[fam].add(info["action"] if fam != "misere" else info["kind"])
            if fam == "solo" and info["action"] == "solo" and info["level"] > expert_bot.estimate_tricks(hand, info["suit"]) + J.BID_SLACK["solo"]:
                bad += 1
    oks.append(check("biais : annonces de la famille seulement", (kinds["emballage"] <= {"propose", "emballer"},
                     kinds["solo"] <= {"solo", "propose"}, kinds["misere"] <= {"etalee", "grande"}, bool(kinds["misere"]), bad),
                     (True, True, True, True, 0)))

    adapted = borrowed_left = 0
    for seed in range(12):
        m = Match({"kind": "elite", "act": 3, "eye": "points", "deck": [], "mise": 0}, 100 + seed)
        for _ in range(3):
            hand = m.next_hand()
            for s in m.adapts:
                if m.state.borrowed[s]:
                    adapted += 1
                    if s in J.attackers(hand.contract):
                        borrowed_left += 1
        m.state.end_hand()
        borrowed_left += sum(len(b) for b in m.state.borrowed)
    oks.append(check("acte III : des elites s'adaptent, jamais en attaque", (adapted > 0, borrowed_left), (True, 0)))
    return oks


if __name__ == "__main__":
    main()
