"""Detecteurs d'erreurs d'equipe -- MESURE seulement, jamais une recompense
(demande utilisateur 2026-09-26 : suivre objectivement les erreurs vues en
jeu, sans liste de punitions ; voir DESIGN.md "Erreurs d'equipe").

Surcoupe (definition de l'utilisateur, 2026-09-26) : tous les adversaires
ont deja joue dans le pli, un coequipier le remporte deja, on a au moins
une carte "nulle" (qui ne prend pas le pli) et on joue quand meme une carte
qui prend le pli -- on vole la main a son coequipier. Comptee a CHAQUE
occasion (meme situation, carte nulle disponible) : taux = surcoupes /
occasions.

Equipes : declarant + partenaire (emballage, trou/bouche-trou) contre les
2 autres ; seul contre tous en solitaire/misere (le declarant n'a pas de
coequipier, les defenseurs sont coequipiers entre eux). En misere,
plusieurs declarants restent chacun pour soi.

Deux jeux de donnes, IA greedy (comportement exporte vers Godot) :
  - "reel"  : vraie enchere (meilleur individu de la lignee exportee,
              aux 4 sieges) puis jeu ;
  - "force" : chaque famille Phase 1 a part egale (contrat force), pour
              couvrir aussi les familles que l'enchere choisit rarement.

Usage : venv/Scripts/python.exe scripts/team_errors.py [--deals N] [--seed N]
Appele aussi par scripts/train.py a chaque evaluation (--eval-every).
"""

import argparse
import json
import random
import sys
import time

import torch

sys.path.insert(0, ".")

from agents.play_guards import ai_play_mask
from engine.cards import NUM_PLAYERS, suit_of, trick_winner

TEAM_ERRORS_FILE_DEFAULT = "team_errors.jsonl"
TEAM_ERRORS_DEALS_DEFAULT = 600  # par jeu de donnes ("reel" et "force")
TEAM_ERRORS_SEED = 20260926


def teammates(game, seat):
    """Coequipiers de `seat` (jamais `seat` lui-meme)."""
    if hasattr(game, "partner"):  # TeamTrumpDeal : emballage, trou
        side = {game.declarer, game.partner}
        mine = side if seat in side else set(range(NUM_PLAYERS)) - side
    else:  # LoneDeclarerDeal : solitaire, misere -- chaque declarant seul
        declarers = set(game.declarers)
        mine = {seat} if seat in declarers else set(range(NUM_PLAYERS)) - declarers
    return mine - {seat}


def surcoupe_situation(game, seat):
    """None si ce n'est pas une occasion de surcoupe, sinon l'ensemble des
    cartes legales qui voleraient le pli au coequipier."""
    if getattr(game, "pending_discards", None) or not game.current_trick:
        return None
    mates = teammates(game, seat)
    already = {s for s, _c in game.current_trick}
    still_to_play = set(range(NUM_PLAYERS)) - already - {seat}
    if not still_to_play <= mates:  # un adversaire joue encore apres nous
        return None
    if trick_winner(game.current_trick, game.trump_suit) not in mates:
        return None
    mask = game.legal_action_mask(seat)
    stealing = {c for c, ok in enumerate(mask)
                if ok and trick_winner(game.current_trick + [(seat, c)], game.trump_suit) == seat}
    if not stealing or len(stealing) == sum(mask):  # aucun choix possible
        return None
    return stealing


def coupe_situation(game, seat):
    """Renonce non coupee (signalement utilisateur 2026-10-22) : un coequipier
    a entame dans une couleur ou l'on n'a plus de carte, il y a un atout et on
    tient au moins un atout qui prendrait le pli maintenant. None si ce n'est
    pas une occasion, sinon (atouts gagnants, coequipier deja maitre ?)."""
    trump = game.trump_suit
    if trump is None or getattr(game, "pending_discards", None) or not game.current_trick:
        return None
    mates = teammates(game, seat)
    leader, lead_card = game.current_trick[0]
    if leader not in mates:
        return None
    hand = game.hands[seat]
    if any(suit_of(c) == suit_of(lead_card) for c in hand) or suit_of(lead_card) == trump:
        return None
    winning = {c for c in hand
               if suit_of(c) == trump and trick_winner(game.current_trick + [(seat, c)], trump) == seat}
    if not winning:
        return None
    return winning, trick_winner(game.current_trick, trump) in mates


MISERE_FAMILIES = ("piccolo", "petite_misere", "grande_misere", "misere_etalee")


def new_tally():
    return {}


def _bump(tally, key, occasions, errors):
    entry = tally.setdefault(key, {"occasions": 0, "surcoupes": 0})
    entry["occasions"] += occasions
    entry["surcoupes"] += errors


def _bump_coupe(tally, key, err):
    entry = tally.setdefault(key, {"occasions": 0, "defausses": 0})
    entry["occasions"] += 1
    entry["defausses"] += err


def atout_situation(game, seat):
    """Atouts pas tires (signalement 2026-10-23) : contrat a deux (emballage,
    trou), le camp DECLARANT a la main, tient de l'atout, et un adversaire en
    a encore (vraies mains : mesure). Pas toujours une erreur -> comparer a
    ExpertBot."""
    if not hasattr(game, "partner") or game.trump_suit is None or game.current_trick:
        return False
    if getattr(game, "pending_discards", None) or seat not in (game.declarer, game.partner):
        return False
    trump = game.trump_suit
    if not any(suit_of(c) == trump for c in game.hands[seat]):
        return False
    opp = [s for s in range(NUM_PLAYERS) if s not in (game.declarer, game.partner)]
    return any(suit_of(c) == trump for s in opp for c in game.hands[s])


def ruffed_by_defence(game, seat, card):
    """Le pli que `card` complete est-il coupe par la defense (entame hors
    atout, gagne par un atout d'un adversaire du camp declarant) ?"""
    if not hasattr(game, "partner") or game.trump_suit is None or len(game.current_trick) != 3:
        return False
    trick = game.current_trick + [(seat, card)]
    winner = trick_winner(trick, game.trump_suit)
    if winner in (game.declarer, game.partner) or suit_of(trick[0][1]) == game.trump_suit:
        return False
    return any(s == winner and suit_of(c) == game.trump_suit for s, c in trick)


def net_chooser(nets):
    def choose(game, seat):
        obs = torch.tensor(game.observation(seat), dtype=torch.float32)
        mask = torch.tensor(ai_play_mask(game, seat), dtype=torch.bool)
        return nets[game.role_of(seat)].act_greedy(obs, mask)
    return choose


def expert_chooser():
    from agents import expert_bot
    return expert_bot.choose_card


def play_greedy_and_count(game, choose, tally, family):
    """Joue la donne jusqu'au bout (`choose(game, seat)` : IA argmax comme
    Godot, ou ExpertBot) en comptant les erreurs d'equipe par (famille, camp)."""
    team = hasattr(game, "partner") and game.trump_suit is not None
    if team:
        _bump_coupe(tally.setdefault("_defense_coupe", {}), "total", 0)  # occasions = donnes a deux
    while not game.done:
        seat = game.turn
        role = game.role_of(seat)
        card = choose(game, seat)
        if atout_situation(game, seat):
            _bump_coupe(tally.setdefault("_atout", {}), "total", 0 if suit_of(card) == game.trump_suit else 1)
        if team and ruffed_by_defence(game, seat, card):
            tally["_defense_coupe"]["total"]["defausses"] += 1
        stealing = surcoupe_situation(game, seat)
        if stealing is not None:
            camp = "defense" if role == "defenseur" else "declarant"
            err = 1 if card in stealing else 0
            if family in MISERE_FAMILIES:
                # Defense de misere : le pli reste a la defense quoi qu'on joue,
                # prendre avec une grosse carte = s'en debarrasser (deblocage),
                # souvent le bon coup -> affiche a part, hors total (2026-10-24).
                _bump(tally, "%s/%s (hors total)" % (family, camp), 1, err)
            else:
                _bump(tally, "%s/%s" % (family, camp), 1, err)
                _bump(tally, "total/%s" % camp, 1, err)
                _bump(tally, "total", 1, err)
        coupe = coupe_situation(game, seat)
        if coupe is not None:
            winning, mate_master = coupe
            camp = "defense" if role == "defenseur" else "declarant"
            err = 0 if suit_of(card) == game.trump_suit else 1
            ct = tally.setdefault("_coupe", {})
            _bump_coupe(ct, "%s/%s" % (family, camp), err)
            _bump_coupe(ct, "total/%s" % ("coequipier_maitre" if mate_master else "coequipier_battu"), err)
            # Ne pas couper un coequipier deja maitre est souvent juste : seul le
            # cas coequipier battu est compte comme erreur (2026-10-24).
            if not mate_master:
                _bump_coupe(ct, "total", err)
        game.step(seat, card)


def run_real(choose, population, rng, deals, tally):
    from agents import evolution
    from engine.whisted import DeckChain
    from scripts.train import (MEASURE_WHISTED_RATIO, declarer_role_and_seats, dispatch_play_phase2,
                               run_bidding_phase)
    best = evolution.best_per_lineage(population)
    ind = best.get(evolution.EXPORTED_LINEAGES[0]) or next(iter(best.values()))
    bidder_for_seat = {seat: ind.net for seat in range(4)}
    chain = DeckChain(MEASURE_WHISTED_RATIO, rng)
    for _ in range(deals):
        hands, dealer, whisted = chain.deal(rng)
        bidding, _log = run_bidding_phase(bidder_for_seat, hands, dealer)
        if bidding.general_pass:
            chain.finish_unplayed(hands, dealer)
            continue
        contract = bidding.locked_contract
        role, _seats = declarer_role_and_seats(contract)
        game = dispatch_play_phase2(contract, hands, dealer, bidding.bid_summary, whisted)
        play_greedy_and_count(game, choose, tally, role)
        chain.finish_played(game, dealer, contract["type"])


def run_forced(choose, rng, deals, tally):
    from engine.cards import deal
    from scripts.train import DECLARER_FAMILIES, build_game
    for i in range(deals):
        family = DECLARER_FAMILIES[i % len(DECLARER_FAMILIES)]
        game, _seats = build_game(family, deal(rng), rng)
        play_greedy_and_count(game, choose, tally, family)


def rate(entry):
    return entry["surcoupes"] / entry["occasions"] if entry["occasions"] else None


def measure(nets, population, deals, total_episodes=0, elapsed=0.0, out_file=TEAM_ERRORS_FILE_DEFAULT,
            seed=TEAM_ERRORS_SEED, player="ia"):
    """Mesure complete (graine fixe -> comparable d'une fois sur l'autre),
    imprime et ajoute un record JSON Lines. Retourne le record. `player` :
    "ia" (reseaux greedy) ou "expert" (ExpertBot, memes donnes, reference)."""
    choose = net_chooser(nets) if player == "ia" else expert_chooser()
    record = {"total_donnes": total_episodes, "elapsed_s": round(elapsed, 1), "joueur": player,
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "deals": deals, "sets": {}}
    for name in ("reel", "force"):
        rng = random.Random(seed)
        tally = new_tally()
        if name == "reel":
            run_real(choose, population, rng, deals, tally)
        else:
            run_forced(choose, rng, deals, tally)
        coupe = tally.pop("_coupe", {})
        for key in ("_atout", "_defense_coupe"):
            v = tally.pop(key, {}).get("total")
            if v:
                record.setdefault(key[1:], {})[name] = dict(v, taux=round(v["defausses"] / max(1, v["occasions"]), 4))
        record["sets"][name] = {k: dict(v, taux=None if rate(v) is None else round(rate(v), 4))
                                for k, v in sorted(tally.items())}
        record.setdefault("renonce_non_coupee", {})[name] = {
            k: dict(v, taux=round(v["defausses"] / v["occasions"], 4)) for k, v in sorted(coupe.items())}
    print("\n=== Erreurs d'equipe : SURCOUPE du coequipier (IA greedy, %d donnes/jeu) ===" % deals, flush=True)
    for name, tally in record["sets"].items():
        print("  [%s]" % name, flush=True)
        for key, v in tally.items():
            if v["occasions"]:
                print("    %-28s %5.1f%%  (%d/%d)" % (key, 100 * v["taux"], v["surcoupes"], v["occasions"]),
                      flush=True)
    print("=== RENONCE NON COUPEE : coequipier entame dans notre renonce, on defausse au lieu de couper (total = coequipier battu) ===",
          flush=True)
    for name, tally in record["renonce_non_coupee"].items():
        print("  [%s]" % name, flush=True)
        for key, v in tally.items():
            print("    %-28s %5.1f%%  (%d/%d)" % (key, 100 * v["taux"], v["defausses"], v["occasions"]),
                  flush=True)
    print("=== ATOUTS PAS TIRES : camp declarant (a deux) a la main, adversaires encore a l'atout, entame hors atout ===",
          flush=True)
    for name, v in record.get("atout", {}).items():
        print("  [%s] %5.1f%%  (%d/%d)" % (name, 100 * v["taux"], v["defausses"], v["occasions"]), flush=True)
    print("=== COUPES DE LA DEFENSE : plis hors atout coupes par la defense, par donne a deux ===", flush=True)
    for name, v in record.get("defense_coupe", {}).items():
        print("  [%s] %.3f / donne  (%d plis / %d donnes)" % (name, v["taux"], v["defausses"], v["occasions"]), flush=True)
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("  (ajoute a %s)\n" % out_file, flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=TEAM_ERRORS_DEALS_DEFAULT)
    parser.add_argument("--seed", type=int, default=TEAM_ERRORS_SEED)
    parser.add_argument("--out-file", default=TEAM_ERRORS_FILE_DEFAULT)
    parser.add_argument("--players", default="ia", help="ia, expert ou ia,expert (memes donnes)")
    args = parser.parse_args()
    from scripts.phase3 import load_exported_ai
    nets, population, _generation = load_exported_ai()
    start = time.time()
    for player in args.players.split(","):
        print("##### joueur : %s" % player, flush=True)
        measure(nets, population, args.deals, out_file=args.out_file, seed=args.seed, player=player)
    print("Termine en %.0fs." % (time.time() - start), flush=True)


if __name__ == "__main__":
    main()
