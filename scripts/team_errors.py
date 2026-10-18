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
from engine.cards import NUM_PLAYERS, trick_winner

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


def new_tally():
    return {}


def _bump(tally, key, occasions, errors):
    entry = tally.setdefault(key, {"occasions": 0, "surcoupes": 0})
    entry["occasions"] += occasions
    entry["surcoupes"] += errors


def play_greedy_and_count(game, nets, tally, family):
    """Joue la donne jusqu'au bout (argmax, comme Godot) en comptant les
    surcoupes par (famille, camp)."""
    while not game.done:
        seat = game.turn
        role = game.role_of(seat)
        obs = torch.tensor(game.observation(seat), dtype=torch.float32)
        mask = torch.tensor(ai_play_mask(game, seat), dtype=torch.bool)
        card = nets[role].act_greedy(obs, mask)
        stealing = surcoupe_situation(game, seat)
        if stealing is not None:
            camp = "defense" if role == "defenseur" else "declarant"
            err = 1 if card in stealing else 0
            _bump(tally, "%s/%s" % (family, camp), 1, err)
            _bump(tally, "total/%s" % camp, 1, err)
            _bump(tally, "total", 1, err)
        game.step(seat, card)


def run_real(nets, population, rng, deals, tally):
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
        play_greedy_and_count(game, nets, tally, role)
        chain.finish_played(game, dealer, contract["type"])


def run_forced(nets, rng, deals, tally):
    from engine.cards import deal
    from scripts.train import DECLARER_FAMILIES, build_game
    for i in range(deals):
        family = DECLARER_FAMILIES[i % len(DECLARER_FAMILIES)]
        game, _seats = build_game(family, deal(rng), rng)
        play_greedy_and_count(game, nets, tally, family)


def rate(entry):
    return entry["surcoupes"] / entry["occasions"] if entry["occasions"] else None


def measure(nets, population, deals, total_episodes=0, elapsed=0.0, out_file=TEAM_ERRORS_FILE_DEFAULT,
            seed=TEAM_ERRORS_SEED):
    """Mesure complete (graine fixe -> comparable d'une fois sur l'autre),
    imprime et ajoute un record JSON Lines. Retourne le record."""
    record = {"total_donnes": total_episodes, "elapsed_s": round(elapsed, 1),
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "deals": deals, "sets": {}}
    for name in ("reel", "force"):
        rng = random.Random(seed)
        tally = new_tally()
        if name == "reel":
            run_real(nets, population, rng, deals, tally)
        else:
            run_forced(nets, rng, deals, tally)
        record["sets"][name] = {k: dict(v, taux=None if rate(v) is None else round(rate(v), 4))
                                for k, v in sorted(tally.items())}
    print("\n=== Erreurs d'equipe : SURCOUPE du coequipier (IA greedy, %d donnes/jeu) ===" % deals, flush=True)
    for name, tally in record["sets"].items():
        print("  [%s]" % name, flush=True)
        for key, v in tally.items():
            if v["occasions"]:
                print("    %-28s %5.1f%%  (%d/%d)" % (key, 100 * v["taux"], v["surcoupes"], v["occasions"]),
                      flush=True)
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("  (ajoute a %s)\n" % out_file, flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=TEAM_ERRORS_DEALS_DEFAULT)
    parser.add_argument("--seed", type=int, default=TEAM_ERRORS_SEED)
    parser.add_argument("--out-file", default=TEAM_ERRORS_FILE_DEFAULT)
    args = parser.parse_args()
    from scripts.phase3 import load_exported_ai
    nets, population, _generation = load_exported_ai()
    start = time.time()
    measure(nets, population, args.deals, out_file=args.out_file, seed=args.seed)
    print("Termine en %.0fs." % (time.time() - start), flush=True)


if __name__ == "__main__":
    main()
