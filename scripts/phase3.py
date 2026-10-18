"""Phase 3 (2026-09-24) : parties REELLES, sans aucune contrainte -- ce que
l'IA choisit vraiment. Demande explicite de l'utilisateur : "Les stats
devraient etre celles que l'ia choisit volontairement sans contrainte".

Pourquoi les stats Phase 2 ne suffisent pas : elles sont mesurees PENDANT
l'entrainement, sur toute la population d'enchere (40 individus dont ~32
enfants mutes chaque generation),
avec un jeu de carte ECHANTILLONNE (net.act, exploration PPO). Ce n'est
pas l'IA qui joue dans Godot.

Ici, seule l'IA EXPORTEE joue, exactement comme dans le jeu (niveau
"maitre", temperature 0) :
  - enchere : le meilleur individu de la lignee (evolution.best_per_lineage,
    le meme que scripts/export_to_godot.py), sur les 4 sieges ;
  - jeu de carte : les 9 specialistes, GREEDY (act_greedy = argmax, comme
    game/whist_ai_policy.gd) ;
  - donnes : suite continue 50% deck whiste (engine/whisted.py:DeckChain) ;
  - aucun contrat force, aucun faconnage, aucune mise a jour : mesure pure.

Une table par lignee exportee (4 sieges `encherisseur` depuis la
suppression de `encherisseur_amical` le 2026-09-27 ; la table "mixte"
encherisseur/amical a disparu avec elle), + une table "expert" (2026-09-28,
demande whist-4-it/utilisateur) : 4 ExpertBots (agents/expert_bot.py) pour
l'enchere ET le jeu de la carte, reference affichee a cote de l'IA. Toutes
les tables partent de la meme graine de donnes (donnes identiques tant que
la suite de decks whistes, qui depend du jeu, ne diverge pas). Chaque table
produit un bloc au MEME format que les records Phase 2 (contracts,
general_pass, bid_actions, families + by_level), lu par whist-4-it
(onglet "Stats IA", whist_ai_training_stats_phase3.jsonl).

Usage autonome : venv/Scripts/python.exe scripts/phase3.py [--deals N]
  [--stats-file X] [--no-godot-sync]
Appele aussi par scripts/train.py avant chaque export Godot.
"""

import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, ".")

from agents.play_guards import ai_play_mask
from agents import evolution, expert_bot, specialists
from agents.specialists import PLAY_ROLES
from engine.bidding import ACTION_RAISE, Bidding, bid_action_to_dict
from engine.trou import detect_trou
from engine.whisted import DeckChain
from scripts.train import (POPULATION_FILE_DEFAULT, WHISTED_RATIO_DEFAULT, declarer_role_and_seats,
                           describe_contract, dispatch_play_phase2, new_phase2_stats_accumulator,
                           phase2_family_blocks, real_points_by_seat, run_bidding_phase, update_phase2_stats)

STATS_FILE_PHASE3_DEFAULT = "training_stats_phase3.jsonl"
PHASE3_DEALS_DEFAULT = 1000  # donnes PAR TABLE (une par lignee exportee)


def play_out_greedy(game, nets):
    while not game.done:
        seat = game.turn
        net = nets[game.role_of(seat)]
        obs = torch.tensor(game.observation(seat), dtype=torch.float32)
        mask = torch.tensor(ai_play_mask(game, seat), dtype=torch.bool)
        game.step(seat, net.act_greedy(obs, mask))


EXPERT = "expert"  # bidder_for_seat / nets : l'ExpertBot a la place d'un reseau


def play_out_expert(game):
    while not game.done:
        seat = game.turn
        game.step(seat, expert_bot.choose_card(game, seat))


def run_expert_bidding(hands, dealer_index):
    """Comme train.run_bidding_phase (Trou auto-detecte d'abord), 4 ExpertBots."""
    bidding = Bidding(dealer_index)
    trou_info = detect_trou(hands)
    if trou_info is not None:
        bidding.seed_trou(trou_info)
    action_log = []
    while not bidding.finished:
        seat = bidding.turn
        choice = expert_bot.apply_bid(bidding, seat, hands[seat])
        action_log.append((seat, ACTION_RAISE if isinstance(choice, tuple) else choice))
    return bidding, action_log


def run_real_deal(nets, bidder_for_seat, rng, chain, stats):
    """Une donne complete sans contrainte : vraie enchere greedy -> jeu
    greedy. N'entraine rien, alimente seulement `stats` (format Phase 2).
    bidder_for_seat == nets == EXPERT : 4 ExpertBots (enchere et jeu)."""
    hands, dealer, whisted = chain.deal(rng)
    if bidder_for_seat == EXPERT:
        bidding, action_log = run_expert_bidding(hands, dealer)
    else:
        bidding, action_log = run_bidding_phase(bidder_for_seat, hands, dealer)
    for _seat, action_id in action_log:
        name = bid_action_to_dict(action_id)["action"]
        stats["bid_actions"][name] = stats["bid_actions"].get(name, 0) + 1
    stats["donnes"] += 1
    if bidding.general_pass:
        stats["general_pass"] += 1
        chain.finish_unplayed(hands, dealer)
        return
    contract = bidding.locked_contract
    role, declarer_seats = declarer_role_and_seats(contract)
    game = dispatch_play_phase2(contract, hands, dealer, bidding.bid_summary, whisted)
    if nets == EXPERT:
        play_out_expert(game)
    else:
        play_out_greedy(game, nets)
    chain.finish_played(game, dealer, contract["type"])
    key = describe_contract(contract)
    stats["contracts"][key] = stats["contracts"].get(key, 0) + 1
    update_phase2_stats(stats, role, game, declarer_seats, contract)
    # vrais points PAR JOUEUR declarant (demande whist-4-it 28/09) : somme sur
    # les sieges declarants / nombre de sieges declarants (misere comprise)
    pts = real_points_by_seat(role, game, contract)
    acc = stats.setdefault("contract_points", {}).setdefault(key, [0.0, 0])
    acc[0] += sum(pts[s] for s in declarer_seats)
    acc[1] += len(declarer_seats)


def run_phase3(nets, population, rng, deals_per_table, total_episodes, elapsed, stats_file, generation=None):
    """Joue `deals_per_table` donnes par table (une par lignee exportee),
    imprime et ajoute UN record a `stats_file`. Retourne le record."""
    best = evolution.best_per_lineage(population)
    record = {"phase": 3, "total_donnes": total_episodes, "elapsed_s": round(elapsed, 1),
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "bidder_generation": generation,
              "deals_per_table": deals_per_table, "tables": {}}
    print("\n=== PHASE 3 : parties reelles de l'IA exportee (greedy, sans contrainte), %d donnes/table ===" % (
        deals_per_table), flush=True)
    tables = []  # (cle, libelle, individu par siege)
    for lineage in evolution.EXPORTED_LINEAGES:
        if lineage not in best:
            print("  %-20s absente de la population -- table ignoree" % lineage, flush=True)
            continue
        tables.append((lineage, "4 x %s" % lineage, {seat: best[lineage] for seat in range(4)}))
    tables.append((EXPERT, "4 x ExpertBot", None))
    deal_seed = rng.randrange(2 ** 31)
    for key, label, individual_for_seat in tables:
        stats = new_phase2_stats_accumulator()
        table_rng = random.Random(deal_seed)
        chain = DeckChain(WHISTED_RATIO_DEFAULT, table_rng)
        if individual_for_seat is None:
            table_nets = bidder_for_seat = EXPERT
            ids, who = [], "regles"
        else:
            table_nets = nets
            bidder_for_seat = {seat: evolution.policy_of(ind) for seat, ind in individual_for_seat.items()}
            ids = sorted({ind.id for ind in individual_for_seat.values()})
            who = "individu(s) " + ", ".join("#%d" % i for i in ids)
        for _ in range(deals_per_table):
            run_real_deal(table_nets, bidder_for_seat, table_rng, chain, stats)
        print("  -- table %s (%s) : redonnes=%d, contrats=%s" % (
            label, who, stats["general_pass"], dict(sorted(stats["contracts"].items()))), flush=True)
        record["tables"][key] = {
            "bidder_id": (ids[0] if len(ids) == 1 else ids) if ids else None, "phase2_donnes": stats["donnes"],
            "general_pass": stats["general_pass"],
            "contracts": dict(sorted(stats["contracts"].items())),
            "bid_actions": dict(sorted(stats["bid_actions"].items())),
            "families": phase2_family_blocks(stats),
            "contract_points": {k: round(total / n, 2)
                                for k, (total, n) in sorted(stats.get("contract_points", {}).items())}}
    with open(stats_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("  (ajoute a %s)\n" % stats_file, flush=True)
    return record


def load_exported_ai(population_file=POPULATION_FILE_DEFAULT):
    """Charge les checkpoints tels qu'exportes (roles de jeu + population)."""
    nets = {}
    for role in PLAY_ROLES:
        net, _optimizer = specialists.load_or_create(role)
        net.eval()
        nets[role] = net
    population, generation = evolution.load_population(population_file)
    return nets, population, generation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=PHASE3_DEALS_DEFAULT, help="donnes par table")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stats-file", default=STATS_FILE_PHASE3_DEFAULT)
    parser.add_argument("--population-file", default=POPULATION_FILE_DEFAULT)
    parser.add_argument("--no-godot-sync", action="store_true",
                        help="ne pas copier le fichier de stats Phase 3 dans whist-4-it/game/")
    args = parser.parse_args()
    nets, population, generation = load_exported_ai(args.population_file)
    start = time.time()
    run_phase3(nets, population, random.Random(args.seed), args.deals, 0, 0.0, args.stats_file, generation)
    print("Phase 3 terminee en %.0fs." % (time.time() - start), flush=True)
    if not args.no_godot_sync:
        from scripts.godot_sync import publish_stats_file
        publish_stats_file(args.stats_file, "phase3")


if __name__ == "__main__":
    main()
