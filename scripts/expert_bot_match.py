"""Match ExpertBot (agents/expert_bot.py, regles de l'utilisateur) contre
l'IA et contre RandomBot, EN DUPLICATE : les memes mains et le meme contrat
(choisi par l'enchere du bot, 4 sieges bot) sont rejoues avec des joueurs
differents, pour que seule la qualite du jeu de la carte varie.

Variantes par contrat :
  bot/bot     : camp declarant bot   vs defense bot
  ia/bot      : camp declarant IA    vs defense bot     (l'IA declare-t-elle mieux que le bot ?)
  bot/ia      : camp declarant bot   vs defense IA      (l'IA defend-elle mieux que le bot ?)
  bot/random  : camp declarant bot   vs defense RandomBot
  ia/random   : camp declarant IA    vs defense RandomBot
Reussite = critere reel du contrat pour le camp declarant.

But : savoir OU le bot joue mieux que l'IA avant de s'en servir comme
professeur (on n'imite que la ou le professeur est meilleur).

Les reseaux sont lus depuis une COPIE des checkpoints (--ai-dir) : ne pas
lire checkpoints_v2 pendant que l'entrainement y sauvegarde (verrou
Windows 1224, voir agents/checkpoint_io.py).

Usage : venv/Scripts/python.exe scripts/expert_bot_match.py --ai-dir <copie> [--contracts 500] [--seed 1]
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
from agents import expert_bot, specialists
from agents.specialists import PLAY_ROLES
from engine.bidding import Bidding
from engine.trou import detect_trou
from engine.whisted import DeckChain

VARIANTS = ("bot/bot", "ia/bot", "bot/ia", "bot/random", "ia/random")


def run_bot_bidding(hands, dealer):
    bidding = Bidding(dealer)
    trou = detect_trou(hands)
    if trou is not None:
        bidding.seed_trou(trou)
    while not bidding.finished:
        seat = bidding.turn
        expert_bot.apply_bid(bidding, seat, hands[seat])  # leve si illegal
    return bidding


def bot_player(game, seat):
    return expert_bot.choose_card(game, seat)


def random_player(rng):
    def play(game, seat):
        return rng.choice([c for c, ok in enumerate(game.legal_action_mask(seat)) if ok])
    return play


def ai_player(nets):
    def play(game, seat):
        obs = torch.tensor(game.observation(seat), dtype=torch.float32)
        mask = torch.tensor(ai_play_mask(game, seat), dtype=torch.bool)
        return nets[game.role_of(seat)].act_greedy(obs, mask)
    return play


def play_out(game, players):
    while not game.done:
        seat = game.turn
        card = players[seat](game, seat)
        if not game.legal_action_mask(seat)[card]:
            raise AssertionError("carte illegale %d siege %d" % (card, seat))
        game.step(seat, card)
    return game


EXPERT_MATCH_FILE_DEFAULT = "expert_match.jsonl"
EXPERT_MATCH_CONTRACTS_DEFAULT = 300  # a chaque evaluation de train.py (~30 s)
EXPERT_MATCH_SEED = 20260927  # fixe : memes donnes a chaque mesure, seuls les reseaux changent
TRAIN_VARIANTS = ("bot/bot", "ia/bot", "bot/ia")


def run_match(nets, rng, contracts, variants=VARIANTS):
    """-> (tallies {(variante, contrat): [reussites, essais, plis camp]},
    contrats {cle: n}, passes generales). `nets` None : variantes sans IA
    ignorees."""
    from scripts.train import (MEASURE_WHISTED_RATIO, declarer_role_and_seats, describe_contract,
                               dispatch_play_phase2)
    from scripts.eval_vs_random import declarer_successes

    players = {
        "bot": {"decl": bot_player, "def": bot_player},
        "random": {"def": random_player(rng)},
    }
    if nets:
        players["ia"] = {"decl": ai_player(nets), "def": ai_player(nets)}
    variants = [v for v in variants if all(p in players for p in v.split("/"))]

    chain = DeckChain(MEASURE_WHISTED_RATIO, rng)
    tallies = {}
    kinds = {}
    passes = played = 0
    with torch.no_grad():
        while played < contracts:
            hands, dealer, whisted = chain.deal(rng)
            bidding = run_bot_bidding(hands, dealer)
            if bidding.general_pass:
                passes += 1
                chain.finish_unplayed(hands, dealer)
                continue
            contract = bidding.locked_contract
            _role, decl = declarer_role_and_seats(contract)
            key = describe_contract(contract)
            family = contract["type"]
            kinds[key] = kinds.get(key, 0) + 1
            first = None
            for variant in variants:
                d, f = variant.split("/")
                game = dispatch_play_phase2(contract, hands, dealer, bidding.bid_summary, whisted)
                seats = [players[d]["decl"] if s in decl else players[f]["def"] for s in range(4)]
                play_out(game, seats)
                oks = declarer_successes(contract, game, decl)
                for k in (key, "_" + family, "_TOUT"):
                    t = tallies.setdefault((variant, k), [0, 0, 0])
                    t[0] += sum(oks)
                    t[1] += len(oks)
                    t[2] += sum(game.tricks_won[s] for s in decl)
                first = first or game
            chain.finish_played(first, dealer, contract["type"])
            played += 1
    return tallies, kinds, passes, variants


def print_table(tallies, variants):
    print("Reussite du CAMP DECLARANT (plis moyens du camp)", flush=True)
    rows = sorted({k for (_v, k) in tallies})
    print("%-16s" % "contrat" + "".join("%18s" % v for v in variants))
    for k in rows:
        line = "%-16s" % k.lstrip("_")
        for v in variants:
            t = tallies.get((v, k))
            line += "%18s" % ("%5.1f%% (%4.1f) %4d" % (100 * t[0] / t[1], t[2] / t[1], t[1]) if t else "-")
        print(line, flush=True)


def measure_and_record(nets, total_episodes, elapsed, contracts=EXPERT_MATCH_CONTRACTS_DEFAULT,
                       out_file=EXPERT_MATCH_FILE_DEFAULT):
    """Mesure periodique de train.py (imitation de l'ExpertBot, 2026-09-27) :
    l'IA progresse-t-elle vers le bot ? Graine fixe. Ecrit une ligne JSON."""
    t0 = time.time()
    tallies, kinds, passes, variants = run_match(nets, random.Random(EXPERT_MATCH_SEED), contracts, TRAIN_VARIANTS)
    print("\n=== Duplicate vs ExpertBot (%d contrats, %.0fs) ===" % (contracts, time.time() - t0), flush=True)
    print_table({k: v for k, v in tallies.items() if k[1].startswith("_")}, variants)
    record = {"total_donnes": total_episodes, "elapsed_s": round(elapsed, 1),
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "contracts": contracts, "general_passes": passes,
              "results": {"%s|%s" % (v, k.lstrip("_")): {"success": round(t[0] / t[1], 4),
                                                          "tricks": round(t[2] / t[1], 3), "n": t[1]}
                          for (v, k), t in tallies.items() if k.startswith("_")}}
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("  (ajoute a %s)" % out_file, flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ai-dir", default=None, help="copie des checkpoints ; sans : bot seul")
    args = parser.parse_args()

    nets = None
    if args.ai_dir:
        specialists.CHECKPOINT_DIR = args.ai_dir
        missing = [r for r in PLAY_ROLES if not os.path.exists(specialists.checkpoint_path(r))]
        if missing:  # load_or_create creerait un reseau NEUF sans rien dire
            raise SystemExit("checkpoints absents de %s : %s" % (args.ai_dir, missing))
        nets = {role: specialists.load_or_create(role)[0] for role in PLAY_ROLES}
    start = time.time()
    tallies, kinds, passes, variants = run_match(nets, random.Random(args.seed), args.contracts)
    print("%d contrats joues (%d passes generales), %.0fs" % (args.contracts, passes, time.time() - start))
    print("Contrats choisis par le bot : " + ", ".join(
        "%s %d" % kv for kv in sorted(kinds.items(), key=lambda kv: -kv[1])))
    print()
    print_table(tallies, variants)


if __name__ == "__main__":
    main()
