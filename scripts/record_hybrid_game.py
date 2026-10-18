"""Replay d'une partie ou les 4 sieges ENCHERISSENT avec le bot hybride
(lignee encherisseur_expert : regles de l'ExpertBot + reseau, voir
agents/hybrid_bidder.py) -- meilleur individu de la population courante --
puis jouent la carte avec les specialistes actuels (checkpoints_v2/,
greedy comme Godot). Contrairement aux autres replays, les evenements
"bid" sont enregistres (format game/replay_format.gd de whist-4-it).

Les donnes a Trou (detection automatique, evenements a part) et les
passes generales sont redonnees sans etre enregistrees, comme le fait le
format pour la passe generale.

Usage : venv/Scripts/python.exe scripts/record_hybrid_game.py --out <fichier.json> [--hands 4] [--seed N]
"""

import argparse
import json
import random
import sys
import time

import torch

sys.path.insert(0, ".")

from agents import evolution, specialists
from agents.specialists import PLAY_ROLES
from engine.bidding import bid_action_to_dict
from engine.cards import deal
from engine.trou import detect_trou
from scripts import train
from scripts.record_game import SUIT_SYMBOLS, play_and_record
from scripts.replay_same_deals import Greedy


def with_symbol(d):
    d = dict(d)
    if isinstance(d.get("suit"), int):
        d["suit"] = SUIT_SYMBOLS[d["suit"]]
    return d


def contract_event(contract):
    c = {k: v for k, v in with_symbol(contract).items() if k in ("type", "suit", "level", "declarer", "partner", "kind", "declarers")}
    if "declarers" in c:
        c["declarers"] = list(c["declarers"])
    return c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--hands", type=int, default=4)
    parser.add_argument("--seed", type=int, default=int(time.time()))
    args = parser.parse_args()

    population, generation = evolution.load_population(train.POPULATION_FILE_DEFAULT)
    best = max((i for i in population if i.lineage == evolution.HYBRID_LINEAGE), key=lambda i: i.snapshot_fitness())
    bidder = evolution.policy_of(best)
    print("bot hybride : individu #%d (generation %d, derniere fitness %+.2f), graine %d" % (
        best.id, generation, best.snapshot_fitness(), args.seed))
    nets = {r: Greedy(specialists.load_or_create(r)[0]) for r in PLAY_ROLES}
    rng = random.Random(args.seed)

    recorded, totals, dealer = [], [0, 0, 0, 0], rng.randrange(4)
    while len(recorded) < args.hands:
        hands = deal(rng)
        if detect_trou(hands) is not None:
            continue
        bidding, action_log = train.run_bidding_phase({s: bidder for s in range(4)}, hands, dealer)
        if bidding.general_pass:
            continue
        contract = bidding.locked_contract
        role, decl = train.declarer_role_and_seats(contract)
        game = train.dispatch_play_phase2(contract, hands, dealer, bidding.bid_summary, False)
        with torch.no_grad():
            play_events = play_and_record(game, contract_event(contract), nets)
        bid_events = [{"type": "bid", "seat": s, "action": with_symbol(bid_action_to_dict(a))} for s, a in action_log]
        deltas = [int(round(v)) for v in train.real_points_by_seat(role, game, contract)]
        totals = [t + d for t, d in zip(totals, deltas)]
        play_events.append({"type": "hand_finished", "tricks_won": list(game.tricks_won), "deltas": deltas,
                            "total_score": list(totals)})
        recorded.append({"dealer": dealer, "deal": [list(h) for h in hands], "events": bid_events + play_events})
        print("donne %d : %s, plis camp declarant %d, points declarant %+d" % (
            len(recorded), train.describe_contract(contract), sum(game.tricks_won[s] for s in decl), deltas[min(decl)]))
        dealer = (dealer + 1) % 4

    out = {"format_version": 1, "generated_by": "whist-ai-bot-hybride",
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "hands": recorded}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("ecrit : %s" % args.out)


if __name__ == "__main__":
    main()
