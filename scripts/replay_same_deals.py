"""Rejoue les MEMES donnes (memes mains, meme donneur, meme contrat, memes
sieges) que des replays existants de whist-4-it, avec l'IA ACTUELLE
(checkpoints_v2/, greedy comme Godot) -- pour voir si un comportement
remarque dans un ancien replay a change apres entrainement (2026-09-27 :
les 7 replays "mauvais jeu" du 26/09 rejoues apres l'imitation de
l'ExpertBot).

Usage : venv/Scripts/python.exe scripts/replay_same_deals.py --out-dir <dossier> <replay.json> [...]
Imprime, par replay, les plis du camp declarant avant -> apres.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, ".")

from agents import specialists
from agents.specialists import PLAY_ROLES
from engine.bidding import first_leader
from engine.emballage import new_emballage_deal
from engine.misere import new_misere_deal
from engine.solitaire import new_solitaire_deal
from engine.trou import detect_trou, new_trou_deal
from scripts.record_game import play_and_record
from scripts.train import declarer_role_and_seats, real_points_by_seat

SUIT_INDEX = {"♥": 0, "♦": 1, "♣": 2, "♠": 3}


class Greedy:
    def __init__(self, net):
        self.net = net

    def act(self, obs, mask):
        return self.net.act_greedy(obs, mask)


def internal_contract(event_contract):
    c = dict(event_contract)
    if "suit" in c:
        c["suit"] = SUIT_INDEX.get(c["suit"])  # "" = Trou pas encore fixe (regle du 2026-10-01)
    if c["type"] == "trou":
        c["royal"] = c.pop("is_royal", False)
    return c


def build(contract, hands, dealer):
    leader = first_leader(dealer, contract)
    t = contract["type"]
    if t == "solo":
        return new_solitaire_deal(hands, contract["declarer"], contract["suit"], contract["level"], leader)
    if t == "emballage":
        return new_emballage_deal(hands, contract["declarer"], contract["partner"], contract["suit"],
                                  contract["level"], leader)
    if t == "trou":
        # l'atout/niveau se refixent a l'entame rejouee (carte d'ancrage via detect_trou)
        return new_trou_deal(hands, contract["declarer"], contract["partner"], detect_trou(hands)["anchor_card"],
                             leader, royal=contract["royal"])
    return new_misere_deal(hands, contract["declarers"], contract["kind"], leader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--generated-by", default="whist-ai-rejeu-meme-donne")
    parser.add_argument("replays", nargs="+")
    args = parser.parse_args()
    nets = {r: Greedy(specialists.load_or_create(r)[0]) for r in PLAY_ROLES}
    os.makedirs(args.out_dir, exist_ok=True)
    for path in args.replays:
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        hand = old["hands"][0]
        event_contract = hand["events"][0]["contract"]
        contract = internal_contract(event_contract)
        role, decl = declarer_role_and_seats(contract)
        game = build(contract, [list(h) for h in hand["deal"]], hand["dealer"])
        game.set_public_context(dealer=hand["dealer"], bid_summary=None, whisted=False, level_known=True)
        with torch.no_grad():
            events = play_and_record(game, event_contract, nets)
        deltas = [int(round(v)) for v in real_points_by_seat(role, game, contract)]
        events.append({"type": "hand_finished", "tricks_won": list(game.tricks_won), "deltas": deltas,
                       "total_score": deltas})
        out = {"format_version": 1, "generated_by": args.generated_by,
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "hands": [{"dealer": hand["dealer"], "deal": hand["deal"], "events": events}]}
        name = os.path.basename(path)
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        before = hand["events"][-1]["tricks_won"]
        print("%-60s plis camp declarant %2d -> %2d ; points declarant %+d -> %+d" % (
            name, sum(before[s] for s in decl), sum(game.tricks_won[s] for s in decl),
            hand["events"][-1]["deltas"][min(decl)], deltas[min(decl)]))


if __name__ == "__main__":
    main()
