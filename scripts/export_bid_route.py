"""Export du routage d'enchere vers Godot (15e conseil, etape 2, 2026-11-14 ;
voir agents/bid_route.py). A lancer sur le cloud, etat ppo_free (le meme que
free_match --bid-cells, donc le MEME encherisseur que celui mesure).

1) Encherisseur mesure (scripts/train_free.load_bidder : lignee exportee +
   regle d'abondance + garde-fous) compare aux poids publies dans le jeu
   (--ingame, copie de whist-4-it/game) : IDENTIQUE ou DIFFERENT.
2) Export de son reseau -> <out>/whist_ai_weights_data_v2_encherisseur.gd.
3) Cas de parite du routage -> <out>/bid_route_cases.json : donnes completes
   (sans Trou) ou les 4 sieges encherissent "routage" (ExpertBot, reseau dans
   les cellules BID_ROUTE), poids float16 tels que publies. Chaque decision :
   [siege, expert, reseau, jouee] (id d'action, ou ["raise_kind", misere] /
   ["raise_to", niveau]). Rejoues par whist-4-it tests/bid_route_parity_runner.gd.

Usage (cloud) : ci/run_bid_route_export.sh
"""

import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, ".")

from agents import bid_route, evolution, expert_bot
from agents.abondance_rule import RuledBidder
from engine.bidding import Bidding
from engine.cards import deal
from engine.trou import detect_trou
from scripts import train_free
from scripts.export_to_godot import _export_net, fp16_rounded
from scripts.godot_weights import load_role

SEED = 20261114
ROUTED_DEALS = 400   # donnes avec au moins une decision routee
PLAIN_DEALS = 100    # donnes sans decision routee
MAX_DEALS = 200000


def encode(choice):
    if isinstance(choice, tuple):
        return ["raise_kind" if choice[0] == expert_bot.RAISE_KIND else "raise_to", choice[1]]
    return int(choice)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingame", default="godot_weights_ingame")
    ap.add_argument("--out", default="godot_export_bid")
    args = ap.parse_args()

    bidder = train_free.load_bidder()
    net = bidder.net
    published = fp16_rounded(net)
    ingame = load_role(args.ingame, "encherisseur", evolution.OBS_DIM)
    diff = max((published.state_dict()[k] - v).abs().max().item() for k, v in ingame.state_dict().items()
               if not k.startswith("value_head"))
    print("ENCHERISSEUR mesure vs jeu (%s) : %s (ecart max %.3g)" % (
        args.ingame, "IDENTIQUE" if diff == 0 else "DIFFERENT", diff), flush=True)

    path, size = _export_net("encherisseur", net, evolution.OBS_DIM, evolution.ACTION_DIM, args.out)
    print("EXPORT %s (%d octets de poids)" % (path, size), flush=True)

    ruled = RuledBidder(published)
    rng = random.Random(SEED)
    kept, routed_deals, plain_deals, cells = [], 0, 0, {}
    with torch.no_grad():
        for _ in range(MAX_DEALS):
            if routed_deals >= ROUTED_DEALS and plain_deals >= PLAIN_DEALS:
                break
            hands = deal(rng)
            if detect_trou(hands) is not None:
                continue
            dealer = rng.randrange(4)
            b = Bidding(dealer)
            steps, routed, rare = [], 0, False
            while not b.finished:
                seat = b.turn
                e = expert_bot.choose_bid(b, seat, hands[seat])
                n = ruled.decide(b, seat, hands[seat])
                played = bid_route.route(e, n)
                if played is n and bid_route.bid_label(n) != bid_route.bid_label(e):
                    routed += 1
                    cell = (bid_route.bid_label(e, False), bid_route.bid_label(n, False))
                    rare = rare or cells.get(cell, 0) < 15
                    cells[cell] = cells.get(cell, 0) + 1
                steps.append([seat, encode(e), encode(n), encode(played)])
                assert expert_bot.apply_choice(b, seat, played), (seat, played)
            if routed and (routed_deals < ROUTED_DEALS or rare):
                routed_deals += 1
            elif not routed and plain_deals < PLAIN_DEALS:
                plain_deals += 1
            else:
                continue
            kept.append({"dealer": dealer, "hands": [sorted(h) for h in hands], "bids": steps,
                         "general_pass": b.general_pass})
    out = os.path.join(args.out, "bid_route_cases.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"source": "whist-ai scripts/export_bid_route.py", "seed": SEED,
                   "route": sorted("%s>%s" % c for c in bid_route.BID_ROUTE), "deals": kept}, f, separators=(",", ":"))
    print("CAS %d donnes (%d routees, %d sans routage) -> %s" % (len(kept), routed_deals, plain_deals, out), flush=True)
    for cell, k in sorted(cells.items(), key=lambda kv: -kv[1]):
        print("  cellule %-16s > %-16s : %d" % (cell[0], cell[1], k))
    missing = [c for c in bid_route.BID_ROUTE if c not in cells]
    print("cellules jamais vues dans les cas : %s" % (missing or "aucune"), flush=True)


if __name__ == "__main__":
    main()
