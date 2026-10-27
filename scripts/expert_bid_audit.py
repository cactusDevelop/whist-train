"""Audit des ENCHERES de l'ExpertBot (signalement 2026-10-26 : « l'encherisseur
est defectueux, il propose des contrats impossibles pour lui et surencherit
abusivement » -- GATES-signalements.md, G1). Mesure seule, 4 sieges bot.

Par contrat verrouille (cle describe_contract) :
  n, reussite (jeu ExpertBot des deux camps), plis du camp,
  impossible = le camp declarant ne fait PAS le niveau meme a cartes ouvertes
               (DDS, contrats a atout hors trou),
  ecart moyen niveau - DDS, part des contrats montes par relance,
  relance fatale = DDS >= niveau d'avant la derniere relance mais < niveau.
Plus le nombre moyen de relances par donne et la repartition par type.

Usage (cloud, ci/run_expert_bid_audit.sh) : python -u scripts/expert_bid_audit.py --deals 20000
"""

import argparse
import json
import random
import sys
import time

sys.path.insert(0, ".")

from agents import expert_bot
from engine import dds
from engine.bidding import Bidding, bid_action_to_dict
from engine.trou import detect_trou
from engine.whisted import DeckChain
from scripts.eval_vs_random import declarer_successes
from scripts.expert_bot_match import bot_player, play_out
from scripts.train import MEASURE_WHISTED_RATIO, declarer_role_and_seats, describe_contract, dispatch_play_phase2


def bid_with_log(hands, dealer):
    """Enchere ExpertBot 4 sieges -> (bidding, nb de relances, niveau du
    contrat final avant sa derniere relance ou None)."""
    bidding = Bidding(dealer)
    trou = detect_trou(hands)
    if trou is not None:
        bidding.seed_trou(trou)
    raises = 0
    levels = {}  # index de contrat -> niveaux successifs
    while not bidding.finished:
        seat = bidding.turn
        ci = bidding.lie_contract_index[seat]
        before = bidding.contracts[ci].get("level") if ci != -1 else None
        choice = expert_bot.apply_bid(bidding, seat, hands[seat])
        is_raise = isinstance(choice, tuple) or bid_action_to_dict(choice)["action"] == "raise"
        if is_raise and ci != -1:
            raises += 1
            levels.setdefault(ci, []).append(before)
    pre = None
    final = bidding.locked_contract
    if final is not None:
        for ci, c in enumerate(bidding.contracts):
            if c is final and ci in levels:
                pre = levels[ci][-1]
    return bidding, raises, pre


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--dds-node-limit", type=int, default=20_000_000)
    parser.add_argument("--params", default="{}",
                        help="reglages JSON (expert_bot.set_bid_params) des 4 sieges, {} = defauts")
    parser.add_argument("--profile", default=None, help="profil de expert_bot.PROFILES (avant --params)")
    args = parser.parse_args()
    expert_bot.set_bid_params(dict(expert_bot.PROFILES[args.profile] if args.profile else {},
                                   **json.loads(args.params)))
    print("profil %s, reglages %s" % (args.profile, args.params), flush=True)

    rng = random.Random(args.seed)
    chain = DeckChain(MEASURE_WHISTED_RATIO, rng)
    rows = {}  # cle -> compteurs
    passes = total_raises = deals = 0
    t0 = time.time()
    while deals < args.deals:
        hands, dealer, whisted = chain.deal(rng)
        bidding, raises, pre = bid_with_log(hands, dealer)
        deals += 1
        total_raises += raises
        if bidding.general_pass:
            passes += 1
            chain.finish_unplayed(hands, dealer)
            continue
        contract = bidding.locked_contract
        _role, decl = declarer_role_and_seats(contract)
        keys = (describe_contract(contract), "_" + contract["type"], "_TOUT")
        game = dispatch_play_phase2(contract, hands, dealer, bidding.bid_summary, whisted)
        dd = None
        if contract["type"] in ("emballage", "solo") and dds.supported(game):
            dd = dds.solve(game, node_limit=args.dds_node_limit)
        play_out(game, [bot_player] * 4)
        oks = declarer_successes(contract, game, decl)
        tricks = sum(game.tricks_won[s] for s in decl)
        for k in keys:
            r = rows.setdefault(k, {"n": 0, "ok": 0, "tries": 0, "tricks": 0, "dd_n": 0, "imp": 0, "gap": 0,
                                    "raised": 0, "fatal": 0})
            r["n"] += 1
            r["ok"] += sum(oks)
            r["tries"] += len(oks)
            r["tricks"] += tricks
            r["raised"] += pre is not None
            if dd is not None:
                r["dd_n"] += 1
                r["imp"] += dd < contract["level"]
                r["gap"] += contract["level"] - dd
                r["fatal"] += pre is not None and pre <= dd < contract["level"]
        chain.finish_played(game, dealer, contract["type"])
        if deals % 2000 == 0:
            print("  %d donnes, %.0fs" % (deals, time.time() - t0), flush=True)

    print("\n=== Audit encheres ExpertBot : %d donnes, %d passes generales, %.2f relances/donne, %.0fs ==="
          % (deals, passes, total_raises / deals, time.time() - t0))
    print("%-16s %6s %6s %7s %6s %8s %8s %7s %7s" % ("contrat", "n", "part", "reussi", "plis",
                                                     "imposs.", "niv-DDS", "releve", "fatale"))
    played = deals - passes
    for k in sorted(rows, key=lambda k: (not k.startswith("_"), -rows[k]["n"])):
        r = rows[k]
        dn = r["dd_n"]
        print("%-16s %6d %5.1f%% %6.1f%% %6.2f %8s %8s %6.1f%% %7s" % (
            k.lstrip("_"), r["n"], 100 * r["n"] / played, 100 * r["ok"] / r["tries"], r["tricks"] / r["n"],
            "%.1f%%" % (100 * r["imp"] / dn) if dn else "-", "%+.2f" % (r["gap"] / dn) if dn else "-",
            100 * r["raised"] / r["n"], "%.1f%%" % (100 * r["fatal"] / dn) if dn else "-"))
    print("AUDIT_DONE")


if __name__ == "__main__":
    main()
