"""Diagnostic (MESURE seulement, cloud) : ou l'IA progresse-t-elle face a la
championne, contrat par contrat ET camp par camp ? (2026-10-26 : 12 portes,
emballage plat -0.05 pt/donne alors que trou/miseres progressent.)

La porte (champion_play) melange les reseaux dans un meme camp (sieges
{0,2} contre {1,3}, alors que le partenaire d'emballage est n'importe quel
siege). Ici, apres l'enchere (la meme pour les 3 parties, greedy), on joue
la donne 3 fois :
  - base    : championne aux 4 sieges ;
  - decl    : IA courante au camp DECLARANT, championne en defense ;
  - defense : championne au camp declarant, IA courante en defense.
Gain declarant = points du camp declarant (decl - base) ; gain defense =
points du camp de defense (defense - base), en vrais points, par donne.

Usage (cloud, whist-train mode ppo_camp_diag) :
  python scripts/champion_by_camp.py [--deals 20000]
"""

import argparse
import math
import random
import sys
import time

import torch

sys.path.insert(0, ".")

from agents import evolution
from agents.play_guards import ai_play_mask
from engine.bidding import Bidding
from engine.trou import detect_trou
from engine.whisted import DeckChain
from scripts.champion_play import CHAMPION_SEED, load_champion


def _play_out(contract, hands, dealer, bid_summary, whisted, nets_by_seat):
    from scripts.train import declarer_role_and_seats, dispatch_play_phase2, real_points_by_seat
    role, _declarers = declarer_role_and_seats(contract)
    game = dispatch_play_phase2(contract, [list(h) for h in hands], dealer, bid_summary, whisted)
    while not game.done:
        seat = game.turn
        obs = torch.tensor(game.observation(seat), dtype=torch.float32)
        mask = torch.tensor(ai_play_mask(game, seat), dtype=torch.bool)
        game.step(seat, nets_by_seat[seat][game.role_of(seat)].act_greedy(obs, mask))
    return real_points_by_seat(role, game, contract)


def stats(xs):
    n = len(xs)
    if not n:
        return 0.0, 0.0
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, n - 1))
    return m, sd / math.sqrt(n)


def main():
    from scripts.duel_expert import _bid
    from scripts.phase3 import load_exported_ai
    from scripts.train import declarer_role_and_seats
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=CHAMPION_SEED)
    args = parser.parse_args()
    new, population, _gen = load_exported_ai()
    champ, meta = load_champion()
    if champ is None:
        sys.exit("pas de championne")
    print("Championne depuis %s ; %d donnes" % (meta.get("since"), args.deals), flush=True)
    bidder = evolution.policy_of(evolution.best_per_lineage(population)[evolution.EXPORTED_LINEAGES[0]])
    rng = random.Random(args.seed)
    chain = DeckChain(0.5, rng)
    rows = {}
    t0 = time.time()
    with torch.no_grad():
        for i in range(args.deals):
            hands, dealer, whisted = chain.deal(rng)
            chain.finish_unplayed(hands, dealer)
            bidding = Bidding(dealer)
            trou = detect_trou(hands)
            if trou is not None:
                bidding.seed_trou(trou)
            _bid(bidding, hands, {0, 1, 2, 3}, bidder)
            if bidding.general_pass:
                continue
            contract = bidding.locked_contract
            role, decl = declarer_role_and_seats(contract)
            defense = [s for s in range(4) if s not in decl]
            play = lambda new_seats: _play_out(contract, hands, dealer, bidding.bid_summary, whisted,
                                               [new if s in new_seats else champ for s in range(4)])
            base = play(set())
            with_decl = play(set(decl))
            with_def = play(set(defense))
            r = rows.setdefault(role, {"decl": [], "def": []})
            r["decl"].append(sum(with_decl[s] for s in decl) - sum(base[s] for s in decl))
            r["def"].append(sum(with_def[s] for s in defense) - sum(base[s] for s in defense))
            if (i + 1) % 2000 == 0:
                print("  %d donnes (%.0f s)" % (i + 1, time.time() - t0), flush=True)
    print("\n=== IA courante - championne, vrais points du camp par donne (+- 2 erreurs types) ===")
    tot = {"decl": [], "def": []}
    for role, r in sorted(rows.items(), key=lambda kv: -len(kv[1]["decl"])):
        md, sd = stats(r["decl"])
        mf, sf = stats(r["def"])
        tot["decl"] += r["decl"]
        tot["def"] += r["def"]
        print("  %-14s %6d donnes   declarant %+.3f +- %.3f   defense %+.3f +- %.3f   (parties changees : decl %.0f %%, def %.0f %%)"
              % (role, len(r["decl"]), md, 2 * sd, mf, 2 * sf,
                 100 * sum(x != 0 for x in r["decl"]) / len(r["decl"]),
                 100 * sum(x != 0 for x in r["def"]) / len(r["def"])))
    md, sd = stats(tot["decl"])
    mf, sf = stats(tot["def"])
    print("  %-14s %6d donnes   declarant %+.3f +- %.3f   defense %+.3f +- %.3f" % ("TOUT", len(tot["decl"]), md, 2 * sd, mf, 2 * sf))


if __name__ == "__main__":
    main()
