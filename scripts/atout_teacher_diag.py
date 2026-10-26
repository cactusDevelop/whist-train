"""Diagnostic (MESURE seulement, cloud) : le professeur DDS apprend-il a
l'IA a ne PAS tirer les atouts ? (2026-10-26 : "atouts pas tires" passe de
49-55 % a 60 % depuis le professeur DDS.)

Soupcon : le professeur moyenne des mondes resolus A CARTES OUVERTES ; dans
chaque monde il sait ou sont les atouts adverses, donc tirer les atouts lui
parait souvent inutile (biais de "fusion de strategies" du PIMC).

A chaque occasion "atouts pas tires" (team_errors.atout_situation, camp
declarant a la main, entame possible a l'atout ET hors atout) sur des
donnes jouees par l'IA greedy, on compare la meilleure entame A L'ATOUT et
la meilleure HORS ATOUT (en plis du camp declarant) selon :
  - pimc   : le professeur tel qu'a l'entrainement (dds_values, mondes tires
             du point de vue du joueur) -- c'est lui qui choisit la cible ;
  - ouvert : DDS sur les VRAIES mains (borne cartes ouvertes) ;
  - ia     : vraies mains, la carte puis la suite jouee par l'IA greedy aux
             4 sieges (ce qui se passe vraiment en jeu) ;
  - expert : idem, suite jouee par l'ExpertBot.
Ecart > 0 = l'atout rapporte plus. Si pimc < 0 alors que ia/expert > 0, le
professeur enseigne le mauvais coup.

Usage (cloud, whist-train mode ppo_atout_diag) :
  python scripts/atout_teacher_diag.py [--positions 400] [--samples 8] [--workers 4]
"""

import argparse
import multiprocessing
import random
import sys
import time

sys.path.insert(0, ".")

from agents import rollouts
from engine import dds
from engine.cards import suit_of
from scripts.team_errors import atout_situation, net_chooser


def side_tricks(game):
    return sum(game.tricks_won[s] for s in set(dds.side_seats(game)))


def best_by_class(legal, q, trump):
    t = [v for c, v in zip(legal, q) if suit_of(c) == trump]
    n = [v for c, v in zip(legal, q) if suit_of(c) != trump]
    return max(t), max(n), legal[q.index(max(q))]


def class_cards(legal, q, trump):
    """Meilleure carte (selon q) a l'atout et hors atout."""
    best = {}
    for c, v in zip(legal, q):
        k = suit_of(c) == trump
        if k not in best or v > best[k][1]:
            best[k] = (c, v)
    return best[True][0], best[False][0]


def play_from(game, seat, card, finish):
    g = rollouts.clone_with_hands(game, [list(h) for h in game.hands])
    g.step(seat, card)
    finish(g)
    return side_tricks(g)


def analyse(game, seat, ai_card, nets, rng, samples, out):
    legal = [c for c, ok in enumerate(game.legal_action_mask(seat)) if ok]
    trump = game.trump_suit
    if not any(suit_of(c) == trump for c in legal) or all(suit_of(c) == trump for c in legal):
        return
    score_fn = lambda g: [side_tricks(g)] * 4
    q = rollouts.dds_values(game, seat, legal, score_fn, rng, samples)
    if q is None:
        return
    pimc_t, pimc_n, teacher_card = best_by_class(legal, q, trump)
    # Vraies mains, cartes ouvertes
    open_q = []
    for card in legal:
        g = rollouts.clone_with_hands(game, [list(h) for h in game.hands])
        g.step(seat, card)
        open_q.append(side_tricks(g) + (0 if g.done else dds.solve(g)))
    open_t, open_n, _ = best_by_class(legal, open_q, trump)
    # Suites reelles : la meilleure carte de chaque classe selon le professeur
    card_t, card_n = class_cards(legal, q, trump)
    ia = lambda g: rollouts.play_out_greedy_batched([g], nets)
    ex = lambda g: rollouts.play_out_expert([g])
    out.append({
        "role": game.role_of(seat),
        "trick": len(game.play_log) // 4,
        "ia_trump": suit_of(ai_card) == trump,
        "teacher_trump": suit_of(teacher_card) == trump,
        "pimc": pimc_t - pimc_n,
        "open": open_t - open_n,
        "ia": play_from(game, seat, card_t, ia) - play_from(game, seat, card_n, ia),
        "expert": play_from(game, seat, card_t, ex) - play_from(game, seat, card_n, ex),
    })


def collect(nets, population, rng, positions, samples):
    from agents import evolution
    from engine.whisted import DeckChain
    from scripts.train import (MEASURE_WHISTED_RATIO, build_game, declarer_role_and_seats,
                               dispatch_play_phase2, run_bidding_phase)
    from engine.cards import deal
    choose = net_chooser(nets)
    best = evolution.best_per_lineage(population)
    ind = best.get(evolution.EXPORTED_LINEAGES[0]) or next(iter(best.values()))
    bidders = {s: ind.net for s in range(4)}
    chain = DeckChain(MEASURE_WHISTED_RATIO, rng)
    rows = {"reel": [], "force": []}
    deals = 0
    t0 = time.time()
    while min(len(r) for r in rows.values()) < positions // 2:
        deals += 1
        name = "reel" if deals % 2 else "force"
        if len(rows[name]) >= positions // 2:
            continue
        if name == "reel":
            hands, dealer, whisted = chain.deal(rng)
            bidding, _log = run_bidding_phase(bidders, hands, dealer)
            if bidding.general_pass:
                chain.finish_unplayed(hands, dealer)
                continue
            contract = bidding.locked_contract
            game = dispatch_play_phase2(contract, hands, dealer, bidding.bid_summary, whisted)
        else:
            game, _seats = build_game("emballage", deal(rng), rng)
        while not game.done:
            seat = game.turn
            card = choose(game, seat)
            if atout_situation(game, seat) and dds.supported(game) and len(rows[name]) < positions // 2:
                analyse(game, seat, card, nets, rng, samples, rows[name])
            game.step(seat, card)
        if name == "reel":
            chain.finish_played(game, dealer, contract["type"])
        if deals % 20 == 0:
            print("  %d donnes, %d/%d positions (%.0f s)" % (deals, len(rows["reel"]), len(rows["force"]),
                                                          time.time() - t0), flush=True)
    return rows


def report(name, rows):
    n = len(rows)
    if not n:
        print("[%s] aucune position" % name)
        return
    mean = lambda k, rs=rows: sum(r[k] for r in rs) / max(1, len(rs))
    print("\n=== [%s] %d positions 'atouts pas tires' ===" % (name, n))
    print("  IA entame atout          : %5.1f %%" % (100 * mean("ia_trump")))
    print("  professeur entame atout  : %5.1f %%" % (100 * mean("teacher_trump")))
    print("  ecart atout - hors atout (plis du camp declarant), moyenne :")
    for k in ("pimc", "open", "ia", "expert"):
        pos = sum(r[k] > 0.25 for r in rows) / n
        neg = sum(r[k] < -0.25 for r in rows) / n
        print("    %-7s %+.3f   (atout mieux %4.1f %% / hors atout mieux %4.1f %%)" % (k, mean(k), 100 * pos, 100 * neg))
    dis = [r for r in rows if not r["teacher_trump"]]
    if dis:
        print("  quand le professeur dit HORS ATOUT (%d) : suite IA %+.3f, suite expert %+.3f, ouvert %+.3f"
              % (len(dis), mean("ia", dis), mean("expert", dis), mean("open", dis)))
    for role in sorted({r["role"] for r in rows}):
        rs = [r for r in rows if r["role"] == role]
        print("  %-12s %3d pos : prof atout %4.1f %%, pimc %+.2f, ia %+.2f, expert %+.2f"
              % (role, len(rs), 100 * mean("teacher_trump", rs), mean("pimc", rs), mean("ia", rs), mean("expert", rs)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=400)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    from scripts.phase3 import load_exported_ai
    nets, population, _generation = load_exported_ai()
    if args.workers > 1:
        rollouts.set_pool(multiprocessing.Pool(args.workers), args.workers)
    rows = collect(nets, population, random.Random(args.seed), args.positions, args.samples)
    for name, rs in rows.items():
        report(name, rs)
    report("total", rows["reel"] + rows["force"])


if __name__ == "__main__":
    main()
