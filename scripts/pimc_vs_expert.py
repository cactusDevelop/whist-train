"""PIMC (dds sur mondes echantillonnes) contre ExpertBot, en duplicate.
Mesure seule, cloud (conseil 09/11 : « verifier que la recherche bat
ExpertBot AVANT de la faire imiter par le reseau »).

Encheres ExpertBot reelles. Pour chaque donne a atout (emballage, solo,
trou ; dds.supported), la meme donne est jouee 3 fois :
  base : 4 ExpertBot ;
  def  : les defenseurs jouent PIMC (carte de Q max, Q = vrais points moyens
         sur N mondes compatibles avec ce que le siege sait, suite parfaite
         a cartes ouvertes : agents/rollouts.dds_values), le reste ExpertBot ;
  decl : idem pour le camp declarant.
Gain = points moyens par siege PIMC (variante - base), meme donne.
Sort aussi les situations ou PIMC s'ecarte d'ExpertBot (pistes de regles).

Usage : python -u scripts/pimc_vs_expert.py --minutes 60 --workers 4 --samples 8
"""

import argparse
import math
import multiprocessing as mp
import random
import sys
import time

sys.path.insert(0, ".")

from agents import expert_bot
from agents.rollouts import dds_values
from engine import dds
from engine.cards import NUM_PLAYERS
from engine.whisted import DeckChain
from scripts.expert_bot_match import run_bot_bidding
from scripts.expert_play_audit import card_txt, describe, situation
from scripts.train import MEASURE_WHISTED_RATIO, declarer_role_and_seats, dispatch_play_phase2, real_points_by_seat


def new_game(contract, hands, dealer, bidding, whisted):
    return dispatch_play_phase2(contract, [list(h) for h in hands], dealer, bidding.bid_summary, whisted)


def play(game, pimc_seats, score_fn, rng, samples, diffs):
    """Joue la donne ; `pimc_seats` jouent PIMC. -> nombre de decisions PIMC."""
    n = 0
    while not game.done:
        seat = game.turn
        card = expert_bot.choose_card(game, seat)
        trou_lead = getattr(game, "trou_pending", lambda: False)() and seat == game.partner
        if seat in pimc_seats and not trou_lead:  # l'entame du bouche-trou fixe le Trou (regle) : jamais PIMC
            legal = [c for c, ok in enumerate(game.legal_action_mask(seat)) if ok]
            if len(legal) > 1 and game.trump_suit is not None and dds.supported(game):
                q = dds_values(game, seat, legal, score_fn, rng, samples)
                if q is not None:
                    n += 1
                    best = max(q)
                    qe = q[legal.index(card)]
                    if best - qe > 1e-6:  # ExpertBot garde la main a egalite
                        pick = legal[q.index(best)]
                        diffs.append((situation(game, seat, card), best - qe,
                                      "%s | siege %d | pli %s | main %s | expert %s (%.1f) pimc %s (%.1f)" % (
                                          describe(game.public_contract()), seat,
                                          " ".join("%d:%s" % (s, card_txt(c)) for s, c in game.current_trick) or "-",
                                          " ".join(card_txt(c) for c in sorted(game.hands[seat])),
                                          card_txt(card), qe, card_txt(pick), best)))
                        card = pick
        game.step(seat, card)
    return n


def worker(args):
    seed, samples, deadline = args
    rng = random.Random(seed)
    play_rng = random.Random(seed + 1)
    chain = DeckChain(MEASURE_WHISTED_RATIO, rng)
    rows = []  # (variante, type contrat, gain moyen par siege PIMC, decisions)
    diffs = {"def": [], "decl": []}
    while time.time() < deadline:
        hands, dealer, whisted = chain.deal(rng)
        bidding = run_bot_bidding(hands, dealer)
        if bidding.general_pass:
            chain.finish_unplayed(hands, dealer)
            continue
        contract = bidding.locked_contract
        role, _decl = declarer_role_and_seats(contract)
        base = new_game(contract, hands, dealer, bidding, whisted)
        play(base, (), None, play_rng, samples, [])
        chain.finish_played(base, dealer, contract["type"])
        if contract["type"] not in ("emballage", "solo", "trou") or base.trump_suit is None:
            continue
        pts_base = real_points_by_seat(role, base, contract)
        side = set(dds.side_seats(base))

        def score_fn(g):
            return real_points_by_seat(role, g, contract)
        for name, seats in (("def", [s for s in range(NUM_PLAYERS) if s not in side]), ("decl", sorted(side))):
            g = new_game(contract, hands, dealer, bidding, whisted)
            n = play(g, set(seats), score_fn, play_rng, samples, diffs[name])
            pts = real_points_by_seat(role, g, contract)
            gain = sum(pts[s] - pts_base[s] for s in seats) / len(seats)
            rows.append((name, contract["type"], gain, n))
            if time.time() >= deadline:
                break
    return rows, diffs


def stats(xs):
    n = len(xs)
    if n == 0:
        return 0, 0.0, 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / max(1, n - 1)
    return n, m, math.sqrt(var / n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    t0 = time.time()
    deadline = t0 + args.minutes * 60
    with mp.Pool(args.workers) as pool:
        parts = pool.map(worker, [(args.seed * 1000 + w, args.samples, deadline) for w in range(args.workers)])
    rows = [r for p, _ in parts for r in p]
    diffs = {k: [d for _, ds in parts for d in ds[k]] for k in ("def", "decl")}
    print("=== PIMC (%d mondes) vs ExpertBot, duplicate, %.0f s ===" % (args.samples, time.time() - t0))
    print("%-6s %-10s %6s %9s %7s %8s" % ("camp", "contrat", "n", "gain/sieg", "se", "decis."))
    for name in ("def", "decl"):
        for typ in ("emballage", "solo", "trou", None):
            sel = [r for r in rows if r[0] == name and (typ is None or r[1] == typ)]
            n, m, se = stats([r[2] for r in sel])
            dec = sum(r[3] for r in sel)
            print("%-6s %-10s %6d %+9.3f %7.3f %8d" % (name, typ or "TOUT", n, m, se, dec))
    for name in ("def", "decl"):
        agg = {}
        for sit, gap, ex in diffs[name]:
            a = agg.setdefault(sit, {"n": 0, "gap": 0.0, "ex": []})
            a["n"] += 1
            a["gap"] += gap
            if len(a["ex"]) < 4 and gap >= 1.0:
                a["ex"].append(ex)
        print("\n=== %s : ecarts PIMC / ExpertBot par situation (Q gap cumule) ===" % name)
        for sit in sorted(agg, key=lambda k: -agg[k]["gap"])[:12]:
            a = agg[sit]
            print("%-42s n %5d  gap moyen %.2f  cumul %.1f" % (sit, a["n"], a["gap"] / a["n"], a["gap"]))
            for ex in a["ex"]:
                print("     " + ex)
    print("PIMC_VS_EXPERT_DONE")


if __name__ == "__main__":
    main()
