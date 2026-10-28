"""Audit du JEU DE LA CARTE de l'ExpertBot (IA de tous les sieges en jeu
depuis whist-4-it b8435f4) -- plaintes 2026-10-26 : « l'IA joue mal au jeu
de la carte ». Mesure seule, cloud.

Encheres ExpertBot reelles, contrats a atout (emballage, solo, trou). A chaque
decision a plusieurs cartes legales, Q de chaque carte = vrais points moyens
sur N mondes compatibles avec ce que le siege sait (agents/rollouts.dds_values :
suite jouee parfaitement a cartes ouvertes). Erreur = la carte de l'ExpertBot
perd >= SEUIL points face a la meilleure. Tableau par situation + exemples.

Usage : python -u scripts/expert_play_audit.py --deals 1500 --workers 4
"""

import argparse
import multiprocessing as mp
import random
import sys
import time

sys.path.insert(0, ".")

from agents import expert_bot
from agents.rollouts import dds_values
from engine import dds
from engine.cards import NUM_PLAYERS, rank_of, suit_of, trick_winner
from engine.whisted import DeckChain
from scripts.expert_bot_match import run_bot_bidding
from scripts.team_errors import teammates
from scripts.train import MEASURE_WHISTED_RATIO, declarer_role_and_seats, dispatch_play_phase2, real_points_by_seat

SUITS = "CKTP"  # coeur, carreau, trefle, pique (engine/cards.py)
RANKS = "23456789XVDRA"


def card_txt(c):
    return RANKS[rank_of(c)] + SUITS[suit_of(c)]


def situation(game, seat, card):
    """Categorie de la decision (position + ce que la carte fait)."""
    trick = game.current_trick
    mates = teammates(game, seat)
    side = "decl" if seat in set(dds.side_seats(game)) else "def"
    trump = game.trump_suit
    if not trick:
        kind = "entame atout" if suit_of(card) == trump else "entame"
        return "%s/%s" % (side, kind)
    led = suit_of(trick[0][1])
    pos = len(trick) + 1
    winner = trick_winner(trick, trump)
    mate_wins = winner in mates
    void = not any(suit_of(c) == led for c in game.hands[seat])
    wins = trick_winner(trick + [(seat, card)], trump) == seat
    if void:
        can_ruff = any(suit_of(c) == trump and trick_winner(trick + [(seat, c)], trump) == seat
                       for c in game.hands[seat]) if trump is not None and led != trump else False
        if can_ruff:
            act = "coupe" if suit_of(card) == trump else "defausse"
            return "%s/renonce %de/%s %s" % (side, pos, "coeq gagne" if mate_wins else "adv gagne", act)
        return "%s/renonce %de sans coupe" % (side, pos)
    if mate_wins:
        return "%s/fournit %de coeq gagne %s" % (side, pos, "PREND" if wins else "laisse")
    return "%s/fournit %de adv gagne %s" % (side, pos, "prend" if wins else "petite")


def worker(args):
    seed, n_deals, samples, threshold = args
    rng = random.Random(seed)
    chain = DeckChain(MEASURE_WHISTED_RATIO, rng)
    rows = []  # (situation, perte, exemple ou None)
    deals = 0
    while deals < n_deals:
        hands, dealer, whisted = chain.deal(rng)
        bidding = run_bot_bidding(hands, dealer)
        if bidding.general_pass:
            chain.finish_unplayed(hands, dealer)
            continue
        contract = bidding.locked_contract
        role, _decl = declarer_role_and_seats(contract)
        game = dispatch_play_phase2(contract, [list(h) for h in hands], dealer, bidding.bid_summary, whisted)
        if contract["type"] not in ("emballage", "solo", "trou"):
            from scripts.expert_bot_match import bot_player, play_out
            play_out(game, [bot_player] * 4)
            chain.finish_played(game, dealer, contract["type"])
            continue
        deals += 1

        def score_fn(g):
            return real_points_by_seat(role, g, contract)
        while not game.done:
            seat = game.turn
            card = expert_bot.choose_card(game, seat)
            legal = [c for c, ok in enumerate(game.legal_action_mask(seat)) if ok]
            if len(legal) > 1 and game.trump_suit is not None and dds.supported(game):
                q = dds_values(game, seat, legal, score_fn, rng, samples)
                if q is not None:
                    best = max(q)
                    loss = best - q[legal.index(card)]
                    sit = situation(game, seat, card)
                    ex = None
                    if loss >= threshold:
                        bc = legal[q.index(best)]
                        ex = "%s | %s | pli %s | main %s | joue %s (%.1f) au lieu de %s (%.1f)" % (
                            describe(contract), "siege %d" % seat,
                            " ".join("%d:%s" % (s, card_txt(c)) for s, c in game.current_trick) or "-",
                            " ".join(card_txt(c) for c in sorted(game.hands[seat])),
                            card_txt(card), q[legal.index(card)], card_txt(bc), best)
                    rows.append((sit, loss, ex))
            game.step(seat, card)
        chain.finish_played(game, dealer, contract["type"])
    return rows


def describe(c):
    s = c["type"]
    if "level" in c:
        s += "-%s" % c["level"]
    if c.get("suit") is not None:
        s += " atout %s" % SUITS[c["suit"]] if isinstance(c["suit"], int) else " atout %s" % c["suit"]
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=2.0, help="perte (points) comptee comme erreur")
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    t0 = time.time()
    per = args.deals // args.workers
    with mp.Pool(args.workers) as pool:
        parts = pool.map(worker, [(args.seed * 1000 + w, per, args.samples, args.threshold) for w in range(args.workers)])
    rows = [r for p in parts for r in p]
    agg = {}
    for sit, loss, ex in rows:
        for k in (sit, "_" + sit.split("/")[0], "_TOUT"):
            a = agg.setdefault(k, {"n": 0, "err": 0, "loss": 0.0, "ex": []})
            a["n"] += 1
            a["loss"] += loss
            if ex is not None:
                a["err"] += 1
                if len(a["ex"]) < 6 and not k.startswith("_"):
                    a["ex"].append(ex)
    print("=== Audit jeu de la carte ExpertBot : %d donnes a atout, %d decisions, %d mondes, seuil %.1f pts, %.0fs ==="
          % (per * args.workers, len(rows), args.samples, args.threshold, time.time() - t0))
    print("%-42s %7s %7s %8s %9s" % ("situation", "n", "err%", "perte", "perte/don"))
    n_deals = per * args.workers
    for k in sorted(agg, key=lambda k: (not k.startswith("_"), -agg[k]["loss"])):
        a = agg[k]
        print("%-42s %7d %6.1f%% %8.2f %9.3f" % (k.lstrip("_"), a["n"], 100 * a["err"] / a["n"],
                                               a["loss"] / a["n"], a["loss"] / n_deals))
    print("\n=== Exemples (situations les plus couteuses) ===")
    for k in sorted((k for k in agg if not k.startswith("_")), key=lambda k: -agg[k]["loss"])[:8]:
        print("## %s" % k)
        for ex in agg[k]["ex"]:
            print("   " + ex)
    print("PLAY_AUDIT_DONE")


if __name__ == "__main__":
    main()
