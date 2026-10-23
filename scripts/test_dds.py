"""Tests + chronos du solveur double-mort (engine/dds.c). Cloud seulement
(whist-train, mode dds_test) :
  python scripts/test_dds.py [--positions 400] [--full 30]

1. Exactitude : sur des fins de donne aleatoires (1 a 3 plis restants, pli
   en cours entame ou non, camps voisins / en face / 1 contre 3, avec ou
   sans atout), dds.solve == minimax Python exhaustif.
2. Cas evident : le camp qui tient tous les atouts maitres et les plus
   longues couleurs.
3. Chronos : donnes entieres (13 plis), mi-donne, et une decision complete
   du professeur (rollouts.dds_values, 16 mondes) sur de vraies donnes.
"""
import argparse
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import dds
from engine.cards import NUM_PLAYERS, deal, legal_plays, suit_of, trick_winner
from engine.team_trump import TeamTrumpDeal


def make_game(rng, side_kind):
    hands = deal(rng)
    declarer = rng.randrange(NUM_PLAYERS)
    if side_kind == "voisins":
        partner = (declarer + 1) % NUM_PLAYERS
    elif side_kind == "face":
        partner = (declarer + 2) % NUM_PLAYERS
    else:
        partner = declarer  # 1 contre 3 (comme un solo)
    trump = None if rng.random() < 0.2 else rng.randrange(4)
    return TeamTrumpDeal(hands, declarer, partner, trump, 8, rng.randrange(NUM_PLAYERS), "d", "p", "x")


def random_play(g, rng, n):
    for _ in range(n):
        seat = g.turn
        g.step(seat, rng.choice([c for c, ok in enumerate(g.legal_action_mask(seat)) if ok]))


def brute(hands, trump, side, leader, trick):
    """Minimax exhaustif (alpha-beta exact) : plis restants du camp `side`."""
    turn = (leader + len(trick)) % NUM_PLAYERS
    if not trick and not hands[leader]:
        return 0
    led = suit_of(trick[0][1]) if trick else None
    best = None
    maximizing = turn in side
    for c in legal_plays(hands[turn], led):
        hands[turn].remove(c)
        t = trick + [(turn, c)]
        if len(t) == NUM_PLAYERS:
            w = trick_winner(t, trump)
            v = (1 if w in side else 0) + brute(hands, trump, side, w, [])
        else:
            v = brute(hands, trump, side, leader, t)
        hands[turn].append(c)
        hands[turn].sort()
        if best is None or (v > best if maximizing else v < best):
            best = v
    return best


def test_exact(rng, positions):
    bad = 0
    kinds = ("voisins", "face", "seul")
    for i in range(positions):
        g = make_game(rng, kinds[i % 3])
        left = rng.choice((1, 2, 3))
        random_play(g, rng, (13 - left) * 4 + rng.randrange(4))
        side = set(dds.side_seats(g))
        trick = list(g.current_trick)
        leader = trick[0][0] if trick else g.turn
        want = brute([list(h) for h in g.hands], g.trump_suit, side, leader, trick)
        got = dds.solve(g)
        if got != want:
            bad += 1
            if bad <= 5:
                print("ECART", kinds[i % 3], "atout", g.trump_suit, "camp", side, "pli", trick, "mains", g.hands,
                      "dds", got, "minimax", want, flush=True)
    print("exactitude : %d/%d positions identiques" % (positions - bad, positions), flush=True)
    return bad == 0


def test_obvious():
    # camp 0+2 : tous les piques (atout) et tous les coeurs -> 13 plis
    hands = [list(range(39, 52)), list(range(13, 26)), list(range(0, 13)), list(range(26, 39))]
    g = TeamTrumpDeal(hands, 0, 2, 3, 8, 1, "d", "p", "x")
    got = dds.solve(g)
    print("cas evident (13 atouts chez le declarant) : %d plis (attendu 13)" % got, flush=True)
    return got == 13


def chrono_solve(rng, n, played_tricks):
    times, nodes = [], []
    kinds = ("voisins", "face", "seul")
    for i in range(n):
        g = make_game(rng, kinds[i % 3])
        random_play(g, rng, played_tricks * 4)
        n0 = dds.nodes()
        t0 = time.time()
        dds.solve(g)
        times.append(time.time() - t0)
        nodes.append(dds.nodes() - n0)
    print("solve apres %d plis : moyenne %.3fs, mediane %.3fs, max %.3fs, noeuds moyens %d" % (
        played_tricks, statistics.mean(times), statistics.median(times), max(times), statistics.mean(nodes)),
        flush=True)


def chrono_teacher(rng, n, samples):
    from agents import rollouts
    from engine.emballage import final_scores
    times = []
    for i in range(n):
        g = make_game(rng, ("voisins", "face")[i % 2])
        if g.trump_suit is None:
            g.trump_suit = 0
        random_play(g, rng, rng.randrange(0, 24))
        seat = g.turn
        legal = [c for c, ok in enumerate(g.legal_action_mask(seat)) if ok]
        t0 = time.time()
        rollouts.dds_values(g, seat, legal, final_scores, rng, samples)
        times.append(time.time() - t0)
    print("decision du professeur (%d mondes, plis 0-6) : moyenne %.2fs, max %.2fs" % (
        samples, statistics.mean(times), max(times)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=400)
    ap.add_argument("--full", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    ok = test_exact(rng, args.positions) & test_obvious()
    for k in (6, 3, 0):
        chrono_solve(rng, args.full if k else min(args.full, 10), k)
    chrono_teacher(rng, 20, 16)
    print("DDS_OK" if ok else "DDS_ECHEC", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
