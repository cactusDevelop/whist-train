"""A/B de vitesse du solveur double-mort dans UN meme job (cloud seulement,
mode dds_bench de whist-train) : memes mondes, memes cartes, 4 variantes
  compilation -O3 / -O3 -march=native  x  premiere sonde normale / a la
  valeur de la carte precedente (dds_set_guess, comme agents/rollouts.py).
Chaque variante resout la meme liste de (position, cartes legales) ; les
valeurs doivent etre identiques (sinon exception), on compare noeuds et temps.
  python scripts/bench_dds.py [--decisions 40] [--tricks 1,2,3]
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.rollouts import clone_with_hands, determinize
from engine import dds
from scripts.test_dds import make_game, random_play

NODE_LIMIT = 15_000_000  # comme rollouts.DDS_WORLD_NODES, par carte ici


def build_cases(rng, decisions, tricks):
    cases = []
    kinds = ("voisins", "face", "seul")
    while len(cases) < decisions:
        g = make_game(rng, kinds[len(cases) % 3])
        if g.trump_suit is None:
            continue
        random_play(g, rng, rng.choice(tricks) * 4 + rng.randrange(4))
        seat = g.turn
        legal = [c for c, ok in enumerate(g.legal_action_mask(seat)) if ok]
        if len(legal) < 2:
            continue
        cases.append((g, seat, legal, determinize(g, seat, rng)))
    return cases


def run(cases, use_guess):
    t0 = time.time()
    n0 = dds.nodes()
    values = []
    for game, seat, legal, hands in cases:
        prev, first = None, True
        for card in legal:
            g = clone_with_hands(game, hands)
            g.step(seat, card)
            if g.done:
                values.append(None)
                continue
            r = dds.solve(g, new_gen=first, node_limit=NODE_LIMIT, guess=prev if use_guess else None)
            first = False
            values.append(r)
            prev = r
    return values, time.time() - t0, dds.nodes() - n0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=int, default=40)
    parser.add_argument("--tricks", default="1,2,3")
    parser.add_argument("--seed", type=int, default=1102)
    args = parser.parse_args()
    cases = build_cases(random.Random(args.seed), args.decisions, [int(x) for x in args.tricks.split(",")])
    ref = None
    for flags in (["-O3", "-shared", "-fPIC"], ["-O3", "-march=native", "-shared", "-fPIC"]):
        dds._CFLAGS, dds._LIB = flags, None
        for use_guess in (False, True):
            values, secs, nodes = run(cases, use_guess)
            if ref is None:
                ref = values
            elif any(a is not None and b is not None and a != b for a, b in zip(values, ref)):
                raise AssertionError("valeurs differentes : %s guess=%s" % (" ".join(flags), use_guess))
            print("%-28s guess=%-5s %7.2fs %12d noeuds  (%d hors budget)" % (
                " ".join(flags[:-2]), use_guess, secs, nodes, sum(v is None for v in values)), flush=True)
    print("DDS_BENCH_OK", flush=True)


if __name__ == "__main__":
    main()
