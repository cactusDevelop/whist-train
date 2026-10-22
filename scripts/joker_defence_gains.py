"""Gains des jokers des bots par role (defenseur / declarant) et par type de
contrat, pour regler EliteDecks.DEFENCE (whist-4-it). Memes rollouts que
l'entrainement (train_jokers.gen_worker, horizon donne, politique du
champion) : gain = Q(joker) - Q(aucun cast), ecart de points au joueur sur
la donne. joker_profit ne croise pas les dimensions, d'ou ce script.

Cloud seulement (whist-train, mode jokers_defence) :
  python scripts/joker_defence_gains.py --hands 2500 --workers 4
"""
import argparse
import collections
import math
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from agents.network import PolicyValueNet
from engine import joker_obs as O
from engine import jokers as J
from scripts import joker_profit as P
from scripts import train_jokers as T

FAMILIES = ("emballage", "solo", "misere", "trou")
ROLES = ("defenseur", "declarant")


def worker(args):
    seed, hands, weights = args
    rows = []
    orig = P.bot_records

    def recording(hand, seat, mask, q, chosen, config):
        recs = orig(hand, seat, mask, q, chosen, config)
        rows.extend((s["role"], s["famille"], jid, gain, picked) for jid, gain, s, picked in recs)
        return recs

    P.bot_records = recording
    T.gen_worker((seed, hands, weights, "hand"))
    return rows


def stat(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(var / n), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=T.CHECKPOINT)
    ap.add_argument("--hands", type=int, default=2500, help="donnes par worker")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261022)
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = PolicyValueNet(O.OBS_DIM, J.ACTION_DIM)
    net.load_state_dict(ck["net"])
    weights = T.net_to_numpy(net)
    print("champion iter %s, spec %s, %d donnes" % (ck.get("iter"), ck.get("spec_hash"), args.hands * args.workers))
    with mp.Pool(args.workers) as pool:
        parts = pool.map(worker, [(args.seed * 1000 + w, args.hands, weights) for w in range(args.workers)])
    occ = collections.defaultdict(list)
    took = collections.defaultdict(list)
    for part in parts:
        for role, fam, jid, gain, picked in part:
            occ[(role, fam, jid)].append(gain)
            if picked:
                took[(role, fam, jid)].append(gain)
    for role in ROLES:
        print("\n## %s" % role)
        print("| contrat | joker | gain toutes occasions | n | gain quand l'IA le prend | n |")
        print("|---|---|---|---|---|---|")
        for fam in FAMILIES:
            keys = sorted((k for k in occ if k[0] == role and k[1] == fam), key=lambda k: -stat(occ[k])[0])
            for k in keys:
                m, se, n = stat(occ[k])
                pm, pse, pn = stat(took.get(k, []))
                print("| %s | %s | %+.2f ± %.2f | %d | %+.2f ± %.2f | %d |" % (fam, k[2], m, se, n, pm, pse, pn))


if __name__ == "__main__":
    main()
