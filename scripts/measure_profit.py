"""Mesure de rentabilite seule, sans entrainement (whist-4-it 31/10, droite cout/force) :
scripts/joker_profit.profit_worker sur le champion courant, repartant de zero, sous l'empreinte
courante -> godot_export/joker_profit_stats.json et joker_profit_summary.md. Cloud uniquement.

Usage : python scripts/measure_profit.py --minutes 150 --workers 4
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, ".")

from engine import joker_obs as O
from engine import jokers as J
from scripts import joker_profit as P

BATCH = 400


def main():
    import torch
    from agents.network import PolicyValueNet
    from scripts.train_jokers import CHECKPOINT, net_to_numpy
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=150)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=93000000)
    args = parser.parse_args()
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    net = PolicyValueNet(O.OBS_DIM, J.ACTION_DIM)
    net.load_state_dict(ckpt["net"])
    net.eval()
    w = net_to_numpy(net)
    agg, done, end = P.Agg(), 0, time.time() + args.minutes * 60
    with Pool(args.workers) as pool:
        while time.time() < end:
            seeds = list(range(args.seed + done, args.seed + done + BATCH))
            for data in pool.map(P.profit_worker, [(seeds[i::args.workers], w) for i in range(args.workers)]):
                agg.merge(P.Agg(data))
            done += BATCH
            print("matchs %d, %.0f min restantes" % (done, (end - time.time()) / 60), flush=True)
    rows = P.table(agg)
    P.render({"iterations": [ckpt.get("iter", -1)], "profit_matches": done, "spec_hash": J.JOKER_SPEC_HASH}, rows)
    print("valeur d'un point d'energie :", P.energy_value(rows))
    keys = ("id", "cost", "suspicion", "rarity", "n", "gain", "se", "gain_source", "cast_rate", "target_cost",
            "target_cost_low", "target_cost_high", "depends_on_player", "recommended_cost", "note", "verdict")
    for r in rows:
        if r["side"] == "joueur":
            print(json.dumps({k: r.get(k) for k in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
