"""Mesure ciblee des 4 jokers de misere (65a7/1f49, demande whist-4-it 28/10) :
le test de rentabilite (scripts/joker_profit.profit_worker, contrefactuel
apparie, joueur style expert) sur des matchs d'elites ou CHAQUE donne est une
misere imposee -- declarant = le joueur (Defausse Discrete, Meche Courte dans
son deck) ou un bot (La Tournee, Cadeau Empoisonne). Mesure seule, cloud.

Usage : python scripts/misere_jokers.py --matches 1200 --workers 4
"""

import argparse
import json
import os
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, ".")

from engine import jokers as J
from scripts import joker_profit as P

ATTACK = ("defausse_discrete", "meche_courte")
DEFENCE = ("la_tournee", "cadeau_empoisonne")
MISERE = ATTACK + DEFENCE
OUT = "misere_jokers.json"


def misere_overrides(cfg, rng):
    """Moitie misere du joueur (jokers d'attaque x2 dans son deck), moitie
    misere d'un bot (jokers de defense x2)."""
    who = "player" if rng.random() < 0.5 else "bot"
    extra = ATTACK if who == "player" else DEFENCE
    act = rng.choice((1, 2, 3))
    return {"kind": "elite", "act": act, "eye": J.ELITE_EYE, "boss": "", "force_misere": who,
            "deck": [j for j in cfg["deck"] if J.base_id(j) not in MISERE] + [j for j in extra for _ in range(2)]}


def worker(args):
    seeds, weights = args
    return P.profit_worker((seeds, weights, misere_overrides))


def main():
    import torch
    from agents.network import PolicyValueNet
    from engine import joker_obs as O
    from scripts.train_jokers import CHECKPOINT, net_to_numpy
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=91000000)
    args = parser.parse_args()
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    net = PolicyValueNet(O.OBS_DIM, J.ACTION_DIM)
    net.load_state_dict(ckpt["net"])
    net.eval()
    w = net_to_numpy(net)
    seeds = list(range(args.seed, args.seed + args.matches))
    agg = P.Agg()
    with Pool(args.workers) as pool:
        for data in pool.map(worker, [(seeds[i::args.workers], w) for i in range(args.workers)]):
            agg.merge(P.Agg(data))
    rows = [r for r in P.table(agg) if r["side"] == "joueur" and r["id"] in MISERE]
    keys = ("id", "cost", "rarity", "n", "gain", "se", "gain_per_energy", "win_rate", "loss_rate", "note",
            "recommended_cost", "verdict", "contract_rule", "strong_in", "best", "worst")
    out = [{**{k: r.get(k) for k in keys}, "by_contrat": r.get("by", {}).get("contrat"), "by_regle": r.get("by", {}).get("regle")}
           for r in rows]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"spec_hash": J.JOKER_SPEC_HASH, "matches": args.matches, "rows": out}, f, ensure_ascii=False, indent=1)
    for r in out:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
