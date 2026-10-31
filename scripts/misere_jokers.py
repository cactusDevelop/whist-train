"""Mesure ciblee des jokers de misere (65a7/1f49, demande whist-4-it 28/10 ;
8f41 : + Pot-de-Vin, et --set atout pour Main de Fer / Brouillard) :
le test de rentabilite (scripts/joker_profit.profit_worker, contrefactuel
apparie, joueur style expert) sur des matchs d'elites ou CHAQUE donne est une
misere imposee -- declarant = le joueur (Defausse Discrete, Meche Courte dans
son deck) ou un bot (La Tournee, Cadeau Empoisonne). Mesure seule, cloud.

Usage : python scripts/misere_jokers.py --matches 1200 --workers 4 [--set atout] [--no-eye]
--no-eye : Oeil des elites coupe (sanction -20 pts), pour separer le cout du Soupcon de l'effet.
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

ATTACK = ("defausse_discrete", "meche_courte", "pot_de_vin")
DEFENCE = ("la_tournee", "cadeau_empoisonne")
MISERE = ATTACK + DEFENCE
ATOUT = ("main_de_fer", "brouillard_de_la_senne")
OEIL = ("horloge_arretee", "brouillard_de_la_senne", "silence_du_cercle", "treizieme_carte", "manche_truquee")
OUT = "misere_jokers.json"


def misere_overrides(cfg, rng):
    """Moitie misere du joueur (jokers d'attaque x2 dans son deck), moitie
    misere d'un bot (jokers de defense x2)."""
    who = "player" if rng.random() < 0.5 else "bot"
    extra = ATTACK if who == "player" else DEFENCE
    act = rng.choice((1, 2, 3))
    return {"kind": "elite", "act": act, "eye": J.ELITE_EYE, "boss": "", "force_misere": who,
            "deck": [j for j in cfg["deck"] if J.base_id(j) not in MISERE] + [j for j in extra for _ in range(2)]}


def atout_overrides(cfg, rng):
    """8f41 : donnes normales (encheres), Main de Fer et Brouillard x2 dans le deck du joueur."""
    act = rng.choice((1, 2, 3))
    return {"kind": "elite", "act": act, "eye": J.ELITE_EYE, "boss": "",
            "deck": [j for j in cfg["deck"] if J.base_id(j) not in ATOUT] + [j for j in ATOUT for _ in range(2)]}


def oeil_overrides(cfg, rng):
    """7567 (Synergies) : les 5 negatifs a Soupcon, x2 dans le deck, avec / sans l'Oeil."""
    act = rng.choice((1, 2, 3))
    return {"kind": "elite", "act": act, "eye": J.ELITE_EYE, "boss": "",
            "deck": [j for j in cfg["deck"] if J.base_id(j) not in OEIL] + [j for j in OEIL for _ in range(2)]}


SETS = {"misere": (misere_overrides, MISERE), "atout": (atout_overrides, ATOUT), "oeil": (oeil_overrides, OEIL)}


def _overrides(name, no_eye):
    fn = SETS[name][0]
    return lambda cfg, rng: dict(fn(cfg, rng), **({"eye": ""} if no_eye else {}))


def worker(args):
    seeds, weights, name, no_eye = args
    return P.profit_worker((seeds, weights, _overrides(name, no_eye)))


def main():
    import torch
    from agents.network import PolicyValueNet
    from engine import joker_obs as O
    from scripts.train_jokers import CHECKPOINT, net_to_numpy
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=91000000)
    parser.add_argument("--set", choices=sorted(SETS), default="misere")
    parser.add_argument("--no-eye", action="store_true")
    args = parser.parse_args()
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    net = PolicyValueNet(O.OBS_DIM, J.ACTION_DIM)
    net.load_state_dict(ckpt["net"])
    net.eval()
    w = net_to_numpy(net)
    seeds = list(range(args.seed, args.seed + args.matches))
    agg = P.Agg()
    with Pool(args.workers) as pool:
        for data in pool.map(worker, [(seeds[i::args.workers], w, args.set, args.no_eye) for i in range(args.workers)]):
            agg.merge(P.Agg(data))
    rows = [r for r in P.table(agg) if r["side"] == "joueur" and r["id"] in SETS[args.set][1]]
    keys = ("id", "cost", "rarity", "n", "gain", "se", "gain_per_energy", "win_rate", "loss_rate", "note",
            "recommended_cost", "verdict", "contract_rule", "strong_in", "best", "worst")
    out = [{**{k: r.get(k) for k in keys}, "by_contrat": r.get("by", {}).get("contrat"), "by_regle": r.get("by", {}).get("regle")}
           for r in rows]
    out_file = OUT if args.set == "misere" and not args.no_eye else "%s_jokers%s.json" % (args.set, "_sans_oeil" if args.no_eye else "")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"spec_hash": J.JOKER_SPEC_HASH, "matches": args.matches, "set": args.set,
                   "eye": "" if args.no_eye else J.ELITE_EYE, "rows": out}, f, ensure_ascii=False, indent=1)
    for r in out:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
