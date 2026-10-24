"""Porte « IA contre IA championne » du jeu de la carte, en DUPLICATE et en
VRAIS POINTS (2026-10-24, demande utilisateur : savoir si l'IA progresse,
jamais en se comparant a ExpertBot).

Chaque donne est jouee deux fois avec les memes mains : les 4 sieges
encherissent avec le MEME encherisseur (meilleur individu de la lignee
exportee, greedy) -> meme contrat ; puis l'IA courante joue la carte aux
sieges 0/2 et la championne aux sieges 1/3, puis l'inverse. Seul le jeu de
la carte differe : ecart = points (courante) - points (championne) par
donne, moyenne des deux positions.

Porte : si l'ecart depasse 2 erreurs-types, la courante devient la
championne (checkpoints_v2/champion_play/<role>.pt). Pas de retour arriere
de l'entrainement : la porte sert a MESURER et a garder la meilleure IA.

Usage : venv/Scripts/python.exe scripts/champion_play.py [--deals 400]
Appele aussi par scripts/train.py a chaque evaluation (champion_play.jsonl).
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, ".")

from agents import evolution, specialists
from agents.network import PolicyValueNet
from agents.play_guards import ai_play_mask
from agents.specialists import PLAY_ROLES
from engine.bidding import Bidding
from engine.trou import detect_trou
from engine.whisted import DeckChain

CHAMPION_FILE_DEFAULT = "champion_play.jsonl"
CHAMPION_DEALS_DEFAULT = 400  # donnes (x2 en duplicate) a chaque evaluation
CHAMPION_SEED = 20261024  # fixe : memes donnes a chaque mesure
GATE_Z = 2.0
NEW_SEATS = ({0, 2}, {1, 3})


def champion_dir():
    return os.path.join(specialists.CHECKPOINT_DIR, "champion_play")


def _path(role):
    return os.path.join(champion_dir(), "%s.pt" % role)


def save_champion(nets, meta):
    os.makedirs(champion_dir(), exist_ok=True)
    for role in PLAY_ROLES:
        specialists.safe_save({"net": nets[role].state_dict()}, _path(role))
    with open(os.path.join(champion_dir(), "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)


def load_champion():
    """-> (reseaux de la championne, meta) ou (None, None) si absente."""
    if not all(os.path.exists(_path(role)) for role in PLAY_ROLES):
        return None, None
    nets = {}
    for role in PLAY_ROLES:
        obs_dim, action_dim = specialists.IMPLEMENTED[role]
        net = PolicyValueNet(obs_dim, action_dim)
        net.load_state_dict(torch.load(_path(role), map_location="cpu")["net"])
        net.eval()
        nets[role] = net
    meta_path = os.path.join(champion_dir(), "meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    return nets, meta


def ensure_champion(nets, total_episodes=0):
    """Premier lancement : la championne = l'IA telle qu'elle demarre."""
    champ, _meta = load_champion()
    if champ is None:
        save_champion(nets, {"since": time.strftime("%Y-%m-%dT%H:%M:%S"), "total_donnes": total_episodes,
                             "origine": "IA au demarrage (aucune championne)"})
        print("Championne du jeu de la carte creee : %s (IA au demarrage)." % champion_dir(), flush=True)


def _play(hands, dealer, whisted, bidder, new_seats, new_nets, champ_nets):
    from scripts.train import declarer_role_and_seats, dispatch_play_phase2, real_points_by_seat
    bidding = Bidding(dealer)
    trou = detect_trou(hands)
    if trou is not None:
        bidding.seed_trou(trou)
    from scripts.duel_expert import _bid
    _bid(bidding, hands, {0, 1, 2, 3}, bidder)
    if bidding.general_pass:
        return 0.0, None
    contract = bidding.locked_contract
    role, _declarers = declarer_role_and_seats(contract)
    game = dispatch_play_phase2(contract, [list(h) for h in hands], dealer, bidding.bid_summary, whisted)
    while not game.done:
        seat = game.turn
        nets = new_nets if seat in new_seats else champ_nets
        obs = torch.tensor(game.observation(seat), dtype=torch.float32)
        mask = torch.tensor(ai_play_mask(game, seat), dtype=torch.bool)
        game.step(seat, nets[game.role_of(seat)].act_greedy(obs, mask))
    pts = real_points_by_seat(role, game, contract)
    return sum(pts[s] for s in new_seats) - sum(pts[s] for s in range(4) if s not in new_seats), role


def run_match(new_nets, champ_nets, population, deals, seed=CHAMPION_SEED):
    best = evolution.best_per_lineage(population)[evolution.EXPORTED_LINEAGES[0]]
    bidder = evolution.policy_of(best)
    rng = random.Random(seed)
    chain = DeckChain(0.5, rng)
    diffs = []
    by_role = {}
    with torch.no_grad():
        for _ in range(deals):
            hands, dealer, whisted = chain.deal(rng)
            chain.finish_unplayed(hands, dealer)
            diff, role = 0.0, None
            for seats in NEW_SEATS:
                d, role = _play(hands, dealer, whisted, bidder, seats, new_nets, champ_nets)
                diff += d
            if role is None:
                continue
            diffs.append(diff / 2)
            r = by_role.setdefault(role, [0, 0.0])
            r[0] += 1
            r[1] += diff / 2
    n = len(diffs)
    mean = sum(diffs) / max(1, n)
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / max(1, n - 1))
    return {"deals": n, "new_minus_champion": round(mean, 3), "std_error": round(sd / math.sqrt(max(1, n)), 3),
            "by_contract": {k: {"donnes": v[0], "ecart": round(v[1] / v[0], 3)} for k, v in sorted(by_role.items())}}


def measure_and_gate(nets, population, total_episodes, elapsed, deals=CHAMPION_DEALS_DEFAULT,
                     out_file=CHAMPION_FILE_DEFAULT):
    t0 = time.time()
    champ, meta = load_champion()
    if champ is None:
        ensure_champion(nets, total_episodes)
        return None
    current = {role: copy.deepcopy(nets[role]).eval() for role in PLAY_ROLES}
    r = run_match(current, champ, population, deals)
    accepted = r["new_minus_champion"] - GATE_Z * r["std_error"] > 0
    r.update({"total_donnes": total_episodes, "elapsed_s": round(elapsed, 1), "accepted": accepted,
              "champion_since": meta.get("since"), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
    if accepted:
        save_champion(nets, {"since": r["timestamp"], "total_donnes": total_episodes,
                             "ecart_vs_ancienne": r["new_minus_champion"], "std_error": r["std_error"]})
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(r) + "\n")
    print("=== Porte IA vs championne (duplicate, %d donnes x2, %.0fs) : ecart %+.2f +- %.2f pts/donne -> %s ===" % (
        r["deals"], time.time() - t0, r["new_minus_champion"], GATE_Z * r["std_error"],
        "NOUVELLE CHAMPIONNE" if accepted else "championne gardee (depuis %s)" % meta.get("since")), flush=True)
    print("    par contrat : %s" % r["by_contract"], flush=True)
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=CHAMPION_DEALS_DEFAULT)
    args = parser.parse_args()
    nets = {role: specialists.load_or_create(role)[0] for role in PLAY_ROLES}
    population, _gen = evolution.load_population(os.path.join(specialists.CHECKPOINT_DIR, "encherisseur_population.pt"))
    measure_and_gate(nets, population, 0, 0.0, args.deals)


if __name__ == "__main__":
    main()
