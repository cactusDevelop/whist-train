"""Porte « IA contre IA championne » du jeu de la carte, en VRAIS POINTS
(2026-10-24, demande utilisateur : savoir si l'IA progresse, jamais en se
comparant a ExpertBot).

Chaque donne : les 4 sieges encherissent avec le MEME encherisseur (meilleur
individu de la lignee exportee, greedy) -> meme contrat, puis la carte est
jouee 3 fois (camp_rows) : championne partout, IA courante au camp
declarant, IA courante en defense. Gain declarant / defense = points du camp
concerne moins ceux de la partie de base, par donne ; ecart total = somme.

Porte (par camp depuis 2026-10-26 : l'ancienne porte melangee cachait un
declarant en recul de -0.35 pt/donne en emballage, compense par la defense) :
ecart total > 2 erreurs-types ET aucun camp en recul de plus de 2
erreurs-types -> la courante devient la championne
(checkpoints_v2/champion_play/<role>.pt). Pas de retour arriere de
l'entrainement : la porte sert a MESURER et a garder la meilleure IA.

Usage : venv/Scripts/python.exe scripts/champion_play.py [--deals 1000]
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
CHAMPION_DEALS_DEFAULT = 1000  # donnes (x3 : base / declarant / defense) a chaque evaluation
CHAMPION_SEED = 20261024  # fixe : memes donnes a chaque mesure
GATE_Z = 2.0


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


def _play_out(contract, hands, dealer, bid_summary, whisted, nets_by_seat):
    from scripts.train import declarer_role_and_seats, dispatch_play_phase2, real_points_by_seat
    role, _declarers = declarer_role_and_seats(contract)
    game = dispatch_play_phase2(contract, [list(h) for h in hands], dealer, bid_summary, whisted)
    while not game.done:
        seat = game.turn
        obs = torch.tensor(game.observation(seat), dtype=torch.float32)
        mask = torch.tensor(ai_play_mask(game, seat), dtype=torch.bool)
        game.step(seat, nets_by_seat[seat][game.role_of(seat)].act_greedy(obs, mask))
    return real_points_by_seat(role, game, contract)


def mean_se(xs):
    n = len(xs)
    if not n:
        return 0.0, 0.0
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, n - 1))
    return m, sd / math.sqrt(n)


def camp_rows(new_nets, champ_nets, population, deals, seed=CHAMPION_SEED, progress=False):
    """-> {role: {"decl": [gain du camp declarant], "def": [gain de la defense]}}, vrais
    points par donne (voir docstring de module)."""
    from scripts.duel_expert import _bid
    from scripts.train import declarer_role_and_seats
    bidder = evolution.policy_of(evolution.best_per_lineage(population)[evolution.EXPORTED_LINEAGES[0]])
    rng = random.Random(seed)
    chain = DeckChain(0.5, rng)
    rows = {}
    t0 = time.time()
    with torch.no_grad():
        for i in range(deals):
            hands, dealer, whisted = chain.deal(rng)
            chain.finish_unplayed(hands, dealer)
            bidding = Bidding(dealer)
            trou = detect_trou(hands)
            if trou is not None:
                bidding.seed_trou(trou)
            _bid(bidding, hands, {0, 1, 2, 3}, bidder)
            if bidding.general_pass:
                continue
            contract = bidding.locked_contract
            role, decl = declarer_role_and_seats(contract)
            defense = [s for s in range(4) if s not in decl]

            def play(new_seats):
                return _play_out(contract, hands, dealer, bidding.bid_summary, whisted,
                                 [new_nets if s in new_seats else champ_nets for s in range(4)])
            base = play(set())
            with_decl = play(set(decl))
            with_def = play(set(defense))
            r = rows.setdefault(role, {"decl": [], "def": []})
            r["decl"].append(sum(with_decl[s] for s in decl) - sum(base[s] for s in decl))
            r["def"].append(sum(with_def[s] for s in defense) - sum(base[s] for s in defense))
            if progress and (i + 1) % 2000 == 0:
                print("  %d donnes (%.0f s)" % (i + 1, time.time() - t0), flush=True)
    return rows


def run_match(new_nets, champ_nets, population, deals, seed=CHAMPION_SEED):
    rows = camp_rows(new_nets, champ_nets, population, deals, seed)
    decl = [x for r in rows.values() for x in r["decl"]]
    defense = [x for r in rows.values() for x in r["def"]]
    m_decl, se_decl = mean_se(decl)
    m_def, se_def = mean_se(defense)
    m_tot, se_tot = mean_se([a + b for a, b in zip(decl, defense)])
    by_contract = {}
    for role, r in sorted(rows.items()):
        md, sd = mean_se(r["decl"])
        mf, sf = mean_se(r["def"])
        by_contract[role] = {"donnes": len(r["decl"]), "declarant": round(md, 3), "decl_se": round(sd, 3),
                             "defense": round(mf, 3), "def_se": round(sf, 3)}
    return {"deals": len(decl), "new_minus_champion": round(m_tot, 3), "std_error": round(se_tot, 3),
            "declarant": round(m_decl, 3), "declarant_se": round(se_decl, 3),
            "defense": round(m_def, 3), "defense_se": round(se_def, 3), "by_contract": by_contract}


def measure_and_gate(nets, population, total_episodes, elapsed, deals=CHAMPION_DEALS_DEFAULT,
                     out_file=CHAMPION_FILE_DEFAULT):
    t0 = time.time()
    champ, meta = load_champion()
    if champ is None:
        ensure_champion(nets, total_episodes)
        return None
    current = {role: copy.deepcopy(nets[role]).eval() for role in PLAY_ROLES}
    r = run_match(current, champ, population, deals)
    # gain total significatif ET aucun camp en recul significatif
    accepted = (r["new_minus_champion"] - GATE_Z * r["std_error"] > 0
                and r["declarant"] + GATE_Z * r["declarant_se"] >= 0
                and r["defense"] + GATE_Z * r["defense_se"] >= 0)
    r.update({"total_donnes": total_episodes, "elapsed_s": round(elapsed, 1), "accepted": accepted,
              "champion_since": meta.get("since"), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
    if accepted:
        save_champion(nets, {"since": r["timestamp"], "total_donnes": total_episodes,
                             "ecart_vs_ancienne": r["new_minus_champion"], "std_error": r["std_error"],
                             "declarant": r["declarant"], "defense": r["defense"]})
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(r) + "\n")
    print("=== Porte IA vs championne (par camp, %d donnes x3, %.0fs) : ecart %+.2f +- %.2f pts/donne "
          "(declarant %+.2f +- %.2f, defense %+.2f +- %.2f) -> %s ===" % (
              r["deals"], time.time() - t0, r["new_minus_champion"], GATE_Z * r["std_error"],
              r["declarant"], GATE_Z * r["declarant_se"], r["defense"], GATE_Z * r["defense_se"],
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
