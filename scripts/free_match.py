"""IA LIBRE contre references (2026-11-01) : la porte de train_free.py ne mesure
les membres que contre la championne. Ici, chaque ligne remplace une base par un
candidat au jeu de la carte, sur des donnes neuves, sans rien ecrire.

Meme protocole que la porte (3 parties par donne, vrais points) : base aux 4
sieges, candidat au camp declarant, candidat en defense ; gain = somme des deux
ecarts. Le contrat est le meme dans les 3 parties : seule la carte differe.
Deux encheres : `net` (l'encherisseur fixe de l'entrainement) et `expert`
(ExpertBot, celle du jeu actuel). Joueurs : expert, hasard, championne (avec ses
garde-fous, comme en jeu), best et membres de checkpoints_free (cartes legales).

Usage (cloud seulement) : ci/run_free_match.sh, mode free_match de train.yml.
"""

import argparse
import json
import multiprocessing
import os
import random
import sys
import time

import torch

sys.path.insert(0, ".")

from agents import expert_bot
from agents.play_guards import ai_play_mask
from engine.bidding import Bidding
from engine.trou import detect_trou
from engine.whisted import DeckChain
from scripts import champion_play, train_free
from scripts.duel_expert import _bid
from scripts.train import MEASURE_WHISTED_RATIO, declarer_role_and_seats, dispatch_play_phase2, real_points_by_seat

SEED = 20261101  # differentes de la porte : ni selection ni surapprentissage de ses donnes
BIDS = ("net", "expert")
PAIRS = (  # (candidat, base) ; expert contre hasard donne l'echelle
    ("best", "expert"), ("championne", "expert"), ("zero_0", "expert"), ("zero_1", "expert"),
    ("best", "championne"), ("expert", "hasard"), ("zero_0", "hasard"), ("zero_1", "hasard"),
)

NETS = {}  # herites par fork
BIDDER = None


def choose(player, game, seat, rng):
    if player == "expert":
        return expert_bot.choose_card(game, seat)
    if player == "hasard":
        return rng.choice([a for a, ok in enumerate(game.legal_action_mask(seat)) if ok])
    mask = ai_play_mask(game, seat) if player == "championne" else game.legal_action_mask(seat)
    obs = torch.tensor(game.observation(seat), dtype=torch.float32)
    return NETS[player][game.role_of(seat)].act_greedy(obs, torch.tensor(mask, dtype=torch.bool))


def match_chunk(args):
    cand, base, bid, deals, seed = args
    rng = random.Random(seed)
    play_rng = random.Random(seed + 1)
    chain = DeckChain(MEASURE_WHISTED_RATIO, rng)
    rows = {}

    def play(contract, role, hands, dealer, bidding, whisted, cand_seats):
        game = dispatch_play_phase2(contract, [list(h) for h in hands], dealer, bidding.bid_summary, whisted)
        while not game.done:
            seat = game.turn
            game.step(seat, choose(cand if seat in cand_seats else base, game, seat, play_rng))
        return real_points_by_seat(role, game, contract)

    with torch.no_grad():
        for _ in range(deals):
            hands, dealer, whisted = chain.deal(rng)
            chain.finish_unplayed(hands, dealer)
            bidding = Bidding(dealer)
            trou = detect_trou(hands)
            if trou is not None:
                bidding.seed_trou(trou)
            _bid(bidding, hands, {0, 1, 2, 3} if bid == "net" else set(), BIDDER)
            if bidding.general_pass:
                continue
            contract = bidding.locked_contract
            role, decl = declarer_role_and_seats(contract)
            defense = {s for s in range(4) if s not in decl}
            ref = play(contract, role, hands, dealer, bidding, whisted, set())
            with_decl = play(contract, role, hands, dealer, bidding, whisted, set(decl))
            with_def = play(contract, role, hands, dealer, bidding, whisted, defense)
            r = rows.setdefault(role, {"decl": [], "def": []})
            r["decl"].append(sum(with_decl[s] - ref[s] for s in decl))
            r["def"].append(sum(with_def[s] - ref[s] for s in defense))
    return rows


def run_pair(pool, cand, base, bid, deals, procs):
    parts = pool.map(match_chunk, [(cand, base, bid, deals // procs, SEED + 10 * i) for i in range(procs)], chunksize=1)
    rows = {}
    for part in parts:
        for role, r in part.items():
            into = rows.setdefault(role, {"decl": [], "def": []})
            into["decl"] += r["decl"]
            into["def"] += r["def"]
    decl = [x for r in rows.values() for x in r["decl"]]
    defense = [x for r in rows.values() for x in r["def"]]
    gain, se = champion_play.mean_se([a + b for a, b in zip(decl, defense)])
    m_decl, se_decl = champion_play.mean_se(decl)
    m_def, se_def = champion_play.mean_se(defense)
    by_role = {role: {"n": len(r["decl"]), "gain": round(champion_play.mean_se(
        [a + b for a, b in zip(r["decl"], r["def"])])[0], 2)} for role, r in sorted(rows.items())}
    return {"candidat": cand, "base": base, "enchere": bid, "donnes": len(decl), "gain": round(gain, 3),
            "se": round(se, 3), "declarant": round(m_decl, 3), "declarant_se": round(se_decl, 3),
            "defense": round(m_def, 3), "defense_se": round(se_def, 3), "par_contrat": by_role}


def main():
    global BIDDER
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    NETS["championne"], meta = champion_play.load_champion()
    if NETS["championne"] is None:
        raise SystemExit("pas de championne dans checkpoints_v2/champion_play")
    BIDDER = train_free.load_bidder()
    NETS["best"] = train_free.load_nets(torch.load(os.path.join(train_free.ROOT, "best.pt"), map_location="cpu",
                                                   weights_only=False))
    for name in ("zero_0", "zero_1"):
        member = torch.load(train_free.member_path(name), map_location="cpu", weights_only=False)
        NETS[name] = train_free.load_nets(member["nets"])
        print("%s : %d donnes d'entrainement" % (name, member["deals"]), flush=True)
    print("Championne : %s ; %d donnes par ligne, %d enchere(s)." % (meta, args.deals, len(BIDS)), flush=True)

    pool = multiprocessing.get_context("fork").Pool(args.workers, train_free.init_worker)
    for bid in BIDS:
        for cand, base in PAIRS:
            t0 = time.time()
            r = run_pair(pool, cand, base, bid, args.deals, args.workers)
            verdict = "DEVANT" if r["gain"] - 2 * r["se"] > 0 else ("DERRIERE" if r["gain"] + 2 * r["se"] < 0 else "egalite")
            print("=== enchere %-6s %-10s a la place de %-10s : %+.2f +- %.2f (declarant %+.2f, defense %+.2f, "
                  "%d donnes, %.0fs) -> %s ===" % (bid, cand, base, r["gain"], 2 * r["se"], r["declarant"], r["defense"],
                                                   r["donnes"], time.time() - t0, verdict), flush=True)
            print("RESULT " + json.dumps(r), flush=True)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
