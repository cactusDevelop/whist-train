"""Duel IA contre ExpertBot, en DUPLICATE et en VRAIS POINTS (2026-09-27 soir,
objectif de l'utilisateur : "j'attends que l'IA que j'entraine depasse
ExpertBot et domine la population").

Chaque donne est jouee deux fois avec les memes mains et le meme donneur :
  1. IA aux sieges 0 et 2, ExpertBot aux sieges 1 et 3 ;
  2. sieges echanges.
L'IA = ce qui est exporte vers Godot : le meilleur individu de la lignee
exportee (enchere greedy) + les 9 reseaux de jeu (greedy). L'ExpertBot
encherit et joue avec ses regles (agents/expert_bot.py). Le score d'un camp
= somme des vrais points (engine/*.final_scores) de ses deux sieges ; on
compare la moyenne par donne. Le duplicate annule la chance des cartes :
seules les decisions (enchere + jeu) font la difference.

Usage : venv/Scripts/python.exe scripts/duel_expert.py [--deals 400] [--ai-dir <copie de checkpoints_v2>]
Appele aussi par scripts/train.py a chaque evaluation (duel_expert.jsonl).
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, ".")

from agents.play_guards import ai_play_mask
from agents import evolution, expert_bot, specialists
from agents.specialists import PLAY_ROLES
from engine.bidding import Bidding, legal_action_mask_with_hand, observation as bid_observation
from engine.trou import detect_trou
from engine.whisted import DeckChain

DUEL_FILE_DEFAULT = "duel_expert.jsonl"
DUEL_DEALS_DEFAULT = 300  # donnes (x2 en duplicate) a chaque evaluation de train.py
DUEL_SEED = 20260928  # fixe : memes donnes a chaque mesure, seule l'IA change
AI_SEATS = ({0, 2}, {1, 3})


def _bid(bidding, hands, ai_seats, ai_bidder):
    """Enchere complete ; -> journal des decisions [(siege, action)], qui
    determine entierement la suite quand le jeu de la carte est greedy."""
    log = []
    while not bidding.finished:
        seat = bidding.turn
        hand = hands[seat]
        if seat in ai_seats:
            obs = torch.tensor(bid_observation(bidding, seat, hand), dtype=torch.float32)
            mask = torch.tensor(legal_action_mask_with_hand(bidding, seat, hand), dtype=torch.bool)
            action = ai_bidder.decide(bidding, seat, hand) if getattr(ai_bidder, "hybrid", False) \
                else ai_bidder.act_greedy(obs, mask)
            if not bidding.apply(seat, action):
                raise AssertionError("IA : enchere illegale %r" % action)
        else:
            action = expert_bot.apply_bid(bidding, seat, hand)
        log.append((seat, action))
    return tuple(log)


def play_deal(hands, dealer, whisted, ai_seats, ai_bidder, nets, bot_plays_cards=True, cache=None, deal_key=None):
    """-> (points du camp IA, points du camp bot, cle du contrat, camp declarant).
    bot_plays_cards=False : les 4 sieges jouent la carte avec les reseaux de
    l'IA (seule l'ENCHERE differe entre les camps -- fitness d'enchere) ; le
    jeu etant alors deterministe, `cache` memorise les points par (donne,
    journal d'enchere) : deux individus qui encherissent pareil ne rejouent
    pas la donne."""
    from scripts.train import declarer_role_and_seats, describe_contract, dispatch_play_phase2, real_points_by_seat
    bidding = Bidding(dealer)
    trou = detect_trou(hands)
    if trou is not None:
        bidding.seed_trou(trou)
    log = _bid(bidding, hands, ai_seats, ai_bidder)
    if bidding.general_pass:
        return 0.0, 0.0, "general_pass", None
    contract = bidding.locked_contract
    role, declarers = declarer_role_and_seats(contract)
    key = (deal_key, log)
    if cache is not None and key in cache:
        pts = cache[key]
    else:
        game = dispatch_play_phase2(contract, [list(h) for h in hands], dealer, bidding.bid_summary, whisted)
        while not game.done:
            seat = game.turn
            if seat in ai_seats or not bot_plays_cards:
                obs = torch.tensor(game.observation(seat), dtype=torch.float32)
                mask = torch.tensor(ai_play_mask(game, seat), dtype=torch.bool)
                card = nets[game.role_of(seat)].act_greedy(obs, mask)
            else:
                card = expert_bot.choose_card(game, seat)
            game.step(seat, card)
        pts = real_points_by_seat(role, game, contract)
        if cache is not None:
            cache[key] = pts
    ai = sum(pts[s] for s in range(4) if s in ai_seats)
    bot = sum(pts[s] for s in range(4) if s not in ai_seats)
    side = "ia" if declarers <= ai_seats else ("bot" if not declarers & ai_seats else "mixte")
    return ai, bot, describe_contract(contract), side


def run_duel(nets, population, deals, seed=DUEL_SEED):
    best = evolution.best_per_lineage(population)[evolution.EXPORTED_LINEAGES[0]]
    ai_bidder = evolution.policy_of(best)
    rng = random.Random(seed)
    chain = DeckChain(0.5, rng)
    diffs = []
    contracts = {}
    ai_total = bot_total = 0.0
    with torch.no_grad():
        for _ in range(deals):
            hands, dealer, whisted = chain.deal(rng)
            chain.finish_unplayed(hands, dealer)  # suite de donnes independante des resultats
            diff = 0.0
            for seats in AI_SEATS:
                ai, bot, key, side = play_deal(hands, dealer, whisted, seats, ai_bidder, nets)
                ai_total += ai
                bot_total += bot
                diff += ai - bot
                if side is not None:
                    k = "%s:%s" % (side, key)
                    contracts[k] = contracts.get(k, 0) + 1
            diffs.append(diff / 2)
    n = len(diffs)
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / max(1, n - 1))
    return {"deals": n, "bidder_id": best.id, "ai_points_per_deal": round(ai_total / (2 * n), 3),
            "bot_points_per_deal": round(bot_total / (2 * n), 3), "ai_minus_bot": round(mean, 3),
            "std_error": round(sd / math.sqrt(n), 3), "contracts": contracts}


def evaluate_bidders(population, nets, rng, deals):
    """FITNESS D'ENCHERE (2026-09-27, decision utilisateur : "fais le
    duplicate contre ExpertBot") : chaque individu encherit sur les MEMES
    `deals` donnes que tous les autres, aux sieges 0/2 puis 1/3, contre
    l'ExpertBot aux deux autres sieges ; les 4 sieges jouent ensuite la
    carte avec les memes reseaux (seule l'enchere differe). Fitness d'une
    donne = (points de ses sieges - points des sieges ExpertBot), moyenne
    des deux positions. Le duplicate et les donnes communes retirent la
    chance des cartes : avant, 50 donnes ou l'individu ne prenait presque
    aucun contrat lui-meme faisaient une note surtout due au hasard, et la
    selection degradait l'enchere (elite a -7 pts/donne contre le bot, sa
    graine a -1). -> nombre de donnes rejouees (hors cache)."""
    chain = DeckChain(0.5, rng)
    table = []
    for _ in range(deals):
        hands, dealer, whisted = chain.deal(rng)
        chain.finish_unplayed(hands, dealer)
        table.append((hands, dealer, whisted))
    cache = {}
    with torch.no_grad():
        for ind in population:
            bidder = evolution.policy_of(ind)
            for i, (hands, dealer, whisted) in enumerate(table):
                diff = 0.0
                for seats in AI_SEATS:
                    ai, bot, _key, _side = play_deal(hands, dealer, whisted, seats, bidder, nets,
                                                     bot_plays_cards=False, cache=cache, deal_key=i)
                    diff += ai - bot
                ind.record_game(diff / 2)
    return len(cache)


def measure_and_record(nets, population, total_episodes, elapsed, deals=DUEL_DEALS_DEFAULT,
                       duel_file=DUEL_FILE_DEFAULT):
    t0 = time.time()
    r = run_duel(nets, population, deals)
    r.update({"total_donnes": total_episodes, "elapsed_s": round(elapsed, 1),
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
    with open(duel_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(r) + "\n")
    verdict = "IA DEVANT" if r["ai_minus_bot"] - 2 * r["std_error"] > 0 else (
        "ExpertBot devant" if r["ai_minus_bot"] + 2 * r["std_error"] < 0 else "egalite (dans le bruit)")
    print("=== Duel IA vs ExpertBot (duplicate, %d donnes x2, %.0fs) : IA %+.2f / bot %+.2f pts par donne, "
          "ecart %+.2f +- %.2f -> %s ===" % (r["deals"], time.time() - t0, r["ai_points_per_deal"],
                                              r["bot_points_per_deal"], r["ai_minus_bot"], 2 * r["std_error"],
                                              verdict), flush=True)
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=400)
    parser.add_argument("--ai-dir", default=None, help="copie de checkpoints_v2 (ne pas lire pendant une sauvegarde)")
    parser.add_argument("--out", default=None, help="fichier JSON Lines (par defaut : rien n'est ecrit)")
    args = parser.parse_args()
    if args.ai_dir:
        specialists.CHECKPOINT_DIR = args.ai_dir
    nets = {role: specialists.load_or_create(role)[0] for role in PLAY_ROLES}
    population, _gen = evolution.load_population(os.path.join(specialists.CHECKPOINT_DIR, "encherisseur_population.pt"))
    if args.out:
        measure_and_record(nets, population, 0, 0.0, args.deals, args.out)
    else:
        r = run_duel(nets, population, args.deals)
        print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
