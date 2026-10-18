"""Banc d'essai des GROS contrats rares (demande utilisateur 2026-10-03 :
"les gros contrats sont rares. L'IA doit savoir reconnaitre la main qui
pourrait gagner ces gros contrats rares et jouer parfaitement dessus").

Quels gros contrats ? Le bareme (engine/scoring.py, verifie contre Godot)
ne paie PAS plus un emballage plus haut a plis egaux (5 + 2 x (plis -
niveau)) : emballage 11-13 ne sert qu'a passer au-dessus d'un adversaire.
Ceux qui paient sont l'ABONDANCE (solo 9-12, 5 x niveau - 35, x3 pour le
soliste : 30 a 75) et le SOLO CHELEM (13, 90) -- plat, tout ou rien. Ce
banc mesure donc les mains d'ABONDANCE : un siege a au moins
STRONG_SOLO_MIN plis surs (agents/expert_bot.py:estimate_tricks) dans une
couleur.

Donnes : suite de donnes en deck whiste (ratio --whisted-ratio), chaque
donne JOUEE par 4 ExpertBot pour que le paquet suivant soit realiste (le
deck whiste regroupe les couleurs : ~2x plus de mains d'abondance qu'en
melange uniforme). Graine fixe : memes mains a chaque mesure.

Sur chaque main forte (siege S, couleur T) :
  1. JEU -- abondance-9 IMPOSEE a S (il entame), atout T, defense
     ExpertBot : plis faits par l'IA declarante et par l'ExpertBot
     declarant. Potentiel de la main = max des deux (>= 9 : abondance
     jouable a ce niveau).
  2. RECONNAISSANCE -- table 4 x IA (enchere + jeu greedy, l'IA
     exportee) et table 4 x ExpertBot : contrat de S (abondance/chelem ?
     a quel niveau vs son potentiel ?), reussite, points de S.

Lecture seule : ne touche ni l'entrainement ni les checkpoints.

Usage : venv/Scripts/python.exe scripts/big_contracts_bench.py [--deals 1000] [--ai-dir checkpoints_v2] [--out big_contracts.jsonl]
"""

import argparse
import json
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
from engine.cards import NUM_PLAYERS, NUM_SUITS
from engine.scoring import ABONDANCE_MIN_LEVEL
from engine.trou import detect_trou
from engine.whisted import DeckChain

STRONG_SOLO_MIN = 9
BENCH_SEED = 20261003
BENCH_FILE_DEFAULT = "big_contracts.jsonl"


def best_lone(hands):
    """-> (plis surs estimes, siege, couleur) du meilleur solo de la donne."""
    return max((expert_bot.estimate_tricks(hands[s], t), s, t) for s in range(NUM_PLAYERS) for t in range(NUM_SUITS))


def run_bidding(hands, dealer, bidder):
    bidding = Bidding(dealer)
    trou = detect_trou(hands)
    if trou is not None:
        bidding.seed_trou(trou)
    while not bidding.finished:
        seat = bidding.turn
        bidder(bidding, seat, hands[seat])
    return bidding


def ai_bid(bidder):
    def bid(bidding, seat, hand):
        if getattr(bidder, "hybrid", False):
            action = bidder.decide(bidding, seat, hand)
        else:
            obs = torch.tensor(bid_observation(bidding, seat, hand), dtype=torch.float32)
            mask = torch.tensor(legal_action_mask_with_hand(bidding, seat, hand), dtype=torch.bool)
            action = bidder.act_greedy(obs, mask)
        if not bidding.apply(seat, action):
            raise AssertionError("IA : enchere illegale %r" % action)
    return bid


def bot_bid(bidding, seat, hand):
    expert_bot.apply_bid(bidding, seat, hand)


def ai_card(nets):
    def play(game, seat):
        obs = torch.tensor(game.observation(seat), dtype=torch.float32)
        mask = torch.tensor(ai_play_mask(game, seat), dtype=torch.bool)
        return nets[game.role_of(seat)].act_greedy(obs, mask)
    return play


def bot_card(game, seat):
    return expert_bot.choose_card(game, seat)


def play(contract, bid_summary, hands, dealer, whisted, decl_player, def_player):
    """-> (jeu termine, points par siege)."""
    from scripts.train import declarer_role_and_seats, dispatch_play_phase2, real_points_by_seat
    role, decl = declarer_role_and_seats(contract)
    game = dispatch_play_phase2(contract, [list(h) for h in hands], dealer, bid_summary, whisted)
    while not game.done:
        seat = game.turn
        game.step(seat, (decl_player if seat in decl else def_player)(game, seat))
    return game, real_points_by_seat(role, game, contract)


def strong_deals(rng, n, whisted_ratio):
    """n donnes fortes d'une suite de donnes jouees par 4 ExpertBot."""
    chain = DeckChain(whisted_ratio, rng)
    out, seen = [], 0
    while len(out) < n:
        hands, dealer, whisted = chain.deal(rng)
        seen += 1
        if best_lone(hands)[0] >= STRONG_SOLO_MIN:
            out.append(([list(h) for h in hands], dealer, whisted))
        bidding = run_bidding(hands, dealer, bot_bid)
        if bidding.general_pass:
            chain.finish_unplayed(hands, dealer)
            continue
        game, _pts = play(bidding.locked_contract, bidding.bid_summary, hands, dealer, whisted, bot_card, bot_card)
        chain.finish_played(game, dealer, bidding.locked_contract["type"])
    return out, seen


def lone_level(contract, seat):
    """Niveau du solo/abondance de `seat`, ou None."""
    if contract is not None and contract["type"] == "solo" and contract["declarer"] == seat:
        return contract["level"]
    return None


def run_bench(nets, ai_bidder, deals, rng, whisted_ratio):
    from scripts.train import describe_contract
    table, seen = strong_deals(rng, deals, whisted_ratio)
    ia, bot = ai_card(nets), bot_card
    r = {"deals": deals, "seen": seen, "whisted_ratio": whisted_ratio,
         "forced": {"ia_tricks": 0, "bot_tricks": 0, "ia_ge9": 0, "bot_ge9": 0, "ia_ge_bot": 0},
         "potential": {}, "tables": {}}
    for name in ("ia", "bot"):
        r["tables"][name] = {"abondance": 0, "abondance_made": 0, "abondance_on_jouable": 0, "jouable": 0,
                             "under": 0, "exact": 0, "over": 0, "seat_points": 0.0, "contracts": {}}
    with torch.no_grad():
        for hands, dealer, whisted in table:
            _est, s, t = best_lone(hands)
            forced = {"type": "solo", "suit": t, "level": ABONDANCE_MIN_LEVEL, "declarer": s}
            empty = Bidding(dealer).bid_summary
            gi, _p = play(forced, empty, hands, dealer, whisted, ia, bot)
            gb, _p = play(forced, empty, hands, dealer, whisted, bot, bot)
            ti, tb = gi.tricks_won[s], gb.tricks_won[s]
            f = r["forced"]
            f["ia_tricks"] += ti
            f["bot_tricks"] += tb
            f["ia_ge9"] += ti >= ABONDANCE_MIN_LEVEL
            f["bot_ge9"] += tb >= ABONDANCE_MIN_LEVEL
            f["ia_ge_bot"] += ti >= tb
            potential = max(ti, tb)
            r["potential"][str(potential)] = r["potential"].get(str(potential), 0) + 1
            jouable = potential >= ABONDANCE_MIN_LEVEL
            for name, bidder, card in (("ia", ai_bid(ai_bidder), ia), ("bot", bot_bid, bot)):
                tb_ = r["tables"][name]
                bidding = run_bidding(hands, dealer, bidder)
                contract = None if bidding.general_pass else bidding.locked_contract
                key = "passe" if contract is None else describe_contract(contract)
                tb_["contracts"][key] = tb_["contracts"].get(key, 0) + 1
                pts = [0.0] * NUM_PLAYERS
                game = None
                if contract is not None:
                    game, pts = play(contract, bidding.bid_summary, hands, dealer, whisted, card, card)
                tb_["seat_points"] += pts[s]
                level = lone_level(contract, s)
                big = level is not None and level >= ABONDANCE_MIN_LEVEL
                tb_["jouable"] += jouable
                if big:
                    tb_["abondance"] += 1
                    tb_["abondance_made"] += game.tricks_won[s] >= level
                    tb_["abondance_on_jouable"] += jouable
                if jouable:
                    lvl = level if big else 0
                    tb_["under" if lvl < potential else ("exact" if lvl == potential else "over")] += 1
    return r


def summarize(r):
    n = r["deals"]
    f = r["forced"]
    lines = ["Mains d'abondance (un siege >= %d plis surs) : %d sur %d donnes (%.2f %%), deck whiste %.0f %%" % (
        STRONG_SOLO_MIN, n, r["seen"], 100.0 * n / r["seen"], 100 * r["whisted_ratio"])]
    lines.append("  JEU -- abondance-9 imposee, defense ExpertBot : IA %.2f plis (>= 9 : %.1f %%) | ExpertBot %.2f plis "
                 "(>= 9 : %.1f %%) | IA >= bot sur %.1f %% des mains" % (
                     f["ia_tricks"] / n, 100.0 * f["ia_ge9"] / n, f["bot_tricks"] / n, 100.0 * f["bot_ge9"] / n,
                     100.0 * f["ia_ge_bot"] / n))
    lines.append("  potentiel (max plis IA/bot) : " + ", ".join(
        "%s plis %d" % kv for kv in sorted(r["potential"].items(), key=lambda kv: int(kv[0]))))
    for name, label in (("ia", "table 4 x IA"), ("bot", "table 4 x ExpertBot")):
        t = r["tables"][name]
        top = ", ".join("%s %d" % kv for kv in sorted(t["contracts"].items(), key=lambda kv: -kv[1])[:8])
        j = max(1, t["jouable"])
        lines.append("  %-20s abondance/chelem du siege fort : %d (reussie %.0f %%) ; sur %d mains jouables : "
                     "annoncee %.0f %%, sous le potentiel %.0f %%, au potentiel %.0f %%, au-dessus %.0f %% ; "
                     "points du siege fort %+.1f/donne | %s" % (
                         label, t["abondance"], 100.0 * t["abondance_made"] / max(1, t["abondance"]), t["jouable"],
                         100.0 * t["abondance_on_jouable"] / j, 100.0 * t["under"] / j, 100.0 * t["exact"] / j,
                         100.0 * t["over"] / j, t["seat_points"] / n, top))
    return "\n".join(lines)


def load_ai(ai_dir):
    if ai_dir:
        specialists.CHECKPOINT_DIR = ai_dir
    missing = [r for r in PLAY_ROLES if not os.path.exists(specialists.checkpoint_path(r))]
    if missing:  # load_or_create creerait un reseau NEUF sans rien dire
        raise SystemExit("checkpoints absents de %s : %s" % (specialists.CHECKPOINT_DIR, missing))
    nets = {role: specialists.load_or_create(role)[0] for role in PLAY_ROLES}
    population, _gen = evolution.load_population(os.path.join(specialists.CHECKPOINT_DIR, "encherisseur_population.pt"))
    best = evolution.best_per_lineage(population)[evolution.EXPORTED_LINEAGES[0]]
    return nets, evolution.policy_of(best)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=BENCH_SEED)
    parser.add_argument("--whisted-ratio", type=float, default=1.0)
    parser.add_argument("--ai-dir", default=None)
    parser.add_argument("--out", default=None, help="fichier JSON Lines (par defaut : rien n'est ecrit)")
    args = parser.parse_args()
    nets, bidder = load_ai(args.ai_dir)
    t0 = time.time()
    r = run_bench(nets, bidder, args.deals, random.Random(args.seed), args.whisted_ratio)
    print(summarize(r))
    print("(%.0fs)" % (time.time() - t0))
    if args.out:
        with open(args.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **r}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
