"""Ou l'IA libre perd ses points en EMBALLAGE (06/11, demande user : BILAN
emballage -3.7 vs ExpertBot malgre la defense dds ciblee). Mesure seule, cloud.

Meme protocole que le BILAN de train_free (enchere net, vrais points) mais
limite aux emballages : partie de reference 4 ExpertBot, puis best au camp
declarant, puis best en defense. Dans les parties avec best, a chaque decision
de best on demande aussi sa carte a ExpertBot ; si elles different, le dds
note les deux cartes (mondes vus du siege = information reelle, et vraies
mains = omniscient). Regret = Q(carte expert) - Q(carte best), en points du
siege ; somme / donnes d'emballage = points/donne comparables au BILAN.

Tableaux par camp : regret par pli, par position dans le pli, par paire
(type de carte best -> type de carte expert), atouts en main ; pires exemples.

Usage (cloud seulement) : ci/run_free_emb_diag.sh, mode free_emb_diag de train.yml.
"""

import argparse
import collections
import multiprocessing
import os
import random
import sys
import time

import torch

sys.path.insert(0, ".")

from agents import expert_bot, rollouts
from engine.bidding import Bidding
from engine.cards import rank_of, suit_of, trick_winner
from engine.trou import detect_trou
from engine.whisted import DeckChain
from scripts import champion_play, train_free
from scripts.duel_expert import _bid
from scripts.train import MEASURE_WHISTED_RATIO, declarer_role_and_seats, dispatch_play_phase2, real_points_by_seat

SEED = 20261106
RANKS = "23456789TJQKA"
SUITS = "CKTP"  # Coeur, Carreau, Trefle, Pique
NETS = {}
BIDDER = None
ARGS = None


def name(card):
    return RANKS[rank_of(card)] + SUITS[suit_of(card)]


def kind(game, seat, card, camp):
    """Type de carte jouee, du point de vue du siege."""
    t, trick = game.trump_suit, game.current_trick
    s, high = suit_of(card), rank_of(card) >= 9  # valet ou plus
    if not trick:
        return "entame %s %s" % ("atout" if s == t else "couleur", "haute" if high else "basse")
    led = suit_of(trick[0][1])
    ally = trick_winner(trick, t) in camp
    tag = "allie maitre" if ally else "adverse maitre"
    beats = trick_winner(trick + [(seat, card)], t) == seat
    if s == led:
        what = "fournit %s" % ("et prend" if beats else ("haute" if high else "basse"))
    elif s == t:
        trumped = any(suit_of(c) == t for _x, c in trick)
        what = ("surcoupe" if trumped else "coupe") if beats else "sous-coupe"
    else:
        what = "defausse %s" % ("haute" if high else "basse")
    return "%s, %s" % (what, tag)


def q_values(game, seat, cards, worlds):
    """Q (points du siege, dds) de chaque carte, ou None si trop de mondes hors budget."""
    role, _decl = declarer_role_and_seats(game_contract(game))
    score_fn = lambda g: real_points_by_seat(role, g, game_contract(g))
    ws = [(game, seat, cards, h) for h in worlds]
    return rollouts._dds_q(game, seat, cards, score_fn, len(ws), ws, [rollouts._dds_world(w) for w in ws])


def game_contract(game):
    return game._diag_contract


def chunk(args):
    seed, deals, deadline = args
    rng = random.Random(seed)
    play_rng = random.Random(seed + 1)
    chain = DeckChain(MEASURE_WHISTED_RATIO, rng)
    out = {"n": 0, "pts": {"decl": [], "def": []}, "made": collections.Counter(), "audits": [], "skipped": 0,
           "trumps_def": collections.Counter()}

    def play(contract, role, hands, dealer, bidding, whisted, free_seats, audit_camp):
        game = dispatch_play_phase2(contract, [list(h) for h in hands], dealer, bidding.bid_summary, whisted)
        game._diag_contract = contract
        while not game.done:
            seat = game.turn
            if seat not in free_seats:
                game.step(seat, expert_bot.choose_card(game, seat))
                continue
            obs = torch.tensor(game.observation(seat), dtype=torch.float32)
            mask = torch.tensor(game.legal_action_mask(seat), dtype=torch.bool)
            b = NETS["best"][game.role_of(seat)].act_greedy(obs, mask)
            e = expert_bot.choose_card(game, seat)
            if b != e and time.time() < deadline:
                camp = free_seats
                trick_no = game.trick_count + 1
                worlds = [rollouts.determinize(game, seat, rng) for _ in range(ARGS.samples)]
                q = q_values(game, seat, [b, e], worlds)
                qo = q_values(game, seat, [b, e], [game.hands])
                if q is None:
                    out["skipped"] += 1
                else:
                    trumps = sum(1 for c in game.hands[seat] if suit_of(c) == game.trump_suit)
                    out["audits"].append({
                        "camp": audit_camp, "pli": trick_no, "pos": len(game.current_trick) + 1,
                        "kb": kind(game, seat, b, camp), "ke": kind(game, seat, e, camp),
                        "atouts": min(trumps, 5), "regret": q[1] - q[0],
                        "omni": None if qo is None else qo[1] - qo[0],
                        "ex": "niv %d atout %s pli %d | main %s | pli en cours %s | best %s expert %s" % (
                            contract["level"], SUITS[game.trump_suit], trick_no,
                            " ".join(name(c) for c in sorted(game.hands[seat], key=lambda c: (suit_of(c), -rank_of(c)))),
                            " ".join(name(c) for _s, c in game.current_trick) or "-", name(b), name(e))})
            game.step(seat, b)
        return game, real_points_by_seat(role, game, contract)

    with torch.no_grad():
        while out["n"] < deals and time.time() < deadline:
            hands, dealer, whisted = chain.deal(rng)
            chain.finish_unplayed(hands, dealer)
            bidding = Bidding(dealer)
            trou = detect_trou(hands)
            if trou is not None:
                bidding.seed_trou(trou)
            _bid(bidding, hands, {0, 1, 2, 3}, BIDDER)
            if bidding.general_pass or bidding.locked_contract["type"] != "emballage":
                continue
            contract = bidding.locked_contract
            role, decl = declarer_role_and_seats(contract)
            defense = {s for s in range(4) if s not in decl}
            g_ref, ref = play(contract, role, hands, dealer, bidding, whisted, set(), None)
            g_decl, with_decl = play(contract, role, hands, dealer, bidding, whisted, set(decl), "decl")
            g_def, with_def = play(contract, role, hands, dealer, bidding, whisted, defense, "def")
            out["n"] += 1
            out["pts"]["decl"].append(sum(with_decl[s] - ref[s] for s in decl))
            out["pts"]["def"].append(sum(with_def[s] - ref[s] for s in defense))
            made = lambda g: sum(g.tricks_won[s] for s in decl) >= contract["level"]
            out["made"][("def", made(g_ref), made(g_def))] += 1
            out["made"][("decl", made(g_ref), made(g_decl))] += 1
            for s in defense:
                out["trumps_def"][min(5, sum(1 for c in hands[s] if suit_of(c) == contract["suit"]))] += 1
    return out


def table(title, audits, n, key):
    groups = collections.defaultdict(list)
    for a in audits:
        groups[key(a)].append(a)
    print("  -- %s --" % title)
    print("     %-48s %6s %9s %9s %9s" % ("", "ecarts", "pts/donne", "moy", "omni moy"))
    for k, rows in sorted(groups.items(), key=lambda kv: sum(r["regret"] for r in kv[1]), reverse=True)[:ARGS.top]:
        tot = sum(r["regret"] for r in rows)
        om = [r["omni"] for r in rows if r["omni"] is not None]
        print("     %-48s %6d %+9.3f %+9.2f %+9.2f" % (str(k)[:48], len(rows), tot / n, tot / len(rows),
                                                      sum(om) / len(om) if om else float("nan")))


def main():
    global BIDDER, ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=4000, help="emballages vises (total)")
    parser.add_argument("--minutes", type=float, default=150)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--top", type=int, default=15)
    ARGS = parser.parse_args()

    BIDDER = train_free.load_bidder()
    NETS["best"] = train_free.load_nets(torch.load(os.path.join(train_free.ROOT, "best.pt"), map_location="cpu",
                                                   weights_only=False))
    deadline = time.time() + 60 * ARGS.minutes
    t0 = time.time()
    pool = multiprocessing.get_context("fork").Pool(ARGS.workers, train_free.init_worker)
    parts = pool.map(chunk, [(SEED + 10 * i, ARGS.deals // ARGS.workers, deadline) for i in range(ARGS.workers)])
    pool.close()
    pool.join()

    n = sum(p["n"] for p in parts)
    audits = [a for p in parts for a in p["audits"]]
    made = sum((p["made"] for p in parts), collections.Counter())
    print("=== %d emballages, %d ecarts notes au dds (%d hors budget), %.0f s ===" % (
        n, len(audits), sum(p["skipped"] for p in parts), time.time() - t0))
    for camp in ("decl", "def"):
        pts = [x for p in parts for x in p["pts"][camp]]
        m, se = champion_play.mean_se(pts)
        a = [x for x in audits if x["camp"] == camp]
        tot = sum(x["regret"] for x in a)
        om = [x["omni"] for x in a if x["omni"] is not None]
        print("\n=== best au camp %s : %+.2f +- %.2f pts/donne vs ExpertBot ; %d ecarts, regret dds %+.3f pts/donne "
              "(omniscient %+.3f) ===" % (camp, m, 2 * se, len(a), tot / max(1, n), sum(om) / max(1, n)))
        print("  contrat fait ? (ref ExpertBot -> avec best) : " + ", ".join(
            "%s->%s %d" % ("fait" if r else "chute", "fait" if w else "chute", c)
            for (cp, r, w), c in sorted(made.items()) if cp == camp))
        print("  pertes %d / egalites %d / gains %d donnes" % (
            sum(1 for x in pts if x < 0), sum(1 for x in pts if x == 0), sum(1 for x in pts if x > 0)))
        table("par pli", a, n, lambda x: "pli %2d" % x["pli"])
        table("par position dans le pli", a, n, lambda x: "position %d" % x["pos"])
        table("par atouts en main (5 = 5+)", a, n, lambda x: "%d atouts" % x["atouts"])
        table("par carte best (regret = expert mieux)", a, n, lambda x: x["kb"])
        table("par paire best -> expert", a, n, lambda x: "%s -> %s" % (x["kb"], x["ke"]))
        print("  -- pires ecarts --")
        for x in sorted(a, key=lambda x: x["regret"], reverse=True)[:ARGS.top]:
            print("     %+6.1f (omni %s) %s" % (x["regret"], "?" if x["omni"] is None else "%+.1f" % x["omni"], x["ex"]))
    print("\n  atouts des defenseurs (5 = 5+) : %s" % dict(sorted(sum((p["trumps_def"] for p in parts),
                                                                      collections.Counter()).items())))


if __name__ == "__main__":
    main()
