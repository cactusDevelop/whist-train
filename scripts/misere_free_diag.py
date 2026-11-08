"""Ou l'IA libre perd ses points en MISERE ETALEE (08/11, demande user : BILAN
misere_etalee -8.9 vs ExpertBot, cycle 276). Mesure seule, cloud.

Meme protocole que scripts/emballage_free_diag.py (partie de reference 4
ExpertBot, puis best au camp declarant, puis best en defense ; a chaque
decision de best on demande aussi sa carte a ExpertBot) mais le dds ne sait
pas jouer la misere (engine/dds.py : contrats a atout seulement). Les deux
cartes sont donc notees par des suites jouees par ExpertBot aux 4 sieges :
mondes vus du siege (information reelle, --samples) et vraies mains
(omniscient, 1 suite). Regret = Q(carte expert) - Q(carte best), en points du
siege ; somme / donnes = points/donne comparables au BILAN. Limite : la note
vaut ce que vaut ExpertBot pour finir la donne.

Usage (cloud seulement) : ci/run_free_misere_diag.sh, mode free_misere_diag de train.yml.
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
from engine.cards import rank_of, suit_of
from engine.trou import detect_trou
from engine.whisted import DeckChain
from scripts import champion_play, train_free
from scripts.duel_expert import _bid
from scripts.emballage_free_diag import kind, name
from scripts.train import MEASURE_WHISTED_RATIO, declarer_role_and_seats, dispatch_play_phase2, real_points_by_seat

SEED = 20261108
NETS = {}
BIDDER = None
ARGS = None


def q_values(game, seat, cards, worlds, contract):
    """Q (points du siege) de chaque carte, suite jouee par ExpertBot dans chaque monde."""
    role, _decl = declarer_role_and_seats(contract)
    q = [0.0] * len(cards)
    for hands in worlds:
        for i, card in enumerate(cards):
            g = rollouts.clone_with_hands(game, hands)
            g.step(seat, card)
            rollouts.play_out_expert([g])
            q[i] += real_points_by_seat(role, g, contract)[seat] / len(worlds)
    return q


def chunk(args):
    seed, deals, deadline = args
    rng = random.Random(seed)
    chain = DeckChain(MEASURE_WHISTED_RATIO, rng)
    out = {"n": 0, "seen": 0, "pts": {"decl": [], "def": []}, "made": collections.Counter(), "audits": [],
           "ndecl": collections.Counter()}

    def play(contract, role, hands, dealer, bidding, whisted, free_seats, audit_camp):
        game = dispatch_play_phase2(contract, [list(h) for h in hands], dealer, bidding.bid_summary, whisted)
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
                trick_no = game.trick_count + 1
                worlds = [rollouts.determinize(game, seat, rng) for _ in range(ARGS.samples)]
                q = q_values(game, seat, [b, e], worlds, contract)
                qo = q_values(game, seat, [b, e], [game.hands], contract)
                out["audits"].append({
                    "camp": audit_camp, "pli": trick_no, "pos": len(game.current_trick) + 1,
                    "kb": kind(game, seat, b, free_seats), "ke": kind(game, seat, e, free_seats),
                    "plis": min(game.tricks_won[seat], 3), "regret": q[1] - q[0], "omni": qo[1] - qo[0],
                    "ex": "%d declarant(s) pli %d | plis du siege %d | main %s | pli en cours %s | best %s expert %s" % (
                        len(contract["declarers"]), trick_no, game.tricks_won[seat],
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
            out["seen"] += 1
            if bidding.general_pass:
                continue
            contract = bidding.locked_contract
            if contract["type"] != "misere" or contract["kind"] != "etalee":
                continue
            role, decl = declarer_role_and_seats(contract)
            defense = {s for s in range(4) if s not in decl}
            g_ref, ref = play(contract, role, hands, dealer, bidding, whisted, set(), None)
            g_decl, with_decl = play(contract, role, hands, dealer, bidding, whisted, set(decl), "decl")
            g_def, with_def = play(contract, role, hands, dealer, bidding, whisted, defense, "def")
            out["n"] += 1
            out["ndecl"][len(decl)] += 1
            out["pts"]["decl"].append(sum(with_decl[s] - ref[s] for s in decl))
            out["pts"]["def"].append(sum(with_def[s] - ref[s] for s in defense))
            made = lambda g: sum(1 for s in decl if g.tricks_won[s] == 0)  # declarants sans pli
            out["made"][("def", made(g_ref), made(g_def))] += 1
            out["made"][("decl", made(g_ref), made(g_decl))] += 1
    return out


def table(title, audits, n, key):
    groups = collections.defaultdict(list)
    for a in audits:
        groups[key(a)].append(a)
    print("  -- %s --" % title)
    print("     %-48s %6s %9s %9s %9s" % ("", "ecarts", "pts/donne", "moy", "omni moy"))
    for k, rows in sorted(groups.items(), key=lambda kv: sum(r["regret"] for r in kv[1]), reverse=True)[:ARGS.top]:
        tot = sum(r["regret"] for r in rows)
        om = sum(r["omni"] for r in rows) / len(rows)
        print("     %-48s %6d %+9.3f %+9.2f %+9.2f" % (str(k)[:48], len(rows), tot / n, tot / len(rows), om))


def main():
    global BIDDER, ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=2000, help="misere etalees visees (total)")
    parser.add_argument("--minutes", type=float, default=45)
    parser.add_argument("--samples", type=int, default=16)
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
    print("=== %d misere etalees sur %d donnes, %d ecarts notes (suites ExpertBot), %.0f s ; declarants : %s ===" % (
        n, sum(p["seen"] for p in parts), len(audits), time.time() - t0,
        dict(sorted(sum((p["ndecl"] for p in parts), collections.Counter()).items()))))
    for camp in ("decl", "def"):
        pts = [x for p in parts for x in p["pts"][camp]]
        m, se = champion_play.mean_se(pts)
        a = [x for x in audits if x["camp"] == camp]
        tot = sum(x["regret"] for x in a)
        om = sum(x["omni"] for x in a)
        print("\n=== best au camp %s : %+.2f +- %.2f pts/donne vs ExpertBot ; %d ecarts, regret %+.3f pts/donne "
              "(omniscient %+.3f) ===" % (camp, m, 2 * se, len(a), tot / max(1, n), om / max(1, n)))
        print("  declarants sans pli (ref ExpertBot -> avec best) : " + ", ".join(
            "%d->%d %d" % (r, w, c) for (cp, r, w), c in sorted(made.items()) if cp == camp))
        print("  pertes %d / egalites %d / gains %d donnes" % (
            sum(1 for x in pts if x < 0), sum(1 for x in pts if x == 0), sum(1 for x in pts if x > 0)))
        table("par pli", a, n, lambda x: "pli %2d" % x["pli"])
        table("par position dans le pli", a, n, lambda x: "position %d" % x["pos"])
        table("par plis deja pris par le siege (3 = 3+)", a, n, lambda x: "%d plis" % x["plis"])
        table("par carte best (regret = expert mieux)", a, n, lambda x: x["kb"])
        table("par paire best -> expert", a, n, lambda x: "%s -> %s" % (x["kb"], x["ke"]))
        print("  -- pires ecarts --")
        for x in sorted(a, key=lambda x: x["regret"], reverse=True)[:ARGS.top]:
            print("     %+6.1f (omni %+.1f) %s" % (x["regret"], x["omni"], x["ex"]))


if __name__ == "__main__":
    main()
