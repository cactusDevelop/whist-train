"""A/B duplicate des regles de misere de l'ExpertBot (signalement 2026-11-24
« Petite Misere en Practice », GATES-signalements.md). Mesure seule, cloud.

Donnes : enchere du jeu (ExpertBot + BID_ROUTE aux 4 sieges, scripts/ai_insights.run_bidding), 100 % whistees ;
on garde les contrats de misere (toutes sortes, petite en tete). Chaque donne est jouee une fois par variante, dans
deux champs :
  - "expert" : ExpertBot aux 4 sieges (conseil Practice d'avant dc5e5b5, Revue) ;
  - "jeu"    : la table du jeu (reseaux libre_* la ou hybrid_applies -- defense de la petite misere -- ExpertBot ailleurs).
Gain d'une variante = points reels du camp concerne (declarants pour H3, defenseurs pour H1/H2) - reference,
meme donne ; IC par donne (appariement). Aussi : taux de chute des declarants (>= 1 pli de trop).

Critere fixe avant tout resultat (24/11) : une regle est PROMUE si, sur la petite misere, champ "expert",
gain du camp concerne z > 3, ET aucune sorte de misere ni le champ "jeu" ne la fait perdre z < -2.
IDENTITE doit donner exactement 0.

Usage (cloud) : python -u scripts/petite_misere_ab.py --minutes 300 --workers 4
"""

import argparse
import collections
import math
import multiprocessing as mp
import random
import sys
import time

sys.path.insert(0, ".")

import torch

from agents import expert_bot
from engine.trou import detect_trou
from engine.whisted import DeckChain
from scripts import ai_insights as game_ai
from scripts.expert_bid_ab import load_game_ai
from scripts.train import declarer_role_and_seats, dispatch_play_phase2, real_points_by_seat

SEED = 20261124
VARIANTS = {
    "IDENTITE": {},
    "H1_FORCED_HIGH": {"MIS_DEF_FORCED_HIGH": True},
    "H2_MATE_HIGH": {"MIS_DEF_MATE_HIGH": True},
    "H1H2": {"MIS_DEF_FORCED_HIGH": True, "MIS_DEF_MATE_HIGH": True},
    "H3_LEAD_GUARD": {"MIS_DECL_LEAD_GUARD": True},
}
CAMP = {"IDENTITE": "decl", "H1_FORCED_HIGH": "def", "H2_MATE_HIGH": "def", "H1H2": "def", "H3_LEAD_GUARD": "decl"}
FLAGS = ("MIS_DEF_FORCED_HIGH", "MIS_DEF_MATE_HIGH", "MIS_DECL_LEAD_GUARD")
FIELDS = ("expert", "jeu")


def set_flags(params):
    for k in FLAGS:
        setattr(expert_bot, k, params.get(k, False))


def play(bidding, hands, dealer, whisted, field):
    contract = bidding.locked_contract
    role, decl = declarer_role_and_seats(contract)
    game = dispatch_play_phase2(contract, [list(h) for h in hands], dealer, bidding.bid_summary, whisted)
    with torch.no_grad():
        while not game.done:
            seat = game.turn
            if field == "jeu" and game_ai.hybrid_applies(contract, seat):
                mask = torch.tensor(game.legal_action_mask(seat), dtype=torch.bool)
                obs = torch.tensor(game.observation(seat), dtype=torch.float32)
                card = game_ai.NETS[game.role_of(seat)].act_greedy(obs, mask)
            else:
                card = expert_bot.choose_card(game, seat)
            game.step(seat, card)
    pts = real_points_by_seat(role, game, contract)
    limit = 1 if contract["kind"] == "piccolo" else 0
    falls = sum(1 for s in decl if game.tricks_won[s] != limit)
    d = sum(pts[s] for s in decl)
    return d, sum(pts[s] for s in range(4) if s not in decl), falls, len(decl)


def chunk(args):
    seed, deadline = args
    torch.set_num_threads(1)
    rng = random.Random(seed)
    chain = DeckChain(game_ai.WHISTED_RATIO, rng)
    # (sorte, champ, variante) -> listes ; "seen" = donnes tirees
    out = {"seen": 0, "n": collections.Counter(), "rows": collections.defaultdict(list)}
    while time.time() < deadline:
        hands, dealer, whisted = chain.deal(rng)
        chain.finish_unplayed(hands, dealer)
        out["seen"] += 1
        set_flags({})
        bidding, _ = game_ai.run_bidding(hands, dealer, detect_trou(hands))
        if bidding.general_pass or bidding.locked_contract["type"] != "misere":
            continue
        kind = bidding.locked_contract["kind"]
        out["n"][kind] += 1
        for field in FIELDS:
            res = {}
            for name, params in VARIANTS.items():
                set_flags(params)
                res[name] = play(bidding, hands, dealer, whisted, field)
            set_flags({})
            ref = res["IDENTITE"]
            for name, r in res.items():
                out["rows"][(kind, field, name)].append(
                    (r[0] - ref[0], r[1] - ref[1], r[2], ref[2], r[3], ref[0]))
    return out


def stats(xs):
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(var / n)
    return m, se, (m / se if se > 0 else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=300)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--weights", default="godot_weights_ingame")
    a = p.parse_args()
    load_game_ai(a.weights)
    t0 = time.time()
    deadline = t0 + 60 * a.minutes
    with mp.get_context("fork").Pool(a.workers) as pool:
        parts = pool.map(chunk, [(SEED + 1000 * i, deadline) for i in range(a.workers)])
    seen = sum(x["seen"] for x in parts)
    n = sum((x["n"] for x in parts), collections.Counter())
    rows = collections.defaultdict(list)
    for x in parts:
        for k, v in x["rows"].items():
            rows[k].extend(v)
    print("=== %d donnes, contrats de misere : %s ; %.0f min ===" % (seen, dict(n), (time.time() - t0) / 60))
    kinds = ["petite"] + sorted(k for k in n if k != "petite")
    verdict = {}
    for kind in kinds:
        for field in FIELDS:
            print("\n--- misere %s, champ %s (%d donnes) ---" % (kind, field, n[kind]))
            print("  %-15s %-5s %16s %16s %12s %12s" % ("variante", "camp", "gain camp", "d decl pts", "chutes ref", "chutes var"))
            for name in VARIANTS:
                r = rows.get((kind, field, name), [])
                if not r:
                    continue
                camp = CAMP[name]
                g = stats([x[0] if camp == "decl" else x[1] for x in r])
                dd = stats([x[0] for x in r])
                decl_n = sum(x[4] for x in r)
                fv = sum(x[2] for x in r) / max(1, decl_n)
                fr = sum(x[3] for x in r) / max(1, decl_n)
                print("  %-15s %-5s %+7.3f z %+6.1f %+7.3f z %+6.1f %11.1f%% %11.1f%%" % (
                    name, camp, g[0], g[2], dd[0], dd[2], 100 * fr, 100 * fv))
                verdict.setdefault(name, []).append((kind, field, g[2]))
    print("\n=== critere (petite/expert z > 3, aucune cellule z < -2) ===")
    for name, cells in verdict.items():
        if name == "IDENTITE":
            continue
        main_z = next((z for k, f, z in cells if k == "petite" and f == "expert"), 0.0)
        worst = min(cells, key=lambda c: c[2])
        ok = main_z > 3 and worst[2] >= -2
        print("  %-15s petite/expert z %+.1f ; pire %s/%s z %+.1f -> %s" % (
            name, main_z, worst[0], worst[1], worst[2], "PROMUE" if ok else "rejetee"))


if __name__ == "__main__":
    main()
