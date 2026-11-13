"""IA LIBRE contre references (2026-11-01) : la porte de train_free.py ne mesure
les membres que contre la championne. Ici, chaque ligne remplace une base par un
candidat au jeu de la carte, sur des donnes neuves, sans rien ecrire.

Meme protocole que la porte (3 parties par donne, vrais points) : base aux 4
sieges, candidat au camp declarant, candidat en defense ; gain = somme des deux
ecarts. Le contrat est le meme dans les 3 parties : seule la carte differe.
Deux encheres : `net` (l'encherisseur fixe de l'entrainement) et `expert`
(ExpertBot, celle du jeu actuel). Joueurs : expert, hasard, championne (avec ses
garde-fous, comme en jeu), best et membres de checkpoints_free (cartes legales).

HYBRIDE (2026-11-08, conseil) : joueur "hybride" = best sur les contrats de
--hybrid-roles (ex. piccolo,grande_misere), ExpertBot ailleurs ; c'est ce qu'on
brancherait en jeu. --pairs choisit les lignes, --seed des donnes neuves
(ne pas juger l'hybride sur les donnes qui ont servi a choisir ses contrats).

ROUTAGE (2026-11-09, conseil) : joueur "routage" = best seulement sur les couples
(contrat, camp) de --route (ex. piccolo,emballage:decl,solitaire:def ; un contrat
seul = les deux camps), ExpertBot ailleurs. --auto-route : 1) best a la place
d'ExpertBot sur --seed, par contrat ET par camp ; 2) garde les couples
gain - 2 se > 0 (n >= 200) ; 3) mesure routage contre ExpertBot et contre
l'hybride actuel sur --seed + 1000003 (donnes jamais vues par la selection).

CHAMP (2026-11-13, 15e conseil, critique) : --others p1,p2 (profils de
expert_bot.PROFILES) = les sieges ADVERSES du camp mesure jouent (et tout le monde
encherit) avec ce profil ; le camp mesure joue candidat puis base (ExpertBot par
defaut), adversaires identiques dans les 2 parties. Mesure si le gain du routage
tient contre autre chose qu'ExpertBot (acte1, joueur_0927, bruites).

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
HYBRID_ROLES = set()  # herite par fork
CUR_ROLE = [None]  # contrat de la donne en cours (par processus)
CUR_DECL = [frozenset()]  # sieges du camp declarant de la donne en cours
ROUTE = [frozenset()]  # couples (contrat, "decl"/"def") du joueur routage, passe par match_chunk


def parse_route(spec):
    out = set()
    for tok in (t for t in spec.split(",") if t):
        role, _, side = tok.partition(":")
        out.update([(role, side)] if side else [(role, "decl"), (role, "def")])
    if any(side not in ("decl", "def") for _, side in out):
        raise SystemExit("--route : camp decl ou def")
    return frozenset(out)


def choose(player, game, seat, rng):
    if player == "expert":
        return expert_bot.choose_card(game, seat)
    if player in ("hybride", "routage"):
        if player == "hybride":
            use = CUR_ROLE[0] in HYBRID_ROLES
        else:
            use = (CUR_ROLE[0], "decl" if seat in CUR_DECL[0] else "def") in ROUTE[0]
        if not use:
            return expert_bot.choose_card(game, seat)
        player = "best"
    if player == "hasard":
        return rng.choice([a for a, ok in enumerate(game.legal_action_mask(seat)) if ok])
    mask = ai_play_mask(game, seat) if player == "championne" else game.legal_action_mask(seat)
    obs = torch.tensor(game.observation(seat), dtype=torch.float32)
    return NETS[player][game.role_of(seat)].act_greedy(obs, torch.tensor(mask, dtype=torch.bool))


def match_chunk(args):
    cand, base, bid, deals, seed, route, others = args
    ROUTE[0] = route
    prof = expert_bot.PROFILES[others] if others else None
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

    def play_field(contract, role, hands, dealer, bidding, whisted, team, player):
        """CHAMP : team joue player (reglages par defaut), les autres = ExpertBot au profil du champ."""
        game = dispatch_play_phase2(contract, [list(h) for h in hands], dealer, bidding.bid_summary, whisted)
        while not game.done:
            seat = game.turn
            if seat in team:
                expert_bot.set_bid_params({})
                card = choose(player, game, seat, play_rng)
            else:
                expert_bot.set_bid_params(prof)
                card = expert_bot.choose_card(game, seat)
            game.step(seat, card)
        expert_bot.set_bid_params({})
        return real_points_by_seat(role, game, contract)

    with torch.no_grad():
        for _ in range(deals):
            hands, dealer, whisted = chain.deal(rng)
            chain.finish_unplayed(hands, dealer)
            bidding = Bidding(dealer)
            trou = detect_trou(hands)
            if trou is not None:
                bidding.seed_trou(trou)
            if prof is not None:
                expert_bot.set_bid_params(prof)
            _bid(bidding, hands, {0, 1, 2, 3} if bid == "net" else set(), BIDDER)
            expert_bot.set_bid_params({})
            if bidding.general_pass:
                continue
            contract = bidding.locked_contract
            role, decl = declarer_role_and_seats(contract)
            defense = {s for s in range(4) if s not in decl}
            CUR_ROLE[0] = role
            CUR_DECL[0] = frozenset(decl)
            r = rows.setdefault(role, {"decl": [], "def": []})
            if prof is not None:
                deal = (contract, role, hands, dealer, bidding, whisted)
                for side, team in (("decl", set(decl)), ("def", defense)):
                    a = play_field(*deal, team, cand)
                    b = play_field(*deal, team, base)
                    r[side].append(sum(a[s] - b[s] for s in team))
                continue
            ref = play(contract, role, hands, dealer, bidding, whisted, set())
            with_decl = play(contract, role, hands, dealer, bidding, whisted, set(decl))
            with_def = play(contract, role, hands, dealer, bidding, whisted, defense)
            r["decl"].append(sum(with_decl[s] - ref[s] for s in decl))
            r["def"].append(sum(with_def[s] - ref[s] for s in defense))
    return rows


def run_pair(pool, cand, base, bid, deals, procs, seed=SEED, route=frozenset(), others=None):
    parts = pool.map(match_chunk, [(cand, base, bid, deals // procs, seed + 10 * i, route, others)
                                   for i in range(procs)],
                     chunksize=1)
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
    by_role = {}
    for role, r in sorted(rows.items()):
        g, g_se = champion_play.mean_se([a + b for a, b in zip(r["decl"], r["def"])])
        d, d_se = champion_play.mean_se(r["decl"])
        f, f_se = champion_play.mean_se(r["def"])
        by_role[role] = {"n": len(r["decl"]), "gain": round(g, 2), "se": round(g_se, 2), "decl": round(d, 3),
                         "decl_se": round(d_se, 3), "def": round(f, 3), "def_se": round(f_se, 3)}
    return {"candidat": cand, "base": base, "enchere": bid, "champ": others or "-", "donnes": len(decl), "gain": round(gain, 3),
            "se": round(se, 3), "declarant": round(m_decl, 3), "declarant_se": round(se_decl, 3),
            "defense": round(m_def, 3), "defense_se": round(se_def, 3), "par_contrat": by_role}


def main():
    global BIDDER
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pairs", default="", help="lignes candidat:base separees par des virgules (vide = toutes)")
    parser.add_argument("--bids", default=",".join(BIDS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--hybrid-roles", default="", help="voir HYBRIDE")
    parser.add_argument("--route", default="", help="voir ROUTAGE")
    parser.add_argument("--auto-route", action="store_true", help="voir ROUTAGE (enchere expert)")
    parser.add_argument("--others", default="", help="voir CHAMP : profils separes par des virgules")
    parser.add_argument("--seeds", default="", help="graines separees par des virgules (CHAMP ; defaut --seed)")
    args = parser.parse_args()
    fields = [x for x in args.others.split(",") if x]
    for f in fields:
        if f not in expert_bot.PROFILES:
            raise SystemExit("--others : profil inconnu %s" % f)
    if fields and (args.auto_route or args.bids != "expert"):
        raise SystemExit("--others : --bids expert seulement, sans --auto-route")
    route = parse_route(args.route)
    if args.auto_route and not args.hybrid_roles:
        raise SystemExit("--auto-route compare a l'hybride : donner --hybrid-roles")
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",")] if args.pairs else list(PAIRS)
    HYBRID_ROLES.update(x for x in args.hybrid_roles.split(",") if x)
    if any("hybride" in p for p in pairs) and not HYBRID_ROLES:
        raise SystemExit("hybride sans --hybrid-roles")

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

    def line(cand, base, bid, seed, route, others=None):
        t0 = time.time()
        r = run_pair(pool, cand, base, bid, args.deals, args.workers, seed, route, others)
        verdict = "DEVANT" if r["gain"] - 2 * r["se"] > 0 else ("DERRIERE" if r["gain"] + 2 * r["se"] < 0 else "egalite")
        z = r["gain"] / r["se"] if r["se"] else 0.0
        print("=== enchere %-6s champ %-12s graine %d %-10s a la place de %-10s : %+.3f +- %.3f (z %+.1f, "
              "declarant %+.3f, defense %+.3f, %d donnes, %.0fs) -> %s ===" % (
                  bid, others or "-", seed, cand, base, r["gain"], r["se"], z, r["declarant"], r["defense"],
                  r["donnes"], time.time() - t0, verdict), flush=True)
        print("RESULT " + json.dumps(r), flush=True)
        return r

    if args.auto_route:
        sel = line("best", "expert", "expert", args.seed, route)
        picked = set()
        for role, c in sel["par_contrat"].items():
            for side in ("decl", "def"):
                ok = c["n"] >= 200 and c[side] - 2 * c[side + "_se"] > 0
                print("selection %-16s %-4s n %6d : %+.3f +- %.3f%s" % (role, side, c["n"], c[side], 2 * c[side + "_se"],
                                                                     "  <- GARDE" if ok else ""), flush=True)
                if ok:
                    picked.add((role, side))
        route = frozenset(picked)
        print("ROUTE " + ",".join("%s:%s" % rs for rs in sorted(route)), flush=True)
        seed2 = args.seed + 1000003
        line("routage", "expert", "expert", seed2, route)
        line("routage", "hybride", "expert", seed2, route)
        line("hybride", "expert", "expert", seed2, route)
    elif fields:
        seeds = [int(x) for x in args.seeds.split(",") if x] or [args.seed]
        for seed in seeds:
            for f in fields:
                for cand, base in pairs:
                    line(cand, base, "expert", seed, route, f)
    else:
        for bid in args.bids.split(","):
            for cand, base in pairs:
                line(cand, base, bid, args.seed, route)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
