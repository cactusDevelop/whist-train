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

ENCHERE PAR CELLULE (2026-11-13, 15e conseil, etape 2) : --bid-cells. Cartes =
routage aux 4 sieges (comme en jeu), encheres ExpertBot. 1) sur --seed, a chaque
decision d'enchere ou l'encherisseur reseau differe d'ExpertBot : on rejoue la donne
avec la decision reseau a CE seul point (ExpertBot ailleurs) ; gain = vrais points
du siege qui a devie. Cellule = (categorie ExpertBot, categorie reseau), sans
couleur. 2) garde les cellules gain - 2 se > 0 (n >= 200). 3) sur --seed + 1000003 :
un siege encherit "routage d'enchere" (reseau seulement dans les cellules gardees)
contre ExpertBot, 4 rotations ; critere gain > 3 SE. --seeds : graines de
confirmation (defaut --seed + 1000003) ; --others : confirmation aussi face a ces
champs (les 3 autres sieges encherissent ET jouent ExpertBot au profil du champ,
le siege mesure joue la carte en routage ; base = meme siege avec l'enchere ExpertBot).

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
from agents.bid_route import bid_label
from agents.play_guards import ai_play_mask
from engine.bidding import Bidding, bid_action_to_dict, legal_action_mask_with_hand, observation as bid_observation
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


DEALER = [0]  # donneur de la donne en cours (par processus)
BID_ROUTE = [frozenset()]  # cellules (cat ExpertBot, cat reseau) ou le reseau encherit


def net_bid(bidding, seat, hand):
    if getattr(BIDDER, "hybrid", False):
        return BIDDER.decide(bidding, seat, hand)
    obs = torch.tensor(bid_observation(bidding, seat, hand), dtype=torch.float32)
    mask = torch.tensor(legal_action_mask_with_hand(bidding, seat, hand), dtype=torch.bool)
    return BIDDER.act_greedy(obs, mask)


BASE_ROUTE = [None]  # --bid-iterate : routage d'enchere en place (agents/bid_route.BID_ROUTE), sinon None


def run_bidding(hands, dealer, trou, override=None, routed_seat=None, trace=None, prof=None, me=None):
    """ExpertBot partout ; override=(siege, k, choix) : k-ieme decision du siege remplacee ;
    routed_seat : ce siege suit BID_ROUTE ; trace : liste remplie de (siege, k, cellule, alternative, signe) ;
    prof : profil ExpertBot des sieges autres que me (champ). BASE_ROUTE (--bid-iterate) : les autres
    sieges (seulement me face a un champ) suivent le routage en place."""
    bidding = Bidding(dealer)
    if trou is not None:
        bidding.seed_trou(trou)
    count = [0, 0, 0, 0]
    while not bidding.finished:
        seat = bidding.turn
        hand = hands[seat]
        k = count[seat]
        count[seat] += 1
        if override is not None and override[0] == seat and override[1] == k:
            choice = override[2]
        else:
            if prof is not None:
                expert_bot.set_bid_params({} if seat == me else prof)
            choice = expert_bot.choose_bid(bidding, seat, hand)
            if prof is not None:
                expert_bot.set_bid_params({})
            if seat == routed_seat:
                cells = BID_ROUTE[0]
            elif BASE_ROUTE[0] is not None and (prof is None or seat == me):
                cells = BASE_ROUTE[0]
            else:
                cells = None
            if trace is not None or cells is not None:
                n = net_bid(bidding, seat, hand)
                if bid_label(n) != bid_label(choice):
                    cell = (bid_label(choice, False), bid_label(n, False))
                    expert = choice
                    if cells is not None and cell in cells:
                        choice = n
                    if trace is not None:
                        # gain toujours oriente "reseau moins ExpertBot" : deviation vers l'autre choix
                        trace.append((seat, k, cell, expert, -1) if choice is n else (seat, k, cell, n, 1))
        if not expert_bot.apply_choice(bidding, seat, choice):
            raise AssertionError("enchere illegale %r siege %d" % (choice, seat))
    return bidding


def bid_cells_chunk(args):
    deals, seed, route, confirm, others, base_route = args
    BASE_ROUTE[0] = base_route
    prof = expert_bot.PROFILES[others] if others else None
    ROUTE[0] = route
    rng = random.Random(seed)
    play_rng = random.Random(seed + 1)
    chain = DeckChain(MEASURE_WHISTED_RATIO, rng)
    cells = {}
    per_deal = []

    def points(bidding, hands, whisted, me=None):
        if bidding.general_pass:
            return [0.0] * 4
        contract = bidding.locked_contract
        role, decl = declarer_role_and_seats(contract)
        CUR_ROLE[0] = role
        CUR_DECL[0] = frozenset(decl)
        game = dispatch_play_phase2(contract, [list(h) for h in hands], DEALER[0], bidding.bid_summary, whisted)
        while not game.done:
            seat = game.turn
            if prof is not None and seat != me:
                expert_bot.set_bid_params(prof)
                card = expert_bot.choose_card(game, seat)
                expert_bot.set_bid_params({})
            else:
                card = choose("routage", game, seat, play_rng)
            game.step(seat, card)
        return real_points_by_seat(role, game, contract)

    with torch.no_grad():
        for _ in range(deals):
            hands, dealer, whisted = chain.deal(rng)
            chain.finish_unplayed(hands, dealer)
            DEALER[0] = dealer
            trou = detect_trou(hands)
            if confirm:
                base = None if prof is not None else points(run_bidding(hands, dealer, trou), hands, whisted)
                g = 0.0
                for s in range(4):
                    if prof is not None:
                        ref = points(run_bidding(hands, dealer, trou, prof=prof, me=s), hands, whisted, s)
                    else:
                        ref = base
                    b = run_bidding(hands, dealer, trou, routed_seat=s, prof=prof, me=s)
                    g += points(b, hands, whisted, s if prof is not None else None)[s] - ref[s]
                per_deal.append(g)
                continue
            trace = []
            base = points(run_bidding(hands, dealer, trou, trace=trace), hands, whisted)
            for seat, k, cell, alt, sign in trace:
                b = run_bidding(hands, dealer, trou, override=(seat, k, alt))
                cells.setdefault(cell, []).append(sign * (points(b, hands, whisted)[seat] - base[seat]))
    return cells, per_deal


def run_bid_cells(pool, deals, procs, seed, route, confirm, others=None):
    parts = pool.map(bid_cells_chunk, [(deals // procs, seed + 10 * i, route, confirm, others, BASE_ROUTE[0]) for i in range(procs)],
                     chunksize=1)
    cells, per_deal = {}, []
    for c, p in parts:
        for k, v in c.items():
            cells.setdefault(k, []).extend(v)
        per_deal += p
    return cells, per_deal


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
    parser.add_argument("--bid-cells", action="store_true", help="voir ENCHERE PAR CELLULE (cartes = --route)")
    parser.add_argument("--min-n", type=int, default=200, help="--bid-cells : decisions minimum par cellule (smoke : petit)")
    parser.add_argument("--bid-iterate", action="store_true",
                        help="--bid-cells en partant du routage d'enchere en place (agents/bid_route.BID_ROUTE)")
    parser.add_argument("--seeds", default="", help="graines separees par des virgules (CHAMP ; defaut --seed)")
    args = parser.parse_args()
    fields = [x for x in args.others.split(",") if x]
    for f in fields:
        if f not in expert_bot.PROFILES:
            raise SystemExit("--others : profil inconnu %s" % f)
    if fields and not args.bid_cells and (args.auto_route or args.bids != "expert"):
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

    if args.bid_iterate:
        if not args.bid_cells:
            raise SystemExit("--bid-iterate va avec --bid-cells")
        from agents.bid_route import BID_ROUTE as CURRENT_ROUTE
        BASE_ROUTE[0] = CURRENT_ROUTE
        print("ITERATION depuis le routage en place (%d cellules) : base = routage aux 4 sieges ; cellule = gain "
              "reseau moins ExpertBot dans ce contexte ; confirmation = nouveau routage a la place de l'ancien" % len(
                  CURRENT_ROUTE), flush=True)
    if args.bid_cells:
        t0 = time.time()
        cells, _ = run_bid_cells(pool, args.deals, args.workers, args.seed, route, False)
        n_dec = sum(len(v) for v in cells.values())
        print("CELLULES graine %d : %d decisions divergentes sur %d donnes (%.0fs)" % (
            args.seed, n_dec, args.deals, time.time() - t0), flush=True)
        picked = set()
        for cell, xs in sorted(cells.items(), key=lambda kv: -len(kv[1])):
            g, se = champion_play.mean_se(xs)
            ok = len(xs) >= args.min_n and g - 2 * se > 0
            if BASE_ROUTE[0] is not None and cell in BASE_ROUTE[0]:
                ok = not (len(xs) >= args.min_n and g + 2 * se < 0)  # en place : retiree seulement si nettement negative
            if ok:
                picked.add(cell)
            print("cellule expert %-22s reseau %-22s n %7d : %+.3f +- %.3f (z %+.1f)%s" % (
                cell[0], cell[1], len(xs), g, se, g / se if se else 0.0, "  <- GARDE" if ok else ""), flush=True)
            print("CELL " + json.dumps({"expert": cell[0], "reseau": cell[1], "n": len(xs), "gain": round(g, 3),
                                        "se": round(se, 3)}), flush=True)
        print("BID_ROUTE " + ",".join("%s>%s" % c for c in sorted(picked)), flush=True)
        if BASE_ROUTE[0] is not None:
            picked |= {c for c in BASE_ROUTE[0] if c not in cells}  # jamais vue : gardee
            print("AJOUTEES " + ",".join("%s>%s" % c for c in sorted(picked - BASE_ROUTE[0])), flush=True)
            print("RETIREES " + ",".join("%s>%s" % c for c in sorted(BASE_ROUTE[0] - picked)), flush=True)
            if picked == BASE_ROUTE[0]:
                print("ROUTAGE INCHANGE : rien a confirmer", flush=True)
                picked = set()
        if picked:
            BID_ROUTE[0] = frozenset(picked)
            pool.close()
            pool.join()
            pool = multiprocessing.get_context("fork").Pool(args.workers, train_free.init_worker)
            seeds = [int(x) for x in args.seeds.split(",") if x] or [args.seed + 1000003]
            for seed2 in seeds:
                for f in [None] + fields:
                    t0 = time.time()
                    _, per_deal = run_bid_cells(pool, args.deals, args.workers, seed2, route, True, f)
                    g, se = champion_play.mean_se(per_deal)
                    print("=== CONFIRMATION graine %d champ %-12s : routage d'enchere a la place d'ExpertBot (4 rotations) : "
                          "%+.3f +- %.3f (z %+.1f, %d donnes, %.0fs) -> %s ===" % (
                              seed2, f or "-", g, se, g / se if se else 0.0, len(per_deal), time.time() - t0,
                              "DEVANT" if g - 3 * se > 0 else ("DERRIERE" if g + 3 * se < 0 else "egalite")), flush=True)
    elif args.auto_route:
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
