"""PIMC (dds sur mondes echantillonnes) contre ExpertBot, en duplicate.
Mesure seule, cloud (conseil 09/11 : « verifier que la recherche bat
ExpertBot AVANT de la faire imiter par le reseau »).

Encheres ExpertBot reelles. Pour chaque donne a atout (emballage, solo,
trou ; dds.supported), la meme donne est jouee 3 fois :
  base : 4 ExpertBot ;
  def  : les defenseurs jouent PIMC (carte de Q max, Q = vrais points moyens
         sur N mondes compatibles avec ce que le siege sait, suite parfaite
         a cartes ouvertes : agents/rollouts.dds_values), le reste ExpertBot ;
  decl : idem pour le camp declarant.
Gain = points moyens par siege PIMC (variante - base), meme donne.
Sort aussi les situations ou PIMC s'ecarte d'ExpertBot (pistes de regles).

Usage : python -u scripts/pimc_vs_expert.py --minutes 60 --workers 4 --samples 8

--oracle (10e conseil 11/11, apres l'echec de la distillation) : DEFENSE
D'EMBALLAGE seule, par profil adverse (--others, expert_bot.PROFILES), avec
deux variantes sur la meme donne :
  def  : PIMC a toutes les decisions de defense (plafond) ;
  trig : PIMC seulement quand le pli est entame et que l'adversaire le gagne
         (fournir/renoncer en 2e-4e : la ou l'eleve perdait le plus) ;
         ExpertBot ailleurs.
Si ni def ni trig ne gagne > 3 SE sur les profils -> piste PIMC close,
pivot encheres. Sinon : extraire des regles des ecarts (trig) -> ExpertBot.

--camps decl --validate 32 (conseil 14/11 soir, critique) : le gap max(Q)-Q(expert)
sur les memes 8 mondes est biaise (max de bruit, toujours > 0, A/B 24 trompe).
A chaque ecart, Q(expert) et Q(pimc) sont re-estimes sur 32 mondes NEUFS
(independants du choix) -> gap valide sans biais, peut etre negatif ; table par
situation avec SE. Situation retenue si gap valide > 3 SE avec n >= 100.

--expert-q (conseil 16/11 apres-midi, palier emb_draw : gap Q z+14 mais gain reel z+0.7) :
avec --validate, le gap est AUSSI re-estime sur les MEMES mondes neufs avec une suite
jouee par ExpertBot aux 4 sieges (Q realiste) au lieu du dds a cartes ouvertes.
Tableau par donne CHANGEE (>= 1 carte PIMC) : gain reel, somme des gaps dds / expert,
pente gain ~ somme des gaps. Variante "rnd" (--control) : carte legale au hasard
(autre que celle d'ExpertBot) aux memes declenchements -> calibre ce que bouge 1 carte.
Pente ~0 et gap expert ~0 -> strategy fusion, piste close.
"""

import argparse
import math
import multiprocessing as mp
import random
import sys
import time

sys.path.insert(0, ".")

from agents import expert_bot
from agents.rollouts import clone_with_hands, dds_values, determinize
from engine import dds
from engine.cards import NUM_PLAYERS, suit_of, trick_winner
from engine.whisted import DeckChain
from scripts.expert_bot_match import run_bot_bidding
from scripts.expert_play_audit import card_txt, describe, situation
from scripts.team_errors import teammates
from scripts.train import MEASURE_WHISTED_RATIO, declarer_role_and_seats, dispatch_play_phase2, real_points_by_seat


def new_game(contract, hands, dealer, bidding, whisted):
    return dispatch_play_phase2(contract, [list(h) for h in hands], dealer, bidding.bid_summary, whisted)


def emb_draw(game, seat, card=None):
    """Conseil 16/11 (plafond) : PIMC seulement quand l'emballeur ExpertBot ENTAMERAIT atout."""
    return (not game.current_trick and card is not None and suit_of(card) == game.trump_suit
            and game.public_contract().get("type") == "emballage" and seat in set(dds.side_seats(game)))


def emb_draw2(game, seat, card=None):
    """Conseil 16/11 soir : emb_draw restreint a l'emballeur avec exactement 2 atouts en main
    (seule situation dont le gap tient avec suite ExpertBot : +0.43 z7.6 / +0.52 z12.8, run 43883236250).
    Confirmation sur donnes neuves, critere fixe d'avance : gain decl z > 2 ET gain donnes changees > +0.10."""
    return (emb_draw(game, seat, card)
            and sum(1 for c in game.hands[seat] if suit_of(c) == game.trump_suit) == 2)


def emb_def_lead0(game, seat, card=None):
    """Conseil 17/11 : defenseur d'emballage SANS atout a l'entame, pli 6+ (8 cartes ou moins) -- seul gap
    defense RETENU (run 44055110607 : +0.26 +- 0.06 z 4.2 suite ExpertBot). Confirmation sur donnes neuves,
    critere fige : gain def emballage sur donnes changees > +0.10 ET z > 2, controle rnd tres negatif."""
    return (not game.current_trick and game.trump_suit is not None
            and game.public_contract().get("type") == "emballage" and seat not in set(dds.side_seats(game))
            and len(game.hands[seat]) <= 8 and not any(suit_of(c) == game.trump_suit for c in game.hands[seat]))


TRIGGERS = {"all": None, "emb_draw": emb_draw, "emb_draw2": emb_draw2, "emb_def_lead0": emb_def_lead0}


def adv_wins_trick(game, seat, card=None):
    """Pli entame et gagne (pour l'instant) par un adversaire de `seat`."""
    trick = game.current_trick
    return bool(trick) and trick_winner(trick, game.trump_suit) not in teammates(game, seat)


def expert_values(game, seat, cards, score_fn, rng, samples):
    """Q realiste (16/11) : memes mondes que dds_values avec ce rng, suite jouee par ExpertBot aux 4 sieges."""
    q = [0.0] * len(cards)
    for _ in range(samples):
        hands = determinize(game, seat, rng)
        for i, c in enumerate(cards):
            g = clone_with_hands(game, hands)
            g.step(seat, c)
            while not g.done:
                g.step(g.turn, expert_bot.choose_card(g, g.turn))
            q[i] += score_fn(g)[seat] / samples
    return q


DUMP_EX = 4  # --dump N : exemples par situation (N > 4 : tous les ecarts, pas seulement gap >= 1)
BRANCH_KEY = False  # --branch-key (conseil 17/11 soir) : branche d entame ExpertBot (+ _M si maitresse) ajoutee a la situation
PIMC_TIMES = []  # secondes par decision PIMC (dds_values, hors --validate) : budget CPU avant tout port


def play(game, pimc_seats, score_fn, rng, samples, diffs, trigger=None, validate=0, vrng=None, acc=None,
         expert_q=False, randomize=False):
    """Joue la donne ; `pimc_seats` jouent PIMC (seulement si trigger(game, seat) quand il est donne).
    `acc` (dict) recoit changed / vgap / egap sommes sur la donne. randomize : carte legale au hasard
    (autre qu'ExpertBot) au lieu de PIMC (controle). -> nombre de decisions PIMC."""
    n = 0
    while not game.done:
        seat = game.turn
        card = expert_bot.choose_card(game, seat)
        branch = expert_bot.LEAD_BRANCH if BRANCH_KEY and not game.current_trick else None
        trou_lead = getattr(game, "trou_pending", lambda: False)() and seat == game.partner
        if seat in pimc_seats and not trou_lead and (trigger is None or trigger(game, seat, card)):  # l'entame du bouche-trou fixe le Trou (regle) : jamais PIMC
            legal = [c for c, ok in enumerate(game.legal_action_mask(seat)) if ok]
            if randomize and len(legal) > 1:
                card = rng.choice([c for c in legal if c != card])
                if acc is not None:
                    acc["changed"] += 1
            elif len(legal) > 1 and game.trump_suit is not None and dds.supported(game):
                t_dec = time.perf_counter()
                q = dds_values(game, seat, legal, score_fn, rng, samples)
                if q is not None:
                    PIMC_TIMES.append(time.perf_counter() - t_dec)
                    n += 1
                    best = max(q)
                    qe = q[legal.index(card)]
                    if best - qe > 1e-6:  # ExpertBot garde la main a egalite
                        pick = legal[q.index(best)]
                        key = sit_key(game, seat, card) + ("" if branch is None else " b:" + branch)
                        vgap = egap = None
                        if validate:
                            vstate = vrng.getstate()
                            q2 = dds_values(game, seat, [card, pick], score_fn, vrng, validate)
                            if q2 is not None:
                                vgap = q2[1] - q2[0]
                            if expert_q:
                                vrng.setstate(vstate)  # memes mondes neufs que le dds
                                q3 = expert_values(game, seat, [card, pick], score_fn, vrng, validate)
                                egap = q3[1] - q3[0]
                        if acc is not None:
                            acc["changed"] += 1
                            acc["vgap"] += vgap or 0.0
                            acc["egap"] += egap or 0.0
                            if egap is not None:
                                acc["e"].append((key, egap, vgap))
                        diffs.append((key, best - qe, vgap,
                                      "%s | siege %d | pli %s | main %s | expert %s (%.1f) pimc %s (%.1f)" % (
                                          describe(game.public_contract()), seat,
                                          " ".join("%d:%s" % (s, card_txt(c)) for s, c in game.current_trick) or "-",
                                          " ".join(card_txt(c) for c in sorted(game.hands[seat])),
                                          card_txt(card), qe, card_txt(pick), best)
                                      + ("" if vgap is None else " | valide dds %+.1f" % vgap)
                                      + ("" if egap is None else " expert %+.1f" % egap)
                                      + ("" if branch is None else " | atout %d branche %s pimc %s" % (
                                          game.trump_suit, branch, "M" if _is_master_now(game, seat, pick) else "-"))))
                        card = pick
        game.step(seat, card)
    return n


def _is_master_now(game, seat, card):
    """Carte maitresse vue du siege (meme definition qu ExpertBot.PublicView.is_master)."""
    v = expert_bot.PublicView(game, seat)
    return v.is_master(expert_bot.mirror_card(card) if v.mirrored else card)


def sit_key(game, seat, card):
    """10/11 (conseil) : situation + type de contrat ; entames : atouts en main et numero du pli."""
    sit = situation(game, seat, card)
    typ = game.public_contract().get("type", "?") if hasattr(game, "public_contract") else "?"
    if not game.current_trick:
        nt = sum(1 for c in game.hands[seat] if suit_of(c) == game.trump_suit)
        pli = 13 - len(game.hands[seat]) + 1
        sit += " atouts%s pli%s" % (min(nt, 4) if nt < 4 else "4+", "1" if pli == 1 else ("2-5" if pli <= 5 else "6+"))
    return "%s %s" % (typ, sit)


def worker(args):
    seed, samples, deadline, camps, validate, others, trig, expert_q, control = args
    trigger = TRIGGERS[trig]
    expert_bot.set_bid_params(dict(expert_bot.PROFILES[others]))  # ExpertBot (base + sieges non PIMC) sous ce profil
    rng = random.Random(seed)
    play_rng = random.Random(seed + 1)
    vrng = random.Random(seed + 2)
    chain = DeckChain(MEASURE_WHISTED_RATIO, rng)
    rows = []  # (variante, type contrat, gain moyen par siege PIMC, decisions, changees, somme vgap, somme egap)
    diffs = {"def": [], "decl": []}
    egaps = []  # (situation, gap expert, gap dds) sur mondes neufs
    while time.time() < deadline:
        hands, dealer, whisted = chain.deal(rng)
        bidding = run_bot_bidding(hands, dealer)
        if bidding.general_pass:
            chain.finish_unplayed(hands, dealer)
            continue
        contract = bidding.locked_contract
        role, _decl = declarer_role_and_seats(contract)
        base = new_game(contract, hands, dealer, bidding, whisted)
        play(base, (), None, play_rng, samples, [])
        chain.finish_played(base, dealer, contract["type"])
        if contract["type"] not in ("emballage", "solo", "trou") or base.trump_suit is None:
            continue
        if trig in ("emb_draw", "emb_draw2", "emb_def_lead0") and contract["type"] != "emballage":
            continue
        pts_base = real_points_by_seat(role, base, contract)
        side = set(dds.side_seats(base))

        def score_fn(g):
            return real_points_by_seat(role, g, contract)
        variants = [("def", [s for s in range(NUM_PLAYERS) if s not in side]), ("decl", sorted(side))]
        if control:  # controle sur le camp mesure (defense si --camps def seul)
            variants.append(("rnd", sorted(side) if "decl" in camps else variants[0][1]))
        for name, seats in variants:
            if name not in camps and name != "rnd":
                continue
            g = new_game(contract, hands, dealer, bidding, whisted)
            acc = {"changed": 0, "vgap": 0.0, "egap": 0.0, "e": egaps}
            n = play(g, set(seats), score_fn, play_rng, samples, diffs.get(name, []), trigger, validate=validate,
                     vrng=vrng, acc=acc, expert_q=expert_q, randomize=(name == "rnd"))
            pts = real_points_by_seat(role, g, contract)
            gain = sum(pts[s] - pts_base[s] for s in seats) / len(seats)
            rows.append((name, contract["type"], gain, n, acc["changed"], acc["vgap"], acc["egap"]))
            if time.time() >= deadline:
                break
    return rows, diffs, list(PIMC_TIMES), egaps


def oracle_worker(args):
    """--oracle : defense d'emballage, variantes def (PIMC partout) et trig (PIMC si l'adversaire gagne le pli)."""
    seed, samples, deadline, others = args
    expert_bot.set_bid_params(dict(expert_bot.PROFILES[others]))
    rng = random.Random(seed)
    play_rng = random.Random(seed + 1)
    chain = DeckChain(MEASURE_WHISTED_RATIO, rng)
    rows = []  # (variante, "emballage", gain moyen par siege defenseur, decisions PIMC)
    diffs = {"def": [], "trig": []}
    while time.time() < deadline:
        hands, dealer, whisted = chain.deal(rng)
        bidding = run_bot_bidding(hands, dealer)
        if bidding.general_pass:
            chain.finish_unplayed(hands, dealer)
            continue
        contract = bidding.locked_contract
        role, _decl = declarer_role_and_seats(contract)
        base = new_game(contract, hands, dealer, bidding, whisted)
        play(base, (), None, play_rng, samples, [])
        chain.finish_played(base, dealer, contract["type"])
        if contract["type"] != "emballage" or base.trump_suit is None:
            continue
        pts_base = real_points_by_seat(role, base, contract)
        defense = [s for s in range(NUM_PLAYERS) if s not in set(dds.side_seats(base))]

        def score_fn(g):
            return real_points_by_seat(role, g, contract)
        for name, trigger in (("trig", adv_wins_trick), ("def", None)):
            g = new_game(contract, hands, dealer, bidding, whisted)
            n = play(g, set(defense), score_fn, play_rng, samples, diffs[name], trigger)
            pts = real_points_by_seat(role, g, contract)
            rows.append((name, "emballage", sum(pts[s] - pts_base[s] for s in defense) / len(defense), n))
    return rows, diffs


def oracle_main(args):
    profiles = args.others.split(",")
    for others in profiles:
        t0 = time.time()
        deadline = t0 + args.minutes * 60 / len(profiles)
        with mp.Pool(args.workers) as pool:
            parts = pool.map(oracle_worker, [(args.seed * 1000 + w, args.samples, deadline, others)
                                             for w in range(args.workers)])
        rows = [r for p, _ in parts for r in p]
        for name in ("def", "trig"):
            n, m, se = stats([r[2] for r in rows if r[0] == name])
            dec = sum(r[3] for r in rows if r[0] == name)
            print("ORACLE %-16s %-4s emballage def : n %6d gain %+.3f +- %.3f pts/siege (z %+.1f), %d decisions PIMC,"
                  " %d mondes, %.0f s" % (others, name, n, m, se, m / se if se else 0.0, dec, args.samples,
                                         time.time() - t0), flush=True)
        agg = {}
        for sit, gap, _ex in (d for _, ds in parts for d in ds["trig"]):
            a = agg.setdefault(sit, [0, 0.0])
            a[0] += 1
            a[1] += gap
        for sit in sorted(agg, key=lambda k: -agg[k][1])[:12]:
            print("   trig %-58s n %5d  gap Q moyen %.2f" % (sit, agg[sit][0], agg[sit][1] / agg[sit][0]))
    print("PIMC_ORACLE_DONE")


def stats(xs):
    n = len(xs)
    if n == 0:
        return 0, 0.0, 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / max(1, n - 1)
    return n, m, math.sqrt(var / n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--oracle", action="store_true", help="defense d emballage seule, variantes def/trig, par profil")
    parser.add_argument("--camps", default="def,decl", help="camps PIMC mesures (def, decl)")
    parser.add_argument("--validate", type=int, default=0, help="re-estime chaque ecart sur N mondes neufs (gap sans biais)")
    parser.add_argument("--trigger", default="all", choices=sorted(TRIGGERS), help="hors --oracle : quand PIMC remplace ExpertBot (emb_draw = entame d atout de l emballeur, conseil 16/11)")
    parser.add_argument("--expert-q", action="store_true", help="avec --validate : gap aussi re-estime avec suite ExpertBot (Q realiste, conseil 16/11)")
    parser.add_argument("--control", action="store_true", help="variante rnd : carte legale au hasard aux memes declenchements (camp declarant)")
    parser.add_argument("--others", default="expert", help="profils ExpertBot (expert_bot.PROFILES), virgules ; hors --oracle : duree partagee")
    parser.add_argument("--branch-key", action="store_true", help="entames : situation + branche ExpertBot (diagnostic conseil 17/11 soir)")
    parser.add_argument("--dump", type=int, default=4, help="exemples imprimes par situation (conseil 17/11 : toutes les paires)")
    args = parser.parse_args()
    global DUMP_EX, BRANCH_KEY
    DUMP_EX = args.dump
    BRANCH_KEY = args.branch_key
    if args.oracle:
        return oracle_main(args)
    profiles = args.others.split(",")
    for others in profiles:
        run_profile(args, others, args.minutes / len(profiles))
    print("PIMC_VS_EXPERT_DONE")


def run_profile(args, others, minutes):
    """15/11 (conseil) : --others p1,p2 -> duree partagee, une table par profil ExpertBot (base + sieges non PIMC)."""
    t0 = time.time()
    deadline = t0 + minutes * 60
    with mp.Pool(args.workers) as pool:
        parts = pool.map(worker, [(args.seed * 1000 + w, args.samples, deadline, set(args.camps.split(",")), args.validate,
                                   others, args.trigger, args.expert_q, args.control) for w in range(args.workers)])
    rows = [r for p in parts for r in p[0]]
    diffs = {k: [d for p in parts for d in p[1][k]] for k in ("def", "decl")}
    times = sorted(t for p in parts for t in p[2])
    egaps = [e for p in parts for e in p[3]]
    print("=== profil %s : PIMC (%d mondes) vs ExpertBot, duplicate, %.0f s ===" % (others, args.samples, time.time() - t0))
    if times:
        print("temps/decision PIMC (Python, runner) : n %d mediane %.0f ms p90 %.0f ms max %.0f ms" % (
            len(times), 1000 * times[len(times) // 2], 1000 * times[int(len(times) * 0.9)], 1000 * times[-1]))
    print("%-6s %-10s %6s %9s %7s %8s" % ("camp", "contrat", "n", "gain/sieg", "se", "decis."))
    for name in ("def", "decl"):
        for typ in ("emballage", "solo", "trou", None):
            sel = [r for r in rows if r[0] == name and (typ is None or r[1] == typ)]
            n, m, se = stats([r[2] for r in sel])
            dec = sum(r[3] for r in sel)
            print("%-6s %-10s %6d %+9.3f %7.3f %8d  (z %+.1f) profil %s" % (name, typ or "TOUT", n, m, se, dec,
                                                                    m / se if se else 0.0, others))
    print("\n=== donnes CHANGEES seulement (>= 1 carte differente d'ExpertBot), profil %s ===" % others)
    for name in ("def", "decl", "rnd"):
        sel = [r for r in rows if r[0] == name and r[4] > 0]
        if not sel:
            continue
        n, m, se = stats([r[2] for r in sel])
        line = "%-5s n %5d  gain reel %+.3f +- %.3f (z %+.1f)  cartes changees/donne %.2f" % (
            name, n, m, se, m / se if se else 0.0, sum(r[4] for r in sel) / n)
        if name != "rnd":
            for lab, k in (("dds", 5), ("expert", 6)):
                xs = [r[k] for r in sel]
                mx = sum(xs) / n
                sxx = sum((x - mx) ** 2 for x in xs)
                slope = sum((x - mx) * (r[2] - m) for x, r in zip(xs, sel)) / sxx if sxx > 0 else 0.0
                line += " | somme gap %s %+.3f pente %+.3f" % (lab, mx, slope)
        print(line)
    if egaps:
        print("\n=== gap valide dds vs suite ExpertBot (memes mondes neufs) par situation ===")
        agg = {}
        for sit, eg, vg in egaps:
            a = agg.setdefault(sit, ([], []))
            a[0].append(eg)
            if vg is not None:
                a[1].append(vg)
        for sit in sorted(agg, key=lambda k: -len(agg[k][0])):
            en, em, ese = stats(agg[sit][0])
            vn, vm, vse = stats(agg[sit][1])
            print("%-58s n %5d  dds %+.2f +- %.2f  | expert %+.2f +- %.2f (z %+.1f)%s" % (
                sit, en, vm, vse, em, ese, em / ese if ese > 0 else 0.0,
                "  RETENUE" if en >= 100 and ese > 0 and em > 3 * ese else ""))
    for name in ("def", "decl"):
        agg = {}
        for sit, gap, vgap, ex in diffs[name]:
            a = agg.setdefault(sit, {"n": 0, "gap": 0.0, "ex": [], "v": []})
            a["n"] += 1
            a["gap"] += gap
            if vgap is not None:
                a["v"].append(vgap)
            if len(a["ex"]) < DUMP_EX and (gap >= 1.0 or DUMP_EX > 4):
                a["ex"].append(ex)
        print("\n=== %s : ecarts PIMC / ExpertBot par situation (Q gap cumule) ===" % name)
        for sit in sorted(agg, key=lambda k: -agg[k]["gap"])[:30]:
            a = agg[sit]
            line = "%-62s n %5d  gap moyen %.2f  cumul %.1f" % (sit, a["n"], a["gap"] / a["n"], a["gap"])
            if a["v"]:
                vn, vm, vse = stats(a["v"])
                line += "  | VALIDE n %d gap %+.2f +- %.2f (z %+.1f)%s" % (
                    vn, vm, vse, vm / vse if vse > 0 else 0.0, "  RETENUE" if vn >= 100 and vse > 0 and vm > 3 * vse else "")
            print(line)
            for ex in a["ex"]:
                print("     " + ex)


if __name__ == "__main__":
    main()
