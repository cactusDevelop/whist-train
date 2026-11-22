"""A/B duplicate des reglages de prudence des encheres de l'ExpertBot
(agents/expert_bot.set_bid_params ; signalement 2026-10-26 « contrats
impossibles, surencheres abusives »). Mesure seule, cloud.

Chaque donne est jouee 1 + 2 x V fois, jeu de la carte ExpertBot partout :
  - reference : 4 sieges aux reglages par defaut ;
  - pour chaque variante : sieges 0/2 variante (1/3 defaut), puis 1/3.
Gain de la variante = points reels des sieges variante - points des memes
sieges dans la reference, somme des deux placements (pts/donne). Par
variante aussi : reussite et frequence des contrats DECLARES par un siege
variante, par type -- un bot trop timide se voit (passes, contrats perdus
par la defense).

Usage : python -u scripts/expert_bid_ab.py --deals 20000 --workers 4
"""

import argparse
import math
import multiprocessing as mp
import os
import random
import sys
import time

sys.path.insert(0, ".")

from agents import expert_bot
from engine.bidding import Bidding
from engine.trou import detect_trou
from engine.whisted import DeckChain
from scripts.eval_vs_random import declarer_successes
from scripts.expert_bot_match import bot_player, play_out
from scripts.train import (MEASURE_WHISTED_RATIO, declarer_role_and_seats, describe_contract,
                           dispatch_play_phase2, real_points_by_seat)

# Jeux de variantes : une variante = dict partiel de expert_bot.set_bid_params, appliquee PAR-DESSUS le profil
# du champ ({**PROFILES[--others], **params}) ; IDENTITE = le profil lui-meme. Contre un profil qui fixe deja le
# knob (ex. acte1 EMBALLER_MIN=5), on compare donc profil vs variante, PAS defaut vs variante (piege a31) : la
# sortie affiche la valeur effective de chaque knob varie. Historique a27-a31 (EMBALLER_MIN 4->3 etc.) : git f3e3e40.
VARIANT_SETS = {}
VARIANT_SETS["a32"] = {  # 12/11 14e conseil : regle STRUCTURELLE de releve OPP_SUIT_CAP (plis de cote plafonnes dans les couleurs adverses)
    "IDENTITE": {},  # controle : doit donner exactement 0
    "OPP_CAP_1": {"OPP_SUIT_CAP": 1},
    "OPP_CAP_0": {"OPP_SUIT_CAP": 0},
}
VARIANT_SETS["a33"] = {  # 14/11 piste ai_insights : proposer plus souvent avec une longue couleur
    "IDENTITE": {},
    "LONG5_MIN0": {"PROPOSE_LONG_TRUMPS": 5, "PROPOSE_LONG_MIN": 0},
    "LONG5_MIN2": {"PROPOSE_LONG_TRUMPS": 5, "PROPOSE_LONG_MIN": 2},
    "LONG6_MIN0": {"PROPOSE_LONG_TRUMPS": 6, "PROPOSE_LONG_MIN": 0},
}
VARIANT_SETS["a34"] = {  # 15/11 conseil 14/11 etape 2 (jeu de la carte, DECLARANT) : entame d atout de l emballeur
    "IDENTITE": {},
    "DRAW_RAW": {"DRAW_COUNT_RAW": True},
    "NO_INFER": {"INFER_OPP_TRUMPS": False},
}
VARIANT_SETS["a35"] = {  # 17/11 nuit emb_draw2 acceptee (decl z~45) : l emballeur avec 2 atouts ne tire pas
    "IDENTITE": {},
    "NODRAW2": {"DRAW_MIN_TRUMPS": 3},
    "NODRAW2_NM": {"DRAW_MIN_TRUMPS": 3, "DRAW_MIN_MASTER_OK": True},
}
VARIANT_SETS["a36"] = {  # 17/11 soir conseil : defenseur SANS atout, pli 6+ (emb_def_lead0 confirme z 28.9, PIMC entame plus bas)
    "IDENTITE": {},
    "CASH_K1": {"DEF0_CASH_MARGIN": 1},
    "PARTNER_LOW": {"DEF0_PARTNER_LOW": True},
    "LONG_LOW": {"DEF0_LONG_LOW": True},
    "LOW_BOTH": {"DEF0_PARTNER_LOW": True, "DEF0_LONG_LOW": True},
}
VARIANT_SETS["a37"] = {  # 17/11 nuit conseil : branche renom (diag 44116701435 : PIMC meme couleur plus haut, gap +1.14)
    "IDENTITE": {},
    "RENOM_HIGH": {"DEF0_RENOM": "high"},
    "RENOM_MASTER": {"DEF0_RENOM": "master"},
    "RENOM_HIGH_ALL": {"DEF0_RENOM": "high_all"},
    "RENOM_HIGH_COND": {"DEF0_RENOM": "high_cond"},
    "RENOM_SKIP": {"DEF0_RENOM": "skip"},
}
VARIANT_SETS["a38"] = {  # 17/11 soir conseil apres a37 accepte : base = high_all ; controle + couleur d entame defense
    "IDENTITE": {},
    "RENOM_NONE": {"DEF0_RENOM": None},  # controle : doit perdre ~0.13..0.20 (a37)
    "RENOM_T2LOW": {"DEF0_RENOM": "high_t2low"},
    "LEAD_SEQ_ANY": {"DEF_LEAD_SEQ_ANY": True},
    "LEAD_PASSIVE": {"DEF_LEAD_PASSIVE": True},
    "LEAD_SHORT": {"DEF_LEAD_SHORT": True},
}
VARIANT_SETS["a39"] = {  # 18/11 apres-midi conseil : diag general 44262105035 (base high_all), fournir coequipier gagnant + defausse sur coupe
    # RESULTAT run 44318627505 (7 champs, annule pour lenteur) : REJETE -- SAFE_LAX -0.06 z -4, UNBLOCK2_LOOSE -0.13 z -8, RUFFED_SHORT -0.03 z -2, LAX_UNBLOCK -0.19 ; controle NO_OVERTAKE -1.46 z -31 (banc OK)
    "IDENTITE": {},
    "CTRL_NO_OVERTAKE": {"NO_OVERTAKE": True},  # controle negatif : doit perdre (A/B 21/22)
    "SAFE_LAX": {"PARTNER_SAFE_LAX": True},
    "UNBLOCK2_LOOSE": {"UNBLOCK_2ND_LOOSE": True},
    "RUFFED_SHORT": {"DISCARD_RUFFED_SHORT": True},
    "LAX_UNBLOCK": {"PARTNER_SAFE_LAX": True, "UNBLOCK_2ND_LOOSE": True},
}
VARIANT_SETS["a40"] = {  # 18/11 soir conseil 2e tour : dump coeq_coupe 44335928563, couvrir le coequipier faible (PIMC)
    "IDENTITE": {},
    "CTRL_NO_OVERTAKE": {"NO_OVERTAKE": True},  # controle negatif (a39 : -1.46 z -31)
    "COVER_BEAT": {"COVER_WEAK_PARTNER": "beat"},
    "COVER_BEAT_LOW": {"COVER_WEAK_PARTNER": "beat_low"},
    "COVER_HIGH": {"COVER_WEAK_PARTNER": "high"},
}
VARIANT_SETS["a41"] = {  # 19/11 matin : nuit pimc 44437340894, decl emballeur brule la plus haute (coequipier gagnant)
    "IDENTITE": {},
    "CTRL_NO_OVERTAKE": {"NO_OVERTAKE": True},  # controle negatif (a39/a40)
    "DECL_SAFE2": {"DECL_OVERTAKE_HIGH": "safe2"},
    "DECL_MASTERS": {"DECL_OVERTAKE_HIGH": "masters"},
    "DECL_BOTH": {"DECL_OVERTAKE_HIGH": "both"},
}
VARIANT_SETS["a42"] = {  # 20/11 : dump def_lead_ts 44693519485, defenseur du trou entame l atout maitre (fin de donne)
    "IDENTITE": {},
    "CTRL_NO_OVERTAKE": {"NO_OVERTAKE": True},  # controle negatif
    "TROU_P6_12": {"DEF_TROU_TRUMP_LEAD": "p6_12"},
    "TROU_P2_12": {"DEF_TROU_TRUMP_LEAD": "p2_12"},
    "TROU_P6_123": {"DEF_TROU_TRUMP_LEAD": "p6_123"},
}
VARIANT_SETS["a43"] = {  # 20/11 soir : dump def_ruff2 44759251024, defenseur 2e en renonce trou/solo defausse au lieu de couper
    "IDENTITE": {},
    "CTRL_NO_OVERTAKE": {"NO_OVERTAKE": True},  # controle negatif
    "DUCK2_LOW": {"DEF_DUCK_RUFF2": "low"},
    "DUCK2_NONHM": {"DEF_DUCK_RUFF2": "nonHM"},
    "DUCK2_ALL": {"DEF_DUCK_RUFF2": "all"},
}
VARIANT_SETS["a44"] = {  # 21/11 : dump def_follow2 44906927444, defenseur d emballage 2e fournit, adversaire gagne
    # RESULTAT 12 runs paralleles 44959372075..44959416817 : REJETE 12/12 -- KEEP -0.04..-0.13 z -5..-7, TOP -0.06..-0.23 z -4..-13, BOTH -0.11..-0.38 ; controle NO_OVERTAKE -0.00..-0.19
    "IDENTITE": {},
    "CTRL_NO_OVERTAKE": {"NO_OVERTAKE": True},  # controle negatif
    "F2_KEEP": {"DEF_FOLLOW2": "keep"},
    "F2_TOP": {"DEF_FOLLOW2": "top"},
    "F2_BOTH": {"DEF_FOLLOW2": "both"},
}
VARIANT_SETS["a45"] = {  # 22/11 : pimc --single (gain reel), emballeur/proposeur 4e defausse, coequipier gagne
    # RESULTAT 12 runs paralleles 45006611454..45006690662 : D4_LOW ACCEPTE 12/12 (+0.029..+0.113 z 4.3..9.5, 0 cellule < -2 SE) ;
    # HIGH 6/12 ; SINGLE 12/12 mais < LOW ; SHORT_WHEN_MATE ~0 (emballage decl +0.096/donne, perdu en defense -0.087)
    "IDENTITE": {},
    "CTRL_NO_OVERTAKE": {"NO_OVERTAKE": True},  # controle negatif
    "D4_LOW": {"DECL_DISCARD4": "low"},
    "D4_HIGH": {"DECL_DISCARD4": "high"},
    "D4_SINGLE": {"DECL_DISCARD4": "single"},
    "SHORT_WHEN_MATE": {"DISCARD_SHORT_WHEN_MATE": True},  # A/B 13 (09/11) : rejete sous placements biaises, jamais retente
}
VARIANT_SETS["a46"] = {  # 22/11 : base = a45 ; defausse courte du camp declarant d emballage etendue
    # RESULTAT 12 runs paralleles 45007930864..45007999738 : REJETE -- MATE_ALL +0.028..+0.085 sur 10 champs MAIS -0.056/-0.051
    # (z -3.7/-3.4) contre expert_tune1027 ; MATE_LAST ~0 ; CTRL_D4_OFF perd 12/12 (-0.032..-0.129 : a45 confirme)
    "IDENTITE": {},
    "CTRL_D4_OFF": {"DECL_DISCARD4": None},  # controle negatif (doit perdre ~0.03..0.11)
    "MATE_ALL": {"DECL_DISCARD_MATE": "all"},
    "MATE_LAST": {"DECL_DISCARD_MATE": "last"},
}
# placements par defaut d un jeu (knob de JEU -> "camp" = vrais camps du contrat de reference ; encheres -> "single")
SET_SEATS = {"a34": "camp", "a35": "camp", "a36": "camp", "a37": "camp", "a38": "camp", "a39": "camp", "a40": "camp", "a41": "camp", "a42": "camp", "a43": "camp", "a44": "camp", "a45": "camp", "a46": "camp"}
# AB_SET (env, herite par les workers) : jeu de variantes ; defaut = le plus recent
VARIANTS = VARIANT_SETS[os.environ.get("AB_SET", max(VARIANT_SETS))]
# 11/11 (11e conseil, critique) : pour un knob d ENCHERES, placer la variante sur les camps du contrat de
# reference est mal pose (la variante change le contrat ; pas de partenaires fixes au whist ; repli (0,2)/(1,3)
# sur les passes generales). SEATS=single : variante sur un seul siege, 4 rotations, gain = points de ce siege
# - reference ; cellules = contrat JOUE par la variante | camp du siege variante.
SEATS = os.environ.get("AB_SEATS", SET_SEATS.get(os.environ.get("AB_SET", max(VARIANT_SETS)), "single"))

def play_deal(hands, dealer, whisted, params_by_seat):
    """-> (points des 4 sieges, contrat ou None, jeu ou None)."""
    bidding = Bidding(dealer)
    trou = detect_trou(hands)
    if trou is not None:
        bidding.seed_trou(trou)
    while not bidding.finished:
        seat = bidding.turn
        expert_bot.set_bid_params(params_by_seat[seat])
        expert_bot.apply_bid(bidding, seat, hands[seat])
    expert_bot.set_bid_params({})
    if bidding.general_pass:
        return [0.0] * 4, None, None
    contract = bidding.locked_contract
    role, _decl = declarer_role_and_seats(contract)
    game = dispatch_play_phase2(contract, [list(h) for h in hands], dealer, bidding.bid_summary, whisted)
    while not game.done:
        seat = game.turn
        expert_bot.set_bid_params(params_by_seat[seat])
        game.step(seat, expert_bot.choose_card(game, seat))
    expert_bot.set_bid_params({})
    return real_points_by_seat(role, game, contract), contract, game


def worker(args):
    seed, n, others = args
    from agents.expert_bot import PROFILES
    other = dict(PROFILES[others])
    if os.environ.get("FOLLOW_TRACE"):
        expert_bot.FOLLOW_TRACE = {}
    rng = random.Random(seed)
    chain = DeckChain(MEASURE_WHISTED_RATIO, rng)
    out = []  # (variante, gain, [(cle, ok, essais)] des contrats declares par la variante)
    base_rows = []
    for _ in range(n):
        hands, dealer, whisted = chain.deal(rng)
        hands = [list(h) for h in hands]
        base, bc, bgame = play_deal(hands, dealer, whisted, [other] * 4)
        if bc is not None:
            _r, decl = declarer_role_and_seats(bc)
            oks = declarer_successes(bc, bgame, decl)
            base_rows.append((describe_contract(bc), bc["type"], sum(oks), len(oks)))
        for name, params in VARIANTS.items():
            gain = 0.0
            rows = []
            cells = []  # 10/11 : (contrat de reference|camp des sieges variante, gain du placement)
            # 10/11 : placements = VRAIS camps du contrat de reference (avant : (0,2)/(1,3) -> declarants voisins
            # = 1 declarant + 1 defenseur par placement, gain nul par construction ; A1/A2/A3 "0 %" = artefact)
            if SEATS == "single":
                placements = ((0,), (1,), (2,), (3,))
            elif bc is not None:
                _r, bdecl0 = declarer_role_and_seats(bc)
                placements = (tuple(sorted(bdecl0)), tuple(s for s in range(4) if s not in bdecl0))
            else:
                placements = ((0, 2), (1, 3))
            for seats in placements:
                pbs = [{**other, **params} if s in seats else other for s in range(4)]
                pts, c, game = play_deal(hands, dealer, whisted, pbs)
                g = sum(pts[s] - base[s] for s in seats)
                gain += g
                if SEATS == "single":
                    if c is None:
                        cells.append(("passe generale", g))
                    else:
                        _r, vdecl = declarer_role_and_seats(c)
                        cells.append(("%s|%s" % (c["type"], "decl" if vdecl & set(seats) else "def"), g))
                elif bc is not None:
                    _r, bdecl = declarer_role_and_seats(bc)
                    cells.append(("%s|%s" % (bc["type"], "decl" if bdecl & set(seats) else "def"), g))
                if c is not None:
                    _r, decl = declarer_role_and_seats(c)
                    if decl & set(seats):
                        oks = declarer_successes(c, game, decl)
                        rows.append((describe_contract(c), c["type"], sum(oks), len(oks)))
            out.append((name, gain, rows, cells))
        if bgame is None:
            chain.finish_unplayed(hands, dealer)
        else:
            chain.finish_played(bgame, dealer, bc["type"])
    if expert_bot.FOLLOW_TRACE is not None:  # diagnostic 10/11 : branches "coequipier gagnant, on fournit"
        for k, c in sorted(expert_bot.FOLLOW_TRACE.items(), key=lambda kv: -kv[1])[:60]:
            print("TRACE %d %s %d" % (seed, " ".join(map(str, k)), c), flush=True)
    return out, base_rows


def stat(xs):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1))
    return m, sd / math.sqrt(len(xs))


def table(rows, n_deals, title):
    agg = {}
    for key, typ, ok, tries in rows:
        for k in (key, "_" + typ, "_TOUT"):
            a = agg.setdefault(k, [0, 0, 0])
            a[0] += 1
            a[1] += ok
            a[2] += tries
    print("  %s : contrats declares / donne, reussite" % title)
    for k in sorted(agg, key=lambda k: (not k.startswith("_"), -agg[k][0])):
        n, ok, tries = agg[k]
        print("    %-16s %6d %6.3f %6.1f%%" % (k.lstrip("_"), n, n / n_deals, 100 * ok / max(1, tries)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--others", default="expert", help="profil (PROFILES) de la reference et des sieges adverses")
    parser.add_argument("--trace", action="store_true", help="diagnostic : compte les branches coequipier gagnant (TRACE)")
    args = parser.parse_args()
    if args.trace:
        os.environ["FOLLOW_TRACE"] = "1"
    t0 = time.time()
    per = args.deals // args.workers
    with mp.Pool(args.workers) as pool:
        parts = pool.map(worker, [(args.seed * 1000 + w, per, args.others) for w in range(args.workers)])
    n = per * args.workers
    print("=== A/B encheres ExpertBot : %d donnes, %d variantes, %.0fs ===" % (n, len(VARIANTS), time.time() - t0))
    table([r for _o, b in parts for r in b], n, "reference (4 sieges)")
    for name, params in VARIANTS.items():
        gains = [g for o, _b in parts for v, g, _r, _c in o if v == name]
        rows = [r for o, _b in parts for v, _g, rs, _c in o if v == name for r in rs]
        cells = {}
        for o, _b in parts:
            for v, _g, _r, cs in o:
                if v == name:
                    for k, g in cs:
                        cells.setdefault(k, []).append(g)
        changed = sum(1 for g in gains if g != 0)
        m, se = stat(gains)
        prof = expert_bot.PROFILES[args.others]
        eff = ["%s %s -> %s" % (k, prof.get(k, getattr(expert_bot, k, None)), v) for k, v in params.items()]
        if eff:
            print("\n  (valeurs effectives, profil %s -> variante : %s)" % (args.others, ", ".join(eff)))
        print("\n%s %s : gain %+.3f +- %.3f pts/donne (z %.1f), donnes changees %.1f%%"
              % (name, params, m, se, m / se if se else 0.0, 100 * changed / len(gains)))
        table(rows, n, "variante (sieges variante, %s)" % ("4 rotations 1 siege" if SEATS == "single" else "2 placements"))
        print("  gain par contrat (%s) | camp" % ("JOUE par la variante" if SEATS == "single" else "de reference") + "  des sieges variante (pts par placement, z ; contribution/donne)")
        for k in sorted(cells, key=lambda k: -len(cells[k])):
            xs = cells[k]
            if len(xs) >= 2:
                cm, cse = stat(xs)
                print("    %-22s n %6d  %+.3f +- %.3f  z %+5.1f  %+.4f/donne" % (
                    k, len(xs), cm, cse, cm / cse if cse else 0.0, sum(xs) / n))
    names = list(VARIANTS)
    by = {name: [g for o, _b in parts for v, g, _r, _c in o if v == name] for name in names}
    print("\n=== ecarts apparies (memes donnes) ===")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m, se = stat([x - y for x, y in zip(by[a], by[b])])
            print("%s - %s : %+.3f +- %.3f (z %.1f)" % (a, b, m, se, m / se if se else 0.0))
    print("AB_DONE")


if __name__ == "__main__":
    main()
