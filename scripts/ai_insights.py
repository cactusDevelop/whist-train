"""Stats IA pour le joueur (demande whist-4-it 14/11, conseil de 3 avis) :
game/ai_insights.json, onglet Parametres > Stats IA. Mesure seule, cloud.

Les 4 sieges jouent COMME DANS LE JEU 1.2.25 : enchere ExpertBot + routage
d'enchere (agents/bid_route.BID_ROUTE, encherisseur = poids publies),
cartes ExpertBot sauf reseaux libre_* sur les miseres (piccolo, grande,
etalee : tous les sieges ; petite : defense seulement) et le camp declarant
du trou (whist_ai_bridge_v2.gd hybrid_applies). Poids lus dans --weights
(copie de whist-4-it/game, fp16 tels que publies). Donnes whistees comme en
jeu (paquet ramasse, coupe).

a) par contrat : n, reussite, EV du declarant (vrais points), cout d'une
   chute, gain d'une reussite, frequence.
b) quand annoncer : reussite/EV par tranche de la main du declarant
   (atouts x honneurs d'atout ; miseres : cartes hautes V+). BIAISE par la
   selection (le bot n'annonce que les bonnes mains) -> voir d).
c) defense : taux de chute par contrat, par type et hauteur d'entame
   (premiere carte de la defense). Correlation, pas causalite.
d) contrefactuel passe vs annonce : 1 siege tire au hasard par donne, a sa
   1re decision d'enchere : on rejoue la donne en le forcant a PASSER et a
   ANNONCER (propose ou solo 6, tire au hasard, dans sa couleur la plus
   longue), le reste de la table inchange ; gain = ses vrais points
   (annonce - passe), par tranche atouts x honneurs. Le trou n'est jamais un
   choix (auto-declenche) : pas de contrefactuel.

Usage (cloud seulement) : python -u scripts/ai_insights.py --deals 100000 --workers 4 --out ai_insights.json
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import random
import subprocess
import sys
import time

import torch

sys.path.insert(0, ".")

from agents import bid_route, evolution, expert_bot, specialists
from agents.abondance_rule import RuledBidder
from engine.bidding import ACTION_PASS, Bidding, _PROPOSE_BASE, _SOLO_BASE, _SOLO_LEVELS
from engine.cards import ACE_RANK, rank_of, suit_of
from engine.trou import detect_trou
from engine.whisted import DeckChain
from scripts.champion_play import mean_se
from scripts.godot_weights import load_role
from scripts.train import declarer_role_and_seats, dispatch_play_phase2, real_points_by_seat

GAME_VERSION = "1.2.25"
SEED = 20261114
WHISTED_RATIO = 1.0  # en jeu, chaque donne vient du paquet ramasse de la precedente
MIN_N = 200  # seuil d'affichage (conseil) : en dessous, la cellule n'est pas exportee
JACK_RANK = 9  # rang 0 = 2 ... 9 = Valet, 12 = As
UNRELIABLE_JOKERS = ["pot_de_vin", "main_de_fer"]  # notes joker_profit : joueur simule qui les joue mal (30/10)

BIDDER = None  # herites par fork
NETS = {}


def hybrid_applies(contract, seat):
    """Port de whist_ai_bridge_v2.gd::hybrid_applies (1.2.25)."""
    t = contract["type"]
    if t == "misere":
        if contract["kind"] == "petite":
            return seat not in contract["declarers"]
        return contract["kind"] in ("petite", "piccolo", "grande", "etalee")
    if t == "trou":
        return seat in (contract["declarer"], contract["partner"])
    return False


def run_bidding(hands, dealer, trou, force=None):
    """Enchere du jeu (ExpertBot + routage aux 4 sieges). force=(siege, choix) : sa 1re decision imposee.
    -> (bidding, {siege: (1re decision jouee, contexte)})."""
    b = Bidding(dealer)
    if trou is not None:
        b.seed_trou(trou)
    first = {}
    while not b.finished:
        seat = b.turn
        hand = hands[seat]
        if seat not in first and force is not None and force[0] == seat:
            choice = force[1]
        else:
            e = expert_bot.choose_bid(b, seat, hand)
            choice = bid_route.route(e, BIDDER.decide(b, seat, hand))
        if seat not in first:
            first[seat] = (choice, "apres_annonce" if b.contracts else ("premier" if not any(b.has_acted) else "apres_passe"))
        if not expert_bot.apply_choice(b, seat, choice):
            raise AssertionError("enchere illegale %r siege %d" % (choice, seat))
    return b, first


def play(bidding, hands, dealer, whisted):
    contract = bidding.locked_contract
    role, decl = declarer_role_and_seats(contract)
    game = dispatch_play_phase2(contract, [list(h) for h in hands], dealer, bidding.bid_summary, whisted)
    with torch.no_grad():
        while not game.done:
            seat = game.turn
            if hybrid_applies(contract, seat):
                r = game.role_of(seat)
                mask = torch.tensor(game.legal_action_mask(seat), dtype=torch.bool)
                card = NETS[r].act_greedy(torch.tensor(game.observation(seat), dtype=torch.float32), mask)
            else:
                card = expert_bot.choose_card(game, seat)
            game.step(seat, card)
    return game, role, decl, real_points_by_seat(role, game, contract)


def seat_points(bidding, hands, dealer, whisted, seat):
    if bidding.general_pass:
        return 0.0
    return play(bidding, hands, dealer, whisted)[3][seat]


def contract_key(c):
    t = c["type"]
    if t == "misere":
        return "misere-" + c["kind"]
    if t == "trou":
        return "trou-royal" if c.get("royal") else "trou"
    return "%s-%d" % (t, c["level"])


def trump_bucket(hand, suit):
    n = sum(1 for c in hand if suit_of(c) == suit)
    h = sum(1 for c in hand if suit_of(c) == suit and rank_of(c) >= JACK_RANK)
    return "atouts%s_honneurs%s" % (n if 4 <= n <= 7 else ("3-" if n < 4 else "8+"), h if h < 3 else "3+")


def misere_bucket(hand):
    h = sum(1 for c in hand if rank_of(c) >= JACK_RANK)
    return "hautes%s" % (h if h <= 4 else "5+")


def best_suit(hand):
    return max(range(4), key=lambda s: (sum(1 for c in hand if suit_of(c) == s),
                                        sum(rank_of(c) for c in hand if suit_of(c) == s and rank_of(c) >= JACK_RANK)))


def lead_cats(game, decl, hand0):
    """Premiere carte jouee par la defense (et si c'est l'entame du 1er pli)."""
    for i, (seat, card) in enumerate(game.play_log):
        if seat in decl:
            continue
        s = suit_of(card)
        n = sum(1 for c in hand0[seat] if suit_of(c) == s)
        trump = getattr(game, "trump_suit", None)
        kind = "atout" if trump is not None and s == trump else ("courte" if n <= 2 else ("longue" if n >= 5 else "moyenne"))
        height = "as" if rank_of(card) == ACE_RANK else ("honneur" if rank_of(card) >= JACK_RANK else "petite")
        return kind, height, i == 0
    return None, None, False


def chunk(args):
    deals, seed = args
    rng = random.Random(seed)
    chain = DeckChain(WHISTED_RATIO, rng)
    acc = {"deals": 0, "general_pass": 0, "contracts": {}, "when": {}, "defense": {}, "cf": {}, "cf_ctx": {}}

    def push(d, k, v):
        d.setdefault(k, []).append(v)

    for _ in range(deals):
        hands, dealer, whisted = chain.deal(rng)
        trou = detect_trou(hands)
        bidding, first = run_bidding(hands, dealer, trou)
        acc["deals"] += 1
        if bidding.general_pass:
            acc["general_pass"] += 1
            chain.finish_unplayed(hands, dealer)
            base_pts = [0.0] * 4
        else:
            c = bidding.locked_contract
            game, role, decl, pts = play(bidding, hands, dealer, whisted)
            base_pts = pts
            chain.finish_played(game, dealer, c["type"])
            key = contract_key(c)
            mains = c["declarers"] if c["type"] == "misere" else [c["declarer"]]
            for s in mains:  # plusieurs miseres simultanees : chaque declarant compte
                made = pts[s] > 0
                push(acc["contracts"], key, (pts[s], made))
                if c["type"] == "misere":
                    bucket = misere_bucket(hands[s])
                elif c["type"] == "trou":
                    bucket = None
                else:
                    bucket = trump_bucket(hands[s], c["suit"])
                if bucket:
                    fam = "abondance" if c["type"] == "solo" and 9 <= c["level"] <= 12 else key
                    push(acc["when"], (fam, bucket), (pts[s], made))
                kind, height, is_opening = lead_cats(game, decl, hands)
                if kind:
                    push(acc["defense"], (key, "type", kind), not made)
                    push(acc["defense"], (key, "hauteur", height), not made)
        # d) contrefactuel : 1 siege, 1re decision, passe vs annonce dans sa couleur la plus longue
        seat = rng.randrange(4)
        if trou is not None or seat not in first:
            continue
        hand = hands[seat]
        suit = best_suit(hand)
        kind = rng.choice(("propose", "solo_6"))
        announce = _PROPOSE_BASE + suit if kind == "propose" else _SOLO_BASE + suit * _SOLO_LEVELS
        played, ctx = first[seat]
        pre = Bidding(dealer)  # legalite de l'annonce : rejouer l'enchere (deterministe) jusqu'a la 1re decision du siege
        while not pre.finished and pre.turn != seat:
            s2 = pre.turn
            e = expert_bot.choose_bid(pre, s2, hands[s2])
            expert_bot.apply_choice(pre, s2, bid_route.route(e, BIDDER.decide(pre, s2, hands[s2])))
        legal = not pre.finished and pre.turn == seat and (
            pre.can_propose(seat, suit) if kind == "propose" else pre.can_solo(seat, suit, 6)) and pre.can_pass(seat)
        if not legal:
            continue
        if played == ACTION_PASS:
            p_pass = base_pts[seat]
        else:
            b, _ = run_bidding(hands, dealer, None, force=(seat, ACTION_PASS))
            p_pass = seat_points(b, hands, dealer, whisted, seat)
        if played == announce:
            p_ann = base_pts[seat]
        else:
            b2, _ = run_bidding(hands, dealer, None, force=(seat, announce))
            p_ann = seat_points(b2, hands, dealer, whisted, seat)
        bot = "passe" if played == ACTION_PASS else ("annonce" if played == announce else "autre")
        push(acc["cf"], (kind, trump_bucket(hand, suit)), (p_ann - p_pass, bot))
        push(acc["cf_ctx"], (kind, ctx), p_ann - p_pass)
    return acc


def merge(parts):
    out = {"deals": 0, "general_pass": 0, "contracts": {}, "when": {}, "defense": {}, "cf": {}, "cf_ctx": {}}
    for p in parts:
        out["deals"] += p["deals"]
        out["general_pass"] += p["general_pass"]
        for k in ("contracts", "when", "defense", "cf", "cf_ctx"):
            for kk, v in p[k].items():
                out[k].setdefault(kk, []).extend(v)
    return out


def r(x, nd=2):
    return round(x, nd)


def summarize(rows, total_deals, min_n):
    pts = [p for p, _ in rows]
    made = [1.0 if m else 0.0 for _, m in rows]
    p, p_se = mean_se(made)
    ev, ev_se = mean_se(pts)
    fail = [x for x, m in rows if not m]
    win = [x for x, m in rows if m]
    cf, cf_se = mean_se(fail)
    gw, gw_se = mean_se(win)
    return {"n": len(rows), "freq": r(len(rows) / total_deals, 4), "reussite": r(p, 3), "reussite_se": r(p_se, 3),
            "ev": r(ev), "ev_se": r(ev_se), "cout_chute": r(cf) if fail else None, "cout_chute_se": r(cf_se) if fail else None,
            "gain_reussite": r(gw) if win else None, "gain_reussite_se": r(gw_se) if win else None}


def fingerprint(weights_dir):
    h = hashlib.sha256()
    files = sorted(os.path.join("engine", f) for f in os.listdir("engine") if f.endswith(".py"))
    files += ["agents/expert_bot.py", "agents/bid_route.py", "agents/abondance_rule.py"]
    files += sorted(os.path.join(weights_dir, f) for f in os.listdir(weights_dir) if f.endswith(".gd"))
    for f in files:
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:8]


def build(acc, min_n, meta):
    total = acc["deals"]
    contracts = {}
    hidden = []
    for k, rows in sorted(acc["contracts"].items()):
        s = summarize(rows, total, min_n)
        (contracts.__setitem__(k, s) if s["n"] >= min_n else hidden.append([k, s["n"]]))
    when = {}
    for (fam, bucket), rows in sorted(acc["when"].items()):
        if len(rows) >= min_n:
            s = summarize(rows, total, min_n)
            s.pop("freq")
            when.setdefault(fam, {})[bucket] = s
    defense = {}
    for (k, axis, cat), xs in sorted(acc["defense"].items()):
        if k in contracts and len(xs) >= min_n:
            m, se = mean_se([1.0 if x else 0.0 for x in xs])
            defense.setdefault(k, {"n": contracts[k]["n"], "taux_chute": r(1 - contracts[k]["reussite"], 3),
                                   "taux_chute_se": contracts[k]["reussite_se"]}).setdefault("par_entame_" + axis, {})[cat] = {
                "n": len(xs), "taux_chute": r(m, 3), "taux_chute_se": r(se, 3)}
    cf = {}
    for (kind, bucket), rows in sorted(acc["cf"].items()):
        if len(rows) < min_n:
            continue
        g, se = mean_se([x for x, _ in rows])
        bot_ann = sum(1 for _, b in rows if b == "annonce") / len(rows)
        verdict = "annoncer" if g - 2 * se > 0 else ("passer" if g + 2 * se < 0 else "limite")
        cf.setdefault(kind, {})[bucket] = {"n": len(rows), "gain_annonce_vs_passe": r(g), "se": r(se),
                                           "conseil": verdict, "bot_annonce": r(bot_ann, 3)}
    cf_ctx = {}
    for (kind, ctx), xs in sorted(acc["cf_ctx"].items()):
        if len(xs) >= min_n:
            g, se = mean_se(xs)
            cf_ctx.setdefault(kind, {})[ctx] = {"n": len(xs), "gain_annonce_vs_passe": r(g), "se": r(se)}
    return dict(meta, **{
        "deals": total, "general_pass_rate": r(acc["general_pass"] / total, 4), "min_n": min_n,
        "contracts": contracts, "hidden_contracts_below_min_n": hidden, "when_to_bid": when, "defense": defense,
        "pass_vs_bid": cf, "pass_vs_bid_by_context": cf_ctx, "unreliable_jokers": UNRELIABLE_JOKERS})


def main():
    global BIDDER
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--weights", default="godot_weights_ingame", help="copie des whist_ai_weights_data_v2_*.gd du jeu")
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--out", default="ai_insights.json")
    args = ap.parse_args()

    BIDDER = RuledBidder(load_role(args.weights, "encherisseur", evolution.OBS_DIM))
    for role in ("defenseur", "piccolo", "grande_misere", "petite_misere", "misere_etalee", "trou", "bouche_trou"):
        NETS[role] = load_role(args.weights, "libre_" + role, specialists.IMPLEMENTED[role][0])
    t0 = time.time()
    per = args.deals // args.workers
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        parts = pool.map(chunk, [(per, args.seed + 10 * i) for i in range(args.workers)], chunksize=1)
    acc = merge(parts)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "?"
    meta = {"source": "whist-ai scripts/ai_insights.py", "commit": commit, "game_version": GAME_VERSION,
            "rules_fingerprint": fingerprint(args.weights), "seed": args.seed,
            "date": time.strftime("%Y-%m-%d"), "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "mention": "ExpertBot contre ExpertBot (IA du jeu 1.2.25 aux 4 sieges), pas contre des humains",
            "definitions": {
                "ev": "vrais points moyens du declarant par contrat annonce (biaise : le bot n'annonce que les bonnes mains)",
                "atouts/honneurs": "main du declarant : cartes dans la couleur d'atout, dont Valet et plus",
                "hautes": "misere : cartes Valet et plus dans la main du declarant",
                "pass_vs_bid": "meme main rejouee : 1re decision forcee a annoncer (propose ou solo 6, couleur la plus longue) "
                               "moins forcee a passer ; vrais points du siege ; conseil = gain hors +-2 se",
                "defense": "par_entame = premiere carte jouee par la defense ; correlation, pas causalite"}}
    out = build(acc, args.min_n, meta)
    out["elapsed_s"] = r(time.time() - t0, 1)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("INSIGHTS %d donnes en %.0fs -> %s (%d contrats affiches, %d masques)" % (
        out["deals"], out["elapsed_s"], args.out, len(out["contracts"]), len(out["hidden_contracts_below_min_n"])), flush=True)
    for k, s in out["contracts"].items():
        print("  %-16s n %6d reussite %.3f ev %+6.2f chute %s" % (k, s["n"], s["reussite"], s["ev"], s["cout_chute"]))
    for kind, buckets in out["pass_vs_bid"].items():
        for b, s in buckets.items():
            print("  CF %-8s %-22s n %6d gain %+6.2f +- %.2f -> %s (bot annonce %.0f%%)" % (
                kind, b, s["n"], s["gain_annonce_vs_passe"], s["se"], s["conseil"], 100 * s["bot_annonce"]))


if __name__ == "__main__":
    main()
