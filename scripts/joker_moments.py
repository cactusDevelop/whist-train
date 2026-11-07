"""Mesure « moment utile » des jokers du joueur (G1 valide 2026-11-06, whist-4-it-c9 +
utilisateur ; CHANTIER-REWORK-JOKERS-PLAN.md Axe 1). Cloud uniquement.

But : ne plus punir un joker de situation que l'IA lance au mauvais moment (Horloge
Arretee, Double Valet, Pot-de-Vin, Collet...).

Methode, a chaque OCCASION (le joker est en main ET lancable a cette decision, pli ou
enchere) : la meme situation (copie exacte, memes donnes de la suite, memes graines) est
rejouee jusqu'a la FIN DU MATCH de trois facons :
  jouer     : le joker est lance maintenant, puis la regle du joueur continue ;
  garder    : il est interdit a cette decision seulement, la regle peut le lancer plus tard ;
  recycler  : le Recycler du jeu (1 energie, une nouvelle pioche, une fois par donne), quand il
              est permis a cette decision ;
  defausser : defausse pour l'energie (+1), puis la regle continue.
Horizon exact (whist-4-it) : la main de jokers part a la defausse en fin de donne, la pioche
et la defausse durent le match, rien ne passe au match suivant.

Bruit / biais de selection : 2K rejouees par branche (graines des jokers et des donnes
suivantes communes aux trois branches d'une rejouee). Les K premieres DECIDENT : alt = la meilleure
des autres branches, utile si jouer > alt. Les K suivantes MESURENT : gain = jouer - alt.
  force       = gain moyen sur les occasions utiles (points du joueur, fin de match) ;
  polyvalence = part des donnes (joker en main et lancable au moins une fois) avec au
                moins une occasion utile ;
  bien_vu     = part des occasions utiles ou la regle du joueur le lance ;
  gain_tout   = gain moyen sur toutes les occasions (comparaison avec l'ancienne mesure).

Valeur d'un ⚡ / d'une pioche : en debut de donne (contrat connu), +1 ⚡ (resp. +1 joker
pioche) contre rien, meme situation rejouee K fois jusqu'a la fin du match.

Echantillon : par donne, UN joker « suivi » tire au hasard dans la main du joueur (enchere
au 1er tour d'enchere, sinon pli) ; toutes ses occasions de la donne sont mesurees.
Deck : celui d'un run (joker_match.player_deck) + INJECT jokers de l'acte, pour que les
rares aient des occasions.

Usage : python scripts/joker_moments.py --minutes 150 --workers 4 [--k 2]
Sorties : joker_moments.json (cumul + par joker), joker_moments_summary.md.
"""

import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, ".")

from engine import joker_match as jm
from engine import jokers as J

OUT_JSON = "joker_moments.json"
OUT_MD = "joker_moments_summary.md"
# 8ebd : --value-only = seulement la valeur d'un ⚡ / d'une pioche (toutes les donnes, aucun rejeu par joker)
VALUE_JSON, VALUE_MD = "joker_values.json", "joker_values_summary.md"
SEED = 95000000
INJECT = 2
VALUE_PROB = 0.25  # donnes ou l'on mesure aussi la valeur d'un ⚡ / d'une pioche
MIN_N = 30
BATCH = 40
UPLOAD_EVERY_S = 900
BRANCHES = ("jouer", "garder", "recycler", "defausser")
ALTS = BRANCHES[1:]
PLAYER = jm.PLAYER


def _measurable(jid):
    return J.jtype(jid) != "passif" and J.base_id(jid) not in J.FROM_DECK_IDS


def _reseed(m2, seed):
    rng = random.Random(seed)
    m2.state.rng = random.Random(rng.random())
    m2.rules.rng = random.Random(rng.random())
    m2.player.rng = random.Random(rng.random())


def _rollout(match, hand, setup, policy, hands_left, deals, seed, lead_deals=()):
    """Points du joueur de maintenant a la fin du match. `hand` None : la donne n'est pas
    encore donnee (occasion d'enchere) : `lead_deals` = la donne reelle (et ses passes
    generales), jouee avec le reglage `setup` et les memes pioches que la ligne principale ;
    les graines de la rejouee ne prennent effet qu'apres l'enchere."""
    m2, h2 = copy.deepcopy((match, hand))
    if h2 is None:
        setup(m2.player, m2.state)
        m2.deals = list(lead_deals) + list(deals)
        h2 = m2.next_hand()
        _reseed(m2, seed)
    else:
        _reseed(m2, seed)
        m2.deals = list(deals)
        setup(m2.player, m2.state)
    ok = m2.player.forced_ok
    total = jm.play_out(h2, policy)[PLAYER]
    for _ in range(hands_left):
        m2.finish(h2)
        h2 = m2.next_hand()
        total += jm.play_out(h2, policy)[PLAYER]
    return total, ok


def _setups(jid):
    def play(p, _s):
        p.forced_ok = None
        p.lead = jid

    def keep(p, _s):
        p.ban = {jid}

    def swap(p, _s):
        p.forced_ok = None
        p.swap = jid
        p.ban = {jid}

    def drop(p, _s):
        p.forced_ok = None
        p.drop = jid
    return {"jouer": play, "garder": keep, "recycler": swap, "defausser": drop}


def _occasion(match, hand, jid, policy, hands_left, base_seed, k, lead_deals=(), can_recycle=True):
    """-> {branche: [points par rejouee]} ou None si le joker est refuse a la copie. Recycler
    (MatchState.recycle_error) seulement quand le joueur le peut a cette decision."""
    deals_seed = random.Random(base_seed)
    setups = _setups(jid)
    branches = [b for b in BRANCHES if b != "recycler" or can_recycle]
    runs = {b: [] for b in branches}
    for r in range(2 * k):
        seed = base_seed * 31 + r
        deals = jm.spare_deals(match, deals_seed.randrange(1 << 30))
        for b in branches:
            pts, ok = _rollout(match, hand, setups[b], policy, hands_left, deals, seed, lead_deals)
            if b != "garder" and ok is False:
                return None  # la copie refuse (cas limite) : occasion ignoree
            runs[b].append(pts)
    return runs


def _judge(runs, k):
    mean = lambda xs: sum(xs) / len(xs)
    first = {b: mean(xs[:k]) for b, xs in runs.items()}
    alt = max((b for b in ALTS if b in runs), key=lambda b: first[b])
    useful = first["jouer"] > first[alt]
    gain = mean(runs["jouer"][k:]) - mean(runs[alt][k:])
    return useful, gain, alt


def _cell():
    return {"n": 0, "s": 0.0, "q": 0.0}


def _add(c, x):
    c["n"] += 1
    c["s"] += x
    c["q"] += x * x


def _merge_cell(d, s):
    for key in ("n", "s", "q"):
        d[key] += s[key]


def _entry():
    return {"occasions": 0, "useful": 0, "rule_cast": 0, "rule_cast_useful": 0, "hands": 0, "hands_useful": 0,
            "force": _cell(), "tout": _cell(), "alt": {b: 0 for b in ALTS}, "by_contract": {}}


def worker(args):
    seeds, weights, k, value_only = args
    from scripts.train_jokers import make_policy
    from agents import joker_expert as X
    from agents.joker_expert import contract_key
    policy = make_policy(weights)
    data = {"jokers": {}, "value": {"energie": _cell(), "pioche": _cell()}, "value_by": {}, "matches": 0}

    def entry(jid):
        return data["jokers"].setdefault(J.base_id(jid), _entry())

    def record(jid, runs, rule_cast, contract, hand_marks):
        useful, gain, alt = _judge(runs, k)
        e = entry(jid)
        e["occasions"] += 1
        _add(e["tout"], gain)
        e["alt"][alt] += 1
        e["rule_cast"] += bool(rule_cast)
        bc = e["by_contract"].setdefault(contract, {"occasions": 0, "useful": 0, "force": _cell()})
        bc["occasions"] += 1
        if useful:
            e["useful"] += 1
            e["rule_cast_useful"] += bool(rule_cast)
            _add(e["force"], gain)
            bc["useful"] += 1
            _add(bc["force"], gain)
            hand_marks["useful"] = True

    for seed in seeds:
        try:
            rng = random.Random(seed)
            cfg = jm.random_config(rng)
            cfg["style"] = "expert"
            pool = sorted(i for i, (_t, r, _c, _b) in J.CATALOG.items()
                          if r in J.RARITIES_BY_ACT[cfg["act"]] and i not in J.REPLACED and _measurable(i))
            cfg["deck"] = list(cfg["deck"]) + [rng.choice(pool) for _ in range(INJECT)]
            match = jm.Match(cfg, rng.randrange(1 << 30))
            for hno in range(jm.HANDS_PER_MATCH):
                hands_left = jm.HANDS_PER_MATCH - 1 - hno
                occ_seed = rng.randrange(1 << 30)
                # 1er tour d'enchere : candidats du joueur (sonde sur une copie)
                focus, bid_done = None, False
                if cfg["kind"] == "elite" and not value_only:
                    probe = copy.deepcopy(match)
                    probe.player.bid_probe = []
                    n_rec = len(probe.recorded)
                    probe_h = probe.next_hand()
                    lead_deals = probe.recorded[n_rec:]
                    cands = probe.player.bid_probe[0] if probe.player.bid_probe else []
                    cands = [j for j in dict.fromkeys(cands) if _measurable(j)]
                    if cands and rng.random() < 0.5:
                        focus = rng.choice(cands)
                        runs = _occasion(match, None, focus, policy, hands_left, occ_seed, k, lead_deals,
                                         probe_h.state.recycle_error(focus, PLAYER) == "")
                        if runs is not None:
                            cards = probe.rules.deal_hands[PLAYER] if probe.rules.deal_hands else []
                            bv = X.bid_view(cards, probe_h.state) if cards else None
                            verdict = X.bid_rule(bv, focus)[0] if bv else X.NO
                            cast = verdict in (X.YES, X.OPEN_YES) and not X.eye_risk(probe_h.state, focus)
                            marks = {"useful": False}
                            record(focus, runs, cast, contract_key(probe_h.contract), marks)
                            e = entry(focus)
                            e["hands"] += 1
                            e["hands_useful"] += marks["useful"]
                        bid_done = True
                hand = match.next_hand()
                ckey = contract_key(hand.contract)
                if value_only:
                    focus = None
                elif not bid_done:
                    held = [j for j in dict.fromkeys(hand.state.joker_hand[PLAYER]) if _measurable(j) and J.jtype(j) != "enchere"]
                    focus = rng.choice(held) if held else None
                # valeur d'un ⚡ / d'une pioche, debut de la donne
                if value_only or rng.random() < VALUE_PROB:
                    deals = jm.spare_deals(match, rng.randrange(1 << 30))
                    vs = rng.randrange(1 << 30)

                    def plus_energy(_p, s):
                        s.energy[PLAYER] += 1

                    def plus_draw(_p, s):
                        s.draw_jokers(PLAYER, 1)
                    for _r in range(k):
                        base = _rollout(match, hand, lambda p, s: None, policy, hands_left, deals, vs)[0]
                        for name, fn in (("energie", plus_energy), ("pioche", plus_draw)):
                            d = _rollout(match, hand, fn, policy, hands_left, deals, vs)[0] - base
                            _add(data["value"][name], d)
                            _add(data["value_by"].setdefault("%s|%s" % (name, ckey), _cell()), d)
                        vs += 1
                marks = {"useful": False, "seen": False}
                n_occ = [0]

                def before_player(h_now):
                    if bid_done or focus is None or focus not in h_now.state.joker_hand[PLAYER]:
                        return
                    cands = h_now.player._candidates(h_now.state, h_now.rules, h_now.boss, False)
                    if focus not in cands:
                        return
                    n_occ[0] += 1
                    runs = _occasion(match, h_now, focus, policy, hands_left, occ_seed + n_occ[0], k,
                                     can_recycle=h_now.state.recycle_error(focus, PLAYER) == "")
                    if runs is None:
                        return
                    cast = X.player_wants(h_now, focus)[0]
                    marks["seen"] = True
                    record(focus, runs, cast, ckey, marks)

                jm.play_out(hand, policy, before_player=before_player)
                if marks["seen"]:
                    e = entry(focus)
                    e["hands"] += 1
                    e["hands_useful"] += marks["useful"]
                match.finish(hand)
        except Exception as exc:  # un match en erreur est compte et journalise, la mesure continue
            import traceback
            data["errors"] = data.get("errors", 0) + 1
            if "first_error" not in data:
                data["first_error"] = "seed %d : %s" % (seed, traceback.format_exc()[-1500:])
            continue
        data["matches"] += 1
    return data


def merge(dst, src):
    dst["matches"] += src["matches"]
    dst["errors"] = dst.get("errors", 0) + src.get("errors", 0)
    if "first_error" in src and "first_error" not in dst:
        dst["first_error"] = src["first_error"]
        print("ERREUR de match (mesure continue) :", src["first_error"], flush=True)
    for name in ("energie", "pioche"):
        _merge_cell(dst["value"][name], src["value"][name])
    for key, c in src["value_by"].items():
        _merge_cell(dst["value_by"].setdefault(key, _cell()), c)
    for jid, e in src["jokers"].items():
        d = dst["jokers"].setdefault(jid, _entry())
        for key in ("occasions", "useful", "rule_cast", "rule_cast_useful", "hands", "hands_useful"):
            d[key] += e[key]
        for key in ("force", "tout"):
            _merge_cell(d[key], e[key])
        for key in ALTS:
            d["alt"][key] += e["alt"].get(key, 0)
        for c, bc in e["by_contract"].items():
            db = d["by_contract"].setdefault(c, {"occasions": 0, "useful": 0, "force": _cell()})
            db["occasions"] += bc["occasions"]
            db["useful"] += bc["useful"]
            _merge_cell(db["force"], bc["force"])


def stat(c):
    if c["n"] == 0:
        return None, None
    m = c["s"] / c["n"]
    var = max(0.0, c["q"] / c["n"] - m * m)
    return m, math.sqrt(var / c["n"]) if c["n"] > 1 else None


def table(agg):
    rows = []
    for jid, e in sorted(agg["jokers"].items()):
        f, fse = stat(e["force"])
        g, gse = stat(e["tout"])
        rows.append({"id": jid, "type": J.jtype(jid), "rarity": J.jrarity(jid), "cost": J.jcost(jid),
                     "suspicion": J.jsuspicion(jid), "occasions": e["occasions"], "useful": e["useful"],
                     "useful_rate": round(e["useful"] / e["occasions"], 3) if e["occasions"] else None,
                     "force": None if f is None else round(f, 3), "force_se": None if fse is None else round(fse, 3),
                     "hands": e["hands"], "polyvalence": round(e["hands_useful"] / e["hands"], 3) if e["hands"] else None,
                     "bien_vu": round(e["rule_cast_useful"] / e["useful"], 3) if e["useful"] else None,
                     "rule_cast_rate": round(e["rule_cast"] / e["occasions"], 3) if e["occasions"] else None,
                     "gain_tout": None if g is None else round(g, 3), "gain_tout_se": None if gse is None else round(gse, 3),
                     "alt": {b: round(e["alt"].get(b, 0) / e["occasions"], 3) for b in ALTS} if e["occasions"] else None,
                     "fiable": e["useful"] >= MIN_N,
                     "by_contract": {c: {"occasions": bc["occasions"], "useful": bc["useful"],
                                         "force": None if stat(bc["force"])[0] is None else round(stat(bc["force"])[0], 2)}
                                     for c, bc in sorted(e["by_contract"].items())}})
    return rows


def render(agg, meta):
    rows = table(agg)
    value = {name: stat(agg["value"][name]) for name in ("energie", "pioche")}
    value_by = {key: {"n": c["n"], "mean": round(stat(c)[0], 3)} for key, c in sorted(agg["value_by"].items()) if c["n"]}
    out = dict(meta, matches=agg["matches"], k=meta["k"], errors=agg.get("errors", 0),
               first_error=agg.get("first_error"),
               value={n: {"n": agg["value"][n]["n"], "mean": None if m is None else round(m, 3),
                          "se": None if se is None else round(se, 3)} for n, (m, se) in value.items()},
               value_by=value_by, rows=rows)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    fmt = lambda x: "-" if x is None else ("%.2f" % x if isinstance(x, float) else str(x))
    lines = ["# Moments utiles des jokers (%s)" % meta["generated"], "",
             "Empreinte %s, %d matchs, K = %d (decision sur K rejouees, gain mesure sur K autres). "
             "Gain = points du joueur jusqu'a la fin du match, jouer maintenant contre la meilleure de garder / recycler." % (
                 meta["spec_hash"], agg["matches"], meta["k"]), "",
             "Valeur d'un ⚡ : %s pts (se %s, n %d) ; d'une pioche : %s pts (se %s, n %d)." % (
                 fmt(value["energie"][0]), fmt(value["energie"][1]), agg["value"]["energie"]["n"],
                 fmt(value["pioche"][0]), fmt(value["pioche"][1]), agg["value"]["pioche"]["n"]), "",
             "| joker | cout | occasions | utiles | force (se) | polyvalence | bien vu | gain toutes occasions | fiable |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -(r["force"] or -99)):
        lines.append("| %s | %d | %d | %s | %s (%s) | %s | %s | %s | %s |" % (
            r["id"], r["cost"], r["occasions"], fmt(r["useful_rate"]), fmt(r["force"]), fmt(r["force_se"]),
            fmt(r["polyvalence"]), fmt(r["bien_vu"]), fmt(r["gain_tout"]), "oui" if r["fiable"] else "non"))
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return rows


def upload():
    if os.environ.get("GH_TOKEN"):
        subprocess.run(["gh", "release", "upload", "checkpoint-latest", OUT_JSON, OUT_MD, "--clobber"], check=False)


def main():
    import torch
    from agents.network import PolicyValueNet
    from engine import joker_obs as O
    from scripts.train_jokers import CHECKPOINT, net_to_numpy
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=150)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--k", type=int, default=2, help="rejouees par moitie (decision / mesure)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--matches", type=int, default=0, help="arret apres ce nombre de matchs (0 = au temps)")
    parser.add_argument("--value-only", action="store_true", help="8ebd : seulement la valeur du ⚡ / de la pioche")
    args = parser.parse_args()
    global OUT_JSON, OUT_MD
    if args.value_only:
        OUT_JSON, OUT_MD = VALUE_JSON, VALUE_MD
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    net = PolicyValueNet(O.OBS_DIM, J.ACTION_DIM)
    net.load_state_dict(ckpt["net"])
    net.eval()
    w = net_to_numpy(net)
    meta = {"spec_hash": J.JOKER_SPEC_HASH, "bots_net_spec_hash": ckpt.get("spec_hash"), "bots_net_iter": ckpt.get("iter"),
            "k": args.k, "inject": INJECT, "value_only": args.value_only}
    agg = {"jokers": {}, "value": {"energie": _cell(), "pioche": _cell()}, "value_by": {}, "matches": 0}
    start = time.time()
    end, next_upload, done = start + args.minutes * 60, start + UPLOAD_EVERY_S, 0
    with Pool(args.workers) as pool:
        while time.time() < end and (args.matches == 0 or done < args.matches):
            n = BATCH if args.matches == 0 else min(BATCH, args.matches - done)
            seeds = list(range(args.seed + done, args.seed + done + n))
            t0 = time.time()
            for data in pool.map(worker, [(seeds[i::args.workers], w, args.k, args.value_only) for i in range(args.workers)]):
                merge(agg, data)
            done += n
            meta["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            meta["elapsed_min"] = round((time.time() - start) / 60, 1)
            render(agg, meta)
            print("matchs %d (%.1f s/match/worker), %d en erreur, %.0f min restantes" % (
                done, (time.time() - t0) * args.workers / n, agg.get("errors", 0), (end - time.time()) / 60), flush=True)
            if agg.get("errors", 0) > max(20, done // 20):
                raise SystemExit("trop de matchs en erreur (%d / %d)" % (agg["errors"], done))
            if time.time() >= next_upload:
                next_upload = time.time() + UPLOAD_EVERY_S
                upload()
    rows = render(agg, meta)
    for r in rows:
        print(json.dumps({k: r[k] for k in ("id", "cost", "occasions", "useful_rate", "force", "force_se", "polyvalence",
                                            "bien_vu", "gain_tout")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
