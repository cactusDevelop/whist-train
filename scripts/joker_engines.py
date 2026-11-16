"""Decomposition de la « valeur de l'avoir » des moteurs (demande whist-4-it 16/11, apres
conseil) : pourquoi Feu de Paille2, L'Ardoise2, Vieille Habitude2, Beau Valet2 valent < 0 en main
(temoins Chope Vide3, Mauvais Perdant2). Mesure seule, cloud.

Par donne : un moteur suivi (tire parmi ceux de la liste que le joueur tient au debut de la donne,
contrat connu). La meme situation (memes donnes de la suite, memes graines) est rejouee K fois
jusqu'a la FIN DU MATCH sous quatre branches appariees :
  avec   : la main reelle, la regle du joueur (style expert) joue normalement ;
  sans   : le moteur retire de la main (= has_value de joker_moments) ;
  jamais : le moteur garde en main mais jamais lance (il peut partir pour l'energie) ;
  strict : il n'est lance que si la pioche couvre TOUT son tirage (Ardoise2 2, Vieille Habitude2
           le manque, Beau Valet2 ses Valets, Chope Vide3 / Feu de Paille2 1 = regle actuelle,
           Mauvais Perdant2 sans tirage = regle actuelle).
Rappel du code : la regle lance deja un moteur seulement si la pioche est non vide (pointless ->
state.can_draw : sous dominion une pioche vide ne reprend la defausse qu'a l'ouverture) et si les ⚡
suffisent (play_error "energy") ; un tirage PARTIEL reste possible (pioche plus courte que le tirage).

Par branche : points du joueur (fin de match), ⚡ depenses et jokers lances (reste du match),
Chaleur (soupcon) en fin de donne. Dans « avec » (et « strict ») : lancers du moteur, jokers demandes
/ obtenus par ses tirages (taux de pioche vide = 1 - obtenus/demandes), parmi les obtenus ceux lances
ensuite dans la donne, Feu de Paille2 : revelations, revelations sur pioche vide, sorts lances gratis,
Chaleur ajoutee par le moteur et par ce qu'il fait lancer.

Usage : python scripts/joker_engines.py --minutes 180 --workers 4 [--k 2]
Sorties : joker_engines.json, joker_engines_summary.md.
"""

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, ".")

from engine import joker_match as jm
from engine import jokers as J
from scripts.joker_moments import _add, _cell, _merge_cell, _ms, _reseed

OUT_JSON = "joker_engines.json"
OUT_MD = "joker_engines_summary.md"
SEED = 96000000
TARGETS = ("feu_de_paille2", "ardoise2", "vieille_habitude2", "beau_valet2", "chope_vide3", "mauvais_perdant2")
INJECT = 2
BATCH = 40
UPLOAD_EVERY_S = 900
BRANCHES = ("avec", "sans", "jamais", "strict")
PLAYER = jm.PLAYER
BRANCH_KEYS = ("pts", "spent", "played", "heat_hand")
FOCUS_KEYS = ("casts", "req", "got", "drawn_played", "reveals", "reveal_empty", "free_casts", "heat_focus")

# --- portes de lancement (seulement le siege du joueur, seulement le moteur suivi) ---
GATE = {"mode": None, "base": None}


def _need(eff, rules):
    name = type(eff).__name__
    if name == "Ardoise2":
        return eff.DRAWS
    if name == "VieilleHabitude2":
        return eff._missing(rules)
    if name == "BeauValet2":
        return eff._jacks(rules)
    return 1 if name in ("ChopeVide3", "FeuDePaille2") else 0


def _patch_gates():
    for base in TARGETS:
        cls = J.EFFECTS[base]
        if getattr(cls, "_gate_patched", False):
            continue
        orig = cls.pointless

        def pointless(self, rules, _orig=orig, _base=base):
            if self.owner == PLAYER and GATE["base"] == _base and GATE["mode"]:
                if GATE["mode"] == "jamais":
                    return True
                if _orig(self, rules):
                    return True
                return len(rules.state.draw_pile[self.owner]) < _need(self, rules)
            return _orig(self, rules)
        cls.pointless = pointless
        cls._gate_patched = True


def _instrument(m2, focus):
    """Compteurs sur la copie (wrappers d'instance de MatchState / JokerRules)."""
    st, ru = m2.state, m2.rules
    fb = J.base_id(focus)
    rec = {k: 0 for k in FOCUS_KEYS + ("spent", "played")}
    inside = [0]
    pending = []
    orig_play, orig_spend, orig_draw, orig_reveal, orig_end = ru.play, st.spend, st.draw_jokers, st.reveal, st.end_hand

    def play(seat, jid, target=-1):
        if seat == PLAYER and J.base_id(jid) == fb and inside[0] == 0:
            inside[0] += 1
            s0 = st.suspicion
            try:
                return orig_play(seat, jid, target)
            finally:
                inside[0] -= 1
                rec["heat_focus"] += st.suspicion - s0
        return orig_play(seat, jid, target)

    def spend(seat, jid, free=False):
        if seat != PLAYER:
            return orig_spend(seat, jid, free)
        cost = 0 if free else st.cost_of(seat, jid)
        s0 = st.suspicion
        out = orig_spend(seat, jid, free)
        rec["spent"] += cost
        rec["played"] += 1
        if jid in pending:
            pending.remove(jid)
            rec["drawn_played"] += 1
        if J.base_id(jid) == fb and not inside[0]:
            rec["casts"] += 1
            rec["heat_focus"] += st.suspicion - s0
        elif inside[0] and free:
            rec["free_casts"] += 1
        return out

    def draw(seat, n, opening=False):
        if seat != PLAYER or not inside[0]:
            return orig_draw(seat, n, opening)
        before = list(st.draw_pile[seat])
        tot0 = len(st.draw_pile[seat]) + len(st.discard[seat])
        out = orig_draw(seat, n, opening)
        got = max(0, tot0 - len(st.draw_pile[seat]) - len(st.discard[seat]))
        rec["req"] += n
        rec["got"] += min(got, n)
        if len(st.draw_pile[seat]) <= len(before) and got:
            pending.extend(before[len(before) - got:])
        return out

    def reveal(seat):
        out = orig_reveal(seat)
        if seat == PLAYER and inside[0]:
            rec["reveals"] += 1
            rec["reveal_empty"] += out == ""
        return out

    def end_hand():
        pending.clear()
        return orig_end()
    ru.play, st.spend, st.draw_jokers, st.reveal, st.end_hand = play, spend, draw, reveal, end_hand
    return rec


def _rollout(match, hand, branch, focus, policy, hands_left, deals, seed):
    m2, h2 = copy.deepcopy((match, hand))
    _reseed(m2, seed)
    m2.deals = list(deals)
    if branch == "sans":
        m2.state.joker_hand[PLAYER].remove(focus)
    rec = _instrument(m2, focus)
    GATE["mode"], GATE["base"] = (branch if branch in ("jamais", "strict") else None), J.base_id(focus)
    try:
        total = jm.play_out(h2, policy)[PLAYER]
        rec["heat_hand"] = m2.state.suspicion
        for _ in range(hands_left):
            m2.finish(h2)
            h2 = m2.next_hand()
            total += jm.play_out(h2, policy)[PLAYER]
    finally:
        GATE["mode"] = None
    rec["pts"] = total
    return rec


def _entry():
    return {"n": 0, "branch": {b: {k: _cell() for k in BRANCH_KEYS} for b in BRANCHES},
            "diff": {d: _cell() for d in ("avec-sans", "avec-jamais", "strict-avec", "jamais-sans")},
            "d_spent": _cell(), "d_played": _cell(), "d_heat": _cell(),
            "focus": {b: {k: 0 for k in FOCUS_KEYS} for b in ("avec", "strict")}, "strict_changed": 0}


def worker(args):
    seeds, weights, k = args
    from scripts.train_jokers import make_policy
    _patch_gates()
    policy = make_policy(weights)
    data = {"jokers": {}, "matches": 0}
    for seed in seeds:
        try:
            rng = random.Random(seed)
            cfg = jm.random_config(rng)
            cfg["style"] = "expert"
            cfg["deck"] = list(cfg["deck"]) + [rng.choice(TARGETS) for _ in range(INJECT)]
            match = jm.Match(cfg, rng.randrange(1 << 30))
            for hno in range(jm.HANDS_PER_MATCH):
                hands_left = jm.HANDS_PER_MATCH - 1 - hno
                hand = match.next_hand()
                held = [j for j in dict.fromkeys(hand.state.joker_hand[PLAYER]) if J.base_id(j) in TARGETS]
                if held:
                    focus = rng.choice(held)
                    e = data["jokers"].setdefault(J.base_id(focus), _entry())
                    deals = jm.spare_deals(match, rng.randrange(1 << 30))
                    vs = rng.randrange(1 << 30)
                    for _r in range(k):
                        res = {b: _rollout(match, hand, b, focus, policy, hands_left, deals, vs) for b in BRANCHES}
                        vs += 1
                        e["n"] += 1
                        for b in BRANCHES:
                            for key in BRANCH_KEYS:
                                _add(e["branch"][b][key], res[b][key])
                        p = {b: res[b]["pts"] for b in BRANCHES}
                        _add(e["diff"]["avec-sans"], p["avec"] - p["sans"])
                        _add(e["diff"]["avec-jamais"], p["avec"] - p["jamais"])
                        _add(e["diff"]["strict-avec"], p["strict"] - p["avec"])
                        _add(e["diff"]["jamais-sans"], p["jamais"] - p["sans"])
                        _add(e["d_spent"], res["avec"]["spent"] - res["sans"]["spent"])
                        _add(e["d_played"], res["avec"]["played"] - res["sans"]["played"])
                        _add(e["d_heat"], res["avec"]["heat_hand"] - res["sans"]["heat_hand"])
                        e["strict_changed"] += res["strict"]["casts"] != res["avec"]["casts"]
                        for b in ("avec", "strict"):
                            for key in FOCUS_KEYS:
                                e["focus"][b][key] += res[b][key]
                jm.play_out(hand, policy)
                match.finish(hand)
        except Exception:
            import traceback
            data["errors"] = data.get("errors", 0) + 1
            data.setdefault("first_error", "seed %d : %s" % (seed, traceback.format_exc()[-1500:]))
            continue
        data["matches"] += 1
    return data


def merge(dst, src):
    dst["matches"] += src["matches"]
    dst["errors"] = dst.get("errors", 0) + src.get("errors", 0)
    if "first_error" in src and "first_error" not in dst:
        dst["first_error"] = src["first_error"]
        print("ERREUR de match (mesure continue) :", src["first_error"], flush=True)
    for jid, e in src["jokers"].items():
        d = dst["jokers"].setdefault(jid, _entry())
        d["n"] += e["n"]
        d["strict_changed"] += e["strict_changed"]
        for b in BRANCHES:
            for key in BRANCH_KEYS:
                _merge_cell(d["branch"][b][key], e["branch"][b][key])
        for key in d["diff"]:
            _merge_cell(d["diff"][key], e["diff"][key])
        for key in ("d_spent", "d_played", "d_heat"):
            _merge_cell(d[key], e[key])
        for b in ("avec", "strict"):
            for key in FOCUS_KEYS:
                d["focus"][b][key] += e["focus"][b][key]


def render(agg, meta):
    rows = []
    for jid in TARGETS:
        e = agg["jokers"].get(jid)
        if not e or not e["n"]:
            continue
        f = e["focus"]["avec"]
        n = e["n"]
        rows.append({"id": jid, "cost": J.jcost(jid), "suspicion": J.jsuspicion(jid), "n": n,
                     "diff": {k: _ms(c) for k, c in e["diff"].items()},
                     "d_spent": _ms(e["d_spent"]), "d_played": _ms(e["d_played"]), "d_heat": _ms(e["d_heat"]),
                     "casts_per": round(f["casts"] / n, 3),
                     "draw_req_per": round(f["req"] / n, 3), "draw_got_per": round(f["got"] / n, 3),
                     "empty_rate": round(1 - f["got"] / f["req"], 3) if f["req"] else None,
                     "drawn_played_rate": round(f["drawn_played"] / f["got"], 3) if f["got"] else None,
                     "reveals_per": round(f["reveals"] / n, 3), "reveal_empty_rate": round(f["reveal_empty"] / f["reveals"], 3) if f["reveals"] else None,
                     "free_casts_per": round(f["free_casts"] / n, 3), "heat_focus_per": round(f["heat_focus"] / n, 3),
                     "strict_changed_rate": round(e["strict_changed"] / n, 3),
                     "focus_strict": e["focus"]["strict"]})
    out = dict(meta, matches=agg["matches"], errors=agg.get("errors", 0), first_error=agg.get("first_error"), rows=rows)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    fmt = lambda x: "-" if x is None else ("%.2f" % x if isinstance(x, float) else str(x))
    ms = lambda c: "%s (%s)" % (fmt(c["mean"]), fmt(c["se"]))
    lines = ["# Moteurs : decomposition de la valeur de l'avoir (%s)" % meta.get("generated", ""), "",
             "Empreinte %s, %d matchs, K = %d, branches appariees (memes donnes, memes graines), points du joueur "
             "jusqu'a la fin du match. Regle actuelle : un moteur n'est lance que pioche non vide et ⚡ suffisants." % (
                 meta["spec_hash"], agg["matches"], meta["k"]), "",
             "| joker | cout | n | avec-sans (has) | avec-jamais (le lancer) | jamais-sans (le tenir) | strict-avec | strict change |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d = r["diff"]
        lines.append("| %s | %d | %d | %s | %s | %s | %s | %s |" % (
            r["id"], r["cost"], r["n"], ms(d["avec-sans"]), ms(d["avec-jamais"]), ms(d["jamais-sans"]),
            ms(d["strict-avec"]), fmt(r["strict_changed_rate"])))
    lines += ["", "| joker | lancers/donne | ⚡ depenses (avec-sans) | jokers lances (avec-sans) | tirage demande / obtenu | pioche vide | pioches lancees ensuite | Feu : revelations (vides) / sorts gratis | Chaleur fin de donne (avec-sans) | Chaleur du moteur /donne |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %s | %s | %s | %s / %s | %s | %s | %s (%s) / %s | %s | %s |" % (
            r["id"], fmt(r["casts_per"]), ms(r["d_spent"]), ms(r["d_played"]), fmt(r["draw_req_per"]), fmt(r["draw_got_per"]),
            fmt(r["empty_rate"]), fmt(r["drawn_played_rate"]), fmt(r["reveals_per"]), fmt(r["reveal_empty_rate"]),
            fmt(r["free_casts_per"]), ms(r["d_heat"]), fmt(r["heat_focus_per"])))
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
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
    parser.add_argument("--minutes", type=float, default=180)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--k", type=int, default=2, help="rejouees appariees par donne")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--matches", type=int, default=0, help="arret apres ce nombre de matchs (0 = au temps)")
    args = parser.parse_args()
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    net = PolicyValueNet(O.OBS_DIM, J.ACTION_DIM)
    net.load_state_dict(ckpt["net"])
    net.eval()
    w = net_to_numpy(net)
    meta = {"spec_hash": J.JOKER_SPEC_HASH, "bots_net_iter": ckpt.get("iter"), "k": args.k, "inject": INJECT,
            "targets": list(TARGETS)}
    agg = {"jokers": {}, "matches": 0}
    start = time.time()
    end, next_upload, done = start + args.minutes * 60, start + UPLOAD_EVERY_S, 0
    with Pool(args.workers) as pool:
        while time.time() < end and (args.matches == 0 or done < args.matches):
            n = BATCH if args.matches == 0 else min(BATCH, args.matches - done)
            seeds = list(range(args.seed + done, args.seed + done + n))
            t0 = time.time()
            for data in pool.map(worker, [(seeds[i::args.workers], w, args.k) for i in range(args.workers)]):
                merge(agg, data)
            done += n
            meta["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            meta["elapsed_min"] = round((time.time() - start) / 60, 1)
            render(agg, meta)
            print("matchs %d (%.1f s/match/worker), %d en erreur, n %s, %.0f min restantes" % (
                done, (time.time() - t0) * args.workers / n, agg.get("errors", 0),
                {j: e["n"] for j, e in agg["jokers"].items()}, (end - time.time()) / 60), flush=True)
            if agg.get("errors", 0) > max(20, done // 20):
                raise SystemExit("trop de matchs en erreur (%d / %d)" % (agg["errors"], done))
            if time.time() >= next_upload:
                next_upload = time.time() + UPLOAD_EVERY_S
                upload()
    render(agg, meta)
    print(open(OUT_MD, encoding="utf-8").read())
    print("JOKER_ENGINES_DONE")


if __name__ == "__main__":
    main()
