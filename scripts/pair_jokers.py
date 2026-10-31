"""Mesure de paires de jokers (Synergies, demande whist-4-it 31/10, empreinte 7567).
Mesure seule, cloud. Pour chaque paire (a, b), sur les memes donnes :

  regles : le joueur style expert, avec aucun / a / b / a+b en main
           -> gain de chacun, interaction = G(ab) - G(a) - G(b).
  oracle : le joueur CHERCHE le combo : meilleur moment de lancer, en connaissant
           la suite de la donne (lancers imposes, tours essayes) -> borne haute.
           Interaction = G*(ab) - G*(a) - G*(b), sur les donnes ou a+b ont ete lances.
  naturel : a et b une fois dans le deck, donnes normales : combien de donnes les
           deux sont en main, payables, et reellement lances.

Gain = points du joueur sur la donne + la suivante (comme scripts/joker_profit.py).
Les jokers de la paire sont mis en main du joueur a chaque donne et les autres
retires (pas de bruit d'un autre joker). Une paire plus chere que l'energie de la
donne est mesuree avec l'energie gonflee dans TOUTES les branches : hypothese.
Limite : un seul lancer par tour de pli du joueur.
6483 (etape 05 Synergies) : Chaleur active comme dans un run ; les passifs de la paire
sont armes en debut de branche, ceux du deck retires ; une paire de LATER mesure le
report : le 1er joker a la donne 1, le 2e seulement a la donne suivante. Seule en main,
une carte comme Feu de Paille recoit toute l'energie : gain a lire comme un plafond.

Usage : python scripts/pair_jokers.py --minutes 15 --workers 4
"""

import argparse
import copy
import json
import math
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, ".")

from engine import joker_match as jm
from engine import jokers as J
from scripts import joker_profit as P

PAIRS = [
    ("complice", "kermesse", ""),
    ("complice", "addition", ""),
    ("complice", "bal_des_valets", ""),
    ("feu_de_paille", "ardoise", ""),
    ("boite_d_allumettes", "feu_de_paille", ""),
    ("main_lourde", "consigne", ""),
    ("main_lourde", "cendrier", ""),
    ("valet_de_chambre", "bal_des_valets", ""),
    ("bal_des_valets", "beau_valet", ""),
    ("refrain", "ardoise", ""),
]
LATER = {("boite_d_allumettes", "feu_de_paille"): "feu_de_paille"}  # le 2e arrive a la donne suivante
SINGLE_TRIES = 6   # moments essayes par joker seul
PAIR_TRIES = 12    # couples de moments essayes pour la paire
NATURAL_SHARE = 0.15  # part du temps pour la frequence naturelle
OUT = "pair_jokers.json"
P0 = jm.PLAYER


def _cell():
    return {"n": 0, "s": 0.0, "q": 0.0}


def _add(cell, x):
    cell["n"] += 1
    cell["s"] += x
    cell["q"] += x * x


def _merge_cells(dst, src):
    for k, v in src.items():
        if isinstance(v, dict):
            d = dst.setdefault(k, _cell())
            for f in ("n", "s", "q"):
                d[f] += v[f]
        else:
            dst[k] = dst.get(k, 0) + v


def _fmt(cell):
    n = cell["n"]
    if n == 0:
        return {"n": 0, "gain": None, "se": None}
    m = cell["s"] / n
    var = max(0.0, cell["q"] / n - m * m) * n / max(1, n - 1)
    return {"n": n, "gain": round(m, 3), "se": round(math.sqrt(var / n), 3)}


def _setup(cfg, rng, force_misere):
    cfg.update({"kind": "elite", "act": rng.choice((1, 2, 3)), "eye": J.ELITE_EYE, "boss": "", "style": "expert", "heat": True})
    if force_misere:
        cfg["force_misere"] = force_misere


def _hook(sched):
    """Lance le joker prevu au premier tour legal a partir de son pli ; rien d'autre."""
    def hook(h):
        p = h.player
        p.force_trick = ""
        cands = p._candidates(h.state, h.rules, h.boss, False)
        for jid, turn in sched:
            if h.game.trick_count >= turn and jid in cands:
                p.force_trick = jid
                return
    return hook


def _branch(match, hand, keep, pair, boost, policy, spare, more, hook=None):
    """-> (points du joueur, jokers de la paire lances)."""
    m, h = copy.deepcopy((match, hand))
    s = h.state
    later = LATER.get(pair, "")
    first = [j for j in keep if j != later]
    for jid in s.joker_hand[P0]:
        if J.jtype(jid) == "passif":
            h.rules.disarm(P0, jid)
    s.joker_hand[P0] = list(first)
    for jid in first:  # un passif agit depuis une case, comme a la pioche
        if J.jtype(jid) == "passif" and h.rules.free_case(P0) >= 0:
            h.rules.play(P0, jid)
    if boost:
        s.energy[P0] = max(s.energy[P0], boost)
    pts = list(jm.play_out(h, policy, before_player=hook))
    cast = [j for j in first if j not in s.joker_hand[P0] or J.base_id(j) == "boite_d_allumettes"]
    s.joker_hand[P0] = [j for j in s.joker_hand[P0] if J.base_id(j) not in pair]
    if more:
        m.deals = list(spare)
        h2 = m.next_hand()
        if later in keep:
            h2.state.joker_hand[P0].append(later)
        pts = [a + b for a, b in zip(pts, jm.play_out(h2, policy))]
        if later in keep and later not in h2.state.joker_hand[P0]:
            cast.append(later)
    return pts[P0], cast


def _measure_hand(match, hand, pair, boost, policy, spare, more, rng, out):
    a, b = pair
    legal = {a: [], b: []}

    def probe(h):
        h.player.force_trick = ""
        for jid in h.player._candidates(h.state, h.rules, h.boss, False):
            if jid in legal and h.game.trick_count not in legal[jid]:
                legal[jid].append(h.game.trick_count)

    base, _c = _branch(match, hand, [a, b], pair, boost, policy, spare, more, probe)
    none, _c = _branch(match, hand, [], pair, boost, policy, spare, more)
    ra, ca = _branch(match, hand, [a], pair, boost, policy, spare, more)
    rb, cb = _branch(match, hand, [b], pair, boost, policy, spare, more)
    rab, cab = _branch(match, hand, [a, b], pair, boost, policy, spare, more)
    _add(out["base_tenus_sans_lancer"], base - none)
    _add(out["regles_a"], ra - none)
    _add(out["regles_b"], rb - none)
    _add(out["regles_ab"], rab - none)
    _add(out["regles_interaction"], rab - ra - rb + none)
    out["regles_lance_a"] = out.get("regles_lance_a", 0) + len(ca)
    out["regles_lance_b"] = out.get("regles_lance_b", 0) + len(cb)
    out["regles_lance_ab"] = out.get("regles_lance_ab", 0) + (len(cab) == 2)
    out["donnes"] = out.get("donnes", 0) + 1
    if not legal[a] or not legal[b]:
        return
    out["donnes_a_et_b_legaux"] = out.get("donnes_a_et_b_legaux", 0) + 1

    def best_single(jid):
        turns = rng.sample(legal[jid], min(SINGLE_TRIES, len(legal[jid])))
        vals = []
        for t in turns:
            v, c = _branch(match, hand, [jid], pair, boost, policy, spare, more, _hook([(jid, t)]))
            if c:
                vals.append(v)
        return max(vals) if vals else None

    combos = [(ta, tb) for ta in legal[a] for tb in legal[b] if ta != tb]
    combos = rng.sample(combos, min(PAIR_TRIES, len(combos)))
    vals = []
    for ta, tb in combos:
        sched = sorted([(a, ta), (b, tb)], key=lambda x: x[1])
        v, c = _branch(match, hand, [a, b], pair, boost, policy, spare, more, _hook(sched))
        if len(c) == 2:
            vals.append(v)
    if not vals:
        return
    out["oracle_paire_lancee"] = out.get("oracle_paire_lancee", 0) + 1
    ga, gb = best_single(a), best_single(b)
    if ga is None or gb is None:
        return
    gab = max(vals)
    _add(out["oracle_a"], ga - none)
    _add(out["oracle_b"], gb - none)
    _add(out["oracle_ab"], gab - none)
    _add(out["oracle_interaction"], gab - ga - gb + none)


def pair_worker(args):
    seeds, weights, a, b, force_misere = args
    from scripts.train_jokers import make_policy
    policy = make_policy(weights)
    pair = (a, b)
    out = {k: _cell() for k in ("base_tenus_sans_lancer", "regles_a", "regles_b", "regles_ab", "regles_interaction",
                                "oracle_a", "oracle_b", "oracle_ab", "oracle_interaction")}
    for seed in seeds:
        rng = random.Random(seed)
        cfg = jm.random_config(rng)
        _setup(cfg, rng, force_misere)
        match = jm.Match(cfg, rng.randrange(1 << 30))
        for k in range(jm.HANDS_PER_MATCH):
            more = k + 1 < jm.HANDS_PER_MATCH
            spare = P._spare(match, random.Random(rng.random()))
            hand = match.next_hand()
            cost = sum(hand.state.cost_of(P0, j) for j in pair)
            boost = cost if cost > hand.state.energy[P0] else 0
            if boost:
                out["donnes_energie_gonflee"] = out.get("donnes_energie_gonflee", 0) + 1
            _measure_hand(match, hand, pair, boost, policy, spare, more, rng, out)
            jm.play_out(hand, policy)
            match.finish(hand)
    return out


def natural_worker(args):
    """a et b une fois dans un deck normal, regles du joueur."""
    seeds, weights, a, b, force_misere = args
    from scripts.train_jokers import make_policy
    policy = make_policy(weights)
    log = []
    orig = jm.Player._cast

    def traced(self, state, rules, eye, jid, target=None):
        played = orig(self, state, rules, eye, jid, target)
        if played is not None:
            log.append(J.base_id(jid))
        return played

    jm.Player._cast = traced
    out = {"donnes": 0, "deux_en_main": 0, "deux_payables": 0, "un_lance": 0, "deux_lances": 0}
    try:
        for seed in seeds:
            rng = random.Random(seed)
            cfg = jm.random_config(rng)
            _setup(cfg, rng, force_misere)
            cfg["deck"] = [j for j in cfg["deck"] if J.base_id(j) not in (a, b)] + [a, b]
            match = jm.Match(cfg, rng.randrange(1 << 30))
            for _k in range(jm.HANDS_PER_MATCH):
                del log[:]
                hand = match.next_hand()
                held = [j for j in hand.state.joker_hand[P0] if J.base_id(j) in (a, b)]
                both = len({J.base_id(j) for j in held}) == 2 or (a in log or b in log) and len(held) == 1
                out["donnes"] += 1
                out["deux_en_main"] += both
                if both and sum(hand.state.cost_of(P0, j) for j in held) <= hand.state.energy[P0]:
                    out["deux_payables"] += 1
                jm.play_out(hand, policy)
                got = {a, b} & set(log)
                out["un_lance"] += len(got) == 1
                out["deux_lances"] += len(got) == 2
                match.finish(hand)
    finally:
        jm.Player._cast = orig
    return out


def _timed(pool, fn, jobs_of, minutes, chunk):
    """Lance fn par lots jusqu'a l'echeance ; -> (resultats, matchs)."""
    end, start, results, done = time.time() + minutes * 60, 0, [], 0
    while True:
        seeds, jobs = jobs_of(start, chunk)
        results += pool.map(fn, jobs)
        start += len(seeds)
        done += len(seeds)
        if time.time() >= end:
            return results, done


def main():
    import torch
    from agents.network import PolicyValueNet
    from engine import joker_obs as O
    from scripts.train_jokers import CHECKPOINT, net_to_numpy
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=93000000)
    args = parser.parse_args()
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    net = PolicyValueNet(O.OBS_DIM, J.ACTION_DIM)
    net.load_state_dict(ckpt["net"])
    net.eval()
    w = net_to_numpy(net)
    print("champion iteration", ckpt.get("iter"), "empreinte", J.JOKER_SPEC_HASH, flush=True)
    rows = []
    with Pool(args.workers) as pool:
        for a, b, misere in PAIRS:
            def jobs(start, chunk):
                seeds = list(range(args.seed + start, args.seed + start + chunk))
                return seeds, [(seeds[i::args.workers], w, a, b, misere) for i in range(args.workers)]
            res, matches = _timed(pool, pair_worker, jobs, args.minutes * (1 - NATURAL_SHARE), args.workers)
            agg = {}
            for r in res:
                _merge_cells(agg, r)
            nat_res, nat_matches = _timed(pool, natural_worker, jobs, args.minutes * NATURAL_SHARE, args.workers * 8)
            nat = {}
            for r in nat_res:
                for k, v in r.items():
                    nat[k] = nat.get(k, 0) + v
            row = {"a": a, "b": b, "misere_imposee": misere, "matchs": matches, "matchs_naturels": nat_matches,
                   "cout": [J.CATALOG[a][2], J.CATALOG[b][2]], "naturel": nat,
                   **{k: (_fmt(v) if isinstance(v, dict) else v) for k, v in agg.items()}}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            with open(OUT, "w", encoding="utf-8") as f:
                json.dump({"spec_hash": J.JOKER_SPEC_HASH, "rows": rows}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
