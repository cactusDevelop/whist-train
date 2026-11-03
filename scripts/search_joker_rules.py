"""Recherche des regles des bots a jokers DIRECTEMENT sur le score de match
(mesure seule, le champion n'est pas modifie).

Pourquoi (03/11) : depuis 1f49, aucune iteration de train_jokers.py ne bat le
champion ; les cibles des rollouts (par donne) contredisent la porte (match),
et les reculs suivent les Kermesses posees en moins. Ici on ne regresse plus
rien : on change UN cas de regle (agents/joker_expert, "joker:cas" -> oui/non)
et on mesure en duplicate contre la meilleure version courante, sur des
matchs neufs a chaque mesure.

  1. decouverte : sur des matchs du champion, les cas OUVERTS (le reseau
     decide) et leur frequence ;
  2. Kermesse en regle fixe (toujours posee quand la regle l'ouvre) ;
  3. descente par coordonnees : pour chaque cas frequent, oui puis non ;
     criblage SCREEN matchs (m > 0 et m > SCREEN_Z se), confirmation
     CONFIRM matchs neufs (m > 2 se) -> garde, sinon rejete.

Usage cloud : python -u scripts/search_joker_rules.py --until 2026-11-03T10:30 --workers 4
"""

import argparse
import datetime
import json
import math
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, ".")

from engine import joker_match as jm
from engine import jokers as J

SEARCH_SEED = 60000000  # hors GATE (30M..50M), PROFIT (77M), OOS (91M), pairs (93M)
SEED_STRIDE = 10007


def _stat(xs):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return m, sd / math.sqrt(len(xs))


def discover_worker(args):
    """-> {cas ouvert: [decisions, poses]} sur les matchs du champion."""
    seeds, w = args
    import numpy as np
    from agents import joker_expert as X
    from engine import joker_obs as O
    from scripts.train_jokers import logits_np
    counts = {}

    def policy(hand, seat):
        mask = hand.legal_casts(seat)
        if not any(mask[1:]):
            return 0
        t = {}
        a = X.bot_action(hand, seat, logits_np(w, np.asarray(O.build(hand, seat))), trace=t)
        if t.get("source") == "reseau":
            for jid, d in t["verdicts"].items():
                if d["verdict"] in X.OPEN:
                    c = counts.setdefault(d["regle"], [0, 0])
                    c[0] += 1
                    c[1] += int(a and J.CASTABLE[a - 1] == jid)
        return a
    for seed in seeds:
        rng = random.Random(seed)
        match = jm.Match(jm.random_config(rng), rng.randrange(1 << 30))
        for _ in range(jm.HANDS_PER_MATCH):
            hand = match.next_hand()
            jm.play_out(hand, policy)
            match.finish(hand)
    return counts


def duel(pool, w, cand, best, n, seed0, workers):
    from scripts.train_jokers import policies_worker
    seeds = list(range(seed0, seed0 + n))
    rows = [r for part in pool.map(policies_worker, [(seeds[i::workers], [(w, cand), (w, best)], False)
                                                     for i in range(workers)]) for r in part]
    diffs = [res[0][0] - res[1][0] for _s, _c, res in rows]
    kinds = {}
    for (_s, cfg, res), d in zip(rows, diffs):
        kinds.setdefault(cfg.get("boss") or "elite_acte%d" % cfg["act"], []).append(d)
    m, se = _stat(diffs)
    return m, se, {k: [round(x, 2) for x in _stat(v)] for k, v in sorted(kinds.items()) if len(v) > 1}


def main():
    import torch
    from agents import joker_expert as X
    from agents.network import PolicyValueNet
    from engine import joker_obs as O
    from scripts.train_jokers import CHECKPOINT, net_to_numpy, widen_obs
    p = argparse.ArgumentParser()
    p.add_argument("--until", required=True, help="heure UTC AAAA-MM-JJTHH:MM")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--discover", type=int, default=300)
    p.add_argument("--screen", type=int, default=400)
    p.add_argument("--confirm", type=int, default=800)
    p.add_argument("--screen-z", type=float, default=1.0)
    p.add_argument("--min-decisions", type=int, default=100)
    a = p.parse_args()
    deadline = datetime.datetime.strptime(a.until, "%Y-%m-%dT%H:%M")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    net = PolicyValueNet(O.OBS_DIM, J.ACTION_DIM)
    net.load_state_dict(widen_obs(ckpt["net"]))
    net.eval()
    w = net_to_numpy(net)
    print("champion iteration", ckpt.get("iter"), "empreinte", J.JOKER_SPEC_HASH, flush=True)
    counter = [0]

    def fresh():
        counter[0] += 1
        return SEARCH_SEED + counter[0] * SEED_STRIDE

    def now():
        return datetime.datetime.utcnow()

    with Pool(a.workers) as pool:
        t0 = time.time()
        seeds = list(range(SEARCH_SEED, SEARCH_SEED + a.discover))
        counts = {}
        for part in pool.map(discover_worker, [(seeds[i::a.workers], w) for i in range(a.workers)]):
            for k, (n, c) in part.items():
                counts.setdefault(k, [0, 0])
                counts[k][0] += n
                counts[k][1] += c
        cases = sorted((k for k, v in counts.items() if v[0] >= a.min_decisions), key=lambda k: -counts[k][0])
        print(json.dumps({"decouverte": {k: {"decisions": v[0], "poses": v[1]} for k, v in
                                         sorted(counts.items(), key=lambda kv: -kv[1][0])},
                          "matchs": a.discover, "s": round(time.time() - t0)}, ensure_ascii=False), flush=True)
        best = {}
        tried = set()
        firsts = [{"kermesse:figures_en_main": X.YES, "kermesse:peu_de_figures": X.YES}]
        firsts += [{k: v} for k in cases for v in (X.YES, X.NO)]
        while now() < deadline:
            accepted = False
            for change in firsts:
                if now() >= deadline:
                    break
                key = json.dumps({**best, **change}, sort_keys=True)
                if key in tried or all(best.get(k) == v for k, v in change.items()):
                    continue
                tried.add(key)
                cand = {**best, **change}
                t = time.time()
                m, se, kinds = duel(pool, w, cand, best, a.screen, fresh(), a.workers)
                rec = {"etape": "criblage", "change": change, "m": round(m, 2), "se": round(se, 2),
                       "par_type": kinds, "s": round(time.time() - t)}
                ok = m > 0 and m > a.screen_z * se
                if ok and now() < deadline:
                    t = time.time()
                    m2, se2, kinds2 = duel(pool, w, cand, best, a.confirm, fresh(), a.workers)
                    rec.update(confirmation={"m": round(m2, 2), "se": round(se2, 2), "par_type": kinds2,
                                             "s": round(time.time() - t)})
                    ok = m2 > 0 and m2 > 2 * se2
                    if ok:
                        best = cand
                        accepted = True
                rec["garde"] = bool(ok and "confirmation" in rec)
                rec["meilleur"] = best
                print(json.dumps(rec, ensure_ascii=False), flush=True)
            if not accepted:
                print("passe complete sans changement garde", flush=True)
                break
        print("=== RESULTAT : cas forces gardes %s (%d mesures) ===" % (json.dumps(best, ensure_ascii=False),
                                                                       counter[0]), flush=True)


if __name__ == "__main__":
    main()
