"""Recherche des regles JOUEUR des jokers (agents/joker_expert, siege 0)
DIRECTEMENT sur le score de match du joueur, en duplicate (mesure seule).

Pourquoi (09/11, conseil) : bien_vu / precision de joker_moments sont bruites
(etiquette « utile » = max de branches bruitees, continuation par la regle
courante -> base mouvante) ; le lot 1 des regles revues a la main ne se juge
pas avec eux. Ici, comme scripts/search_joker_rules.py pour les bots : on
inverse UN cas de regle du joueur (PLAYER_FORCE "joker:cas" -> oui/non) et on
garde l'inversion si le score du joueur monte sur des matchs neufs.

  1. decouverte : cas de regle rencontres par le joueur (vus, lances) ;
  2. descente par coordonnees, cas les plus frequents d'abord : criblage
     SCREEN matchs (m > 0 et m > SCREEN_Z se), confirmation CONFIRM matchs
     NEUFS (m > 2 se) -> garde (nouvelle base), sinon rejete.
Bots : regles + reseau champion (comme en jeu). Le resultat est une table de
cas oui/non, lisible et portable telle quelle en GDScript.

Usage cloud : python -u scripts/search_player_rules.py --until 2026-11-10T05:30 --workers 4
"""

import argparse
import datetime
import json
import math
import random
import subprocess
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, ".")

from engine import joker_match as jm
from engine import jokers as J

SEARCH_SEED = 62000000  # hors GATE (30M..50M), bots (60M), PROFIT (77M), OOS (91M), paires (93M)
SEED_STRIDE = 10007
OUT_JSON = "player_rules_search.json"


def _stat(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, float("inf")
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v / len(xs))


_W = None


def _init(w):
    global _W
    _W = w


def _match(seed, force, trace=None):
    """Score du joueur sur un match (graine), avec PLAYER_FORCE = force."""
    from agents import joker_expert as X
    from scripts.train_jokers import hybrid_policy
    policy = hybrid_policy(_W)
    rng = random.Random(seed)
    cfg, mseed = jm.random_config(rng), rng.randrange(1 << 30)
    match = jm.Match(cfg, mseed)
    X.PLAYER_FORCE, X.PLAYER_TRACE = force, trace
    p0 = 0.0
    try:
        for _ in range(jm.HANDS_PER_MATCH):
            hand = match.next_hand()
            pts = jm.play_out(hand, policy)
            p0 += pts[jm.PLAYER]
            match.finish(hand)
    finally:
        X.PLAYER_FORCE, X.PLAYER_TRACE = {}, None
    return p0


def discover_worker(seeds):
    trace = {}
    for s in seeds:
        _match(s, {}, trace)
    return trace


def duel_worker(args):
    """Meme graine (meme config, memes donnes de depart) pour les deux tables.
    Les donnes suivantes peuvent diverger (le jeu change l'etat du match) :
    la graine commune garde l'essentiel de la reduction de variance."""
    seeds, cand, base = args
    out = []
    for s in seeds:
        try:
            out.append(_match(s, cand) - _match(s, base))
        except Exception as e:  # inversion impossible a jouer -> rejet
            return {"error": "%s: %s" % (type(e).__name__, e)}
    return out


def duel(pool, cand, base, n, seed0, workers):
    seeds = list(range(seed0, seed0 + n))
    parts = pool.map(duel_worker, [(seeds[i::workers], cand, base) for i in range(workers)])
    errs = [p["error"] for p in parts if isinstance(p, dict)]
    if errs:
        return None, None, errs[0]
    return _stat([d for p in parts for d in p]) + (None,)


def flip(verdict):
    from agents import joker_expert as X
    return X.NO if verdict in (X.YES, X.OPEN_YES) else X.YES


def upload(state):
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    subprocess.run(["gh", "release", "upload", "checkpoint-latest", OUT_JSON, "--clobber"], check=False)


def main():
    import torch
    from agents import joker_expert as X
    from agents.network import PolicyValueNet
    from engine import joker_obs as O
    from scripts.train_jokers import CHECKPOINT, net_to_numpy, widen_obs
    p = argparse.ArgumentParser()
    p.add_argument("--until", required=True, help="heure UTC AAAA-MM-JJTHH:MM")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--discover", type=int, default=400)
    p.add_argument("--screen", type=int, default=400)
    p.add_argument("--confirm", type=int, default=800)
    p.add_argument("--screen-z", type=float, default=1.0)
    p.add_argument("--min-seen", type=int, default=60, help="vus par le joueur pendant la decouverte")
    p.add_argument("--no-upload", action="store_true")
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
        return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

    state = {"spec_hash": J.JOKER_SPEC_HASH, "champion_iter": ckpt.get("iter"), "kept": {}, "tried": []}
    with Pool(a.workers, initializer=_init, initargs=(w,)) as pool:
        t0 = time.time()
        seeds = list(range(SEARCH_SEED, SEARCH_SEED + a.discover))
        counts = {}
        for part in pool.map(discover_worker, [seeds[i::a.workers] for i in range(a.workers)]):
            for k, (n, c) in part.items():
                counts.setdefault(k, [0, 0])
                counts[k][0] += n
                counts[k][1] += c
        print("decouverte : %d matchs, %d cas, %.0f s" % (a.discover, len(counts), time.time() - t0), flush=True)
        for k, (n, c) in sorted(counts.items(), key=lambda kv: -kv[1][0])[:60]:
            print("  %-45s vus %6d  lances %6d" % (k, n, c), flush=True)
        state["discovery"] = counts
        cases = [k for k, (n, _c) in sorted(counts.items(), key=lambda kv: -kv[1][0]) if n >= a.min_seen]
        print("%d cas >= %d vus" % (len(cases), a.min_seen), flush=True)
        base = {}
        for rid in cases:
            if now() >= deadline:
                print("arret : heure limite", flush=True)
                break
            n, c = counts[rid]
            # verdict actuel du cas : lance (oui) si la regle l'a pose au moins une fois
            current = X.YES if c > 0 else X.NO
            cand = dict(base)
            cand[rid] = flip(current)
            t1 = time.time()
            m, se, err = duel(pool, cand, base, a.screen, fresh(), a.workers)
            row = {"case": rid, "seen": n, "cast": c, "to": cand[rid]}
            if err:
                row["error"] = err
                print("%-45s -> %s : ERREUR %s" % (rid, cand[rid], err), flush=True)
            else:
                row["screen"] = [round(m, 3), round(se, 3), a.screen]
                print("%-45s -> %-9s criblage %+.2f +- %.2f (%.0f s)" % (rid, cand[rid], m, se, time.time() - t1),
                      flush=True)
                if m > 0 and m > a.screen_z * se and now() < deadline:
                    m2, se2, err2 = duel(pool, cand, base, a.confirm, fresh(), a.workers)
                    if err2 is None:
                        row["confirm"] = [round(m2, 3), round(se2, 3), a.confirm]
                        ok = m2 > 0 and m2 > 2 * se2
                        print("    confirmation %+.2f +- %.2f sur %d matchs neufs -> %s" % (
                            m2, se2, a.confirm, "GARDE" if ok else "rejete"), flush=True)
                        if ok:
                            base = cand
                            state["kept"] = base
            state["tried"].append(row)
            if not a.no_upload:
                upload(state)
    print("FIN : %d inversions gardees : %s" % (len(base), json.dumps(base, ensure_ascii=False)), flush=True)
    if not a.no_upload:
        upload(state)


if __name__ == "__main__":
    main()
