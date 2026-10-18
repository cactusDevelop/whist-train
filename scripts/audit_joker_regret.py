"""Mesure (2026-10-18) : ou l'IA jokers perd-elle des points, et une obs plus
riche aiderait-elle ? A chaque decision de cast d'un bot (politique hybride du
checkpoint, sans epsilon), chaque action legale est rejouee jusqu'a la fin de
la donne (comme scripts/train_jokers.gen_worker) -> Q(a).

  regret(a) = max Q - Q(a)   (oracle a information parfaite : borne haute)

Puis hors ligne, decoupage par match 80/20 : un reseau de meme forme reentraine
sur l'obs actuelle (A) et sur obs + features candidates (B, suivi des cartes,
etat du joueur). B nettement sous A en regret test -> l'obs limite ; A sous le
reseau du checkpoint -> c'est l'entrainement qui limite.

    venv/Scripts/python.exe scripts/audit_joker_regret.py [--hands 12000] [ckpt]
"""
import argparse
import collections
import math
import multiprocessing as mp
import os
import pickle
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import joker_match as jm
from engine import jokers as J
from engine import joker_obs as O
from engine.cards import NUM_PLAYERS, NUM_SUITS, rank_of, suit_of
from engine.observation import void_features
from scripts import train_jokers as T

EXTRA_DIM = 4 + 4 + 2 + 1 + 16 + 2


def extra_features(hand, me):
    """Candidates : non vues par couleur, cartes non vues au-dessus de ma
    meilleure par couleur, je tiens la maitresse (demandee / atout), atouts non
    vus, chicanes (observation.void_features), energie + jokers du joueur."""
    g, state = hand.game, hand.state
    my = g.hands[me]
    seen = set(my) | {c for _s, c in g.play_log}
    unseen = [c for c in range(52) if c not in seen]
    out = []
    for s in range(NUM_SUITS):
        out.append(sum(1 for c in unseen if suit_of(c) == s) / 13.0)
    best = {}
    for c in my:
        best[suit_of(c)] = max(best.get(suit_of(c), -1), rank_of(c))
    for s in range(NUM_SUITS):
        top = best.get(s, -1)
        out.append(sum(1 for c in unseen if suit_of(c) == s and rank_of(c) > top) / 13.0 if top >= 0 else 0.0)

    def master(s):
        if s is None or s not in best:
            return 0.0
        return float(not any(suit_of(c) == s and rank_of(c) > best[s] for c in unseen))
    trick = list(g.current_trick)
    out.append(master(suit_of(trick[0][1]) if trick else None))
    out.append(master(g.trump_suit))
    out.append(sum(1 for c in unseen if suit_of(c) == g.trump_suit) / 13.0 if g.trump_suit is not None else 0.0)
    out.extend(void_features(me, list(g.play_log)))
    out.append(min(1.0, state.energy[jm.PLAYER] / 6.0))
    out.append(min(1.0, len(state.joker_hand[jm.PLAYER]) / 5.0))
    assert len(out) == EXTRA_DIM
    return out


def worker(args):
    seed, n_hands, weights = args
    rng = random.Random(seed)
    future = T.make_policy(weights)
    netp = T.net_policy(weights)
    rows = []
    match = None
    mid = [0]

    def deciding(hand, seat):
        mask = hand.legal_casts(seat)
        if not any(mask[1:]):
            return 0
        q = [None] * J.ACTION_DIM
        for a in range(J.ACTION_DIM):
            if mask[a]:
                q[a] = jm.hand_reward(jm.play_out(jm.clone(hand), future, first_action=a), seat)
        chosen = future(hand, seat)
        c = hand.contract
        rows.append({"match": mid[0], "obs": O.build(hand, seat), "extra": extra_features(hand, seat),
                     "mask": list(mask), "q": q, "hybrid": chosen, "net": netp(hand, seat),
                     "kind": match.config.get("boss") or "elite_acte%d" % match.config["act"],
                     "contract": c["type"], "declares": O.build(hand, seat)[O.OFFSETS["my_side_declares"]] > 0,
                     "trick": hand.game.trick_count, "pos": len(hand.game.current_trick)})
        return chosen

    done = 0
    while done < n_hands:
        mid[0] = seed * 100000 + done
        match = jm.Match(jm.random_config(rng), rng.randrange(1 << 30))
        for _ in range(jm.HANDS_PER_MATCH):
            hand = match.next_hand()
            jm.play_out(hand, deciding)
            match.finish(hand)
            done += 1
    return rows


# --- analyse (tableaux numpy : une copie, tient en memoire) -------------------

GROUP_KEYS = ("kind", "contract", "declares", "pos")


def to_arrays(rows):
    n = len(rows)
    a = {"obs": np.zeros((n, O.OBS_DIM), np.float32), "extra": np.zeros((n, EXTRA_DIM), np.float32),
         "mask": np.zeros((n, J.ACTION_DIM), bool), "q": np.full((n, J.ACTION_DIM), np.nan, np.float32),
         "hybrid": np.zeros(n, np.int16), "net": np.zeros(n, np.int16), "match": np.zeros(n, np.int64)}
    for k in GROUP_KEYS:
        a[k] = np.empty(n, object)
    for i, r in enumerate(rows):
        a["obs"][i] = r["obs"]
        a["extra"][i] = r["extra"]
        a["mask"][i] = r["mask"]
        a["q"][i] = [np.nan if v is None else v for v in r["q"]]
        a["hybrid"][i], a["net"][i], a["match"][i] = r["hybrid"], r["net"], r["match"]
        for k in GROUP_KEYS:
            a[k][i] = r[k]
    return a


def regrets(a, picks, idx=None):
    q = a["q"] if idx is None else a["q"][idx]
    best = np.nanmax(q, axis=1)
    return best - q[np.arange(len(q)), picks]


def mean_se(xs):
    return float(xs.mean()), float(xs.std(ddof=1) / math.sqrt(len(xs)))


def fit_eval(a, tr, te, use_extra, epochs, seed, batch=256):
    import torch
    from agents.network import PolicyValueNet
    torch.manual_seed(seed)
    x = a["obs"] if not use_extra else np.concatenate([a["obs"], a["extra"]], axis=1)
    adv = np.nan_to_num(a["q"] - a["q"][:, :1], nan=0.0)
    xt, mt = torch.from_numpy(x[tr]), torch.from_numpy(a["mask"][tr]).float()
    at = (torch.from_numpy(adv[tr]) / T.ADV_SCALE).clamp(-T.ADV_CLIP, T.ADV_CLIP)
    net = PolicyValueNet(x.shape[1], J.ACTION_DIM)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):  # meme perte que train_jokers.fit
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), batch):
            j = perm[i:i + batch]
            lg, _v = net(xt[j])
            loss = (((lg - at[j]) ** 2) * mt[j]).sum() / mt[j].sum() + (lg[:, 0] ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        lg = net(torch.from_numpy(x[te]))[0].numpy()
    lg[~a["mask"][te]] = -np.inf
    return lg.argmax(axis=1)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=os.path.join(T.CHECKPOINT_DIR, "archive_3e5d_iter29_exported.pt"))
    ap.add_argument("--hands", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--cache", default="joker_regret_rows.pkl")
    args = ap.parse_args()
    torch.set_num_threads(2)
    npz = os.path.splitext(args.cache)[0] + ".npz"
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        a = {k: z[k] for k in z.files}
    else:
        if os.path.exists(args.cache):
            with open(args.cache, "rb") as f:
                rows = pickle.load(f)
        else:
            from agents.network import PolicyValueNet
            ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
            net = PolicyValueNet(O.OBS_DIM, J.ACTION_DIM)
            net.load_state_dict(ck["net"])
            weights = T.net_to_numpy(net)
            per = args.hands // args.workers
            with mp.Pool(args.workers) as pool:
                parts = pool.map(worker, [(77000 + w, per, weights) for w in range(args.workers)])
            rows = [r for p in parts for r in p]
            del parts
        a = to_arrays(rows)
        del rows
        np.savez(npz, **a)
    n = len(a["q"])
    print("%d decisions" % n)

    print("\n== regret par decision (points de recompense, oracle info parfaite) ==")
    reg_h = regrets(a, a["hybrid"])
    for label, picks in (("hybride (exporte)", a["hybrid"]), ("reseau seul", a["net"]), ("jamais", np.zeros(n, int))):
        print("  %-18s %.3f +- %.3f" % ((label,) + mean_se(regrets(a, picks))))
    print("  gain oracle vs jamais : %.3f / decision" % float(np.nanmax(a["q"], axis=1).mean() - a["q"][:, 0].mean()))

    print("\n== regret hybride par groupe (part du regret total) ==")
    total = reg_h.sum()
    for key in GROUP_KEYS:
        print("  par %s :" % key)
        g = collections.defaultdict(list)
        for i, k in enumerate(a[key]):
            g[k].append(i)
        for k, ix in sorted(g.items(), key=lambda kv: -reg_h[kv[1]].sum()):
            xs = reg_h[ix]
            print("    %-14s n=%6d  regret moyen %.3f  part %4.1f%%" % (k, len(xs), xs.mean(), 100 * xs.sum() / total))
    best = np.where(a["mask"], np.nan_to_num(a["q"], nan=-np.inf), -np.inf).argmax(axis=1)
    g = collections.defaultdict(float)
    cnt = collections.Counter()
    name = lambda x: J.CASTABLE[x - 1] if x else "aucun"
    for i in np.nonzero(reg_h > 0)[0]:
        k = (name(a["hybrid"][i]), name(best[i]))
        g[k] += reg_h[i]
        cnt[k] += 1
    print("  pires confusions (choisi -> meilleur) :")
    for k, s in sorted(g.items(), key=lambda kv: -kv[1])[:12]:
        print("    %-34s n=%6d  part %4.1f%%" % ("%s -> %s" % k, cnt[k], 100 * s / total))

    print("\n== test hors ligne obs (reseau seul, meme forme, decoupage par match 80/20) ==")
    ms = np.unique(a["match"])
    np.random.RandomState(5).shuffle(ms)
    te_mask = np.isin(a["match"], ms[:len(ms) // 5])
    tr, te = np.nonzero(~te_mask)[0], np.nonzero(te_mask)[0]
    print("  checkpoint (reseau seul)  %.3f +- %.3f   (n test = %d)" % (mean_se(regrets(a, a["net"][te], te)) + (len(te),)))
    print("  checkpoint (hybride)      %.3f +- %.3f" % mean_se(regrets(a, a["hybrid"][te], te)))
    res = {}
    for use_extra in (False, True):
        for s in (1, 2):
            p = fit_eval(a, tr, te, use_extra, args.epochs, s)
            res[(use_extra, s)] = regrets(a, p, te)
            print("  %s graine %d : %.3f" % ("B obs+extra" if use_extra else "A obs      ", s, res[(use_extra, s)].mean()), flush=True)
    d = (res[(False, 1)] + res[(False, 2)]) / 2 - (res[(True, 1)] + res[(True, 2)]) / 2
    print("  A - B apparie (moyenne des 2 graines) : %+.3f +- %.3f par decision" % mean_se(d))


if __name__ == "__main__":
    main()
