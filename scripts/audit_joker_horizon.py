"""Mesure (2026-10-18) : la recompense par donne (engine/joker_match.hand_reward)
rate-t-elle des couts/gains qui se paient aux donnes SUIVANTES du match
(pioche/defausse des jokers, bonus d'energie, Coup du Chapeau...) ?

A chaque decision de cast d'un bot (politique hybride du checkpoint), chaque
action legale est rejouee :
  qh(a) = recompense de la donne en cours (comme scripts/train_jokers.py) ;
  qm(a) = qh(a) + recompenses des donnes restantes du match, jouees par la
          meme politique sur des donnes FIXES communes a toutes les actions
          (Match.deals : seul l'etat des jokers differe entre actions).

Puis : terme inter-donnes par joker, actions dont l'argmax change, et hors
ligne (decoupage par match 80/20) reseau entraine sur qh vs sur qm, juge au
regret qm sur le test.

    venv/Scripts/python.exe scripts/audit_joker_horizon.py [--hands 4000] [ckpt]
"""
import argparse
import collections
import math
import multiprocessing as mp
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import joker_match as jm
from engine import jokers as J
from engine import joker_obs as O
from scripts import train_jokers as T

def worker(args):
    seed, n_hands, weights = args
    rng = random.Random(seed)
    future = T.make_policy(weights)
    netp = T.net_policy(weights)
    rows = []
    ctx = {}

    def deciding(hand, seat):
        mask = hand.legal_casts(seat)
        if not any(mask[1:]):
            return 0
        qh = [np.nan] * J.ACTION_DIM
        qm = [np.nan] * J.ACTION_DIM
        for a in range(J.ACTION_DIM):
            if mask[a]:
                qh[a], qm[a] = jm.match_rollout(ctx["match"], hand, seat, a, future, ctx["left"], ctx["deals"])
        chosen = future(hand, seat)
        rows.append((ctx["mid"], O.build(hand, seat), list(mask), qh, qm, chosen, netp(hand, seat),
                     ctx["match"].config.get("boss") or "elite_acte%d" % ctx["match"].config["act"],
                     hand.contract["type"], ctx["left"]))
        return chosen

    done = 0
    while done < n_hands:
        ctx["mid"] = seed * 100000 + done
        ctx["match"] = jm.Match(jm.random_config(rng), rng.randrange(1 << 30))
        for i in range(jm.HANDS_PER_MATCH):
            ctx["left"] = jm.HANDS_PER_MATCH - 1 - i
            hand = ctx["match"].next_hand()
            ctx["deals"] = jm.spare_deals(ctx["match"], rng.randrange(1 << 30))
            jm.play_out(hand, deciding)
            ctx["match"].finish(hand)
            done += 1
    return rows


def to_arrays(rows):
    cols = list(zip(*rows))
    return {"match": np.array(cols[0], np.int64), "obs": np.array(cols[1], np.float32),
            "mask": np.array(cols[2], bool), "qh": np.array(cols[3], np.float32), "qm": np.array(cols[4], np.float32),
            "hybrid": np.array(cols[5], np.int16), "net": np.array(cols[6], np.int16),
            "kind": np.array(cols[7], object), "contract": np.array(cols[8], object), "left": np.array(cols[9], np.int16)}


def regrets(q, picks):
    return np.nanmax(q, axis=1) - q[np.arange(len(q)), picks]


def mean_se(xs):
    return float(np.mean(xs)), float(np.std(xs, ddof=1) / math.sqrt(len(xs)))


def fit_picks(a, tr, te, target, epochs, seed, batch=256):
    import torch
    from agents.network import PolicyValueNet
    torch.manual_seed(seed)
    q = a[target]
    adv = np.nan_to_num(q - q[:, :1], nan=0.0)
    xt, mt = torch.from_numpy(a["obs"][tr]), torch.from_numpy(a["mask"][tr]).float()
    at = (torch.from_numpy(adv[tr]) / T.ADV_SCALE).clamp(-T.ADV_CLIP, T.ADV_CLIP)
    net = PolicyValueNet(O.OBS_DIM, J.ACTION_DIM)
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
        lg = net(torch.from_numpy(a["obs"][te]))[0].numpy()
    lg[~a["mask"][te]] = -np.inf
    return lg.argmax(axis=1)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=os.path.join(T.CHECKPOINT_DIR, "archive_3e5d_iter29_exported.pt"))
    ap.add_argument("--hands", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--cache", default="joker_horizon_rows.npz")
    args = ap.parse_args()
    torch.set_num_threads(2)
    if os.path.exists(args.cache):
        z = np.load(args.cache, allow_pickle=True)
        a = {k: z[k] for k in z.files}
    else:
        from agents.network import PolicyValueNet
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        net = PolicyValueNet(O.OBS_DIM, J.ACTION_DIM)
        net.load_state_dict(ck["net"])
        weights = T.net_to_numpy(net)
        per = args.hands // args.workers
        with mp.Pool(args.workers) as pool:
            parts = pool.map(worker, [(88000 + w, per, weights) for w in range(args.workers)])
        a = to_arrays([r for p in parts for r in p])
        del parts
        np.savez(args.cache, **a)
    n = len(a["qh"])
    later = a["left"] > 0
    print("%d decisions, dont %d avec des donnes restantes" % (n, later.sum()))
    name = lambda x: J.CASTABLE[x - 1] if x else "aucun"

    print("\n== terme inter-donnes par joker : (qm(j)-qm(0)) - (qh(j)-qh(0)), decisions avec donnes restantes ==")
    for j in range(1, J.ACTION_DIM):
        ix = later & a["mask"][:, j]
        if ix.sum() < 30:
            continue
        x = (a["qm"][ix, j] - a["qm"][ix, 0]) - (a["qh"][ix, j] - a["qh"][ix, 0])
        print("  %-16s n=%6d  %+.3f +- %.3f   (avantage donne %+.3f)" % ((name(j), ix.sum()) + mean_se(x)
                                                                       + (float(np.mean(a["qh"][ix, j] - a["qh"][ix, 0])),)))

    print("\n== regret au critere match (qm), decisions avec donnes restantes ==")
    qm = a["qm"][later]
    for label, picks in (("hybride", a["hybrid"][later]), ("reseau seul", a["net"][later]),
                         ("jamais", np.zeros(later.sum(), int))):
        print("  %-12s %.3f +- %.3f" % ((label,) + mean_se(regrets(qm, picks))))
    cv = J.CASTABLE.index("coup_de_vent") + 1
    ix = later & (a["hybrid"] == 0) & (a["net"] == cv)
    if ix.sum():
        print("  veto Coup de Vent (hybride aucun, reseau coup_de_vent) n=%d : qh %+.3f +- %.3f, qm %+.3f +- %.3f" % (
            (ix.sum(),) + mean_se(a["qh"][ix, cv] - a["qh"][ix, 0]) + mean_se(a["qm"][ix, cv] - a["qm"][ix, 0])))
    bh = np.where(a["mask"], np.nan_to_num(a["qh"], nan=-np.inf), -np.inf).argmax(axis=1)
    bm = np.where(a["mask"], np.nan_to_num(a["qm"], nan=-np.inf), -np.inf).argmax(axis=1)
    print("  meilleure action differente qh / qm : %.1f %% des decisions avec donnes restantes" % (100 * (bh != bm)[later].mean()))

    print("\n== hors ligne : reseau entraine sur qh vs sur qm (decoupage par match 80/20), juge au regret qm ==")
    ms = np.unique(a["match"])
    np.random.RandomState(5).shuffle(ms)
    te_mask = np.isin(a["match"], ms[:len(ms) // 5])
    tr, te = np.nonzero(~te_mask)[0], np.nonzero(te_mask)[0]
    print("  checkpoint hybride  qm %.3f   reseau seul qm %.3f   (n test = %d)" % (
        regrets(a["qm"][te], a["hybrid"][te]).mean(), regrets(a["qm"][te], a["net"][te]).mean(), len(te)))
    res = {}
    for target in ("qh", "qm"):
        for s in (1, 2):
            p = fit_picks(a, tr, te, target, args.epochs, s)
            res[(target, s)] = regrets(a["qm"][te], p)
            print("  entraine sur %s graine %d : regret qm %.3f, regret qh %.3f" % (
                target, s, res[(target, s)].mean(), regrets(a["qh"][te], p).mean()), flush=True)
    d = (res[("qh", 1)] + res[("qh", 2)]) / 2 - (res[("qm", 1)] + res[("qm", 2)]) / 2
    print("  qh - qm apparie (regret qm, moyenne 2 graines) : %+.3f +- %.3f par decision" % mean_se(d))


if __name__ == "__main__":
    main()
