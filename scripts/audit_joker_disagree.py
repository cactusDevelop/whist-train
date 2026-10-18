"""Desaccords reseau seul / hybride (agents/joker_expert) : a chaque decision
ou ils different, Q(hybride) - Q(reseau) par rollouts, attribue au verdict
de regle du joker que le reseau voulait et a celui que l'hybride a pris.

    venv/Scripts/python.exe scripts/audit_joker_disagree.py [--matches 200]
"""
import argparse
import math
import multiprocessing as mp
import random
import sys

sys.path.insert(0, ".")


def worker(args):
    seeds, weights = args
    import numpy as np
    from engine import joker_match as jm, jokers as J, joker_obs as O
    from agents import joker_expert as X
    from scripts.train_jokers import make_policy, logits_np
    netp = make_policy(weights, "net")
    out = {}

    def deciding(hand, seat):
        mask = hand.legal_casts(seat)
        if not any(mask[1:]):
            return 0
        lg = logits_np(weights, np.asarray(O.build(hand, seat)))
        a_net = netp(hand, seat)
        tr = {}
        a_hyb = X.bot_action(hand, seat, lg, tr)
        if a_net != a_hyb:
            qn = jm.hand_reward(jm.play_out(jm.clone(hand), netp, first_action=a_net), seat)
            qh = jm.hand_reward(jm.play_out(jm.clone(hand), netp, first_action=a_hyb), seat)
            v = X.view(hand, seat)
            key = "reseau=%s [%s] / hybride=%s [%s]" % (
                "aucun" if a_net == 0 else "|".join(X.rule(v, J.CASTABLE[a_net - 1])),
                "", "aucun" if a_hyb == 0 else J.CASTABLE[a_hyb - 1], tr["source"])
            c = out.setdefault(key, [0, 0.0, 0.0])
            c[0] += 1; c[1] += qh - qn; c[2] += (qh - qn) ** 2
        return a_net

    for seed in seeds:
        rng = random.Random(seed)
        match = jm.Match(jm.random_config(rng), rng.randrange(1 << 30))
        for _ in range(jm.HANDS_PER_MATCH):
            hand = match.next_hand()
            jm.play_out(hand, deciding)
            match.finish(hand)
    return out


def main():
    import torch
    from agents.network import PolicyValueNet
    from engine import joker_obs as O, jokers as J
    from scripts import train_jokers as T
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=T.CHECKPOINT)
    ap.add_argument("--matches", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(1)
    net = PolicyValueNet(O.OBS_DIM, J.ACTION_DIM)
    net.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["net"])
    w = T.net_to_numpy(net)
    seeds = [56000000 + i for i in range(a.matches)]
    tot = {}
    with mp.Pool(a.workers) as pool:
        for part in pool.map(worker, [(seeds[i::a.workers], w) for i in range(a.workers)]):
            for k, c in part.items():
                t = tot.setdefault(k, [0, 0.0, 0.0])
                for i in range(3):
                    t[i] += c[i]
    for k, (n, s, q) in sorted(tot.items(), key=lambda kv: kv[1][1]):
        m = s / n
        se = math.sqrt(max(0.0, q / n - m * m) / max(1, n - 1))
        print("%5d  total %+8.1f  moy %+6.2f +- %.2f  %s" % (n, s, m, se, k))


if __name__ == "__main__":
    main()
