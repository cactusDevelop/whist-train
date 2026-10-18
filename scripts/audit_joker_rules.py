"""Audit des regles de agents/joker_expert.py (bots) par rollouts : pour chaque
cas de regle (joker:cas, verdict), avantage moyen Q(joker) - Q(aucun cast)
et ecart a la meilleure action, la suite etant jouee par la politique hybride.

    venv/Scripts/python.exe scripts/audit_joker_rules.py [--matches 300] [ckpt]
"""
import argparse
import math
import multiprocessing as mp
import random
import sys

sys.path.insert(0, ".")


def worker(args):
    seeds, weights = args
    from engine import joker_match as jm, jokers as J
    from agents import joker_expert as X
    from scripts.train_jokers import make_policy
    future = make_policy(weights)
    out = {}

    def deciding(hand, seat):
        mask = hand.legal_casts(seat)
        if not any(mask[1:]):
            return 0
        q = [None] * J.ACTION_DIM
        for a in range(J.ACTION_DIM):
            if mask[a]:
                q[a] = jm.hand_reward(jm.play_out(jm.clone(hand), future, first_action=a), seat)
        v = X.view(hand, seat)
        best = max(x for x in q if x is not None)
        for a in range(1, J.ACTION_DIM):
            if mask[a]:
                verdict, rid = X.rule(v, J.CASTABLE[a - 1])
                c = out.setdefault(rid + "|" + verdict, [0, 0.0, 0.0, 0.0])
                adv = q[a] - q[0]
                c[0] += 1; c[1] += adv; c[2] += adv * adv; c[3] += q[a] - best
        return future(hand, seat)

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
    ap.add_argument("--matches", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=55000000)
    a = ap.parse_args()
    torch.set_num_threads(1)
    net = PolicyValueNet(O.OBS_DIM, J.ACTION_DIM)
    net.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["net"])
    w = T.net_to_numpy(net)
    seeds = [a.seed + i for i in range(a.matches)]
    tot = {}
    with mp.Pool(a.workers) as pool:
        for part in pool.map(worker, [(seeds[i::a.workers], w) for i in range(a.workers)]):
            for k, c in part.items():
                t = tot.setdefault(k, [0, 0.0, 0.0, 0.0])
                for i in range(4):
                    t[i] += c[i]
    for k, (n, s, q, r) in sorted(tot.items()):
        m = s / n
        se = math.sqrt(max(0.0, q / n - m * m) / max(1, n - 1))
        flag = "  <-- A REVOIR" if ((k.endswith("|oui") and m + 2 * se < 0) or (k.endswith("|non") and m - 2 * se > 0)) else ""
        print("%-55s n=%5d  adv %+6.2f +- %.2f  regret %+6.2f%s" % (k, n, m, se, r / n, flag))


if __name__ == "__main__":
    main()
