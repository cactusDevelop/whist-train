"""Diagnostic (mesure seule) : main du joueur simule vide alors que la donne
continue (run 8f41 40783453624, ValueError min() vide dans expert_bot._discard).

Journalise chaque joker lance (JokerRules.play) avec l'etat de la donne, et
vide le journal a la premiere main vide. Bots : moitie champion (hybride, comme
la generation), moitie casts aleatoires (plus de combinaisons).
"""
import argparse
import json
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, ".")

import torch

from engine import jokers as J
from engine import joker_match as jm
from engine.cards import NUM_PLAYERS

PLAYER = jm.PLAYER


class EmptyHand(Exception):
    pass


_play = J.JokerRules.play


def _logged_play(self, seat, jid, target=-1):
    g = self.game
    before = len(g.hands[PLAYER]) if g is not None else None
    res = _play(self, seat, jid, target)
    if g is not None:
        log = self.__dict__.setdefault("_diag", [])
        log.append({
            "seat": seat, "jid": jid, "target": target,
            "res": {k: (v if isinstance(v, (int, str, bool)) or v is None else str(v)) for k, v in res.items()},
            "trick": g.trick_count, "in_trick": len(g.current_trick),
            "player_hand": [before, len(g.hands[PLAYER])], "hand_tricks": self.hand_tricks(),
            "live": [e.id for e in self.live()],
        })
    return res


J.JokerRules.play = _logged_play

_card = jm._player_card


def _checked_card(hand, seat):
    g = hand.game
    if not g.hands[seat]:
        raise EmptyHand(json.dumps({
            "contract": hand.contract, "trick": g.trick_count, "num_tricks": getattr(g, "num_tricks", 13),
            "hand_tricks": hand.rules.hand_tricks(), "hands": [len(h) for h in g.hands],
            "live": [e.id for e in hand.rules.live()], "casts": hand.rules.__dict__.get("_diag", []),
        }, default=str))
    return _card(hand, seat)


jm._player_card = _checked_card


def random_policy(rng):
    def pol(hand, seat):
        mask = hand.legal_casts(seat)
        return rng.choice([a for a, ok in enumerate(mask) if ok]) if rng.random() < 0.5 else 0
    return pol


def worker(args):
    seeds, weights, deadline = args
    from scripts.train_jokers import hybrid_policy
    champ = hybrid_policy(weights)
    found, hands = [], 0
    for seed in seeds:
        if time.time() > deadline or len(found) >= 5:
            break
        rng = random.Random(seed)
        pol = champ if seed % 2 else random_policy(rng)
        match = jm.Match(jm.random_config(rng), rng.randrange(1 << 30))
        for _ in range(jm.HANDS_PER_MATCH):
            hand = match.next_hand()
            try:
                jm.play_out(hand, pol)
            except EmptyHand as e:
                found.append({"seed": seed, "policy": "champion" if seed % 2 else "random", **json.loads(str(e))})
                break
            match.finish(hand)
            hands += 1
    return found, hands


def main():
    from agents.network import PolicyValueNet as Net
    from engine import joker_obs as O
    from scripts.train_jokers import CHECKPOINT, net_to_numpy
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=200000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--minutes", type=float, default=150)
    parser.add_argument("--seed", type=int, default=93000000)
    args = parser.parse_args()
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    net = Net(O.OBS_DIM, J.ACTION_DIM)
    net.load_state_dict(ckpt["net"])
    net.eval()
    w = net_to_numpy(net)
    deadline = time.time() + args.minutes * 60
    seeds = list(range(args.seed, args.seed + args.matches))
    found, hands = [], 0
    with Pool(args.workers) as pool:
        for f, h in pool.map(worker, [(seeds[i::args.workers], w, deadline) for i in range(args.workers)]):
            found += f
            hands += h
    print("donnes jouees : %d, mains vides : %d" % (hands, len(found)))
    for f in found:
        print(json.dumps(f, default=str))


if __name__ == "__main__":
    main()
