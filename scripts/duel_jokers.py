"""Duel de whist-4-it contre les habitues a jokers (rapport 2026-10-23,
empreinte 07ac : « l'IA ne joue pas de jokers, c'est ridicule » + Dette de
Jeu x2 du deck admin peut-etre trop forte).

Une table du duplicate (DuplicateMatch) : le rival au siege 0 (ExpertBot +
regles expert pour ses jokers, comme le joueur simule de joker_profit),
trois habitues aux sieges 1-3 a deck d'elite sans maledictions, spammers,
poses = IA jokers du champion ; 4 donnes non whistees, Oeil « points ».
Chaque duel est rejoue avec plusieurs decks du rival (memes donnes, memes
habitues : duplicate exact) ; on compare ses points sur le duel (paires).

Variantes du deck du rival :
  admin       DuelServer.ADMIN_DECK
  admin_sans  ADMIN_DECK, les 2 Dette de Jeu remplacees par 2 Kermesse
  defaut      DuelRules.DEFAULT_DECK
  aucun       pas de joker
et admin / aucun contre des habitues SANS jokers (la situation d'avant).

Cloud seulement (whist-train, mode jokers_duel) :
  python scripts/duel_jokers.py --duels 500 --workers 4
"""
import argparse
import collections
import math
import multiprocessing as mp
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from agents.network import PolicyValueNet
from engine import joker_match as jm
from engine import joker_obs as O
from engine import jokers as J
from scripts import train_jokers as T

DUEL_HANDS = 4  # DuelRules.DUPLICATE_HANDS
EYE = "points"  # DuelRules.EYE
ADMIN_DECK = ["coup_de_vent", "coup_de_vent", "dette_de_jeu", "dette_de_jeu", "fossoyeur_de_plis", "fossoyeur_de_plis",
              "traitre", "traitre", "cameleon", "cameleon", "valet_borgne", "manche_truquee"]  # duel_server.gd
DEFAULT_DECK = ["cameleon", "cameleon", "coup_de_vent", "coup_de_vent", "kermesse", "kermesse",
                "rachat", "rachat", "inversion", "bluff", "faux_pas", "coup_du_chapeau"]  # duel_rules.gd
ADMIN_SANS = [j for j in ADMIN_DECK if j != "dette_de_jeu"] + ["kermesse", "kermesse"]
VARIANTS = (("admin", ADMIN_DECK, True), ("admin_sans", ADMIN_SANS, True), ("defaut", DEFAULT_DECK, True),
            ("aucun", [], True), ("admin_vs_nus", ADMIN_DECK, False), ("aucun_vs_nus", [], False))
PAIRS = (("admin", "admin_sans", "2 Dette de Jeu (contre 2 Kermesse)"),
         ("admin", "defaut", "deck admin contre deck par defaut"),
         ("admin", "aucun", "deck admin contre aucun joker"),
         ("defaut", "aucun", "deck par defaut contre aucun joker"),
         ("admin_vs_nus", "aucun_vs_nus", "deck admin, habitues SANS jokers (avant)"),
         ("aucun", "aucun_vs_nus", "jokers des habitues, rival sans joker"))


def play_duel(cfg, duel_seed, policy, counts):
    m = jm.Match(cfg, duel_seed)
    total = [0] * 4
    for _ in range(DUEL_HANDS):
        h = m.next_hand()
        pts = jm.play_out(h, policy)
        total = [a + b for a, b in zip(total, pts)]
        m.finish(h)
    return total


def worker(args):
    seed, n, weights = args
    policy = T.make_policy(weights)
    counts = collections.Counter()  # (variante, siege, joker) -> casts ; (variante, "dette") -> points transferes
    current = [None]
    orig_play, orig_debt = J.JokerRules.play, J.DetteDeJeu.on_trick_resolved

    def play(self, seat, jid, target=-1):
        out = orig_play(self, seat, jid, target)
        if not out.get("error"):
            counts[(current[0], "habitues" if seat else "rival", J.base_id(jid))] += 1
        return out

    def debt(self, winner, plays, rules):
        if winner == self.target:
            counts[(current[0], "dette_points", "")] += self.DEBT
        return orig_debt(self, winner, plays, rules)

    J.JokerRules.play, J.DetteDeJeu.on_trick_resolved = play, debt
    rows = []
    rng = random.Random(seed)
    for _ in range(n):
        duel_seed = rng.randrange(1 << 30)
        habitues = J.habitue_families(random.Random(duel_seed))
        row = {}
        for name, deck, armed in VARIANTS:
            current[0] = name
            cfg = {"kind": "duel", "act": 1, "eye": EYE, "mise": 0, "suspicion": 0, "style": "expert",
                   "deck": list(deck), "duel_seed": duel_seed,
                   "habitues": habitues if armed else {}}
            pts = play_duel(cfg, duel_seed, policy, counts)
            row[name] = (pts[0], pts[0] - max(pts[1:]))
        rows.append(row)
    return rows, counts


def stat(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(var / n), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=T.CHECKPOINT)
    ap.add_argument("--duels", type=int, default=500, help="duels par worker")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = PolicyValueNet(O.OBS_DIM, J.ACTION_DIM)
    net.load_state_dict(ck["net"])
    weights = T.net_to_numpy(net)
    print("champion iter %s, spec %s, %d duels de %d donnes" % (ck.get("iter"), ck.get("spec_hash"), args.duels * args.workers, DUEL_HANDS))
    with mp.Pool(args.workers) as pool:
        parts = pool.map(worker, [(args.seed * 1000 + w, args.duels, weights) for w in range(args.workers)])
    rows = [r for p, _c in parts for r in p]
    counts = collections.Counter()
    for _r, c in parts:
        counts.update(c)
    n = len(rows)
    print("\n## Points du rival par duel (4 donnes)")
    print("| deck | points | ecart au meilleur habitue |")
    print("|---|---|---|")
    for name, _d, _a in VARIANTS:
        m, se, _ = stat([r[name][0] for r in rows])
        g, gse, _ = stat([r[name][1] for r in rows])
        print("| %s | %+.2f ± %.2f | %+.2f ± %.2f |" % (name, m, se, g, gse))
    print("\n## Differences appariees (meme duel)")
    print("| comparaison | gain points | gain ecart |")
    print("|---|---|---|")
    for a, b, label in PAIRS:
        m, se, _ = stat([r[a][0] - r[b][0] for r in rows])
        g, gse, _ = stat([r[a][1] - r[b][1] for r in rows])
        print("| %s | %+.2f ± %.2f | %+.2f ± %.2f |" % (label, m, se, g, gse))
    print("\n## Jokers poses par duel")
    for name, _d, _a in VARIANTS:
        hab = sum(v for k, v in counts.items() if k[0] == name and k[1] == "habitues")
        riv = sum(v for k, v in counts.items() if k[0] == name and k[1] == "rival")
        print("- %s : habitues %.2f (3 sieges), rival %.2f, Dette de Jeu %.2f pts transferes"
              % (name, hab / n, riv / n, counts[(name, "dette_points", "")] / n))
    print("\n## Detail des poses (variante admin)")
    for (name, who, jid), v in sorted(counts.items(), key=lambda kv: -kv[1]):
        if name == "admin" and who in ("habitues", "rival"):
            print("- %s %s : %.2f / duel" % (who, jid, v / n))


if __name__ == "__main__":
    main()
