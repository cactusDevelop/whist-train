"""Verification de l'observation v6 (bloc P, mes allies) -- cloud seulement
(mode obs_check de train.yml, lit l'etat ppo_free + ppo sans l'ecrire).

1. Bloc P sur de vraies donnes (emballage, solitaire, miseres a 1-3
   declarants, trou) jouees au hasard : valeurs attendues a chaque decision,
   et build() == build_fast_into() au float pres.
2. Migration v5 -> v6 sur TOUS les vrais checkpoints de jeu trouves
   (checkpoints_v2, championne, evaluateurs omniscients, population libre,
   best) : le reseau elargi donne exactement la meme sortie que l'ancien,
   et Adam fait un pas sans erreur de forme.
Sortie : "OBS_V6_OK" ou exception.
"""

import glob
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from agents.network import PolicyValueNet, widen_adam_state
from agents.oracle_critic import ORACLE_DIM, OracleCritic
from agents.specialists import PLAY_ACTION_DIM
from engine.cards import NUM_PLAYERS, deal
from engine.emballage import new_emballage_deal
from engine.misere import new_misere_deal
from engine.observation import ALLY_DIM, OBS_DIM, OBS_DIM_V5, _OFF, build_fast_into
from engine.solitaire import new_solitaire_deal
from engine.trou import new_trou_deal

FAILS = []


def check(label, cond):
    if not cond:
        FAILS.append(label)
        print("ECHEC", label, flush=True)


def ally_block(obs):
    return [int(v) for v in obs[OBS_DIM - ALLY_DIM:]]


def expected_allies(seat, side_of):
    return [1 if side_of[(seat + r) % NUM_PLAYERS] == side_of[seat] and side_of[seat] is not None else 0
            for r in (1, 2, 3)]


def play_out(game, side_of, label, rng):
    n = 0
    while not game.done:
        seat = game.turn
        obs = game.observation(seat)
        fast = np.zeros(OBS_DIM, dtype=np.float32)
        build_fast_into(fast, game, seat)
        check("%s rapide == reference" % label, np.array_equal(np.asarray(obs, dtype=np.float32), fast))
        check("%s allies siege %d : %s attendu %s" % (label, seat, ally_block(obs), expected_allies(seat, side_of)),
              ally_block(obs) == expected_allies(seat, side_of))
        mask = game.legal_action_mask(seat)
        game.step(seat, rng.choice([c for c, ok in enumerate(mask) if ok]))
        n += 1
    return n


def check_block_p(rng, deals=60):
    total = 0
    for i in range(deals):
        hands = deal(rng)
        leader = rng.randrange(4)
        d, p = rng.sample(range(4), 2)
        game = new_emballage_deal([list(h) for h in hands], d, p, rng.randrange(4), 8, leader)
        game.set_public_context(dealer=(leader - 1) % 4, bid_summary=None, whisted=False, level_known=True)
        total += play_out(game, ["att" if s in (d, p) else "def" for s in range(4)], "emballage", rng)

        game = new_solitaire_deal([list(h) for h in hands], d, rng.randrange(4), 6, leader)
        game.set_public_context(dealer=(leader - 1) % 4, bid_summary=None, whisted=False, level_known=True)
        total += play_out(game, [None if s == d else "def" for s in range(4)], "solitaire", rng)

        decl = rng.sample(range(4), rng.choice([1, 2, 3]))
        kind = rng.choice(["piccolo", "grande", "etalee"])
        try:
            game = new_misere_deal([list(h) for h in hands], decl, kind, leader)
        except Exception as e:  # sorte inconnue du moteur : on essaie la suivante
            print("misere %s ignoree : %s" % (kind, e))
            continue
        game.set_public_context(dealer=(leader - 1) % 4, bid_summary=None, whisted=False, level_known=True)
        total += play_out(game, [None if s in decl else "def" for s in range(4)], "misere x%d" % len(decl), rng)

        anchor = next(c for c in hands[p])
        try:
            game = new_trou_deal([list(h) for h in hands], d, p, anchor, p)
        except Exception as e:
            print("trou ignore : %s" % e)
            continue
        game.set_public_context(dealer=(leader - 1) % 4, bid_summary=None, whisted=False, level_known=True)
        total += play_out(game, ["att" if s in (d, p) else "def" for s in range(4)], "trou", rng)
    print("bloc P : %d decisions verifiees, offset P = %d" % (total, _OFF["P"]), flush=True)


def check_net(label, sd, oracle=False):
    first = sd["body.0.weight" if oracle else "trunk.0.weight"]
    width = first.shape[1] - (ORACLE_DIM if oracle else 0)
    if width == OBS_DIM:
        print("%-40s deja v6" % label)
        return
    check("%s largeur v5" % label, width == OBS_DIM_V5)
    if oracle:
        old, new = OracleCritic(OBS_DIM_V5), OracleCritic()
    else:
        old, new = PolicyValueNet(OBS_DIM_V5, PLAY_ACTION_DIM), PolicyValueNet(OBS_DIM, PLAY_ACTION_DIM)
    old.load_state_dict({k: v.clone() for k, v in sd.items()})
    new.load_state_dict({k: v.clone() for k, v in sd.items()})
    x = (torch.rand(256, OBS_DIM + (ORACLE_DIM if oracle else 0)) < 0.2).float()
    x_old = torch.cat([x[:, :OBS_DIM_V5], x[:, OBS_DIM:]], dim=1)
    with torch.no_grad():
        if oracle:
            diff = (old(x_old) - new(x)).abs().max().item()
        else:
            (pa, va), (pb, vb) = old(x_old), new(x)
            diff = max((pa - pb).abs().max().item(), (va - vb).abs().max().item())
    check("%s sortie identique (ecart %.2e)" % (label, diff), diff < 1e-5)
    print("%-40s v5 -> v6, ecart max %.2e" % (label, diff), flush=True)
    return old, new


def check_adam(label, sd, opt_state):
    old_net = PolicyValueNet(OBS_DIM_V5, PLAY_ACTION_DIM)
    old_net.load_state_dict({k: v.clone() for k, v in sd.items()})
    new_net = PolicyValueNet(OBS_DIM, PLAY_ACTION_DIM)
    new_net.load_state_dict({k: v.clone() for k, v in sd.items()})
    opt = torch.optim.Adam(new_net.parameters(), lr=1e-4)
    opt.load_state_dict(opt_state)
    widen_adam_state(opt)
    logits, value = new_net(torch.rand(8, OBS_DIM))
    (logits.mean() + value.mean()).backward()
    opt.step()
    print("%-40s pas Adam OK" % label, flush=True)


def check_migrations():
    n = 0
    for path in sorted(glob.glob("checkpoints_v2/*.pt") + glob.glob("checkpoints_v2/champion_play/*.pt")):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("net") if isinstance(ckpt, dict) else None
        if not isinstance(sd, dict):
            continue
        oracle = "body.0.weight" in sd
        if not oracle and ("trunk.0.weight" not in sd or sd["trunk.0.weight"].shape[1] not in (OBS_DIM_V5, OBS_DIM)):
            continue  # encherisseur / autres observations
        check_net(path, sd, oracle)
        if "optimizer" in ckpt and not oracle and sd["trunk.0.weight"].shape[1] == OBS_DIM_V5:
            check_adam(path, sd, ckpt["optimizer"])
        n += 1
    for path in sorted(glob.glob("checkpoints_free/*.pt") + glob.glob("checkpoints_free/best/*.pt")):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        nets = ckpt.get("nets", ckpt) if isinstance(ckpt, dict) else {}
        for role, sd in nets.items():
            if isinstance(sd, dict) and "trunk.0.weight" in sd:
                check_net("%s:%s" % (path, role), sd)
                opts = ckpt.get("opts") or {}
                if role in opts and sd["trunk.0.weight"].shape[1] == OBS_DIM_V5:
                    check_adam("%s:%s" % (path, role), sd, opts[role])
                n += 1
    print("%d reseaux verifies" % n, flush=True)
    check("au moins un checkpoint trouve", n > 0)


def main():
    check_block_p(random.Random(6))
    check_migrations()
    if FAILS:
        print("%d ECHEC(S)" % len(FAILS))
        sys.exit(1)
    print("OBS_V6_OK")


if __name__ == "__main__":
    main()
