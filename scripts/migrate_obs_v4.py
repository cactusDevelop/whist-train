"""Migre les checkpoints des roles de JEU de l'observation v3 vers v4
(bloc N ajoute A LA FIN, voir engine/observation.py) sans perdre
l'entrainement : la premiere couche recoit des colonnes nulles pour les
nouvelles entrees -> le reseau migre calcule EXACTEMENT la meme chose
qu'avant (verifie ici sur des observations reelles), puis apprend a se
servir du bloc N. Moments Adam des nouvelles colonnes a zero aussi.

Idempotent : un checkpoint deja en v4 est laisse tel quel.

Usage : venv/Scripts/python.exe scripts/migrate_obs_v4.py [--dir checkpoints_v2]
"""

import argparse
import os
import random
import sys

import torch

sys.path.insert(0, ".")

from agents.network import PolicyValueNet
from agents.specialists import PLAY_ACTION_DIM, PLAY_ROLES
from engine.observation import OBS_DIM_V3, OBS_DIM_V4 as OBS_DIM  # cible historique de cette migration : v4

FIRST = "trunk.0.weight"


def widen(t):
    return torch.cat([t, torch.zeros(t.shape[0], OBS_DIM - OBS_DIM_V3, dtype=t.dtype)], dim=1)


def migrate(path):
    ckpt = torch.load(path, map_location="cpu")
    w = ckpt["net"][FIRST]
    if w.shape[1] == OBS_DIM:
        return "deja v4"
    assert w.shape[1] == OBS_DIM_V3, (path, tuple(w.shape))
    old = PolicyValueNet(OBS_DIM_V3, PLAY_ACTION_DIM)
    old.load_state_dict(ckpt["net"])
    ckpt["net"][FIRST] = widen(w)
    new = PolicyValueNet(OBS_DIM, PLAY_ACTION_DIM)
    new.load_state_dict(ckpt["net"])
    # meme sortie sur des entrees v3 quelconques + bloc N quelconque
    x = torch.rand(64, OBS_DIM)
    with torch.no_grad():
        a = old.trunk(x[:, :OBS_DIM_V3])
        b = new.trunk(x)
    assert (a - b).abs().max().item() < 1e-4, (path, (a - b).abs().max().item())
    # moments Adam : le parametre de la 1re couche est l'etat d'index 0
    state = ckpt.get("optimizer", {}).get("state", {})
    for st in state.values():
        for k in ("exp_avg", "exp_avg_sq"):
            if k in st and st[k].dim() == 2 and st[k].shape[1] == OBS_DIM_V3:
                st[k] = widen(st[k])
    torch.save(ckpt, path)
    return "migre %d -> %d" % (OBS_DIM_V3, OBS_DIM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="checkpoints_v2")
    args = parser.parse_args()
    for role in PLAY_ROLES:
        path = os.path.join(args.dir, "%s.pt" % role)
        if os.path.exists(path):
            print("%-14s %s" % (role, migrate(path)), flush=True)


if __name__ == "__main__":
    main()
