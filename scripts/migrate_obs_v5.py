"""Migre les checkpoints des roles de JEU de l'observation v4 vers v5
(bloc O des renonces ajoute A LA FIN, voir engine/observation.py) sans
perdre l'entrainement : la premiere couche recoit des colonnes nulles pour
les 16 nouvelles entrees -> le reseau migre calcule EXACTEMENT la meme chose
qu'avant (verifie ici), puis apprend a se servir du bloc O. Moments Adam des
nouvelles colonnes a zero aussi. Meme methode que scripts/migrate_obs_v4.py.

Idempotent : un checkpoint deja en v5 est laisse tel quel.

Usage : venv/Scripts/python.exe scripts/migrate_obs_v5.py [--dir checkpoints_v2]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, ".")

from agents.network import PolicyValueNet
from agents.specialists import PLAY_ACTION_DIM, PLAY_ROLES
from agents.checkpoint_io import safe_save
from engine.observation import OBS_DIM, OBS_DIM_V4

FIRST = "trunk.0.weight"


def widen(t):
    return torch.cat([t, torch.zeros(t.shape[0], OBS_DIM - OBS_DIM_V4, dtype=t.dtype)], dim=1)


def migrate(path):
    ckpt = torch.load(path, map_location="cpu")
    w = ckpt["net"][FIRST]
    if w.shape[1] == OBS_DIM:
        return "deja v5"
    assert w.shape[1] == OBS_DIM_V4, (path, tuple(w.shape))
    old = PolicyValueNet(OBS_DIM_V4, PLAY_ACTION_DIM)
    old.load_state_dict(ckpt["net"])
    ckpt["net"][FIRST] = widen(w)
    new = PolicyValueNet(OBS_DIM, PLAY_ACTION_DIM)
    new.load_state_dict(ckpt["net"])
    x = torch.rand(64, OBS_DIM)
    with torch.no_grad():
        a = old.trunk(x[:, :OBS_DIM_V4])
        b = new.trunk(x)
    assert (a - b).abs().max().item() < 1e-4, (path, (a - b).abs().max().item())
    state = ckpt.get("optimizer", {}).get("state", {})
    for st in state.values():
        for k in ("exp_avg", "exp_avg_sq"):
            if k in st and st[k].dim() == 2 and st[k].shape[1] == OBS_DIM_V4:
                st[k] = widen(st[k])
    safe_save(ckpt, path)
    return "migre %d -> %d" % (OBS_DIM_V4, OBS_DIM)


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
