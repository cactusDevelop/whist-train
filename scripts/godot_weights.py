"""Relit un whist_ai_weights_data_v2_<role>.gd (format de export_to_godot.py :
w0/b0/w2/b2/wp/bp en float16 little-endian, base64) en PolicyValueNet, pour
generer des cas de parite avec EXACTEMENT les poids publies dans le jeu
(2026-11-08 : les cas faits depuis l'etat cloud ne correspondaient plus a
l'export du 22/10 present dans whist-4-it/game). Le head de valeur n'est pas
publie : il reste aleatoire (jamais utilise par les cas)."""

import base64
import os
import re

import numpy as np
import torch

from agents.network import PolicyValueNet


def load_gd(path, obs_dim=None):
    src = open(path, encoding="utf-8").read()
    dims = {k: int(re.search(r"const %s := (\d+)" % k, src).group(1)) for k in ("OBS_DIM", "HIDDEN", "ACTION_DIM")}
    blob = base64.b64decode(re.search(r'BASE64_WEIGHTS := "([^"]*)"', src).group(1))
    o, h, a = dims["OBS_DIM"], dims["HIDDEN"], dims["ACTION_DIM"]
    sizes = [h * o, h, h * h, h, a * h, a]
    item = 2 if len(blob) == 2 * sum(sizes) else 4
    dtype = "<f2" if item == 2 else "<f4"
    arr = np.frombuffer(blob, dtype=dtype).astype(np.float32)
    assert arr.size == sum(sizes), (path, arr.size, sum(sizes))
    parts, i = [], 0
    for n in sizes:
        parts.append(arr[i:i + n])
        i += n
    sd = {"trunk.0.weight": parts[0].reshape(h, o), "trunk.0.bias": parts[1],
          "trunk.2.weight": parts[2].reshape(h, h), "trunk.2.bias": parts[3],
          "policy_head.weight": parts[4].reshape(a, h), "policy_head.bias": parts[5]}
    sd = {k: torch.from_numpy(np.array(v)) for k, v in sd.items()}
    net = PolicyValueNet(obs_dim or o, a)
    assert net.trunk[0].out_features == h, "HIDDEN different de agents/network.py"
    # obs plus petite (export d'avant le bloc P, v5) : meme migration que les checkpoints (colonnes nulles)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith("value_head") for k in missing), (missing, unexpected)
    net.eval()
    return net


def load_role(godot_dir, role, obs_dim=None):
    return load_gd(os.path.join(godot_dir, "whist_ai_weights_data_v2_%s.gd" % role), obs_dim)
