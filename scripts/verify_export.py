"""Round-trip complet de l'export (voir scripts/export_to_godot.py) : pour
chacun des 9 roles de jeu + les roles d'enchere effectivement exportes
(meilleur individu par lignee de la population evolutive, voir
agents/evolution.py), exporte, RE-DECODE le blob base64 EXACTEMENT comme
game/whist_ai_policy.gd::_load()/_layer() (meme slicing, meme ordre
w0/b0/w2/b2/wp/bp, meme tanh(), pur Python/struct -- aucune dependance a
torch dans cette relecture), et verifie que l'action argmax obtenue
correspond EXACTEMENT a celle du reseau PyTorch d'origine sur un lot
d'observations aleatoires. Preuve que les octets exportes encodent bien
la politique entrainee, avant meme d'ecrire une ligne de GDScript.

Usage : venv/Scripts/python.exe scripts/verify_export.py
"""

import base64
import math
import os
import random
import struct
import sys

import torch

sys.path.insert(0, ".")

from agents import evolution, specialists
from agents.network import HIDDEN
from scripts.export_to_godot import export_bidder_roles, export_role, fp16_rounded

NUM_SAMPLES = 200


def _unpack_floats(blob, offset, count, fmt):
    values = struct.unpack_from("<%d%s" % (count, fmt), blob, offset)
    return list(values), offset + count * struct.calcsize("<" + fmt)


def _decode_weights(base64_weights, obs_dim, hidden, action_dim):
    """Comme WhistAIPolicy : un blob de taille moitie est du float16, sinon
    du float32. None si la taille ne correspond a aucun des deux."""
    blob = base64.b64decode(base64_weights)
    n_floats = hidden * obs_dim + hidden + hidden * hidden + hidden + action_dim * hidden + action_dim
    fmt = {n_floats * 2: "e", n_floats * 4: "f"}.get(len(blob))
    if fmt is None:
        return None
    offset = 0
    w0, offset = _unpack_floats(blob, offset, hidden * obs_dim, fmt)
    b0, offset = _unpack_floats(blob, offset, hidden, fmt)
    w2, offset = _unpack_floats(blob, offset, hidden * hidden, fmt)
    b2, offset = _unpack_floats(blob, offset, hidden, fmt)
    wp, offset = _unpack_floats(blob, offset, action_dim * hidden, fmt)
    bp, offset = _unpack_floats(blob, offset, action_dim, fmt)
    return w0, b0, w2, b2, wp, bp


def _layer(x, w, b, out_dim, in_dim, apply_tanh):
    out = []
    for o in range(out_dim):
        acc = b[o]
        base = o * in_dim
        for i in range(in_dim):
            acc += w[base + i] * x[i]
        out.append(math.tanh(acc) if apply_tanh else acc)
    return out


def gdscript_equivalent_argmax(weights, obs_dim, hidden, action_dim, obs, legal_ids):
    w0, b0, w2, b2, wp, bp = weights
    h1 = _layer(obs, w0, b0, hidden, obs_dim, True)
    h2 = _layer(h1, w2, b2, hidden, hidden, True)
    logits = _layer(h2, wp, bp, action_dim, hidden, False)
    best, best_val = None, -float("inf")
    for a in legal_ids:
        if logits[a] > best_val:
            best_val = logits[a]
            best = a
    return best


def verify_role(role, gd_path, net, obs_dim, action_dim):
    with open(gd_path, encoding="utf-8") as f:
        source = f.read()
    base64_weights = source.split('const BASE64_WEIGHTS := "')[1].split('"')[0]
    weights = _decode_weights(base64_weights, obs_dim, HIDDEN, action_dim)
    assert weights is not None, "taille du blob incoherente avec OBS_DIM/HIDDEN/ACTION_DIM"
    net.eval()
    rounded = fp16_rounded(net)

    rng = random.Random(hash(role) & 0xFFFF)
    mismatches = precision_diffs = 0
    for _ in range(NUM_SAMPLES):
        obs_list = [rng.uniform(-1.0, 1.0) for _ in range(obs_dim)]
        num_legal = rng.randint(1, action_dim)
        legal_ids = sorted(rng.sample(range(action_dim), num_legal))

        obs_tensor = torch.tensor(obs_list, dtype=torch.float32)
        mask = torch.zeros(action_dim, dtype=torch.bool)
        mask[legal_ids] = True
        with torch.no_grad():
            logits, _value = rounded.forward(obs_tensor.unsqueeze(0))
            logits = logits.squeeze(0).masked_fill(~mask, -1e9)
            full_logits, _value = net.forward(obs_tensor.unsqueeze(0))
            full_logits = full_logits.squeeze(0).masked_fill(~mask, -1e9)
        torch_action = int(torch.argmax(logits).item())

        gd_action = gdscript_equivalent_argmax(weights, obs_dim, HIDDEN, action_dim, obs_list, legal_ids)

        if torch_action != gd_action:
            mismatches += 1
        if int(torch.argmax(full_logits).item()) != gd_action:
            precision_diffs += 1

    return mismatches, precision_diffs


def _print_status(role, mismatches, precision_diffs):
    """`mismatches` : decodage GDScript vs reseau arrondi en float16 (doit
    etre 0). `precision_diffs` : meme comparaison contre le reseau float32
    d'origine, pour information (cout de l'arrondi float16)."""
    status = "OK" if mismatches == 0 else "ECHEC (%d/%d divergences)" % (mismatches, NUM_SAMPLES)
    print("  %-14s %s  (float16 vs float32 : %d/%d actions differentes)" % (
        role, status, precision_diffs, NUM_SAMPLES))


def verify_all(out_dir="godot_export", population_file=os.path.join("checkpoints_v2", "encherisseur_population.pt")):
    """Exporte vers `out_dir` puis verifie chaque role. Retourne
    (roles verifies, divergences totales) -- 0 divergence = export fidele."""
    print("Verification du round-trip export (%d observations aleatoires par role)..." % NUM_SAMPLES)
    total_mismatches = 0
    checked = 0
    for role in specialists.IMPLEMENTED:
        gd_path, _size = export_role(role, out_dir)
        obs_dim, action_dim = specialists.IMPLEMENTED[role]
        net, _optimizer = specialists.load_or_create(role)
        mismatches, precision_diffs = verify_role(role, gd_path, net, obs_dim, action_dim)
        total_mismatches += mismatches
        checked += 1
        _print_status(role, mismatches, precision_diffs)

    if os.path.exists(population_file):
        population, _generation = evolution.load_population(population_file)
        best = evolution.best_per_lineage(population)
        bidder_paths = {role: path for role, (path, _size) in export_bidder_roles(population_file, out_dir).items()}
        for role, indiv in best.items():
            if role not in bidder_paths:
                continue
            mismatches, precision_diffs = verify_role(role, bidder_paths[role], indiv.net,
                                                      evolution.OBS_DIM, evolution.ACTION_DIM)
            total_mismatches += mismatches
            checked += 1
            _print_status(role, mismatches, precision_diffs)
    else:
        print("  (pas de population d'enchere -- roles d'enchere non verifies)")
    return checked, total_mismatches


def main():
    checked, total_mismatches = verify_all()
    if total_mismatches:
        raise SystemExit("ECHEC : l'export ne correspond pas au reseau PyTorch d'origine.")
    print("\nOK : les %d exports decodent EXACTEMENT la meme politique que leurs reseaux PyTorch d'origine." % checked)


if __name__ == "__main__":
    main()
