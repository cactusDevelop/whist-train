"""Exporte tous les reseaux de jeu (checkpoints_v2/<role>.pt, 9 roles
depuis la Phase 2) + le meilleur individu d'enchere de la population
evolutive courante (checkpoints_v2/encherisseur_population.pt, voir
agents/evolution.py) vers le format GDScript attendu par le moteur
d'inference deja existant et teste cote Godot (game/whist_ai_policy.gd) :
w0/b0/w2/b2/wp/bp en float16 little-endian (depuis le 2026-09-29, demande
whist-4-it pour alleger l'app ; WhistAIPolicy reconnait un blob de taille
moitie et le reconvertit en float32, les anciens blobs float32 restent
acceptes), concatenes et encodes en base64, dans
une classe `WhistAIWeightsDataV2<Role>` -- meme structure que l'ancien
whist_ai_weights_data.gd (jamais copie, juste la MEME convention de
fichier, deja eprouvee : embedde en CODE .gd, pas en asset res:// binaire,
voir ce fichier chez Godot pour le pourquoi).

Le head de VALEUR (entrainement uniquement) n'est jamais exporte -- seul
le tronc (2 couches Tanh) + le head de POLITIQUE, exactement ce que
whist_ai_policy.gd sait charger et executer.

`encherisseur` : le pont Godot (game/whist_ai_bridge_v2.gd) ne sait
charger qu'UN reseau par nom de role -- on exporte donc le MEILLEUR
individu (fitness moyenne de la derniere generation jouee) de chaque
lignee de evolution.EXPORTED_LINEAGES, jamais les 40 individus. Depuis
le 2026-09-27, une seule lignee exportee (`encherisseur_amical`
supprimee, ici et dans Godot). Si une lignee est actuellement EXTINCTE (0
survivant, voir agents/evolution.py), son slot est saute avec un
avertissement explicite plutot que d'exporter silencieusement un
mauvais reseau.

Usage :
  venv/Scripts/python.exe scripts/export_to_godot.py [--out DIR] [--population-file PATH]

Par defaut, ecrit dans un dossier local `godot_export/` (PAS directement
dans whist-4-it/) -- copier a la main dans
whist-4-it/game/ une fois verifie, comme
mesure de securite tant que le pont n'est pas branche au jeu reel.
"""

import argparse
import base64
import copy
import os
import sys

sys.path.insert(0, ".")

from agents import evolution, specialists
from agents.network import HIDDEN

GDSCRIPT_TEMPLATE = '''class_name WhistAIWeightsDataV2{class_suffix}
extends RefCounted
## Poids entraines pour le role `{role}` (voir whist-ai/DESIGN.md), exportes
## par whist-ai/scripts/export_to_godot.py -- NE JAMAIS MODIFIER A LA MAIN.
## Regenerer ce fichier a chaque nouveau checkpoint. Charge par
## game/whist_ai_policy.gd (WhistAIPolicy), le meme moteur d'inference
## Tanh generique reutilise pour les 7 roles.

const OBS_DIM := {obs_dim}
const HIDDEN := {hidden}
const ACTION_DIM := {action_dim}

const BASE64_WEIGHTS := "{base64_weights}"
'''


def _role_class_suffix(role):
    return "".join(part.capitalize() for part in role.split("_"))


def fp16_rounded(net):
    """COPIE du reseau dont les poids exportes (tronc + head de politique)
    sont arrondis en float16 puis re-castes en float32 : exactement ce que
    Godot execute. Sert a la verification et aux fixtures de parite. Le
    reseau d'origine (entrainement) n'est jamais modifie."""
    rounded = copy.deepcopy(net)
    for layer in (rounded.trunk[0], rounded.trunk[2], rounded.policy_head):
        for param in (layer.weight, layer.bias):
            param.data = param.data.half().float()
    rounded.eval()
    return rounded


def _export_net(role, net, obs_dim, action_dim, out_dir):
    net.eval()

    w0 = net.trunk[0].weight.detach().numpy()  # (HIDDEN, obs_dim)
    b0 = net.trunk[0].bias.detach().numpy()  # (HIDDEN,)
    w2 = net.trunk[2].weight.detach().numpy()  # (HIDDEN, HIDDEN)
    b2 = net.trunk[2].bias.detach().numpy()  # (HIDDEN,)
    wp = net.policy_head.weight.detach().numpy()  # (action_dim, HIDDEN)
    bp = net.policy_head.bias.detach().numpy()  # (action_dim,)

    blob = b"".join(arr.astype("<f2").tobytes() for arr in (w0, b0, w2, b2, wp, bp))
    encoded = base64.b64encode(blob).decode("ascii")

    gd_source = GDSCRIPT_TEMPLATE.format(
        class_suffix=_role_class_suffix(role), role=role,
        obs_dim=obs_dim, hidden=HIDDEN, action_dim=action_dim, base64_weights=encoded,
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "whist_ai_weights_data_v2_%s.gd" % role)
    with open(path, "w", encoding="utf-8") as f:
        f.write(gd_source)
    return path, len(blob)


def export_role(role, out_dir):
    obs_dim, action_dim = specialists.IMPLEMENTED[role]
    net, _optimizer = specialists.load_or_create(role)
    return _export_net(role, net, obs_dim, action_dim, out_dir)


def export_bidder_roles(population_file, out_dir):
    """Exporte le MEILLEUR individu de chaque lignee de la population
    d'enchere courante -- voir docstring de module. Retourne {role:
    (path, size)} pour chaque lignee EFFECTIVEMENT exportee (une lignee
    eteinte est absente, jamais une erreur fatale -- voir
    agents/evolution.py:best_per_lineage)."""
    if not os.path.exists(population_file):
        print("  (pas de population d'enchere trouvee a %r -- roles d'enchere non exportes, "
              "aucun entrainement Phase 2 encore lance ?)" % population_file)
        return {}
    population, generation = evolution.load_population(population_file)
    best = evolution.best_per_lineage(population)
    results = {}
    for lineage in evolution.EXPORTED_LINEAGES:
        indiv = best.get(lineage)
        if indiv is None:
            print("  !! %s : lignee EXTINCTE dans cette population (generation %d), role non exporte" % (
                lineage, generation))
            continue
        path, size = _export_net(lineage, indiv.net, evolution.OBS_DIM, evolution.ACTION_DIM, out_dir)
        results[lineage] = (path, size)
        print("  %-14s -> %s (%d octets de poids, individu #%d, derniere fitness connue=%+.3f)" % (
            lineage, path, size, indiv.id, indiv.snapshot_fitness()))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="godot_export")
    parser.add_argument("--population-file", default=os.path.join("checkpoints_v2", "encherisseur_population.pt"),
                         help="fichier de population d'enchere (voir scripts/train.py:--population-file)")
    args = parser.parse_args()

    print("Export des %d roles de jeu vers %s/ ..." % (len(specialists.IMPLEMENTED), args.out))
    for role in specialists.IMPLEMENTED:
        path, size = export_role(role, args.out)
        print("  %-14s -> %s (%d octets de poids)" % (role, path, size))

    print("Export des roles d'enchere (population evolutive) vers %s/ ..." % args.out)
    bidder_results = export_bidder_roles(args.population_file, args.out)

    total = len(specialists.IMPLEMENTED) + len(bidder_results)
    print("\nOK : %d fichiers ecrits. A copier a la main dans "
          "whist-4-it/game/ une fois verifies (voir scripts/verify_export.py)." % total)


if __name__ == "__main__":
    main()
