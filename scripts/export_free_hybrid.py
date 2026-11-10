"""Exporte vers Godot les reseaux de jeu de best.pt (IA libre) utilises par
l'HYBRIDE (2026-11-08, scripts/free_match.py) : en jeu, le reseau ne joue que
les contrats de HYBRID_KINDS (tous les sieges IA de la donne : declarant ET
defense), ExpertBot ailleurs. Mesure : run whist-train 42523974203, hybride
piccolo+grande_misere +0.87 +- 0.06 pt/donne devant ExpertBot (88512 donnes
neuves, enchere ExpertBot). Masque = cartes legales SEULES (pas de garde-fous
ai_play_mask : c'est ce qui a ete mesure).

Fichiers : whist_ai_weights_data_v2_libre_<role>.gd, classe
WhistAIWeightsDataV2Libre<Role> (meme format que export_to_godot.py), pour ne
pas ecraser les poids de la championne.

Usage : venv/Scripts/python.exe scripts/export_free_hybrid.py --best <best.pt> [--out godot_export_libre]
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, ".")

from agents import specialists
from engine.misere import DECLARER_ROLE_BY_KIND
from scripts import train_free
from scripts.export_to_godot import _export_net

HYBRID_KINDS = ("piccolo", "grande", "petite", "etalee")  # miseres : tous les sieges IA (en jeu depuis 1.2.19)
CANDIDATE_KINDS = ()
# ROUTAGE 09/11 (whist-train 42618189678, graine de confirmation) : en TROU, le
# reseau ne joue que le camp DECLARANT (demandeur = role trou, bouche-trou) ;
# la defense du trou reste ExpertBot. Routage vs hybride +0.337 +- 0.042/donne,
# trou declarant +1.79 +- 0.21 (16929 donnes).
DECL_ONLY_ROLES = ("trou", "bouche_trou")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", required=True)
    ap.add_argument("--out", default="godot_export_libre")
    args = ap.parse_args()
    nets = train_free.load_nets(torch.load(args.best, map_location="cpu", weights_only=False))
    roles = ["defenseur"] + [DECLARER_ROLE_BY_KIND[k] for k in HYBRID_KINDS + CANDIDATE_KINDS] + list(DECL_ONLY_ROLES)
    for role in roles:
        obs_dim, action_dim = specialists.IMPLEMENTED[role]
        path, size = _export_net("libre_" + role, nets[role], obs_dim, action_dim, args.out)
        print("%-16s -> %s (%d octets)" % (role, path, size))
    meta = {"hybrid_kinds": list(HYBRID_KINDS), "candidate_kinds": list(CANDIDATE_KINDS),
            "roles": {k: DECLARER_ROLE_BY_KIND[k] for k in HYBRID_KINDS + CANDIDATE_KINDS},
            "decl_only_roles": list(DECL_ONLY_ROLES),
            "defender_role": "defenseur", "mask": "legal_only", "source": "checkpoints_free/best.pt",
            "measure": "whist-train 42618189678 : hybride +0.889 +- 0.031 vs ExpertBot ; "
                       "routage (hybride + trou declarant) +0.337 +- 0.042 vs hybride"}
    with open(os.path.join(args.out, "libre_hybrid.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
