"""Enregistre UNE PARTIE (plusieurs donnes) par affrontement -- une par
famille de declarant (agents/specialists.py:PHASE1_ROLES) : "Solo vs 3 Sans
Contrat", "Emballage vs 2 Sans Contrat", "Piccolo vs 3 Sans Contrat", etc --
a partir des checkpoints ACTUELS (checkpoints_v2/), et envoie les 6 fichiers
resultants DIRECTEMENT dans le projet whist-4-it (recordings/, cree si
absent), un fichier par famille sous un nom FIXE (voir
scripts/record_game.py:sample_filename()) -- un nouvel envoi remplace
l'ancienne partie de cette famille sans toucher aux autres fichiers deja
presents dans ce dossier. Demande explicite de l'utilisateur (2026-09-19).

Lit les checkpoints tels quels sur le disque : marche aussi bien PENDANT
qu'un entrainement (scripts/train.py) tourne en arriere-plan (les
checkpoints sont sauvegardes tous les --save-every lots, voir DESIGN.md)
qu'apres coup, sans avoir besoin de toucher/relancer ce process. Pour un
envoi PERIODIQUE et AUTOMATIQUE integre a un entrainement en cours (meme
cadence que l'evaluation vs RandomBot), voir plutot le hook de
scripts/train.py:main() (reutilise exactement la meme logique, avec les
reseaux d'entrainement deja en memoire).

Usage :
  venv/Scripts/python.exe scripts/record_training_samples.py
  venv/Scripts/python.exe scripts/record_training_samples.py --seed 1 --no-send
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, ".")

from scripts.record_game import (
    DEFAULT_GODOT_RECORDINGS_DIR, SAMPLE_HANDS_PER_FAMILY,
    build_family_sample_games, load_all_nets, send_family_samples_to_godot,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", default=None,
                         help="dossier local pour une copie de chaque partie (defaut: "
                              "recordings/training_samples_<horodatage>/)")
    parser.add_argument("--godot-dir", default=DEFAULT_GODOT_RECORDINGS_DIR,
                         help="dossier whist-4-it ou envoyer les fichiers (defaut: %s)" % DEFAULT_GODOT_RECORDINGS_DIR)
    parser.add_argument("--no-send", action="store_true", help="ne pas envoyer a whist-4-it, seulement local")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    nets = load_all_nets()
    games = build_family_sample_games(nets, rng, generated_by="whist-ai-training-samples-script")

    out_dir = args.out_dir or os.path.join("recordings", "training_samples_%s" % time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    for family, data in games.items():
        with open(os.path.join(out_dir, "%s.json" % family), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    print("%d partie(s) (1 par famille, %d donnes chacune) ecrites dans %s." % (
        len(games), SAMPLE_HANDS_PER_FAMILY, out_dir), flush=True)

    if args.no_send:
        return
    results = send_family_samples_to_godot(games, godot_dir=args.godot_dir)
    for family, path in results.items():
        if path:
            print("  %-14s -> %s" % (family, path), flush=True)
        else:
            print("  %-14s -> ATTENTION : echec de l'envoi vers %r" % (family, args.godot_dir), flush=True)


if __name__ == "__main__":
    main()
