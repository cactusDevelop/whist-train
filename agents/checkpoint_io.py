"""Ecriture sure des checkpoints (2026-09-27).

Nuit du 26 au 27/09 : l'entrainement est mort a la sauvegarde (Windows,
erreur 1224 "fichier ouvert par un autre processus") pendant que git lisait
les .pt. torch.save direct = un verrou passager tue le run, et une ecriture
interrompue laisse un fichier corrompu. Le 27/09 a 08:49, meme erreur sur
un fichier que git ne lisait plus : probablement l'antivirus qui scanne un
.pt fraichement ecrit (un .pt est une archive zip). Ici : ecriture dans un fichier
temporaire voisin, puis os.replace (atomique), avec quelques tentatives."""

import os
import time

import torch

RETRIES = 6
RETRY_DELAY_S = 2.0  # double a chaque essai : 2+4+8+16+32 = ~1 min au total


def safe_save(obj, path, retries=RETRIES, delay=RETRY_DELAY_S):
    tmp = path + ".tmp"
    for attempt in range(retries):
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)
            return
        except (OSError, RuntimeError) as exc:
            if attempt == retries - 1:
                raise
            print("  !! sauvegarde de %s en echec (%r), nouvel essai dans %.0fs" % (path, exc, delay), flush=True)
            time.sleep(delay)
            delay *= 2


def safe_write_json(obj, path, retries=RETRIES, delay=RETRY_DELAY_S):
    """Meme protection pour un petit fichier d'etat JSON (ex. compteur de
    l'imitation de l'ExpertBot, scripts/train.py:TEACHER_STATE_FILE)."""
    import json
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)  # 1er demarrage sur un --checkpoint-dir neuf
    for attempt in range(retries):
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f)
            os.replace(tmp, path)
            return
        except OSError as exc:
            if attempt == retries - 1:
                raise
            print("  !! ecriture de %s en echec (%r), nouvel essai dans %.0fs" % (path, exc, delay), flush=True)
            time.sleep(delay)
            delay *= 2
