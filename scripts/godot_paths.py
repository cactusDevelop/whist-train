"""Localisation du projet Godot whist-4-it (sans dependance torch), partagee
par scripts/godot_sync.py et scripts/record_game.py."""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dans l'ordre : variable d'environnement, dossier voisin de whist-ai.
# Aucun chemin personnel en dur : introuvable -> None (entrainement headless).
GODOT_PROJECT_CANDIDATES = [
    os.environ.get("WHIST_4_IT_DIR"),
    os.path.join(os.path.dirname(REPO_ROOT), "whist-4-it"),
]


def find_godot_project():
    """Premier candidat contenant un project.godot, ou None."""
    for path in GODOT_PROJECT_CANDIDATES:
        if path and os.path.isfile(os.path.join(path, "project.godot")):
            return path
    return None
