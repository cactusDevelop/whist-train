"""Publication automatique vers le projet Godot whist-4-it (demande explicite
de l'utilisateur, 2026-09-23 : "assure toi de faire cet export a chaque
fois") : regenere godot_export/, VERIFIE le round-trip (scripts/
verify_export.py:verify_all), et SEULEMENT si les 11 exports sont fideles,
copie :
  - godot_export/whist_ai_weights_data_v2_*.gd -> whist-4-it/game/
  - les stats Phase 1 / 2 / 3 (JSON Lines)      -> whist-4-it/game/
    whist_ai_training_stats_phase{1,2,3}.jsonl (chemins et format convenus
    avec la session whist-4-it, onglet "Stats IA" de settings_screen.gd) ;
  - l'historique des evaluations (eval_stats.jsonl, une ligne par export :
    dernier duel vs ExpertBot + dernier expert_match + derniere evaluation
    vs RandomBot, voir append_eval_stats)
    -> whist-4-it/game/whist_ai_eval_stats.jsonl (demande whist-4-it 28/09).
Un export divergent n'est JAMAIS copie dans le jeu.

Usage manuel : venv/Scripts/python.exe scripts/godot_sync.py
  [--stats-file X] [--stats-file-phase2 Y] [--stats-file-phase3 Z] [--godot-project DIR]
"""

import argparse
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, ".")

from scripts.godot_paths import GODOT_PROJECT_CANDIDATES, find_godot_project
from scripts.verify_export import verify_all

STATS_TARGETS = {
    "phase1": "whist_ai_training_stats_phase1.jsonl",
    "phase2": "whist_ai_training_stats_phase2.jsonl",
    "phase3": "whist_ai_training_stats_phase3.jsonl",
    "eval": "whist_ai_eval_stats.jsonl",
}
EVAL_STATS_FILE_DEFAULT = "eval_stats.jsonl"


def _last_record(path):
    if not os.path.isfile(path):
        return None
    last = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                last = line
    return json.loads(last) if last else None


def append_eval_stats(eval_stats_file=EVAL_STATS_FILE_DEFAULT, duel_file="duel_expert.jsonl",
                      expert_match_file="expert_match.jsonl", vs_random_file="eval_vs_random.jsonl"):
    """Ajoute UNE ligne a l'historique des evaluations : copies telles
    quelles des derniers records de duel_expert.jsonl, expert_match.jsonl et
    eval_vs_random.jsonl (null si la mesure n'existe pas encore). Appele a
    chaque export."""
    import time
    record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "duel_expert": _last_record(duel_file), "expert_match": _last_record(expert_match_file),
              "vs_random": _last_record(vs_random_file)}
    with open(eval_stats_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def publish_stats_file(stats_file, key, godot_project=None):
    """Copie UN fichier de stats (pas de poids, donc rien a verifier) --
    utilise par scripts/phase3.py en autonome. Retourne True si copie."""
    godot_project = godot_project or find_godot_project()
    if godot_project is None or not os.path.isfile(stats_file):
        print("  !! stats %s non publiees (projet whist-4-it ou fichier introuvable)." % key, flush=True)
        return False
    target = os.path.join(godot_project, "game", STATS_TARGETS[key])
    shutil.copy2(stats_file, target)
    print("  stats %s publiees -> %s" % (key, target), flush=True)
    return True


def publish_to_godot(export_dir="godot_export", stats_file=None, stats_file_phase2=None,
                     godot_project=None, population_file=os.path.join("checkpoints_v2", "encherisseur_population.pt"),
                     stats_file_phase3=None, parity_fixtures=True, eval_stats_file=None):
    """Exporte, verifie, puis copie poids + stats dans whist-4-it/game/.
    `parity_fixtures` : regenere aussi les cas de parite enchere + jeu sur les
    poids qui viennent d'etre publies (voir publish_parity_fixtures).
    Retourne True si la copie a eu lieu. Ne leve jamais d'exception pour un
    projet Godot introuvable (message + False) : l'entrainement continue."""
    godot_project = godot_project or find_godot_project()
    if godot_project is None:
        print("  !! projet whist-4-it introuvable (%s) -- rien copie dans Godot." % [
            p for p in GODOT_PROJECT_CANDIDATES if p], flush=True)
        return False
    checked, mismatches = verify_all(export_dir, population_file)
    if mismatches:
        print("  !! export DIVERGENT (%d divergences) -- rien copie dans Godot." % mismatches, flush=True)
        return False
    game_dir = os.path.join(godot_project, "game")
    weights = sorted(glob.glob(os.path.join(export_dir, "whist_ai_weights_data_v2_*.gd")))
    for path in weights:
        shutil.copy2(path, game_dir)
    copied_stats = []
    for src, key in ((stats_file, "phase1"), (stats_file_phase2, "phase2"), (stats_file_phase3, "phase3"),
                     (eval_stats_file, "eval")):
        if src and os.path.isfile(src):
            shutil.copy2(src, os.path.join(game_dir, STATS_TARGETS[key]))
            copied_stats.append(STATS_TARGETS[key])
    print("  publie dans %s : %d fichiers de poids verifies (%d roles) + stats %s" % (
        game_dir, len(weights), checked, ", ".join(copied_stats) or "(aucune)"), flush=True)
    if parity_fixtures:
        publish_parity_fixtures(godot_project, population_file, export_dir)
    return True


# Fixtures de parite des runners Godot (tests/ai_bridge_parity_v2_{bid,play}_runner.gd) :
# elles doivent venir des MEMES poids que ceux de game/, sinon les runners
# echouent apres chaque publication automatique (constate le 2026-09-27).
PARITY_FIXTURES = (
    ("scripts/dump_bid_parity_fixtures.py", "bid_parity_cases.json", ("tests", "fixtures", "ai_v2_bid")),
    ("scripts/dump_play_parity_fixtures.py", "play_parity_cases.json", ("tests", "fixtures", "ai_v2_play")),
)


def publish_parity_fixtures(godot_project, population_file, export_dir):
    """Regenere les fixtures de parite depuis checkpoints_v2/ (juste sauvegarde
    par l'appelant) et les copie dans whist-4-it/tests/fixtures/. Jamais
    fatal : un echec est imprime, l'entrainement continue."""
    import subprocess
    import sys
    for script, name, target in PARITY_FIXTURES:
        out = os.path.join(export_dir, name)
        cmd = [sys.executable, script, "--out", out]
        if "bid" in name:
            cmd += ["--population", population_file]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("  !! fixture %s non regeneree : %s" % (name, (r.stdout + r.stderr).strip()[-300:]), flush=True)
            continue
        dest_dir = os.path.join(godot_project, *target)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(out, dest_dir)
        print("  fixture de parite %s regeneree -> %s" % (name, dest_dir), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", default="godot_export")
    parser.add_argument("--stats-file", default="training_stats.jsonl")
    parser.add_argument("--stats-file-phase2", default="training_stats_phase2.jsonl")
    parser.add_argument("--stats-file-phase3", default="training_stats_phase3.jsonl")
    parser.add_argument("--eval-stats-file", default=EVAL_STATS_FILE_DEFAULT)
    parser.add_argument("--godot-project", default=None)
    args = parser.parse_args()
    if not publish_to_godot(args.export_dir, args.stats_file, args.stats_file_phase2, args.godot_project,
                            stats_file_phase3=args.stats_file_phase3, eval_stats_file=args.eval_stats_file):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
