"""Controle horaire de l'entrainement (demande utilisateur 2026-09-27) :
l'entrainement doit tourner sans s'arreter, et chaque contrat tente
VOLONTAIREMENT (Phase 2, vraie enchere -- donc hors Trou, detecte
automatiquement, et hors Phase 1 ou le contrat est force) doit reussir
au moins 30 % du temps. Tout ce qui est en dessous est SIGNALE.

Lit :
  - le dernier log train_log_*.txt (fraicheur = entrainement vivant) ;
  - training_stats_phase2_v3.jsonl (une ligne toutes les 10000 donnes,
    stats de la fenetre ecoulee), compare la derniere ligne a celle
    d'environ une heure avant pour la tendance.

Usage : venv/Scripts/python.exe scripts/check_training_health.py
Code retour 0 si rien a signaler, 1 sinon.
"""

import glob
import json
import os
import sys
import time
from datetime import datetime

STATS_FILE = "training_stats_phase2_v3.jsonl"
SUCCESS_MIN = 0.30
MIN_ATTEMPTS = 20  # en dessous, taux trop bruite pour etre signale seul
LOG_STALE_S = 600
SOLO_HIGH = 10


def voluntary_contracts(line):
    """{contrat: (reussite, tentatives)} pour les contrats pris a
    l'enchere (Trou exclu)."""
    fam = line["families"]
    out = {}
    for f in ("solitaire", "emballage"):
        prefix = "solo" if f == "solitaire" else "emballage"
        for level, v in fam.get(f, {}).get("by_level", {}).items():
            out["%s-%s" % (prefix, level)] = (v["success_rate"], v["attempts"])
    for f in ("piccolo", "petite_misere", "grande_misere", "misere_etalee"):
        v = fam.get(f)
        if v:
            out[f] = (v["success_rate"], v["success_attempts"])
    return out


def summary(line):
    c = voluntary_contracts(line)
    attempts = sum(n for _, n in c.values())
    successes = sum(r * n for r, n in c.values())
    solos = {k: n for k, (_, n) in c.items() if k.startswith("solo-")}
    high = sum(n for k, n in solos.items() if int(k.split("-")[1]) >= SOLO_HIGH)
    return {
        "success": successes / attempts if attempts else 0.0,
        "attempts": attempts,
        "solo_high_share": high / attempts if attempts else 0.0,
    }


def main():
    problems = []
    logs = sorted(glob.glob("train_log_*.txt"), key=os.path.getmtime)
    if not logs:
        problems.append("aucun log d'entrainement")
    else:
        log = logs[-1]
        age = time.time() - os.path.getmtime(log)
        with open(log, encoding="utf-8", errors="replace") as fh:
            tail = fh.read()[-4000:]
        print("log : %s (modifie il y a %.0f s)" % (log, age))
        if "Arret. Checkpoints finaux sauvegardes" in tail:
            problems.append("l'entrainement s'est ARRETE (arret propre dans %s)" % log)
        elif age > LOG_STALE_S:
            problems.append("log fige depuis %.0f min -- entrainement probablement mort" % (age / 60))
        err = log + ".err"
        if os.path.exists(err) and os.path.getsize(err) > 0:
            with open(err, encoding="utf-8", errors="replace") as fh:
                last = fh.read()[-500:].strip()
            if "Traceback" in last or "Error" in last:
                problems.append("erreur dans %s : %s" % (err, last.splitlines()[-1]))

    with open(STATS_FILE, encoding="utf-8") as fh:
        lines = [json.loads(x) for x in fh if x.strip()]
    last = lines[-1]
    t_last = datetime.fromisoformat(last["timestamp"])
    ref = next((x for x in reversed(lines[:-1])
                if (t_last - datetime.fromisoformat(x["timestamp"])).total_seconds() >= 3000), lines[0])
    print("stats : %s, generation %d, %d donnes" % (last["timestamp"], last["bidder_generation"], last["total_donnes"]))

    s_now, s_ref = summary(last), summary(ref)
    print("reussite des contrats volontaires : %.1f %% (%d tentatives) -- avant (%s) : %.1f %%" % (
        100 * s_now["success"], s_now["attempts"], ref["timestamp"], 100 * s_ref["success"]))
    print("part des solos >= %d : %.1f %% -- avant : %.1f %%" % (
        SOLO_HIGH, 100 * s_now["solo_high_share"], 100 * s_ref["solo_high_share"]))
    if s_now["success"] < s_ref["success"] - 0.02:
        problems.append("la reussite globale BAISSE (%.1f %% -> %.1f %%)" % (
            100 * s_ref["success"], 100 * s_now["success"]))

    print("lignees : %s" % ", ".join(
        "%s %s" % (k, "n/d" if v["fitness_moy"] is None else "%+.1f" % v["fitness_moy"])
        for k, v in last["bidder_lineages"].items()))
    print("\ncontrat          reussite  tentatives")
    below = []
    for name, (rate, n) in sorted(voluntary_contracts(last).items(), key=lambda kv: -kv[1][1]):
        flag = ""
        if rate < SUCCESS_MIN and n >= MIN_ATTEMPTS:
            flag = "  <-- SOUS 30 %"
            below.append("%s %.0f %% (%d)" % (name, 100 * rate, n))
        print("%-15s %7.1f %% %10d%s" % (name, 100 * rate, n, flag))
    if below:
        problems.append("contrats sous %d %% de reussite : %s" % (100 * SUCCESS_MIN, ", ".join(below)))

    # duel IA vs ExpertBot (scripts/duel_expert.py) : l'objectif est que l'IA passe devant
    if os.path.exists("duel_expert.jsonl"):
        with open("duel_expert.jsonl", encoding="utf-8") as fh:
            duels = [json.loads(x) for x in fh if x.strip()]
        if duels:
            d = duels[-1]
            print("\nduel IA vs ExpertBot (%s, %d donnes x2) : ecart %+.2f +- %.2f pts/donne%s" % (
                d["timestamp"], d["deals"], d["ai_minus_bot"], 2 * d["std_error"],
                "" if len(duels) < 2 else " -- precedent %+.2f" % duels[-2]["ai_minus_bot"]))

    print()
    if problems:
        print("A SIGNALER :")
        for p in problems:
            print("  - " + p)
        return 1
    print("RAS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
