"""Diagnostic (MESURE seulement, cloud) : ou l'IA progresse-t-elle face a la
championne, contrat par contrat ET camp par camp ? (2026-10-26 : 12 portes,
emballage plat -0.05 pt/donne alors que trou/miseres progressent.)

La porte (champion_play) melange les reseaux dans un meme camp (sieges
{0,2} contre {1,3}, alors que le partenaire d'emballage est n'importe quel
siege). Ici, apres l'enchere (la meme pour les 3 parties, greedy), on joue
la donne 3 fois :
  - base    : championne aux 4 sieges ;
  - decl    : IA courante au camp DECLARANT, championne en defense ;
  - defense : championne au camp declarant, IA courante en defense.
Gain declarant = points du camp declarant (decl - base) ; gain defense =
points du camp de defense (defense - base), en vrais points, par donne.

Usage (cloud, whist-train mode ppo_camp_diag) :
  python scripts/champion_by_camp.py [--deals 20000]
"""

import argparse
import sys

sys.path.insert(0, ".")

from scripts.champion_play import CHAMPION_SEED, camp_rows, load_champion, mean_se


def main():
    from scripts.phase3 import load_exported_ai
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=CHAMPION_SEED)
    args = parser.parse_args()
    new, population, _gen = load_exported_ai()
    champ, meta = load_champion()
    if champ is None:
        sys.exit("pas de championne")
    print("Championne depuis %s ; %d donnes" % (meta.get("since"), args.deals), flush=True)
    rows = camp_rows(new, champ, population, args.deals, args.seed, progress=True)
    print("\n=== IA courante - championne, vrais points du camp par donne (+- 2 erreurs types) ===")
    tot = {"decl": [], "def": []}
    for role, r in sorted(rows.items(), key=lambda kv: -len(kv[1]["decl"])):
        md, sd = mean_se(r["decl"])
        mf, sf = mean_se(r["def"])
        tot["decl"] += r["decl"]
        tot["def"] += r["def"]
        print("  %-14s %6d donnes   declarant %+.3f +- %.3f   defense %+.3f +- %.3f   (parties changees : decl %.0f %%, def %.0f %%)"
              % (role, len(r["decl"]), md, 2 * sd, mf, 2 * sf,
                 100 * sum(x != 0 for x in r["decl"]) / len(r["decl"]),
                 100 * sum(x != 0 for x in r["def"]) / len(r["def"])))
    md, sd = mean_se(tot["decl"])
    mf, sf = mean_se(tot["def"])
    print("  %-14s %6d donnes   declarant %+.3f +- %.3f   defense %+.3f +- %.3f" % ("TOUT", len(tot["decl"]), md, 2 * sd, mf, 2 * sf))


if __name__ == "__main__":
    main()
