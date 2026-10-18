"""Injecte la lignee hybride `encherisseur_expert` (agents/hybrid_bidder.py)
dans la population d'enchere, A L'ARRET de l'entrainement (jamais pendant :
train.py reecrit la population a chaque generation).

La graine (reseau distille de l'ExpertBot, scripts/distill_hybrid_bidder.py)
remplace les pires individus de la population (derniere fitness connue) :
la graine intacte + (N-1) enfants mutes (meme sigma que l'evolution). La
taille de la population ne change pas. Les reseaux d'avant le saut du passe
parole sont elargis sans perte au chargement (evolution.widen_bid_state_dict).

Refuse si la lignee est deja presente (sauf --force). Sauvegarde l'ancienne
population a cote (<fichier>.avant_hybride) avant d'ecrire.

Usage : venv/Scripts/python.exe scripts/inject_hybrid_lineage.py --seed-net <hybrid_seed.pt> [--count 8]
"""

import argparse
import os
import shutil
import sys

import torch

sys.path.insert(0, ".")

from agents import evolution

POPULATION_FILE_DEFAULT = os.path.join("checkpoints_v2", "encherisseur_population.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-net", required=True)
    parser.add_argument("--population-file", default=POPULATION_FILE_DEFAULT)
    parser.add_argument("--count", type=int, default=8, help="individus hybrides injectes (graine + mutants)")
    parser.add_argument("--sigma", type=float, default=evolution.MUTATION_SIGMA_DEFAULT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    population, generation = evolution.load_population(args.population_file)
    if any(ind.lineage == evolution.HYBRID_LINEAGE for ind in population) and not args.force:
        raise SystemExit("lignee %s deja presente (--force pour reinjecter)" % evolution.HYBRID_LINEAGE)
    seed = evolution.new_net()
    seed.load_state_dict(torch.load(args.seed_net, map_location="cpu")["state_dict"])

    # jamais le dernier representant d'une lignee existante
    victims = []
    for ind in sorted(population, key=lambda i: i.snapshot_fitness()):
        if len(victims) == args.count:
            break
        if sum(1 for o in population if o.lineage == ind.lineage and o not in victims) > 1:
            victims.append(ind)
    newcomers = [evolution.Individual(seed, evolution.HYBRID_LINEAGE)]
    newcomers += [evolution.Individual(evolution.mutate_net(seed, args.sigma), evolution.HYBRID_LINEAGE)
                  for _ in range(len(victims) - 1)]
    for victim, newcomer in zip(victims, newcomers):
        population[population.index(victim)] = newcomer

    backup = args.population_file + ".avant_hybride"
    shutil.copy2(args.population_file, backup)
    evolution.save_population(args.population_file, population, generation)
    counts = {lin: sum(1 for i in population if i.lineage == lin) for lin in evolution.LINEAGES}
    print("Population generation %d : %d individus remplaces par la lignee %s -> %s (ancienne : %s)" % (
        generation, len(victims), evolution.HYBRID_LINEAGE, counts, backup))


if __name__ == "__main__":
    main()
