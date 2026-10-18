"""Re-amorce la lignee d'enchere `encherisseur` (reseau pur, la seule
exportee vers Godot) par DISTILLATION de l'enchere de l'ExpertBot, et
retire de la population toute lignee qui n'est plus dans
evolution.LINEAGES (2026-09-27 : `encherisseur_amical` supprimee).

Pourquoi : `encherisseur` etait bloquee a -35 points par donne (solo-13,
solo-12 a 0 % de reussite) sans progresser, gardee en vie par la garantie
d'un survivant par lignee (evolution.select_elites) ; mutation seule
(sigma 0.05) incapable d'en sortir. L'ExpertBot bat l'IA (~80 %).

Contrairement a scripts/distill_hybrid_bidder.py (reseau de la lignee
hybride, seulement les decisions laissees libres par les regles), le
reseau appris ici decide SEUL : TOUTES les decisions sont etiquetees, avec
le masque de TOUTES les actions legales. Etats visites : chaque siege est
tenu soit par le bot, soit par un individu hybride de la population, pour
voir aussi des situations hors table de bots (idee DAgger).

Nouvelle population (taille inchangee) : moitie `encherisseur` (le reseau
distille + enfants mutes), moitie hybride (individus hybrides existants +
enfants mutes du meilleur). L'ancien fichier est sauvegarde a cote.

A lancer ENTRAINEMENT ARRETE (le fichier de population est reecrit) :
  venv/Scripts/python.exe scripts/reseed_encherisseur.py [--deals N] [--dry-run]
  (--from-net X : reutilise le reseau distille d'un --dry-run precedent)
"""

import argparse
import random
import shutil
import sys
import time

import torch

sys.path.insert(0, ".")

from agents import evolution, expert_bot
from agents.hybrid_bidder import bot_choice_as_id
from engine.bidding import legal_action_ids_with_hand
from engine.bidding import observation as bid_observation
from engine.cards import deal
from engine.whisted import DeckChain
from scripts.distill_hybrid_bidder import new_bidding, train
from scripts.train import POPULATION_FILE_DEFAULT, WHISTED_RATIO_DEFAULT

BOT_SEAT_PROB = 0.5


def collect(others, deals, rng):
    chain = DeckChain(WHISTED_RATIO_DEFAULT, rng)
    cap = deals * 12
    obs = torch.zeros(cap, evolution.OBS_DIM, dtype=torch.float16)
    masks = torch.zeros(cap, evolution.ACTION_DIM, dtype=torch.bool)
    labels = torch.zeros(cap, dtype=torch.long)
    n = 0
    for _ in range(deals):
        hands, dealer, _whisted = chain.deal(rng)
        b = new_bidding(hands, dealer)
        players = {s: None if rng.random() < BOT_SEAT_PROB else evolution.policy_of(rng.choice(others))
                   for s in range(4)}
        while not b.finished:
            seat = b.turn
            hand = hands[seat]
            bot_id = bot_choice_as_id(b, seat, expert_bot.choose_bid(b, seat, hand))
            legal = legal_action_ids_with_hand(b, seat, hand)
            if len(legal) > 1 and n < cap:
                obs[n] = torch.tensor(bid_observation(b, seat, hand), dtype=torch.float16)
                masks[n, legal] = True
                labels[n] = bot_id
                n += 1
            p = players[seat]
            action = bot_id if p is None else p.decide(b, seat, hand)
            assert b.apply(seat, action), (seat, action)
        chain.finish_unplayed(hands, dealer)
    return obs[:n].float(), masks[:n].clone(), labels[:n].clone()


def agreement_with_bot(net, deals, rng):
    same = total = 0
    for _ in range(deals):
        hands = deal(rng)
        b = new_bidding(hands, rng.randrange(4))
        while not b.finished:
            seat = b.turn
            hand = hands[seat]
            bot_id = bot_choice_as_id(b, seat, expert_bot.choose_bid(b, seat, hand))
            legal = legal_action_ids_with_hand(b, seat, hand)
            mask = torch.zeros(evolution.ACTION_DIM, dtype=torch.bool)
            mask[legal] = True
            o = torch.tensor(bid_observation(b, seat, hand), dtype=torch.float32)
            same += net.act_greedy(o, mask) == bot_id
            total += 1
            assert b.apply(seat, bot_id)
    return same / total, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population-file", default=POPULATION_FILE_DEFAULT)
    parser.add_argument("--deals", type=int, default=40000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=927)
    parser.add_argument("--sigma", type=float, default=evolution.MUTATION_SIGMA_DEFAULT)
    parser.add_argument("--dry-run", action="store_true", help="distille et mesure sans reecrire la population")
    parser.add_argument("--pop-size", type=int, default=evolution.POP_SIZE_DEFAULT)
    parser.add_argument("--from-net", help="reseau deja distille (sortie d'un --dry-run), pas de nouvelle distillation")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)

    tmp = args.population_file + ".reseed_read"
    shutil.copyfile(args.population_file, tmp)  # lecture sur copie (le fichier vivant peut etre reecrit)
    population, generation = evolution.load_population(tmp)
    hybrids = [ind for ind in population if ind.lineage == evolution.HYBRID_LINEAGE]
    if not hybrids:
        raise SystemExit("aucun individu hybride dans la population -- rien pour generer les etats visites")

    net = evolution.new_net()
    if args.from_net:
        net.load_state_dict(torch.load(args.from_net, map_location="cpu")["state_dict"])
        acc = float("nan")
    else:
        t0 = time.time()
        with torch.no_grad():
            obs, masks, labels = collect(hybrids, args.deals, rng)
        print("%d decisions etiquetees (population gen %d), %.0fs" % (len(labels), generation, time.time() - t0),
              flush=True)
        acc = train(net, obs, masks, labels, args.epochs, args.seed)
        del obs, masks, labels
    agree, n = agreement_with_bot(net, 3000, random.Random(args.seed + 1))
    print("Accord reseau pur / ExpertBot sur %d decisions de tables de bots : %.2f%% (validation %.2f%%)" % (
        n, 100 * agree, 100 * acc), flush=True)
    if args.dry_run:
        torch.save({"state_dict": net.state_dict()}, args.population_file + ".reseed_net.pt")
        print("--dry-run : population inchangee, reseau -> %s.reseed_net.pt" % args.population_file)
        return

    size = args.pop_size  # PAS len(population) : les lignees supprimees en ont deja ete retirees
    n_plain = size // 2
    best_hybrid = max(hybrids, key=lambda ind: ind.snapshot_fitness())
    new_pop = [evolution.Individual(evolution.clone_net(net), "encherisseur")]
    new_pop += [evolution.Individual(evolution.mutate_net(net, args.sigma), "encherisseur") for _ in range(n_plain - 1)]
    new_pop += hybrids[:size - n_plain]
    while len(new_pop) < size:
        new_pop.append(evolution.Individual(evolution.mutate_net(best_hybrid.net, args.sigma), evolution.HYBRID_LINEAGE))
    backup = "%s.avant_reseed_gen%d" % (args.population_file, generation)
    shutil.copyfile(args.population_file, backup)
    evolution.save_population(args.population_file, new_pop, generation)
    counts = {lin: sum(1 for i in new_pop if i.lineage == lin) for lin in evolution.LINEAGES}
    print("population reecrite (generation %d, %d individus : %s), ancienne -> %s" % (
        generation, len(new_pop), counts, backup))


if __name__ == "__main__":
    main()
