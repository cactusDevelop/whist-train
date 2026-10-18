"""Cas de parite de l'ENCHERE de l'IA exportee (2026-09-27, demande whist-4-it
pour un runner de parite d'enchere, apres l'ajout des sauts du passe parole,
ids 52..55).

Pour chaque lignee EXPORTEE (evolution.EXPORTED_LINEAGES, meilleur individu =
celui publie dans game/), des etats d'enchere reels : donneur, mains de la
donne, trou eventuel, journal des actions deja jouees (ids reseau, que le
runner Godot REJOUE dans un vrai WhistBidding, comme les fixtures de jeu),
siege, main, observation attendue (floats), ids legaux attendus (avec le
filtre main) et action greedy attendue du reseau.

Pour obtenir des sauts (rares en jeu naturel), une partie des decisions est
remplacee par un coup aleatoire oriente passe parole / emballage ; le cas
enregistre reste toujours la decision du RESEAU dans l'etat atteint.

Usage : venv/Scripts/python.exe scripts/dump_bid_parity_fixtures.py [--population P] [--out PATH]
A lancer juste apres un export : les poids doivent etre ceux publies dans game/.
"""

import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, ".")

from agents import evolution
from scripts.export_to_godot import fp16_rounded
from engine.bidding import (Bidding, _RAISE_TO_BASE, bid_action_to_dict, legal_action_ids_with_hand,
                            legal_action_mask_with_hand, observation)
from engine.cards import deal
from engine.trou import detect_trou

CASES_PER_LINEAGE = 150
JUMP_CASES_MIN = 20
OFF_POLICY_PROB = 0.35


def biased_random(rng, bidding, seat, hand):
    legal = legal_action_ids_with_hand(bidding, seat, hand)
    for kind in ("pass_parole", "emballer", "propose", "raise"):
        pref = [a for a in legal if bid_action_to_dict(a)["action"] == kind]
        if pref and rng.random() < 0.8:
            return rng.choice(pref)
    return rng.choice(legal)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", default=os.path.join("checkpoints_v2", "encherisseur_population.pt"))
    parser.add_argument("--out", default=os.path.join("godot_export", "bid_parity_cases.json"))
    args = parser.parse_args()
    rng = random.Random(20260927)
    population, generation = evolution.load_population(args.population)
    best = evolution.best_per_lineage(population)
    cases = []
    for lineage in evolution.EXPORTED_LINEAGES:
        net = fp16_rounded(best[lineage].net)  # poids tels que publies (float16)
        n_cases = n_jumps = deals = 0
        while (n_cases < CASES_PER_LINEAGE or n_jumps < JUMP_CASES_MIN) and deals < 200000:
            deals += 1
            hands = deal(rng)
            dealer = rng.randrange(4)
            b = Bidding(dealer)
            trou = detect_trou(hands)
            if trou is not None:
                b.seed_trou(trou)
            log = []
            while not b.finished:
                seat = b.turn
                hand = hands[seat]
                obs = observation(b, seat, hand)
                mask = legal_action_mask_with_hand(b, seat, hand)
                with torch.no_grad():
                    action = net.act_greedy(torch.tensor(obs, dtype=torch.float32), torch.tensor(mask))
                legal = [i for i, ok in enumerate(mask) if ok]
                has_jump = any(i >= _RAISE_TO_BASE for i in legal)
                if (has_jump and n_jumps < JUMP_CASES_MIN) or (n_cases < CASES_PER_LINEAGE and rng.random() < 0.08):
                    cases.append({
                        "lineage": lineage, "individual": best[lineage].id, "dealer": dealer,
                        "deal_hands": [sorted(h) for h in hands], "trou": trou,
                        "bid_log": [list(x) for x in log], "seat": seat, "hand": sorted(hand),
                        "expected_obs": obs, "expected_legal": legal, "expected_action": action,
                        "has_jump": has_jump})
                    n_cases += 1
                    n_jumps += has_jump
                if rng.random() < OFF_POLICY_PROB:
                    action = biased_random(rng, b, seat, hand)
                assert b.apply(seat, action), (seat, action)
                log.append((seat, action))
        print("%-20s individu #%d : %d cas dont %d avec saut legal (%d donnes)" % (
            lineage, best[lineage].id, n_cases, n_jumps, deals))
        if n_jumps < JUMP_CASES_MIN:
            raise SystemExit("pas assez de cas avec saut pour %s" % lineage)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"population_generation": generation, "action_dim": evolution.ACTION_DIM, "cases": cases}, f)
    print("OK : %d cas ecrits dans %s" % (len(cases), args.out))


if __name__ == "__main__":
    main()
