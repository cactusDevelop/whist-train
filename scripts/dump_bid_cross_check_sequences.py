"""Genere des sequences d'actions d'enchere JOUEES ET VALIDEES par
engine/bidding.py (hand-agnostic par conception, comme le vrai
game/bidding.gd -- voir ce fichier : "Card-holding is enforced one layer
up, this class stays hand-agnostic by design"), pour verification croisee
REELLE : tests/bid_cross_check_runner.gd (whist-4-it) rejoue chaque
sequence contre le VRAI WhistBidding et verifie que CHAQUE action, deja
validee cote Python, est AUSSI acceptee cote Godot -- et que l'etat final
(type de contrat verrouille / redonne generale) concorde exactement.

C'est une verification de CORRECTION DE LA LEGALITE elle-meme (les deux
moteurs d'enchere sont des implementations INDEPENDANTES des memes
regles), distincte de la parite d'observation/action du reseau (deja
couverte cote jeu par scripts/dump_play_parity_fixtures.py) -- voir
whist-ai/DESIGN.md.

Usage : venv/Scripts/python.exe scripts/dump_bid_cross_check_sequences.py
Ecrit godot_export/bid_cross_check_sequences.json
"""

import json
import os
import random
import sys

sys.path.insert(0, ".")

from engine.bidding import EMBALLAGE_MAX_LEVEL, EMBALLAGE_MIN_LEVEL, Bidding, bid_action_to_dict
from engine.cards import deal
from engine.trou import detect_trou

NUM_SEQUENCES = 300
NUM_PAROLE_SEQUENCES = 100  # orientees passe parole + saut du proposeur (2026-09-27)
MAX_STEPS_PER_SEQUENCE = 200


def random_sequence(rng, seq_id, parole_bias=False):
    dealer = rng.randrange(4)
    b = Bidding(dealer)
    hands = deal(rng)
    trou_info = detect_trou(hands)
    trou_seed = None
    if trou_info is not None:
        b.seed_trou(trou_info)
        # Regle du 2026-10-01 : plus de choix d'atout avant l'enchere -- le
        # Trou est seede sans atout ni niveau (game/bidding.gd::
        # seed_trou_contract(declarer, partner, is_royal, anchor_card)).
        # anchor_card = id de carte 0..51 (suit * 13 + rank, rank 12 = As).
        trou_seed = {
            "declarer": trou_info["declarer"], "partner": trou_info["partner"],
            "anchor_suit": trou_info["anchor_suit"], "anchor_card": trou_info["anchor_card"],
            "is_royal": trou_info["is_royal"],
        }

    steps = []
    guard = 0
    while not b.finished and guard < MAX_STEPS_PER_SEQUENCE:
        guard += 1
        seat = b.turn
        legal = b.legal_action_ids(seat)
        if not legal:
            break
        # Saut du proposeur en passe parole (whist-4-it 9c0dfb3) : "level"
        # n'est emis QUE pour ce saut, jamais pour la relance +1 (contrat
        # du runner Godot : "raise" + "level" -> raise_contract_to).
        jumps = [lv for lv in range(EMBALLAGE_MIN_LEVEL + 1, EMBALLAGE_MAX_LEVEL + 1) if b.can_raise_to(seat, lv)]
        if jumps and rng.random() < 0.5:
            level = rng.choice(jumps)
            steps.append({"seat": seat, "action": {"action": "raise", "level": level}})
            assert b.apply_raise_to(seat, level), "saut juge legal puis refuse -- bug interne"
            continue
        action_id = rng.choice(legal)
        if parole_bias:
            # sequences orientees emballage -> passe parole -> saut, trop
            # rares au hasard pur (3 passe parole sur 300 sequences)
            for kind in ("pass_parole", "emballer", "propose", "raise"):
                preferred = [a for a in legal if bid_action_to_dict(a)["action"] == kind]
                if preferred and rng.random() < 0.8:
                    action_id = rng.choice(preferred)
                    break
        info = bid_action_to_dict(action_id)
        steps.append({"seat": seat, "action": info})
        ok = b.apply(seat, action_id)
        assert ok, "Python a juge %r legal puis l'a refuse -- bug interne" % info

    locked_type = b.locked_contract["type"] if b.locked_contract else None
    return {
        "seq_id": seq_id,
        "dealer_index": dealer,
        "trou_seed": trou_seed,
        "steps": steps,
        "finished": b.finished,
        "general_pass": b.general_pass,
        "locked_contract_type": locked_type,
    }


def main():
    rng = random.Random(20260918)
    sequences = [random_sequence(rng, i) for i in range(NUM_SEQUENCES)]
    sequences += [random_sequence(rng, NUM_SEQUENCES + i, parole_bias=True) for i in range(NUM_PAROLE_SEQUENCES)]
    unfinished = sum(1 for s in sequences if not s["finished"])
    os.makedirs("godot_export", exist_ok=True)
    out_path = os.path.join("godot_export", "bid_cross_check_sequences.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sequences, f)
    print("OK : %d sequences ecrites dans %s (%d non terminees en %d pas, attendu)." % (
        len(sequences), out_path, unfinished, MAX_STEPS_PER_SEQUENCE))


if __name__ == "__main__":
    main()
