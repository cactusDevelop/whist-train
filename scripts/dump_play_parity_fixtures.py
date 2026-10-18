"""Genere les cas de parite du PORT GDScript de engine/observation.py
(observation v3) contre les reseaux PyTorch reels (checkpoints_v2/).

Les etats ne sont PAS inventes : ils viennent de donnes reellement jouees
(Phase 1 toutes familles + Phase 2 avec une vraie enchere de la population
d'encherisseurs, Trou compris), suite de donnes 50% whistees. Pour chaque
cas : les entrees PUBLIQUES brutes (main, journal (siege, carte), donneur,
contrat public, mains etalees, mon ecart, phase d'ecart, whiste), le
journal des actions d'enchere de la donne (le runner Godot le REJOUE dans
un vrai WhistBidding et doit retrouver le meme resume d'annonces),
l'observation Python attendue (indices des 1 -- toutes les valeurs sont 0
ou 1), la legalite et l'action greedy du vrai reseau du role.

tests/ai_bridge_parity_v2_play_runner.gd (whist-4-it) reconstruit tout en
GDScript et verifie l'egalite stricte.

Usage : venv/Scripts/python.exe scripts/dump_play_parity_fixtures.py [--out PATH]
Ecrit godot_export/play_parity_cases.json par defaut.
"""

import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, ".")

from agents.play_guards import ai_play_mask
from agents import evolution, specialists
from agents.specialists import PLAY_ROLES
from scripts.export_to_godot import fp16_rounded
from engine.trou import detect_trou
from engine.whisted import DeckChain
from scripts.train import (DECLARER_FAMILIES, POPULATION_FILE_DEFAULT, build_game, dispatch_play_phase2,
                           run_bidding_phase)

CASES_PER_ROLE = 40
SAMPLE_PROB = 0.15
GUARD_CASES = 40  # cas ou le garde-fou de jeu retire des cartes (agents/play_guards.py)


def case_from(game, seat, nets, bid_log, hands_at_deal, phase):
    obs = game.observation(seat)
    assert all(v in (0.0, 1.0) for v in obs)
    mask = game.legal_action_mask(seat)
    legal = [c for c in range(52) if mask[c]]
    guarded = ai_play_mask(game, seat)  # garde-fous de jeu (agents/play_guards.py)
    role = game.role_of(seat)
    with torch.no_grad():
        action = nets[role].act_greedy(torch.tensor(obs, dtype=torch.float32), torch.tensor(guarded, dtype=torch.bool))
    revealed = {}
    if getattr(game, "reveal_declarer_hands", False):
        revealed = {str(d): sorted(game.hands[d]) for d in game.declarers}
    ct = game.public_contract()
    return {
        "role": role, "phase": phase, "seat": seat, "hand": sorted(game.hands[seat]),
        "play_log": [[s, c] for s, c in game.play_log], "dealer": game.dealer,
        "contract": {"type": ct["type"], "trump": ct["trump"], "level": ct["level"],
                     "kind": ct["kind"] or "", "declarers": ct["declarers"],
                     "partner": -1 if ct["partner"] is None else ct["partner"], "royal": bool(ct["royal"])},
        "bid_log": bid_log, "deal_hands": hands_at_deal,
        "trou": (detect_trou(hands_at_deal) if bid_log is not None else None),
        "expected_bid_summary": game.bid_summary,
        "revealed_hands": revealed,
        "own_discard": getattr(game, "discards", {}).get(seat, -1),
        "is_discard_phase": bool(getattr(game, "pending_discards", [])),
        "whisted": bool(game.whisted),
        "expected_ones": [i for i, v in enumerate(obs) if v == 1.0], "obs_dim": len(obs),
        "expected_legal": legal, "expected_guarded_legal": [c for c in range(52) if guarded[c]],
        "expected_action_id": int(action),
    }


def play_and_sample(game, nets, rng, bid_log, hands_at_deal, phase, counts, cases):
    while not game.done:
        seat = game.turn
        role = game.role_of(seat)
        if counts.get(role, 0) < CASES_PER_ROLE and rng.random() < SAMPLE_PROB:
            cases.append(case_from(game, seat, nets, bid_log, hands_at_deal, phase))
            counts[role] = counts.get(role, 0) + 1
        elif counts.get("_garde", 0) < GUARD_CASES and ai_play_mask(game, seat) != game.legal_action_mask(seat):
            cases.append(case_from(game, seat, nets, bid_log, hands_at_deal, phase))
            counts["_garde"] = counts.get("_garde", 0) + 1
        obs = torch.tensor(game.observation(seat), dtype=torch.float32)
        mask = torch.tensor(ai_play_mask(game, seat), dtype=torch.bool)
        with torch.no_grad():
            action, _l, _v = nets[role].act(obs, mask)
        game.step(seat, action)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join("godot_export", "play_parity_cases.json"))
    args = parser.parse_args()
    rng = random.Random(2026)
    torch.manual_seed(2026)
    nets = {r: fp16_rounded(specialists.load_or_create(r)[0]) for r in PLAY_ROLES}  # poids tels que publies (float16)
    population, _g = evolution.load_population(POPULATION_FILE_DEFAULT)
    chain = DeckChain(0.5, rng)
    counts, cases = {}, {}
    cases = []
    deals = 0
    while (any(counts.get(r, 0) < CASES_PER_ROLE for r in PLAY_ROLES) or counts.get("_garde", 0) < GUARD_CASES)             and deals < 60000:
        deals += 1
        hands, dealer, whisted = chain.deal(rng)
        hands_at_deal = [sorted(h) for h in hands]
        if deals % 2:
            family = rng.choice(DECLARER_FAMILIES)
            game, _ = build_game(family, hands, rng, dealer, whisted)
            play_and_sample(game, nets, rng, None, hands_at_deal, 1, counts, cases)
            chain.finish_played(game, dealer)
            continue
        chosen = rng.sample(population, 4)
        bidding, log = run_bidding_phase({s: evolution.policy_of(chosen[s]) for s in range(4)}, hands, dealer)
        if bidding.general_pass:
            chain.finish_unplayed(hands, dealer)
            continue
        game = dispatch_play_phase2(bidding.locked_contract, hands, dealer, bidding.bid_summary, whisted)
        play_and_sample(game, nets, rng, [[s, a] for s, a in log], hands_at_deal, 2, counts, cases)
        chain.finish_played(game, dealer, bidding.locked_contract["type"])
    missing = [r for r in PLAY_ROLES if counts.get(r, 0) < CASES_PER_ROLE]
    if counts.get("_garde", 0) < GUARD_CASES:
        missing.append("_garde")
    if missing:
        raise SystemExit("roles sans assez de cas apres %d donnes : %r" % (deals, missing))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cases, f)
    print("OK : %d cas ecrits dans %s (%d par role, %d donnes jouees)." % (
        len(cases), args.out, CASES_PER_ROLE, deals))


if __name__ == "__main__":
    main()
