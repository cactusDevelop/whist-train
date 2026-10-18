"""Smoke test Phase 1 (voir DESIGN.md et scripts/train.py) : pour chacune
des 6 familles de declarant, verifie que la recompense produite par
scripts.train.compute_scores() EGALE une formule attendue recalculee
depuis l'etat reel du jeu (tricks_won post-partie), pas un nombre fige a
l'avance -- l'issue d'une partie depend du reseau (encore aleatoire a
l'initialisation), seule la FORMULE est verifiee, jamais un resultat
particulier. Cote DECLARANT, la formule (distance en plis) est
reimplementee INDEPENDAMMENT ici. Cote DEFENSEUR (bareme de points reel
depuis 2026-09-19, voir DESIGN.md), ce test appelle directement
engine.misere/solitaire/emballage.final_scores -- les MEMES fonctions que
compute_scores() -- puisque ce bareme est deja verifie independamment par
scripts/test_engine.py (12738 assertions contre WHIST-RULES-REFERENCE.txt) ;
ici on verifie le CABLAGE (bon kind/level par famille, bonne moitie du
tableau utilisee), pas une 3e reimplementation du bareme.

Verifie aussi que PPO tourne sans crash pour les 7 roles sur le lot
collecte, et que le roster couvre a la fois des configurations a 2
defenseurs (emballage) et a 3 defenseurs (les autres).

Ne met a jour aucun checkpoint sur disque (nets ephemeres, jamais
sauvegardes) -- ceci est une VERIFICATION, pas un entrainement.
"""

import random
import sys

import torch

sys.path.insert(0, ".")

from agents import ppo
from agents.network import PolicyValueNet
from agents.specialists import IMPLEMENTED, PHASE1_ROLES
from engine.scoring import MISERE_MULTIPLIER_TABLE, MISERE_UNIT
from engine.cards import deal
from engine.emballage import final_scores as emballage_final_scores
from engine.misere import final_scores as misere_final_scores
from engine.solitaire import final_scores as solitaire_final_scores
from scripts.train import (
    DECLARER_FAMILIES, MISERE_FAMILIES, SOLITAIRE_LEVEL_PLACEHOLDER,
    build_game, compute_scores, play_out, record,
)

NUM_DEALS_PER_FAMILY = 30
TOLERANCE = 1e-9


def fresh_nets():
    nets = {}
    for role in PHASE1_ROLES:
        obs_dim, action_dim = IMPLEMENTED[role]
        net = PolicyValueNet(obs_dim, action_dim)
        nets[role] = net
        nets[role + "_optimizer"] = torch.optim.Adam(net.parameters(), lr=3e-4)
    return nets


def independent_expected_scores(family, game, declarer_seats):
    """Reimplementation DELIBEREMENT independante de compute_scores() cote
    DECLARANT (boucle explicite par siege, pas de scission equipe/
    individuel partagee) -- voir docstring de module pour le cote
    DEFENSEUR (bareme de points reel, verifie par appel direct aux
    fonctions engine.*.final_scores, pas reimplemente ici)."""
    tricks = list(game.tricks_won)
    declarer_tricks_sum = sum(tricks[s] for s in declarer_seats)
    expected = [0.0] * 4
    if family in ("solitaire", "emballage"):
        team_reward = float(declarer_tricks_sum)  # pas de cible Phase 1 : maximiser
        for seat in declarer_seats:
            expected[seat] = team_reward
    else:  # vrais points depuis le 2026-10-02 : piccolo = EXACTEMENT 1 pli, sinon 0 pli
        kind = MISERE_FAMILIES[family]
        target = 1 if family == "piccolo" else 0
        made = {seat: tricks[seat] == target for seat in declarer_seats}
        table = MISERE_MULTIPLIER_TABLE[(len(declarer_seats), sum(made.values()))]
        for seat in declarer_seats:
            expected[seat] = float(MISERE_UNIT[kind] * table["winner" if made[seat] else "loser"])

    if family == "solitaire":
        defense = solitaire_final_scores(game, SOLITAIRE_LEVEL_PLACEHOLDER)
    elif family == "emballage":
        defense = emballage_final_scores(game)
    else:
        defense = misere_final_scores(game, MISERE_FAMILIES[family])
    for seat in range(4):
        if seat not in declarer_seats:
            expected[seat] = defense[seat]
    return expected


def check_reward_formulas(nets, rng):
    total_deals_total_tricks = {}
    for family in DECLARER_FAMILIES:
        for _ in range(NUM_DEALS_PER_FAMILY):
            hands = deal(rng)
            game, declarer_seats = build_game(family, hands, rng)
            play_out(game, nets)
            got = compute_scores(family, game, declarer_seats)
            want = independent_expected_scores(family, game, declarer_seats)
            for seat in range(4):
                if abs(got[seat] - want[seat]) > TOLERANCE:
                    raise AssertionError(
                        "famille=%r siege=%d : compute_scores=%r != attendu=%r "
                        "(declarers=%r tricks_won=%r)" % (
                            family, seat, got[seat], want[seat], sorted(declarer_seats), game.tricks_won)
                    )
            total_deals_total_tricks.setdefault(family, []).append(sum(game.tricks_won))
    for family, totals in total_deals_total_tricks.items():
        expected_total = 12 if family == "petite_misere" else 13
        bad = [t for t in totals if t != expected_total]
        if bad:
            raise AssertionError("famille=%r : total de plis attendu=%d, observe=%r" % (
                family, expected_total, bad))
    print("OK : formules de recompense verifiees independamment pour %s (%d donnes/famille)." % (
        DECLARER_FAMILIES, NUM_DEALS_PER_FAMILY))


def check_ppo_runs_for_all_roles(nets, rng):
    buffers = {role: {"obs": [], "mask": [], "action": [], "logp": [], "ret": [], "adv": []} for role in PHASE1_ROLES}
    role_returns = {role: [] for role in PHASE1_ROLES}
    defender_counts_seen = set()
    for family in DECLARER_FAMILIES:
        for _ in range(NUM_DEALS_PER_FAMILY):
            hands = deal(rng)
            game, declarer_seats = build_game(family, hands, rng)
            steps = play_out(game, nets)
            record(buffers, role_returns, steps, game,
                   lambda g, f=family, d=declarer_seats: compute_scores(f, g, d))
            defender_counts_seen.add(4 - len(declarer_seats))

    missing = [role for role in PHASE1_ROLES if not buffers[role]["obs"]]
    if missing:
        raise AssertionError("roles sans aucune decision collectee (roster incomplet) : %r" % missing)

    for role in PHASE1_ROLES:
        buf = buffers[role]
        stats = ppo.update(
            nets[role], nets[role + "_optimizer"],
            buf["obs"], buf["mask"], buf["action"], buf["logp"], buf["ret"], buf["adv"],
        )
        print("  %-14s decisions=%-5d ret_moy=%+7.3f ppo_stats=%s" % (
            role, len(buf["obs"]), sum(role_returns[role]) / len(role_returns[role]), stats))

    if {2, 3} - defender_counts_seen:
        raise AssertionError(
            "defenseur n'a pas vu les deux configurations (2 ET 3 adversaires) sur ce lot : "
            "configurations observees=%r" % defender_counts_seen)
    print("OK : PPO tourne sans crash pour les 7 roles ; defenseur a vu des camps "
          "de 2 ET 3 adversaires (%r)." % sorted(defender_counts_seen))


def main():
    rng = random.Random(0)
    nets = fresh_nets()
    check_reward_formulas(nets, rng)
    check_ppo_runs_for_all_roles(nets, rng)
    print("PHASE1_SMOKE_OK")


if __name__ == "__main__":
    main()
