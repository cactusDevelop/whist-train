"""Solitaire : contrats individuels POSITIFS -- Solo/Abondance/Solo Chelem
sont UN SEUL contrat interne (niveau 6-13, voir engine/bidding.py), joue
et relance de la meme facon quel que soit le point d'entree. Seule la
formule de score differe par NIVEAU (pas par type) : gradue en dessous de
9, plat a partir de 9 -- voir engine/scoring.py, verifie contre
game/scoring.gd (`_score_solo_family()`, projet Godot).
"""

from engine.bidding import ABONDANCE_MIN_LEVEL
from engine.lone_declarer import ACTION_DIM, LoneDeclarerDeal, OBS_DIM
from engine.scoring import abondance_result, apply_solo_result, solo_base_value

__all__ = ["ACTION_DIM", "OBS_DIM", "final_scores", "new_solitaire_deal"]

NUM_TRICKS = 13


def new_solitaire_deal(hands, declarer, trump_suit, level, leader):
    return LoneDeclarerDeal(
        hands, [declarer], trump_suit, NUM_TRICKS, leader,
        declarer_role="solitaire", defender_role="defenseur",
        contract_type="solo", level=level,
    )


def final_scores(game, level):
    declarer = game.declarers[0]
    tricks = game.tricks_won[declarer]
    if level >= ABONDANCE_MIN_LEVEL:
        result = abondance_result(level, tricks)
    else:
        result = apply_solo_result(solo_base_value(level, tricks))
    scores = [0.0] * 4
    for s in range(4):
        scores[s] = float(result["soloist"]) if s == declarer else float(result["opponent"])
    return scores
