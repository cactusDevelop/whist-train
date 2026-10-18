"""Emballage : contrat d'equipe avec atout, declarant et partenaire connus
des la fin de l'enchere, PARTAGENT le meme reseau 'emballage' (contrairement
a Trou ou le declarant et le partenaire force ont des reseaux distincts,
voir DESIGN.md)."""

from engine.scoring import emballage_or_trou_team_value
from engine.team_trump import ACTION_DIM, OBS_DIM, TeamTrumpDeal

__all__ = ["ACTION_DIM", "OBS_DIM", "final_scores", "new_emballage_deal"]


def new_emballage_deal(hands, declarer, partner, trump_suit, level, leader):
    return TeamTrumpDeal(
        hands, declarer, partner, trump_suit, level, leader,
        declarer_role="emballage", partner_role="emballage", defender_role="defenseur",
    )


def final_scores(game):
    value = emballage_or_trou_team_value(game.level, game.declarer_side_tricks())
    return game.team_value_scores(value)
