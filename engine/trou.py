"""Trou / Trou Royal : contrat auto-declenche (jamais une decision
d'enchere) quand une main contient 3 ou 4 as, declarant + bouche-trou
(partenaire force, detecte automatiquement) contre 2 defenseurs, avec
atout. Declarant et bouche-trou ont des reseaux DISTINCTS ('trou' et
'bouche_trou') car leur information et leur strategie different
radicalement (voir DESIGN.md).

Regle du 2026-10-01 (demande utilisateur, portee en meme temps dans
whist-4-it) : le BOUCHE-TROU ne choisit plus l'atout avant l'enchere. Le
contrat Trou est seede SANS atout ni niveau (engine/bidding.py:
Bidding.seed_trou), garde son rang 13 sur l'echelle, et c'est le
bouche-trou qui ENTAME le 1er pli (engine/bidding.py:first_leader) : sa
premiere carte fixe le contrat (TeamTrumpDeal._fix_trou_contract) --
carte d'ancrage -> 8 plis dans sa couleur, toute autre carte -> 9 plis
dans la couleur jouee. Carte d'ancrage = l'As manquant (Trou), ou le plus
haut Coeur du bouche-trou (Trou Royal). Ce module DETECTE le declenchement
(detect_trou) et CONSTRUIT le moteur de jeu.
"""

from engine.cards import ACE_RANK, NUM_PLAYERS, rank_of, suit_of
from engine.scoring import trou_result
from engine.team_trump import ACTION_DIM, OBS_DIM, TeamTrumpDeal

__all__ = ["ACTION_DIM", "OBS_DIM", "detect_trou", "final_scores", "new_trou_deal"]

HEART_SUIT = 0


def detect_trou(hands):
    """Retourne {"declarer", "partner", "anchor_suit", "anchor_card",
    "is_royal"} pour la premiere main (par ordre de siege) qui contient 3+
    as, ou None. `anchor_card` = la carte du BOUCHE-TROU qui, jouee en
    entame, fixe le Trou a 8 plis (l'As manquant, ou pour le Trou Royal son
    plus haut Coeur)."""
    for seat in range(NUM_PLAYERS):
        ace_suits = {suit_of(c) for c in hands[seat] if rank_of(c) == ACE_RANK}
        if len(ace_suits) < 3:
            continue
        is_royal = len(ace_suits) == 4
        anchor_suit = HEART_SUIT if is_royal else next(s for s in range(4) if s not in ace_suits)
        partner = -1
        anchor_card = None
        if is_royal:
            best_rank = -1
            for other in range(NUM_PLAYERS):
                if other == seat:
                    continue
                for c in hands[other]:
                    if suit_of(c) == HEART_SUIT and rank_of(c) > best_rank:
                        best_rank = rank_of(c)
                        partner = other
                        anchor_card = c
        else:
            for other in range(NUM_PLAYERS):
                if other == seat:
                    continue
                for c in hands[other]:
                    if suit_of(c) == anchor_suit and rank_of(c) == ACE_RANK:
                        partner = other
                        anchor_card = c
        return {"declarer": seat, "partner": partner, "anchor_suit": anchor_suit, "anchor_card": anchor_card,
                "is_royal": is_royal}
    return None


def new_trou_deal(hands, declarer, partner, anchor_card, leader, royal=False, contract=None):
    """Atout et niveau INCONNUS jusqu'a l'entame du bouche-trou (`leader`
    vaut toujours `partner` dans le vrai jeu, voir first_leader). `contract`
    (dict de l'enchere) recoit "suit"/"level" a la fixation."""
    return TeamTrumpDeal(
        hands, declarer, partner, None, 0, leader,
        declarer_role="trou", partner_role="bouche_trou", defender_role="defenseur",
        contract_type="trou", royal=royal, anchor_card=anchor_card, contract_record=contract,
    )


def final_scores(game):
    value = trou_result(game.level, game.declarer_side_tricks())
    return game.team_value_scores(value)
