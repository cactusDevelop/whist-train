"""Misere : contrats individuels NEGATIFS (petite/piccolo/grande/etalee) --
UN A QUATRE sieges peuvent declarer la MEME famille simultanement, chacun
pour soi (voir WHIST-RULES-REFERENCE.txt) -- sans atout, contre les
sieges restants ("defenseur", partage avec les autres familles). Voir
DESIGN.md et engine/scoring.py:misere_result (depend de n=nombre de
declarants et k=nombre de reussites), verifie contre game/scoring.gd.

Depuis la Phase 1 (2026-09-19, voir DESIGN.md), chaque sous-type a son
PROPRE reseau specialiste (plus un seul role "misere" partage) -- leurs
objectifs sont trop differents pour un seul cerveau : piccolo doit prendre
EXACTEMENT 1 pli (positif jusqu'a 1, puis negatif au-dela), les 3 autres
doivent en prendre 0 (purement negatif). DECLARER_ROLE_BY_KIND route
chaque kind vers son role.

Quirks de regle geres ici :
- petite misere se joue a 12 cartes par joueur : la carte ecartee est une
  vraie decision de chaque siege (jamais au hasard), voir
  engine/lone_declarer.py:LoneDeclarerDeal (needs_discard).
- etalee expose la main de CHAQUE declarant a tous des le debut.
"""

from engine.lone_declarer import ACTION_DIM, LoneDeclarerDeal, OBS_DIM
from engine.scoring import MISERE_HAND_SIZE, MISERE_MAX_TRICKS, MISERE_ORDER, misere_result

__all__ = ["ACTION_DIM", "DECLARER_ROLE_BY_KIND", "KINDS", "OBS_DIM", "final_scores", "new_misere_deal"]

KINDS = MISERE_ORDER

# Un role specialiste distinct par sous-type (voir docstring ci-dessus) --
# "petite" -> petite_misere, "grande" -> grande_misere, "etalee" ->
# misere_etalee, "piccolo" -> piccolo (deja un nom distinctif).
DECLARER_ROLE_BY_KIND = {
    "petite": "petite_misere",
    "piccolo": "piccolo",
    "grande": "grande_misere",
    "etalee": "misere_etalee",
}


def new_misere_deal(hands, declarers, kind, leader):
    """`hands` : mains completes a 13 cartes telles que distribuees et vues
    pendant l'enchere -- l'ecart de la petite misere se fait APRES, comme
    une vraie decision de jeu (voir needs_discard), jamais avant."""
    return LoneDeclarerDeal(
        hands, declarers, trump_suit=None, num_tricks=MISERE_HAND_SIZE[kind], leader=leader,
        declarer_role=DECLARER_ROLE_BY_KIND[kind], defender_role="defenseur",
        reveal_declarer_hands=(kind == "etalee"), needs_discard=(kind == "petite"),
        contract_type="misere", kind=kind,
    )


def final_scores(game, kind):
    """Reussite = EXACTEMENT max_tricks plis, pas "au plus" -- crucial pour
    Piccolo (max_tricks=1) : prendre 0 pli echoue tout autant que 2+ (voir
    net/whist_table.gd, commentaire explicite). Pour petite/grande/etalee
    (max_tricks=0) ca ne change rien (==0 equivaut a <=0)."""
    max_tricks = MISERE_MAX_TRICKS[kind]
    n = len(game.declarers)
    k = sum(1 for d in game.declarers if game.tricks_won[d] == max_tricks)
    result = misere_result(kind, n, k)
    scores = [0.0] * 4
    for s in range(4):
        if s in game.declarers:
            won = game.tricks_won[s] == max_tricks
            scores[s] = float(result["winner"] if won else result["loser"])
        else:
            scores[s] = float(result["opponent"])
    return scores
