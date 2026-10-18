"""Resume PUBLIC de l'enchere, siege par siege -- ce qu'un client Godot a vu
passer pendant l'enchere (bulles d'annonce de chaque joueur), jamais rien de
prive. Alimente l'observation de jeu de la carte (engine/observation.py,
bloc "enchere"). Port GDScript : game/whist_ai_features_v2_play.gd
(`bid_summary_*`), mis a jour par net/whist_table.gd apres chaque action
d'enchere acceptee et apres le choix d'atout d'un Trou.

Par siege ABSOLU :
  - proposed[suit]  : a fait "Proposer" dans cette couleur
  - emballed[suit]  : a fait "Emballer" cette couleur
  - max_rank        : rang (LADDER, 1..19) le plus haut du contrat auquel ce
                      siege a ete lie a un moment de l'enchere (0 = jamais) --
                      couvre solo/abondance/solo chelem/misere/relances/
                      emballage (y compris le proposeur lie par l'emballeur)
                      et Trou.
  - waited / passed / pass_parole / raised : a deja fait cette action.
"""

from engine.cards import NUM_PLAYERS, NUM_SUITS

MAX_RANK = 19  # nombre de marches de engine/bidding.py:LADDER (verifie a l'import de bidding.py)


def new_bid_summary():
    return [{
        "proposed": [False] * NUM_SUITS, "emballed": [False] * NUM_SUITS, "max_rank": 0,
        "waited": False, "passed": False, "pass_parole": False, "raised": False,
    } for _ in range(NUM_PLAYERS)]


def update_bid_summary(summary, bidding, seat, action, suit=None):
    """A appeler APRES chaque action acceptee. `action` : "wait", "pass",
    "pass_parole", "raise", "propose", "emballer", "solo", "abondance",
    "solo_chelem", "misere", "trou" (ces 5 derniers ne laissent
    qu'une trace de rang, lue sur l'etat de l'enchere)."""
    entry = summary[seat]
    if action == "propose":
        entry["proposed"][suit] = True
    elif action == "emballer":
        entry["emballed"][suit] = True
    elif action == "wait":
        entry["waited"] = True
    elif action == "pass":
        entry["passed"] = True
    elif action == "pass_parole":
        entry["pass_parole"] = True
    elif action == "raise":
        entry["raised"] = True
    for s in range(NUM_PLAYERS):
        ci = bidding.lie_contract_index[s]
        if ci != -1:
            rank = bidding.ladder_rank(ci)
            if rank > summary[s]["max_rank"]:
                summary[s]["max_rank"] = rank
