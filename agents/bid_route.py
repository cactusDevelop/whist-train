"""Routage d'enchere (15e conseil, etape 2, 2026-11-14) : ExpertBot encherit,
sauf dans les CELLULES (categorie ExpertBot, categorie reseau) ou
l'encherisseur reseau (lignee exportee + regle d'abondance + garde-fous,
agents/abondance_rule.RuledBidder) a ete mesure meilleur.

Cellules choisies par scripts/free_match.py --bid-cells (whist-train
43518097264, selection graine 28261113 : gain - 2 se > 0, n >= 200) et
confirmees sur les graines 29261113 + 30261113 x 7 champs (-, expert,
bruite10, bruite25, acte1, joueur_0927, joueur_1027) : +2.1 a +4.5 pts/donne,
z +28 a +70. Port Godot : WhistAIBridgeV2.BID_ROUTE / route_bid_action.
"""

from agents import expert_bot
from engine.bidding import bid_action_to_dict

# (categorie ExpertBot, categorie reseau), libelles de bid_label(..., suit=False)
BID_ROUTE = frozenset([
    ("emballer", "misere_grande"), ("emballer", "misere_piccolo"), ("emballer", "solo_6"),
    ("misere_petite", "misere_grande"), ("misere_piccolo", "misere_grande"),
    ("pass", "emballer"), ("pass", "misere_grande"), ("pass", "misere_piccolo"), ("pass", "solo_6"),
    ("propose", "emballer"), ("propose", "misere_grande"), ("propose", "misere_piccolo"),
    ("propose", "propose"), ("propose", "solo_6"),
    ("raise+1", "pass"),
    ("solo_6", "misere_grande"), ("solo_6", "misere_piccolo"), ("solo_6", "solo_6"),
    ("solo_7", "pass"), ("solo_8", "pass"),
])


def bid_label(choice, suit=True):
    """Categorie d'une decision : id d'action, (RAISE_TO, niveau) ou (RAISE_KIND, misere)."""
    if isinstance(choice, tuple):
        if choice[0] == expert_bot.RAISE_KIND:
            return "raise_misere_%s" % choice[1]
        return "raise_to_%d" % choice[1]
    d = bid_action_to_dict(choice)
    a = d["action"]
    if a == "raise":
        return "raise_to_%d" % d["level"] if "level" in d else "raise+1"
    if a == "misere":
        return "misere_%s" % d["kind"]
    if "level" in d:
        a = "%s_%d" % (a, d["level"])
    if suit and "suit" in d:
        a = "%s/%d" % (a, d["suit"])
    return a


def route(expert_choice, net_choice, cells=BID_ROUTE):
    """Decision jouee : celle du reseau si elle differe et que la cellule est routee."""
    if bid_label(net_choice) != bid_label(expert_choice) and \
            (bid_label(expert_choice, False), bid_label(net_choice, False)) in cells:
        return net_choice
    return expert_choice
