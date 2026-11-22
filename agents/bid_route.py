"""Routage d'enchere (15e conseil, etape 2, 2026-11-14) : ExpertBot encherit,
sauf dans les CELLULES (categorie ExpertBot, categorie reseau) ou
l'encherisseur reseau (lignee exportee + regle d'abondance + garde-fous,
agents/abondance_rule.RuledBidder) a ete mesure meilleur.

Cellules choisies par scripts/free_match.py --bid-cells (whist-train
43518097264, selection graine 28261113 : gain - 2 se > 0, n >= 200) et
confirmees sur les graines 29261113 + 30261113 x 7 champs (-, expert,
bruite10, bruite25, acte1, joueur_0927, joueur_1027) : +2.1 a +4.5 pts/donne,
z +28 a +70. Port Godot : WhistAIBridgeV2.BID_ROUTE / route_bid_action.

22/11 : le routage ne joue l'emballage du reseau que si net_emballer_ok (sinon
il coule le proposeur : -1.31 pt/cas). DECISION UTILISATEUR (choix de difficulte)
apres l'A/B a50 rejete au seul critere 5 : humain simule -0.307, seuil -0.30.
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


def route(expert_choice, net_choice, hand, cells=BID_ROUTE):
    """Decision jouee : celle du reseau si elle differe, que la cellule est routee et, pour un emballage,
    que la main apporte assez au proposeur (net_emballer_ok)."""
    if bid_label(net_choice) != bid_label(expert_choice) and \
            (bid_label(expert_choice, False), bid_label(net_choice, False)) in cells:
        if bid_label(net_choice, False) == "emballer" and not net_emballer_ok(
                hand, bid_action_to_dict(net_choice)["suit"]):
            return expert_choice
        return net_choice
    return expert_choice


def net_emballer_ok(hand, suit):
    """a50 (22/11) : l'emballeur route apporte-t-il assez a son partenaire ?
    >= 3 atouts et >= 3 plis surs, ou >= 4 atouts et >= 2 plis surs.
    Adopte par l'utilisateur le 22/11 (A/B a50 rejete au critere 5, voir l'en-tete)."""
    trumps = sum(1 for c in hand if c // 13 == suit)
    sure = expert_bot.estimate_tricks(hand, suit)
    return (trumps >= 3 and sure >= 3) or (trumps >= 4 and sure >= 2)
