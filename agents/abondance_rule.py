"""Regle d'ABONDANCE de l'IA exportee (decision utilisateur 2026-10-03 soir,
apres l'echec de l'apprentissage : imitation 0f1913e revertee, deblocage de
la sortie abondance sans gain).

Mesure (scripts/big_contracts_bench.py, 1000 mains du banc, deck whiste,
abondance imposee jouee par l'IA) : a 10 plis surs, l'abondance-9 rapporte
+28.6 au soliste (98 % de reussite), a 11, l'abondance-10 +35.5 (90 %) ;
l'IA greedy n'en annoncait que 5-6 % et faisait +13/+14 (solo-7/8, trou,
emballage). Au niveau plis surs + 1 : 21-25 % de reussite, -23 a -90. A 9 plis
surs, le solo rapporte plus que l'abondance-9 : la regle ne joue pas.

Regle : plis surs estimes (expert_bot.estimate_tricks, meilleure couleur,
egalite -> couleur d'indice le plus haut) >= ABONDANCE_RULE_MIN_TRICKS et
abondance (meilleure couleur, niveau min(plis surs - 1, 12)) legale -> on
l'annonce ; sinon le reseau decide. Portee a l'identique dans whist-4-it
(game/whist_ai_bridge.gd, decide_bid_action) -- toute modification ici doit
y etre reportee."""

from agents.expert_bot import estimate_tricks, misere_slack
from engine.bidding import (ABONDANCE_MAX_LEVEL, ABONDANCE_MIN_LEVEL, ACTION_RAISE, bid_action_to_dict,
                            legal_action_ids_with_hand)
from engine.cards import NUM_SUITS, suit_of
from engine.scoring import MISERE_ORDER

ABONDANCE_RULE_MIN_TRICKS = 10

# Garde-fou des SOLOS (decision utilisateur 2026-10-04, "attaque les solos 8
# et 9") : l'IA sur-annoncait sur des mains faibles (solo-8 a 7 plis surs :
# 50 % de reussite, -2.5 contre +1.4 pour l'enchere ExpertBot ; solo-9 a 7 :
# 24 %, -15.9). A/B sur 6000 donnes 4 x IA greedy, memes mains : interdire un
# solo de niveau >= 8 au-dessus des plis surs de sa couleur (abondance : plis
# surs + 1) change 3.6 % des donnes, +3.3 pts au siege concerne a chaque fois.
# Plus large (des le niveau 7) : guere mieux (+0.14 vs +0.12 par donne) pour
# 2.5 x plus d'interventions ; des le solo-6 : perd (ces petits solos poussent
# l'adversaire a un emballage plus cher).
SOLO_CAP_MIN_LEVEL = 8
ABONDANCE_CAP_MARGIN = 1

# Garde-fou de l'EMBALLEUR (decision utilisateur 2026-10-05, "il m'a emballe
# avec un seul atout !") : 3000 donnes 4 x IA greedy, l'IA emballait avec un
# seul atout 328 fois (ExpertBot pareil 15 % du temps), 44 % de reussite,
# -2.2 pts. A/B memes mains, 4 x IA greedy : interdire l'emballage avec
# <= 1 atout -> +3.7 +- 1.1 pts pour le duo proposeur + emballeur a chaque
# intervention (10 000 donnes, 1010 cas ; ~10 % des donnes) ; l'emballeur
# seul : neutre. Plus large (<= 2 atouts, ou sous les plis surs d'ExpertBot)
# -> PERD pour l'emballeur (-1.2 / -1.6 par intervention, 3000 donnes).
EMBALLER_MIN_TRUMPS = 2

# ... et sans honneurs (signalement 2026-10-06 09:03, « comportement de bot
# affreux » : emballage 8 sans honneur, 2 petits atouts) : l'IA emballait les
# mains a 0-4 points d'honneur (A 4, R 3, D 2, V 1) 37 a 58 % du temps (25 a
# 35 % de reussite), l'ExpertBot jamais (scripts/gates/
# check_signalement_20261006.py --emballage). A/B memes mains, 4 x IA greedy,
# 4000 donnes : interdire d'emballer avec <= 4 points ET <= 3 atouts -> +7.0
# +- 1.6 pts pour le duo par intervention (256 cas), emballeur +0.6 +- 1.1 ;
# <= 4 points sans condition d'atouts : +5.2 ; <= 6 points / <= 3 atouts :
# +3.6 mais l'emballeur perd (-1.2).
EMBALLER_WEAK_MAX_HCP = 4
EMBALLER_WEAK_MAX_TRUMPS = 3
_HCP = {12: 4, 11: 3, 10: 2, 9: 1}

# Garde-fou de la GRANDE MISERE (controle horaire 2026-10-08 21:13) : l'IA
# greedy annoncait la grande misere ~5 % des donnes a 34 % de reussite
# (-11.8 pts/declarant, 4000 donnes 4 x IA). Par ecart de la main
# (expert_bot.misere_slack) : <= 5 -> 62-100 %, positif ; >= 6 -> 0-22 %,
# -20 a -36 (130 des 193 annonces). L'ExpertBot s'arrete a 3 (calibre avec
# son propre jeu de la carte) ; le jeu de l'IA tient jusqu'a 5.
GRANDE_MISERE_MAX_SLACK = 5

# Garde-fou des EMBALLAGES HAUTS (controle horaire 2026-10-09 23:00, demande
# utilisateur "corrige") : en entrainement, emballage-11 26 % et -12 12 % de
# reussite. Tous atteints en RELANCANT son propre emballage (jamais en entree
# directe : celle-ci a deja sa garde, bid_ppo.emballer_guard_target). Le
# bareme ne paie pas plus un emballage plus haut a plis egaux : la relance ne
# sert qu'a passer au-dessus de l'adversaire. A/B memes mains (30 000 donnes,
# encheres echantillonnees, un camp ExpertBot une donne sur deux) : interdire la
# relance (+1 ou saut) d'un emballage au-dessus de EMBALLAGE_RAISE_MAX_LEVEL ->
# -0.4 +- 1.1 pts pour le duo par intervention (138 cas) : NEUTRE en points
# (l'adversaire garde alors son contrat : solo-8, emballage-10...), mais plus
# de contrat volontaire sous 30 % (regle utilisateur du 2026-09-27). Pas de
# seuil de plis surs qui se degage (bruit).
EMBALLAGE_RAISE_MAX_LEVEL = 10


def abondance_rule(bidding, seat, hand):
    """-> id d'action d'abondance a annoncer, ou None (le reseau decide)."""
    est, suit = max((estimate_tricks(hand, t), t) for t in range(NUM_SUITS))
    if est < ABONDANCE_RULE_MIN_TRICKS:
        return None
    level = min(est - 1, ABONDANCE_MAX_LEVEL)
    for a in legal_action_ids_with_hand(bidding, seat, hand):
        info = bid_action_to_dict(a)
        if info["action"] == "abondance" and info["suit"] == suit and info["level"] == level:
            return a
    return None


def solo_too_high(bidding, seat, hand, action_id):
    """Annonce solo/abondance/solo chelem, ou relance (+1) de son propre solo, a un niveau
    >= SOLO_CAP_MIN_LEVEL au-dessus des plis surs de la couleur (abondance :
    plis surs + ABONDANCE_CAP_MARGIN) ?"""
    info = bid_action_to_dict(action_id)
    if info["action"] in ("solo", "abondance", "solo_chelem"):
        suit, level = info["suit"], info.get("level", 13)  # solo chelem : 13 plis (oublie avant le 2026-10-11)
    elif action_id == ACTION_RAISE and bidding.lie_contract_index[seat] != -1:
        c = bidding.contracts[bidding.lie_contract_index[seat]]
        if c["type"] != "solo":
            return False
        suit, level = c["suit"], c["level"] + 1
    else:
        return False
    if level < SOLO_CAP_MIN_LEVEL:
        return False
    margin = ABONDANCE_CAP_MARGIN if level >= ABONDANCE_MIN_LEVEL else 0
    return level > estimate_tricks(hand, suit) + margin


def emballer_too_short(hand, action_id):
    """Emballage d'une couleur ou l'on a moins de EMBALLER_MIN_TRUMPS cartes,
    ou main faible (<= EMBALLER_WEAK_MAX_HCP points d'honneur et
    <= EMBALLER_WEAK_MAX_TRUMPS atouts) ?"""
    info = bid_action_to_dict(action_id)
    if info["action"] != "emballer":
        return False
    trumps = sum(1 for c in hand if suit_of(c) == info["suit"])
    if trumps < EMBALLER_MIN_TRUMPS:
        return True
    return trumps <= EMBALLER_WEAK_MAX_TRUMPS and sum(_HCP.get(c % 13, 0) for c in hand) <= EMBALLER_WEAK_MAX_HCP


def grande_misere_too_loose(bidding, seat, hand, action_id):
    """Grande misere (annoncee, ou atteinte en relancant un piccolo) avec une
    main d'ecart > GRANDE_MISERE_MAX_SLACK ?"""
    info = bid_action_to_dict(action_id)
    if info["action"] == "misere":
        kind = info["kind"]
    elif action_id == ACTION_RAISE and bidding.lie_contract_index[seat] != -1:
        c = bidding.contracts[bidding.lie_contract_index[seat]]
        if c["type"] != "misere" or c["kind"] == MISERE_ORDER[-1]:
            return False
        kind = MISERE_ORDER[MISERE_ORDER.index(c["kind"]) + 1]
    else:
        return False
    return kind == "grande" and misere_slack(hand) > GRANDE_MISERE_MAX_SLACK


def emballage_raise_too_high(bidding, seat, action_id):
    """Relance (+1 ou saut) de son propre emballage au-dessus de
    EMBALLAGE_RAISE_MAX_LEVEL, ou emballage NEUF qui entrerait directement
    au-dessus (par-dessus une misere : 11/13, seule source restante des
    emballage-11 en entrainement le 2026-10-10) ?"""
    info = bid_action_to_dict(action_id)
    if info["action"] == "emballer":
        entry = bidding.emballage_entry_level(info["suit"])
        return entry is not None and entry > EMBALLAGE_RAISE_MAX_LEVEL
    if info["action"] != "raise" or bidding.lie_contract_index[seat] == -1:
        return False
    c = bidding.contracts[bidding.lie_contract_index[seat]]
    if c["type"] != "emballage":
        return False
    return info.get("level", c["level"] + 1) > EMBALLAGE_RAISE_MAX_LEVEL


def apply_bid_guards(bidding, seat, hand, mask):
    """Masque des garde-fous de l'IA exportee : solos trop hauts, emballages
    trop courts, grandes miseres trop risquees, emballages relances trop haut ;
    inchange s'il ne resterait plus rien."""
    guarded = [ok and not solo_too_high(bidding, seat, hand, a) and not emballer_too_short(hand, a)
               and not grande_misere_too_loose(bidding, seat, hand, a)
               and not emballage_raise_too_high(bidding, seat, a)
               for a, ok in enumerate(mask)]
    return guarded if any(guarded) else list(mask)


def apply_solo_cap(bidding, seat, hand, mask):
    """Masque (liste de bool) sans les solos trop hauts (solo_too_high) ;
    inchange s'il ne resterait plus rien."""
    capped = [ok and not solo_too_high(bidding, seat, hand, a) for a, ok in enumerate(mask)]
    return capped if any(capped) else list(mask)


class RuledBidder:
    """Reseau d'enchere de la lignee exportee + regle d'abondance + garde-fous
    des solos et de l'emballeur : ce que joue l'IA dans le jeu. `hybrid = True` : les appelants passent par
    decide() (meme convention que agents/hybrid_bidder.py)."""

    hybrid = True

    def __init__(self, net):
        self.net = net

    def decide(self, bidding, seat, hand):
        import torch
        from engine.bidding import legal_action_mask_with_hand, observation
        forced = abondance_rule(bidding, seat, hand)
        if forced is not None:
            return forced
        obs = torch.tensor(observation(bidding, seat, hand), dtype=torch.float32)
        mask = torch.tensor(apply_bid_guards(bidding, seat, hand, legal_action_mask_with_hand(bidding, seat, hand)),
                            dtype=torch.bool)
        return self.net.act_greedy(obs, mask)
