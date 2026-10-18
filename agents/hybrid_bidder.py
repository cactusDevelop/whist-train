"""Enchérisseur HYBRIDE reseau + regles (2026-09-27, demande utilisateur :
"il y a certes des decisions claires a prendre, le reste du temps c'est des
decisions plus techniques qui n'ont pas de regles generales, tu peux donc en
faire un hybride IA-bot booste par les regles").

Lignee evolutive `encherisseur_expert` (agents/evolution.py:HYBRID_LINEAGE).
Chaque decision passe d'abord par une couche de REGLES FIXES (jamais mutee),
tiree des regles de l'ExpertBot (agents/expert_bot.py) et de son evaluation
des plis surs d'une main (estimate_tricks) :

1. Decisions CLAIRES, imposees au reseau (FORCE_MARGIN plis au-dessus du
   seuil de l'utilisateur) :
   - emballer si >= 5 plis surs dans la couleur proposee (seuil 3 + 2) ;
   - proposer si >= 7 plis surs (seuil 5 + 2) et pas une main d'abondance ;
   - abondance d'entree si >= 10 plis surs (seuil 9 + 1).
2. GARDE-FOUS : actions clairement hors de portee INTERDITES au reseau
   (tolerance GUARD_TOLERANCE = 1 pli sous le seuil) :
   - proposer < 4 plis surs, emballer < 2 ;
   - solo/abondance/solo chelem de niveau L avec moins de L-1 plis surs --
     c'est ce garde-fou qui empeche l'escalade des solos observee dans la
     population (40 % de solos >= 10 a ~3 % de reussite) ;
   - relance/saut d'emballage au-dela de ce que l'equipe porte (5 plis
     supposes chez le proposeur, niveau-5 montres par l'emballeur) ;
   - misere (ou relance vers la misere suivante) sur une main qui ne la
     porte pas selon le critere calibre de l'ExpertBot (Ecole du whist,
     expert_bot.misere_fits). Jusqu'au 2026-09-27 23:50 : simple compte de
     hautes cartes (A, R, D <= 3/2/1) -- laissait passer des mains
     injouables, 0-2 % des miseres de l'hybride etaient conformes, reussite
     11-22 % en Phase 2.
3. Tout le reste -- le choix entre plusieurs actions permises -- est
   decide par le RESEAU (argmax sur les actions permises), qui evolue.

Le reseau est amorce par distillation de l'enchere de l'ExpertBot
(scripts/distill_hybrid_bidder.py) puis evolue avec la population.
"""

from agents.expert_bot import (
    ABONDANCE_MIN, PROPOSER_ASSUMED, PROPOSE_MIN, emballer_min, estimate_tricks, misere_fits,
)
from engine.bidding import (
    ACTION_PASS, ACTION_RAISE, bid_action_to_dict, legal_action_ids_with_hand,
)
from engine.bidding import observation as bid_observation

FORCE_MARGIN = 2
GUARD_TOLERANCE = 1
ABONDANCE_FORCE = ABONDANCE_MIN + 1
def _team_tricks(bidding, seat, hand, c):
    """Plis que l'equipe d'emballage porte, vus de `seat`."""
    own = estimate_tricks(hand, c["suit"])
    if seat == c.get("spokesperson"):
        return own + PROPOSER_ASSUMED
    return own + (c["level"] - PROPOSER_ASSUMED)  # proposeur : l'emballeur a montre niveau-5


def allowed(bidding, seat, hand, info):
    """Garde-fou : l'action decrite par `info` (bid_action_to_dict) est-elle
    permise au reseau ?"""
    kind = info["action"]
    if kind in ("wait", "pass", "pass_parole", "trou_choice"):
        return True
    if kind == "propose":
        return estimate_tricks(hand, info["suit"]) >= PROPOSE_MIN - GUARD_TOLERANCE
    if kind == "emballer":
        return estimate_tricks(hand, info["suit"]) >= emballer_min(bidding, info["suit"]) - GUARD_TOLERANCE
    if kind in ("solo", "abondance", "solo_chelem"):
        level = info.get("level", 13)
        return estimate_tricks(hand, info["suit"]) >= level - GUARD_TOLERANCE
    if kind == "misere":
        return misere_fits(hand, info["kind"])
    if kind == "raise":
        c = bidding.contracts[bidding.lie_contract_index[seat]]
        target = info.get("level", c.get("level", 0) + 1)
        if c["type"] == "emballage":
            return _team_tricks(bidding, seat, hand, c) >= target - GUARD_TOLERANCE
        if c["type"] == "solo":
            return estimate_tricks(hand, c["suit"]) >= target - GUARD_TOLERANCE
        if c["type"] == "misere":
            from engine.scoring import MISERE_ORDER
            nxt = MISERE_ORDER[MISERE_ORDER.index(c["kind"]) + 1]
            return misere_fits(hand, nxt)
        return False
    return True


def forced(bidding, seat, hand, infos):
    """Decision CLAIRE imposee, ou None. `infos` : {id: dict} des actions legales."""
    if bidding.lie_contract_index[seat] != -1:
        return None
    best = None
    for a, info in infos.items():
        if info["action"] == "emballer":
            est = estimate_tricks(hand, info["suit"])
            if est >= emballer_min(bidding, info["suit"]) + FORCE_MARGIN and (best is None or est > best[0]):
                best = (est, a)
    if best:
        return best[1]
    abond = [(info["level"], a) for a, info in infos.items() if info["action"] == "abondance"
             and info["level"] <= estimate_tricks(hand, info["suit"])
             and estimate_tricks(hand, info["suit"]) >= ABONDANCE_FORCE]
    if abond:
        return min(abond)[1]
    for a, info in infos.items():
        if info["action"] == "propose":
            est = estimate_tricks(hand, info["suit"])
            if PROPOSE_MIN + FORCE_MARGIN <= est < ABONDANCE_MIN and (best is None or est > best[0]):
                best = (est, a)
    return best[1] if best else None


def rule_layer(bidding, seat, hand):
    """-> (action imposee ou None, ids permis au reseau). Jamais vide : si
    les garde-fous interdisent tout, passer (ou toutes les actions legales
    si passer n'est pas legal -- choix d'atout du trou)."""
    legal = legal_action_ids_with_hand(bidding, seat, hand)
    infos = {a: bid_action_to_dict(a) for a in legal}
    force = forced(bidding, seat, hand, infos)
    permitted = [a for a in legal if allowed(bidding, seat, hand, infos[a])]
    if not permitted:
        permitted = [ACTION_PASS] if ACTION_PASS in legal else legal
    return force, permitted


class HybridBidder:
    """Politique d'un individu de la lignee hybride : regles puis reseau."""

    hybrid = True

    def __init__(self, net):
        self.net = net

    def decide(self, bidding, seat, hand):
        import torch
        force, permitted = rule_layer(bidding, seat, hand)
        if force is not None:
            return force
        obs = torch.tensor(bid_observation(bidding, seat, hand), dtype=torch.float32)
        mask = torch.zeros(self.net.policy_head.out_features, dtype=torch.bool)
        mask[permitted] = True
        return self.net.act_greedy(obs, mask)


def bot_choice_as_id(bidding, seat, choice):
    """Decision de l'ExpertBot (id ou (RAISE_TO, niveau)) -> id d'action reseau."""
    if not isinstance(choice, tuple):
        return choice
    from engine.bidding import RAISE_TO_LEVELS, _RAISE_TO_BASE
    level = choice[1]
    current = bidding.contracts[bidding.lie_contract_index[seat]]["level"]
    if level < current + 2:
        return ACTION_RAISE
    return _RAISE_TO_BASE + RAISE_TO_LEVELS.index(level)
