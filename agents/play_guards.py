"""Garde-fous du JEU DE LA CARTE de l'IA exportee (pendant des garde-fous
d'enchere de agents/abondance_rule.py), appliques au masque des reseaux de jeu
a l'entrainement ET dans le jeu (port : game/whist_expert_bot.gd
small_ruff_guard_legal, appele par game/whist_ai_bridge_v2.gd).

Petite coupe (signalement 2026-10-06 15:20, « overkill un pli ami ») : en
renonce dans la couleur demandee, si un atout gagne le pli SANS risque d'etre
surcoupe d'apres l'information publique du siege (meme critere que
ExpertBot._safe_win), interdit de couper avec un atout plus gros non
equivalent (une carte d'atout non vue entre les deux). Mesure avant :
l'IA coupait trop gros 19.5 % du temps dans ces positions (emballage 23 %),
l'ExpertBot 0 % (scripts/gates/check_signalement_20261006b.py --coupe).
La defausse reste permise.

Depasser le coequipier + defausser un As (signalement utilisateur
2026-10-07 08:30 : « un roi joue juste apres que l'equipier joue la dame de
la meme couleur », « la defausse d'un as ») -- mesure avant, 1000 donnes IA
exportee (scripts/gates/check_signalement_20261007.py) : l'IA depasse un
coequipier qui tient le pli sans risque 4.9 % du temps (defenseur 7.2 %),
l'ExpertBot 0 % ; elle defausse un As (hors misere) 5.2 % du temps quand
elle avait une autre defausse, l'ExpertBot 0.4 %.
Dans le JEU : interdits (ai_play_mask). A l'ENTRAINEMENT (demande
utilisateur : « montre a l'ia qu'elle perd de precieux plis ») : laisses
jouables -- le PPO voit les plis perdus -- et la decision recoit en cible
l'ensemble des autres cartes (perte d'imitation, train_play_mask) ; le taux du
reseau BRUT est suivi par check_signalement_20261007.py (sans --guard).

Surcoupe manquee (signalement utilisateur 2026-10-07 07:08 « l'IA ne surcoupe
pas ses adversaires !!! ») -- mesure avant, 1000 donnes IA exportee
(scripts/gates/check_signalement_20261007b.py --surcoupe) : quand un
adversaire tient le pli par une coupe, que le siege renonce, possede un atout
plus fort et qu'AUCUN coequipier ne joue apres lui, l'IA defausse 24 % du
temps et sous-coupe 4 %, l'ExpertBot surcoupe 100 %. Meme traitement que
ci-dessus : interdit dans le jeu, enseigne a l'entrainement.
"""

from engine.cards import rank_of, suit_of, trick_winner

ACE = 12


def small_ruff_forbidden(game, seat, legal):
    """-> ensemble des cartes interdites par la petite coupe (vide le plus
    souvent, sans construire la vue publique)."""
    trump = game.trump_suit
    trick = game.current_trick
    if trump is None or not trick or getattr(game, "pending_discards", None):
        return set()
    led = suit_of(trick[0][1])
    if led == trump:
        return set()
    hand = game.hands[seat]
    if any(suit_of(c) == led for c in hand):
        return set()
    trumps = [c for c in legal if suit_of(c) == trump]
    if len(trumps) < 2:
        return set()
    from agents.expert_bot import PublicView, _safe_win, _wins_now
    v = PublicView(game, seat)
    later = v.still_to_play()
    safe = [c for c in sorted(trumps) if _wins_now(v, c) and _safe_win(v, c, later)]
    if not safe:
        return set()
    low = safe[0]
    unseen = v.unseen_in(trump)
    return {c for c in trumps if rank_of(c) > rank_of(low)
            and any(rank_of(low) < rank_of(u) < rank_of(c) for u in unseen)}


def overtake_forbidden(game, seat, legal):
    """-> cartes qui depassent inutilement un coequipier tenant le pli sans
    risque (critere ExpertBot._partner_safe), hors misere, quand on fournit et qu'une carte
    de la couleur ne le depasse pas ; une carte equivalente (aucune carte non
    vue entre elle et la plus haute qui ne depasse pas) reste permise."""
    trick = game.current_trick
    if not trick or getattr(game, "pending_discards", None) or getattr(game, "contract_type", None) == "misere":
        return set()
    led = suit_of(trick[0][1])
    following = [c for c in legal if suit_of(c) == led]
    if len(following) < 2:
        return set()
    trump = game.trump_suit
    winner = trick_winner(trick, trump)
    over = [c for c in following if trick_winner(trick + [(seat, c)], trump) == seat]
    under = [c for c in following if c not in over]
    if not over or not under:
        return set()
    from agents.expert_bot import PublicView, _partner_safe
    v = PublicView(game, seat)
    mates = v.teammates(seat)
    if winner not in mates:
        return set()
    later = v.still_to_play()
    if any(s not in mates for s in later) and not _partner_safe(v, dict(trick)[winner], led, later):
        return set()
    top = rank_of(max(under, key=rank_of))
    unseen = v.unseen_in(led)
    return {c for c in over if any(top < rank_of(u) < rank_of(c) for u in unseen)}


def ace_discard_forbidden(game, seat, legal):
    """-> As hors atout defausses (renonce, hors misere) quand une autre
    carte hors atout pouvait etre defaussee."""
    trick = game.current_trick
    if not trick or getattr(game, "pending_discards", None) or getattr(game, "contract_type", None) == "misere":
        return set()
    led = suit_of(trick[0][1])
    if any(suit_of(c) == led for c in legal):
        return set()
    side = [c for c in legal if suit_of(c) != game.trump_suit]
    aces = {c for c in side if rank_of(c) == ACE}
    if not aces or len(aces) == len(side):
        return set()
    return aces


def missed_overruff_forbidden(game, seat, legal):
    """-> cartes qui ne surcoupent pas (defausses, sous-coupes) quand un
    adversaire tient le pli par une coupe, que le siege renonce, peut le
    surcouper et qu'aucun coequipier ne joue apres lui (hors misere)."""
    trump = game.trump_suit
    trick = game.current_trick
    if trump is None or not trick or getattr(game, "pending_discards", None)             or getattr(game, "contract_type", None) == "misere":
        return set()
    led = suit_of(trick[0][1])
    if led == trump or any(suit_of(c) == led for c in legal):
        return set()
    winner = trick_winner(trick, trump)
    win_card = dict(trick)[winner]
    if suit_of(win_card) != trump:
        return set()
    over = {c for c in legal if suit_of(c) == trump and rank_of(c) > rank_of(win_card)}
    if not over or len(over) == len(legal):
        return set()
    from agents.expert_bot import PublicView
    v = PublicView(game, seat)
    mates = v.teammates(seat)
    if winner in mates or any(s in mates for s in v.still_to_play()):
        return set()
    return {c for c in legal if c not in over}


def teach_forbidden(game, seat, legal):
    """Coups interdits dans le jeu mais seulement DECONSEILLES a
    l'entrainement (voir docstring de module)."""
    return (overtake_forbidden(game, seat, legal) | ace_discard_forbidden(game, seat, legal)
            | missed_overruff_forbidden(game, seat, legal))


def _without(mask, banned):
    out = [ok and c not in banned for c, ok in enumerate(mask)]
    return out if any(out) else list(mask)


def ai_play_mask(game, seat):
    """Masque legal du jeu de la carte + garde-fous de l'IA exportee."""
    mask = game.legal_action_mask(seat)
    legal = [c for c, ok in enumerate(mask) if ok]
    banned = small_ruff_forbidden(game, seat, legal) | teach_forbidden(game, seat, legal)
    if not banned:
        return mask
    return _without(mask, banned)


def train_play_mask(game, seat):
    """-> (masque d'entrainement, masque cible ou None) : la petite coupe
    reste interdite ; les coups de teach_forbidden restent jouables et la
    cible (cartes acceptables) les exclut."""
    mask = game.legal_action_mask(seat)
    legal = [c for c, ok in enumerate(mask) if ok]
    ruff = small_ruff_forbidden(game, seat, legal)
    if ruff:
        mask = _without(mask, ruff)
        legal = [c for c, ok in enumerate(mask) if ok]
    taught = teach_forbidden(game, seat, legal)
    if not taught:
        return mask, None
    target = _without(mask, taught)
    return mask, (target if target != mask else None)
