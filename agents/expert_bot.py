"""ExpertBot : joueur a REGLES ECRITES, dicte par l'utilisateur (2026-09-27)
pour montrer a l'IA "ce que donne un bon joueur". Ce n'est PAS un reseau :
aucune donnee apprise, uniquement des regles de whist belge courantes.

Regles de l'utilisateur (message du 27/09, a garder a jour ici ET dans
IA_HISTOIRE_ET_FONCTIONNEMENT.md) :

ENCHERE -- trois contrats de predilection : proposition, emballage, solo.
  Jamais de misere (trop technique pour un bot simple). Le trou se joue
  quand il se declenche (automatique).
  - Plis "surs" d'une main pour un atout donne (estimate_tricks) : figures
    d'atout + longueur d'atout, as, series quasi garanties (A-R-D = 3 plis
    meme si un adversaire sec peut couper).
  - Proposer si >= 5 plis surs ; emballer si >= 3 (5 + 3 = emballage 8).
    (Seuils actuels : tune 40411623914 ; ces regles et celles du 27/10
    vivent en profils concurrents, PROFILES.)
  - L'emballeur surencherit d'un pli par pli supplementaire qu'il se voit
    faire (on suppose 5 plis chez le proposeur). Passe parole seulement
    s'il lui manque exactement 1 pli : le proposeur decide alors, une seule
    fois, et peut monter d'AUTANT de plis qu'il veut (regle confirmee par
    l'utilisateur le 27/09, whist-4-it 9c0dfb3, engine raise_contract_to) :
    le bot saute directement au niveau que l'equipe porte.
  - Solo si >= 6 plis seul, en seconde priorite apres proposition/emballage.
  - Abondance (solo 9+) d'entree si >= 9 plis surs (ajout de Claude, valide
    par l'utilisateur le 27/09).

JEU DE LA CARTE (roles solo/emballage/trou, les deux camps ; defense des
misères en version minimale) :
  - Compter : cartes jouees, renoms (un joueur qui ne fournit pas est en
    renom dans la couleur), annonces d'enchere (qui a propose/emballe
    quoi -> longueurs supposees).
  - Se mettre en renom (defausser la plus courte couleur) pour pouvoir
    couper ; jouer le renom du coequipier pour qu'il coupe ; ne pas encaisser
    dans une couleur ou un adversaire est en renom et a encore de l'atout.
  - Ne jamais surcouper ni depasser le coequipier maitre du pli.
  - Declarant d'emballage : tirer les atouts adverses, s'arreter si un des
    deux adversaires n'en a plus (2 atouts pour 1 : pas rentable) ou si le
    compte dit qu'ils n'en ont plus.
  - Position (l'un prend et entame, l'autre finit le pli) : NON encore code,
    sauf "laisser le pli au partenaire qui le tient".
  - Le systeme d'appel (signalisation facon bridge) : NON code (demande de
    l'utilisateur : "tu peux passer ca").

ECOLE DU WHIST (2026-09-27 soir, demande utilisateur : "ameliore l'ExpertBot
avec tous les comportements decrits dans l'Ecole du whist" --
whist-belgique.be/ecole, Thomas Lescot, fiches + exercices corriges). Seuls
les CONSEILS de jeu sont repris ; les regles restent celles de notre moteur
(engine/bidding.py, verifie contre Godot) -- le bot ne choisit jamais qu'une
action legale. Le mot "misere" n'a plus le sens "jamais" : voir plus bas.
  Enchere :
  - Proposition : en plus des 5 plis, une couleur d'au moins 4 cartes
    (« une proposition creuse coute cher au partenaire » ; l'exigence de
    deux honneurs, mesuree contre-productive, n'est pas gardee).
  - Solo : pas de longue sans honneurs (au moins 5 atouts dont 2 parmi
    A-R-D) ; abondance : au moins 3 honneurs a l'atout (« choisir comme
    atout une longue sans honneur » est le piege classique).
  - Emballeur surencheri : peut lacher l'emballage pour un solo dans une
    autre couleur si la main le porte vraiment (sinon relance ou passe).
  - Misere (misere_kinds_for) : « chaque couleur ecoulable de bas en haut »,
    mesure par misere_slack (ecart a l'echelle 2-4-6-8), seuils CALIBRES
    sur 150 000 donnes (MISERE_SLACK_MAX). Petite : apres le meilleur
    ecart. Piccolo : un As (« une seule levee maitrisable ») et le reste
    ecoulable. On vise la mieux payee que la main porte. « On n'ouvre pas la
    parole avec une misere » : le premier a parler attend (wait) s'il le
    peut. Misere surencherie : on monte a la misere suivante si la main la
    porte.
  Jeu de la carte :
  - Misere declarant : ecart = la carte la plus dangereuse, singleton haut
    d'abord ; passer juste sous le pli ; defausser d'abord les hautes cartes
    des couleurs courtes ; compter les cartes tombees. Piccolo : prendre son
    pli le plus tot possible, puis jouer comme en misere.
  - Misere defense : attaquer la couleur ou le declarant est le plus
    faible (main visible en etalee) ; piccolo : l'empecher de faire son
    unique pli, puis lui en faire prendre un deuxieme.
  - Defense avec atout : jamais d'entame d'atout de l'annonceur (« purger,
    c'est lui rendre service »), jouer la longue de la defense pour le
    forcer a couper ; jamais de faux renom (petit singleton devant le
    partenaire), seul un honneur sec est tolere.
  - Camp declarant : le proposeur purge, le partenaire rend l'atout ; en
    emballage, garder un atout de controle.

Le bot ne lit QUE l'information publique + sa propre main (PublicView),
jamais game.hands des autres : il ne peut pas tricher (sauf les mains
etalees, publiques par regle).
"""

import functools

from engine.bidding import (
    ACTION_PASS, ACTION_PASS_PAROLE, ACTION_RAISE, ACTION_WAIT, EMBALLAGE_MAX_LEVEL, EMBALLAGE_MIN_LEVEL,
    bid_action_to_dict,
    legal_action_ids_with_hand,
)
from engine.cards import NUM_PLAYERS, NUM_SUITS, rank_of, suit_of, trick_winner
from engine.scoring import MISERE_ORDER

ACE, KING, QUEEN, JACK = 12, 11, 10, 9

# Seuils regles par scripts/expert_tune.py (cloud 40411623914, recherche coordonnee duplicate ;
# validation 60000 donnes neuves : +1.837 +-0.053 pt/donne vs anciens reglages, z 35).
# Anciennes valeurs (regles utilisateur du 27/09) : profil "joueur_0927" de PROFILES.
PROPOSE_MIN = 3  # plis surs pour proposer
# Bloc emballer + attente : profil expert_emb1027, vainqueur de la ligue 40651271548 (+0.124 +-0.052,
# z 2.4), promu le 30/10 (signalement « emballe avec 1 atout » : 1 pli sur suffisait, sans condition d'atouts).
EMBALLER_MIN = 4  # plis surs pour emballer (au niveau 8)
SOLO_MIN = 6  # plis surs pour un solo seul
ABONDANCE_MIN = 10  # solo 10+ d'entree
PROPOSER_ASSUMED = 7  # ce que l'emballeur suppose chez le proposeur
EMBALLER_ASSUMED = 2  # longueur d'atout supposee chez l'emballeur
# Reglages des profils concurrents (PROFILES) ; neutre pour l'ExpertBot.
PROPOSE_TRUMP_HONOUR_TRICKS = 0  # plis d'honneurs d'atout exiges pour proposer (trump_honour_tricks)
EMBALLER_LONG_TRUMPS = 5  # emballer avec au moins ce nombre d'atouts...
EMBALLER_SHORT_TRUMPS = 4  # ... ou celui-ci avec au moins une figure (0, 0 = sans condition)
WAIT_BALANCED = True  # premier a parler, main moyenne repartie : attendre (pour emballer ensuite)
PROPOSER_TRUMPS_ASSUMED = 5  # longueur d'atout supposee chez le proposeur
# Ecole du whist (voir docstring) :
# « au moins quatre cartes avec deux honneurs » : les deux honneurs ont ete
# MESURES contre-productifs ici (duel duplicate 3000 donnes contre l'ancien
# bot : +1.11 sans eux, +0.73 avec -- trop de passes generales) ; seule la
# longueur est gardee.
PROPOSE_MIN_LENGTH, PROPOSE_MIN_HONOURS = 4, 0
SOLO_MIN_LENGTH, SOLO_MIN_TOP_HONOURS = 5, 1  # pas de « longue sans honneurs »
ABONDANCE_MIN_HONOURS = 3  # « sept a huit atouts avec trois honneurs »
# Prudence des encheres (signalement 2026-10-26 « contrats impossibles, surencheres
# abusives » ; audit scripts/expert_bid_audit.py : solo 67.6 %, emballage 12/13
# releve 44 % / 18 %). Mesure duplicate scripts/expert_bid_ab.py (cloud 40398240427,
# 6000 donnes, paires variante vs defaut, vrais points) : SOLO_MARGIN 1 -0.10 +-0.07,
# RAISE_MARGIN 1 -0.22 +-0.05 (plus prudent = PERD des points), RAISE_CAP 10 -0.01 +-0.02,
# RAISE_CAP 11 +0.012 +-0.007 -> seul le plafond 11 est garde (plus d'emballage 12/13).
SOLO_MARGIN = 0  # un solo de niveau L demande L + SOLO_MARGIN plis estimes
RAISE_MARGIN = 0  # une relance (solo, emballage, saut du proposeur) demande ce nombre de plis en plus
RAISE_CAP = 9  # (tune 40411623914 ; 11 avant) niveau maximal atteint par relance, saut ou emballage d'entree
# Jeu de la carte (plaintes 2026-10-26 « overkill, defausse au lieu de couper ») :
NO_OVERTAKE = False  # coequipier gagnant, on fournit : toujours la plus petite (jamais le depasser)
RUFF_IF_OPP_LATER = False  # renonce, coequipier gagnant hors atout, un adversaire apres peut couper : couper
SECOND_HAND_LOW = False  # 2e a jouer, coequipier apres : petite carte meme avec une maitresse
# Deduction par l'enchere (utilisateur 2026-10-26 : « en comptant les cartes, en supposant la
# main du coequipier [...] le proposant et l'emballant ont d'office plus d'atout ») : camp
# declarant, quand les atouts non vus moins ceux supposes chez le partenaire = 0, les
# adversaires n'ont plus d'atout (ni coupe ni surcoupe a craindre).
INFER_OPP_TRUMPS = True  # duplicate cloud 40404132860 : +0.015 +-0.007 pt/donne (z 2.3)
# tirer l'atout sans maitresse : l'honneur le plus haut (>= valet) suivi de la carte juste en dessous
# (R-D -> R), plutot que la plus petite. Duplicate cloud 40404703553 (40000 donnes) : +0.291 +-0.017 pt/donne.
TRUMP_LEAD_SEQUENCE = True
DRAW_WITH_LAST_TRUMP = False  # tirer l'atout meme avec son dernier atout (duplicate -0.13 : rejete)
SIDE_LEAD_SEQUENCE = True  # duplicate 40406005024 avec LEAD_PARTNER_SUIT : +0.089 +-0.021 ; entame par defaut hors atout : tete de sequence d'honneurs (R-D, D-V, V-10) de la couleur choisie
LEAD_PARTNER_SUIT = True  # defense : entamer la couleur proposee/emballee par un coequipier
BID_PARAMS = ("SOLO_MARGIN", "RAISE_MARGIN", "RAISE_CAP", "NO_OVERTAKE", "RUFF_IF_OPP_LATER", "SECOND_HAND_LOW",
              "INFER_OPP_TRUMPS", "TRUMP_LEAD_SEQUENCE", "DRAW_WITH_LAST_TRUMP", "SIDE_LEAD_SEQUENCE",
              "LEAD_PARTNER_SUIT")

# Profils CONCURRENTS de l'ExpertBot (utilisateur 2026-10-27 : « mets les regles que je viens
# de te donner en concurrent a l'ExpertBot, pas en remplacement, idem pour toutes les autres
# fois. Le but n'est pas de creer artificiellement le meilleur bot mais de trouver la meilleure
# IA. ») Chaque profil = reglages set_bid_params ; {} = l'ExpertBot. Departage :
# scripts/bot_league.py (duplicate, vrais points). Toute nouvelle regle utilisateur = un
# nouveau profil ici, jamais un changement des defauts.
PROFILES = {
    "expert": {},
    # ExpertBot du 27/10 au 30/10 (tune 40411623914) : emballe des 1 pli sur, sans condition d'atouts
    "expert_tune1027": {"EMBALLER_MIN": 1, "EMBALLER_LONG_TRUMPS": 0, "EMBALLER_SHORT_TRUMPS": 0,
                        "WAIT_BALANCED": False},
    # regles du 27/09 (valeurs d'avant le tune 40411623914)
    "joueur_0927": {"PROPOSE_MIN": 5, "EMBALLER_MIN": 3, "EMBALLER_LONG_TRUMPS": 0, "EMBALLER_SHORT_TRUMPS": 0,
                    "WAIT_BALANCED": False, "ABONDANCE_MIN": 9, "PROPOSER_ASSUMED": 5,
                    "EMBALLER_ASSUMED": 3, "SOLO_MIN_TOP_HONOURS": 2, "RAISE_CAP": 11},
    # 27/10 : proposer 6 plis (plancher 5) avec plusieurs grosses figures d'atout comptees en plis
    # reels (R-D-V = 2 : l'As en prend un) ; emballer 4 (plancher 3) avec 5-6 atouts ou 4 dont
    # une figure ; « etre large » (solo au plus bas niveau, deja le cas) ; jamais de bluff ;
    # main moyenne repartie du premier a parler : attendre pour emballer ensuite.
    "joueur_1027": {"PROPOSE_MIN": 6, "PROPOSE_TRUMP_HONOUR_TRICKS": 2, "EMBALLER_MIN": 4,
                    "EMBALLER_LONG_TRUMPS": 5, "EMBALLER_SHORT_TRUMPS": 4, "WAIT_BALANCED": True,
                    "ABONDANCE_MIN": 9, "PROPOSER_ASSUMED": 5, "EMBALLER_ASSUMED": 3,
                    "SOLO_MIN_TOP_HONOURS": 2, "RAISE_CAP": 11},
}


def set_bid_params(params):
    """Fixe des reglages du module (dict partiel de constantes en MAJUSCULES,
    BID_PARAMS ou seuils d'enchere ; {} = defauts). Mesure seule
    (scripts/expert_bid_ab.py, scripts/expert_tune.py)."""
    g = globals()
    for k, v in _BID_DEFAULTS.items():
        if k not in params:
            g[k] = v
    for k, v in params.items():
        if k not in _BID_DEFAULTS:
            _BID_DEFAULTS[k] = g[k]
        g[k] = v


# --- evaluation de main ------------------------------------------------------

_BID_DEFAULTS = {k: globals()[k] for k in BID_PARAMS}


def _ranks(hand, suit):
    return sorted((rank_of(c) for c in hand if suit_of(c) == suit), reverse=True)


def _top_sequence(ranks, top=ACE):
    n = 0
    while top - n in ranks:
        n += 1
    return n


def side_suit_tricks(ranks):
    """As et series quasi garanties : A=1, A-R=2, A-R-D=3... ; R-D-V sans
    l'as = 2 (l'as en prend un)."""
    n = _top_sequence(ranks, ACE)
    if n:
        return n
    k = _top_sequence(ranks, KING)
    return max(0, k - 1)


def trump_tricks(ranks):
    """Figures d'atout gardees (A ; R si 2+ atouts ; D si 3+ ; V si 4+)
    plus la longueur au-dela de 3 atouts, plafonne a la longueur."""
    length = len(ranks)
    honors = sum(1 for r, guard in ((ACE, 1), (KING, 2), (QUEEN, 3), (JACK, 4)) if r in ranks and length >= guard)
    small = length - honors
    return min(length, honors + min(small, max(0, length - 3)))


def _honours(ranks, top=(ACE, KING, QUEEN, JACK)):
    return sum(1 for r in top if r in ranks)


def solo_suit_ok(hand, suit, level):
    """Ecole du whist : un solo ne part pas d'une longue sans honneurs, une
    abondance (9+) demande trois honneurs a l'atout."""
    ranks = _ranks(hand, suit)
    if len(ranks) < SOLO_MIN_LENGTH or _honours(ranks, (ACE, KING, QUEEN)) < SOLO_MIN_TOP_HONOURS:
        return False
    return level < ABONDANCE_MIN or _honours(ranks) >= ABONDANCE_MIN_HONOURS


def propose_suit_ok(hand, suit):
    ranks = _ranks(hand, suit)
    return (len(ranks) >= PROPOSE_MIN_LENGTH and _honours(ranks) >= PROPOSE_MIN_HONOURS
            and trump_honour_tricks(ranks) >= PROPOSE_TRUMP_HONOUR_TRICKS)


def trump_honour_tricks(ranks):
    """Plis reels des figures d'atout, compte de l'utilisateur (27/10) : ce
    qu'on gagne et perd vraiment -- R-D-V = 2 (l'As adverse en prend un),
    A-D-V = 2, A-R = 2, A seul = 1. Figures tenues moins les figures
    manquantes au-dessus de la plus basse tenue, au moins la sequence
    depuis l'As."""
    held = [r for r in (ACE, KING, QUEEN, JACK) if r in ranks]
    if not held:
        return 0
    missing = sum(1 for r in (ACE, KING, QUEEN, JACK) if r > held[-1] and r not in ranks)
    return max(_top_sequence(ranks, ACE), len(held) - missing)


def emballer_suit_ok(hand, suit):
    """5-6 atouts (meme petits), ou 4 avec au moins une figure."""
    ranks = _ranks(hand, suit)
    return len(ranks) >= EMBALLER_LONG_TRUMPS or (len(ranks) >= EMBALLER_SHORT_TRUMPS and _honours(ranks) >= 1)


def balanced_average(hand):
    """Main moyenne assez repartie : pas de couleur de plus de 4 cartes ni de
    chicane, et au moins 3 plis surs dans sa meilleure couleur (de quoi
    emballer une proposition)."""
    lengths = [len(_ranks(hand, s)) for s in range(NUM_SUITS)]
    return max(lengths) <= 4 and min(lengths) >= 1 and max(
        estimate_tricks(hand, s) for s in range(NUM_SUITS)) >= EMBALLER_MIN


def misere_slack(hand):
    """Ecole du whist : en misere, chaque couleur doit etre « ecoulable de
    bas en haut » (une couleur de quatre cartes commencant par le 2 est une
    securite ; « une couleur de trois cartes dont la plus basse est le 6 »,
    un Roi « non accompagne », une couleur de deux cartes hautes ne le sont
    pas). Mesure : la i-eme plus petite carte d'une couleur (rang 0 = le 2)
    comparee a l'echelle 2-4-6-8... -> max(rang - 2i) sur toute la main.
    Plus c'est bas, plus la main passe sous les autres ; une chicane ne
    coute rien. -1 pour une main vide."""
    return _slack_cached(tuple(sorted(hand)))


@functools.lru_cache(maxsize=131072)
def _slack_cached(hand):
    return max(_suit_slack(sorted(_ranks(hand, s))) for s in range(NUM_SUITS))


def _suit_slack(ascending_ranks):
    return max((r - 2 * i for i, r in enumerate(ascending_ranks)), default=-1)


def _petite_ok(hand):
    """Un seul ecart peut-il rendre la main ecoulable ? Il ne corrige qu'UNE
    couleur : au plus une couleur hors seuil, et une de ses cartes a ecarter."""
    th = MISERE_SLACK_MAX["petite"]
    suits = [sorted(_ranks(hand, s)) for s in range(NUM_SUITS)]
    bad = [r for r in suits if _suit_slack(r) > th]
    if not bad:
        return True
    if len(bad) > 1:
        return False
    r = bad[0]
    return any(_suit_slack(r[:j] + r[j + 1:]) <= th for j in range(len(r)))


def _piccolo_ok(hand):
    """Un As prend le pli obligatoire, le reste de la main est ecoulable."""
    return any(rank_of(a) == ACE and misere_slack([c for c in hand if c != a]) <= MISERE_SLACK_MAX["piccolo"]
               for a in hand)


# Seuils d'ecart par contrat negatif, CALIBRES le 27/09 sur 150 000 donnes
# (declarant seul, 4 sieges ExpertBot) : etalee <= 2 -> 90 % de reussite,
# grande <= 3 -> 75 %, petite <= 3 apres l'ecart -> 70 %, piccolo (un As +
# reste <= 4) -> 60 a 77 %. Au-dela, sous 50 % : le bareme est symetrique
# (gain = perte), une misere a moins de 50 % coute des points.
MISERE_SLACK_MAX = {"etalee": 2, "grande": 3, "petite": 3, "piccolo": 4}


def best_misere_discard(hand):
    """Petite misere : l'ecart qui laisse la main la plus ecoulable (Ecole :
    « supprimer la carte la plus dangereuse : le singleton haut avant
    tout »)."""
    return min(hand, key=lambda c: (misere_slack([x for x in hand if x != c]), -rank_of(c)))


def misere_kinds_for(hand):
    """Contrats negatifs que la main porte, du mieux paye au moins paye
    (ordre inverse de engine/scoring.py:MISERE_ORDER)."""
    return _misere_kinds_cached(tuple(sorted(hand)))


@functools.lru_cache(maxsize=65536)
def _misere_kinds_cached(hand):
    slack = misere_slack(hand)
    kinds = []
    if slack <= MISERE_SLACK_MAX["etalee"]:  # « main mathematiquement imprenable »
        kinds.append("etalee")
    if slack <= MISERE_SLACK_MAX["grande"]:
        kinds.append("grande")
    # piccolo : « une seule levee maitrisable et rien d'autre de dangereux »
    # -- un As prend le pli obligatoire (A-R = deux prises ; sans prenante,
    # zero pli fait chuter aussi)
    if _piccolo_ok(hand):
        kinds.append("piccolo")
    if _petite_ok(hand):
        kinds.append("petite")
    return tuple(kinds)


def misere_fits(hand, kind):
    """`kind` in misere_kinds_for(hand), en ne calculant que le necessaire
    (le tirage des mains de misere de scripts/train.py en teste beaucoup)."""
    if kind in ("etalee", "grande"):
        return misere_slack(hand) <= MISERE_SLACK_MAX[kind]
    if kind == "piccolo":
        return _piccolo_ok(hand)
    return _petite_ok(hand)


def misere_danger_score(hand):
    """Cle de tri des declarants de misere plausibles (scripts/train.py) :
    l'ecart apres le meilleur ecart."""
    return misere_slack([c for c in hand if c != best_misere_discard(hand)])


def emballer_min(bidding, suit):
    """Plis surs exiges pour emballer `suit` : EMBALLER_MIN au niveau 8, +1
    par niveau d'entree au-dessus (regle 2026-09-29, voir
    Bidding.emballage_entry_level)."""
    return EMBALLER_MIN + (bidding.emballage_entry_level(suit) - EMBALLAGE_MIN_LEVEL)


def estimate_tricks(hand, trump):
    return _estimate_cached(tuple(sorted(hand)), trump)


@functools.lru_cache(maxsize=65536)
def _estimate_cached(hand, trump):
    total = 0
    for s in range(NUM_SUITS):
        ranks = _ranks(hand, s)
        total += trump_tricks(ranks) if s == trump else side_suit_tricks(ranks)
    return total


# --- enchere -----------------------------------------------------------------

RAISE_TO = "raise_to"  # saut du proposeur en passe parole (hors ids d'action)


def apply_bid(bidding, seat, hand):
    """Decide ET applique ; -> la decision (id d'action ou (RAISE_TO, niveau))."""
    choice = choose_bid(bidding, seat, hand)
    ok = bidding.apply_raise_to(seat, choice[1]) if isinstance(choice, tuple) else bidding.apply(seat, choice)
    if not ok:
        raise AssertionError("ExpertBot : enchere illegale %r siege %d" % (choice, seat))
    return choice


def choose_bid(bidding, seat, hand):
    """-> un id d'action d'enchere TOUJOURS legal (legal_action_ids_with_hand),
    ou (RAISE_TO, niveau) pour le saut du proposeur en passe parole."""
    legal = legal_action_ids_with_hand(bidding, seat, hand)
    if not legal:
        raise ValueError("aucune action legale pour le siege %d" % seat)
    infos = {a: bid_action_to_dict(a) for a in legal}
    by_kind = {}
    for a, info in infos.items():
        by_kind.setdefault(info["action"], []).append((a, info))

    ci = bidding.lie_contract_index[seat]
    if ci != -1:
        return _respond_to_outbid(bidding, seat, hand, bidding.contracts[ci], legal, by_kind)

    restricted = bidding.solo_only_suit[seat]
    if restricted is not None:  # partenaire d'emballage parti : solo dans cette couleur ou rien
        return _cheapest_solo(by_kind, estimate_tricks(hand, restricted), restricted, hand) or ACTION_PASS

    # abondance d'entree (ajout Claude, valide) : la proposer serait brader la main
    for s in range(NUM_SUITS):
        est = estimate_tricks(hand, s)
        if est >= ABONDANCE_MIN:
            choice = _cheapest_solo(by_kind, est, s, hand, families=("abondance", "solo_chelem"))
            if choice is not None:
                return choice
    # 1. emballer une proposition si >= 3 plis (+1 par niveau au-dessus de 8 :
    #    l'emballage part au plus petit niveau qui depasse l'enchere en tete)
    options = [(estimate_tricks(hand, info["suit"]), a, emballer_min(bidding, info["suit"]))
               for a, info in by_kind.get("emballer", [])]
    options = [o[:2] for o in options if o[0] >= o[2] and emballer_suit_ok(hand, infos[o[1]]["suit"])
               and bidding.emballage_entry_level(infos[o[1]]["suit"]) <= RAISE_CAP]
    if options:
        return max(options)[1]
    # 2. proposer si >= 5 plis
    options = [(estimate_tricks(hand, info["suit"]), a) for a, info in by_kind.get("propose", [])
               if propose_suit_ok(hand, info["suit"])]
    options = [o for o in options if o[0] >= PROPOSE_MIN]
    if options:
        return max(options)[1]
    # 3. solo si >= 6 plis (seconde priorite)
    best = _best_solo(by_kind, hand)
    if best is not None:
        return best
    # 3b. main moyenne repartie, premier a parler : attendre pour emballer ensuite
    if WAIT_BALANCED and "wait" in by_kind and balanced_average(hand):
        return ACTION_WAIT
    # 4. misere (Ecole du whist)
    choice = _misere_choice(bidding, by_kind, hand)
    if choice is not None:
        return choice
    return ACTION_PASS


def _best_solo(by_kind, hand, exclude_suit=None):
    best = None
    for s in range(NUM_SUITS):
        if s == exclude_suit:
            continue
        est = estimate_tricks(hand, s)
        if est >= SOLO_MIN:
            choice = _cheapest_solo(by_kind, est, s, hand)
            if choice is not None and (best is None or est > best[0]):
                best = (est, choice)
    return best[1] if best is not None else None


def _colour_announced(bidding):
    return bool(bidding.pending_proposals) or any(c["type"] in ("emballage", "solo") for c in bidding.contracts)


def _misere_choice(bidding, by_kind, hand):
    """La misere la mieux payee que la main porte (misere_kinds_for), parmi
    les legales. « On n'ouvre pas la parole avec une misere : on repond a
    la couleur annoncee par un autre joueur » -> attendre (wait) si
    possible tant qu'aucune couleur n'a ete annoncee ; sinon l'annoncer
    quand meme plutot que de se coucher pour de bon."""
    offered = {info["kind"]: a for a, info in by_kind.get("misere", [])}
    kinds = [k for k in misere_kinds_for(hand) if k in offered]
    if not kinds:
        return None
    if not _colour_announced(bidding) and "wait" in by_kind:
        return ACTION_WAIT
    return offered[kinds[0]]


def _cheapest_solo(by_kind, est, suit, hand, families=("solo", "abondance", "solo_chelem")):
    """Plus petit niveau legal <= plis estimes dans la couleur, ou None
    (couleur d'atout conforme a l'Ecole du whist, voir solo_suit_ok)."""
    if est < SOLO_MIN:
        return None
    choices = []
    for fam in families:
        for a, info in by_kind.get(fam, []):
            level = info.get("level", 13)
            if info["suit"] == suit and level + SOLO_MARGIN <= est and solo_suit_ok(hand, suit, level):
                choices.append((level, a))
    return min(choices)[1] if choices else None


def _respond_to_outbid(bidding, seat, hand, c, legal, by_kind):
    """Le contrat de ce siege vient d'etre depasse (ou la parole lui est
    passee) : relancer d'un pli si la main le porte, sinon passer."""
    target = c.get("level", 0) + 1
    if c["type"] == "misere":
        # « misere surencherie » : monter a la misere suivante si la main la porte
        if ACTION_RAISE in legal and MISERE_ORDER.index(c["kind"]) + 1 < len(MISERE_ORDER):
            if MISERE_ORDER[MISERE_ORDER.index(c["kind"]) + 1] in misere_kinds_for(hand):
                return ACTION_RAISE
        return ACTION_PASS
    if c["type"] == "emballage":
        own = estimate_tricks(hand, c["suit"])
        if seat == c["spokesperson"]:
            team = own + PROPOSER_ASSUMED
            # Ecole : l'emballeur surencheri peut lacher son emballage pour un
            # solo ailleurs, « a condition que sa main justifie reellement ce
            # Solo » -- prefere a une relance dans une couleur ou il n'a rien
            solo = _best_solo(by_kind, hand, exclude_suit=c["suit"])
            if solo is not None and own < EMBALLER_MIN + (target - 8):
                return solo
            if ACTION_RAISE in legal and team >= target + RAISE_MARGIN and target <= RAISE_CAP:
                return ACTION_RAISE
            if solo is not None:
                return solo
            if ACTION_PASS_PAROLE in legal and team == target - 1:
                return ACTION_PASS_PAROLE
            return ACTION_PASS
        # proposeur, parole passee : l'emballeur a montre (niveau - 5) plis ;
        # une seule decision -> sauter directement au niveau que l'equipe porte
        team = min(EMBALLAGE_MAX_LEVEL, RAISE_CAP, own + (c["level"] - PROPOSER_ASSUMED) - RAISE_MARGIN)
        if team >= target and bidding.can_raise_to(seat, team):
            return (RAISE_TO, team)
        return ACTION_PASS
    if c["type"] == "solo":
        own = estimate_tricks(hand, c["suit"])
        return ACTION_RAISE if ACTION_RAISE in legal and own >= target + RAISE_MARGIN and target <= RAISE_CAP else ACTION_PASS
    return ACTION_PASS  # trou (jamais relancable)


# --- vue publique du jeu de la carte ----------------------------------------

class PublicView:
    """Tout ce que le siege `seat` a le droit de savoir. Construite depuis un
    moteur de jeu (TeamTrumpDeal / LoneDeclarerDeal) sans jamais copier les
    mains des autres."""

    def __init__(self, game, seat):
        self.seat = seat
        self.hand = list(game.hands[seat])
        self.trump = game.trump_suit
        self.contract = game.public_contract()
        self.bid_summary = game.bid_summary
        self.play_log = list(game.play_log)
        self.current_trick = list(game.current_trick)
        self.discard_phase = bool(getattr(game, "pending_discards", None))
        declarers = set(self.contract["declarers"])
        if self.contract["partner"] is not None:
            declarers.add(self.contract["partner"])
        self.declarer_side = declarers
        self.is_misere = self.contract["type"] == "misere"
        self.kind = self.contract.get("kind")
        self.team_contract = self.contract["partner"] is not None
        self.legal = [c for c, ok in enumerate(game.legal_action_mask(seat)) if ok]
        # misere etalee : les mains des declarants sont PUBLIQUES (regle du jeu)
        self.revealed = ({d: list(game.hands[d]) for d in self.contract["declarers"]}
                         if getattr(game, "reveal_declarer_hands", False) else {})
        for d in getattr(game, "exposed", ()):  # roguelike : Cartes sur Table (WhistExpertBot._revealed)
            self.revealed[d] = list(game.hands[d])
        self._analyse()

    def teammates(self, seat):
        """Coequipiers de `seat` (hors lui-meme). Misere : chaque declarant
        est seul ; les autres forment la defense."""
        if seat in self.declarer_side:
            if self.is_misere:
                return set()
            return self.declarer_side - {seat}
        return set(range(NUM_PLAYERS)) - self.declarer_side - {seat}

    def _analyse(self):
        self.void = [[False] * NUM_SUITS for _ in range(NUM_PLAYERS)]
        self.tricks_won = [0] * NUM_PLAYERS
        self.played = set()
        self.trumps_played_by = [0] * NUM_PLAYERS
        for seat, card in self.play_log:
            self.played.add(card)
            if self.trump is not None and suit_of(card) == self.trump:
                self.trumps_played_by[seat] += 1
        # renoms : relire pli par pli
        for i in range(0, len(self.play_log), NUM_PLAYERS):
            chunk = self.play_log[i:i + NUM_PLAYERS]
            led = suit_of(chunk[0][1])
            for seat, card in chunk[1:]:
                if suit_of(card) != led:
                    self.void[seat][led] = True
            if len(chunk) == NUM_PLAYERS:
                self.tricks_won[trick_winner(chunk, self.trump)] += 1
        mine = set(self.hand)
        self.unseen = [c for c in range(52) if c not in self.played and c not in mine]

    # --- comptes -------------------------------------------------------

    def unseen_in(self, suit):
        return [c for c in self.unseen if suit_of(c) == suit]

    def is_master(self, card):
        """Aucune carte non vue de la couleur ne la bat."""
        return all(rank_of(c) < rank_of(card) for c in self.unseen_in(suit_of(card)))

    def may_hold(self, seat, suit):
        if seat == self.seat or self.void[seat][suit] or not self.unseen_in(suit):
            return False
        if (INFER_OPP_TRUMPS and suit == self.trump and self.seat in self.declarer_side
                and seat not in self.declarer_side and self.opponent_trumps_outstanding() == 0):
            return False
        return True

    def may_ruff(self, seat, suit):
        return (self.trump is not None and suit != self.trump and self.void[seat][suit]
                and self.may_hold(seat, self.trump))

    def partner_trumps_estimate(self, partner):
        """Longueur d'atout restante supposee du partenaire d'emballage :
        5 s'il a propose l'atout, 3 s'il l'a emballe, moins ce qu'il a joue."""
        if self.void[partner][self.trump]:
            return 0
        entry = self.bid_summary[partner] if self.bid_summary else None
        start = EMBALLER_ASSUMED
        if entry is not None and entry["proposed"][self.trump]:
            start = PROPOSER_TRUMPS_ASSUMED
        return max(0, min(start - self.trumps_played_by[partner], len(self.unseen_in(self.trump))))

    def opponent_trumps_outstanding(self):
        unseen = len(self.unseen_in(self.trump))
        for mate in self.teammates(self.seat):
            unseen -= self.partner_trumps_estimate(mate)
        return max(0, unseen)

    def opponents(self):
        mates = self.teammates(self.seat)
        return [s for s in range(NUM_PLAYERS) if s != self.seat and s not in mates]

    def still_to_play(self):
        started = len(self.current_trick)
        leader = self.current_trick[0][0] if self.current_trick else self.seat
        return [(leader + i) % NUM_PLAYERS for i in range(started + 1, NUM_PLAYERS)]

    def proposed_long(self, seat, suit):
        """Annonce publique : a propose/emballe cette couleur -> long."""
        if not self.bid_summary:
            return False
        e = self.bid_summary[seat]
        return e["proposed"][suit] or e["emballed"][suit]


# --- jeu de la carte -----------------------------------------------------------

def _lowest(cards):
    return min(cards, key=rank_of)


def _highest(cards):
    return max(cards, key=rank_of)


def decide_trou_atout(hand, anchor_suit):
    """Couleur visee par le bouche-trou (game/whist_expert_bot.gd::
    decide_trou_atout) : l'ancre joue a 8 plis, une autre couleur a 9 --
    +1 pli exige ailleurs. Egalite -> la premiere couleur (ordre 0..3)."""
    return max(range(NUM_SUITS), key=lambda s: (estimate_tricks(hand, s) - (0 if s == anchor_suit else 1), -s))


def trou_opening_card(hand, anchor_card):
    """Entame du bouche-trou, qui FIXE le Trou (regle du 2026-10-01) --
    miroir de net/whist_table.gd::_bot_trou_opening_card : couleur de
    decide_trou_atout (l'ancre si la main n'en a pas), puis la carte
    d'ancrage si elle est de cette couleur (8 plis), sinon la plus haute
    carte de la couleur (9 plis)."""
    suit = decide_trou_atout(hand, suit_of(anchor_card))
    in_suit = [c for c in hand if suit_of(c) == suit]
    if not in_suit:
        in_suit = [c for c in hand if suit_of(c) == suit_of(anchor_card)]
    if anchor_card in in_suit:
        return anchor_card
    return _highest(in_suit)


def choose_card(game, seat):
    if getattr(game, "trou_pending", lambda: False)():
        return trou_opening_card(game.hands[seat], game.anchor_card)
    v = PublicView(game, seat)
    if len(v.legal) == 1:
        return v.legal[0]
    if v.discard_phase:
        return _discard_phase(v)
    if v.is_misere:
        if v.seat in v.declarer_side:
            return _misere_declarer(v)
        return _misere_defense(v)
    if not v.current_trick:
        return _lead(v)
    return _follow(v)


def _discard_phase(v):
    """Ecart de la petite misere. Declarant (Ecole du whist) : « l'ecart
    sert a supprimer la carte la plus dangereuse : le singleton haut avant
    tout ». Defenseur : il garde ses petites cartes (pour passer sous le
    declarant) -> ecarte sa plus haute."""
    if v.seat in v.declarer_side:
        return best_misere_discard(v.hand)
    return _highest(v.legal)


def _misere_unload_key(v, c):
    """Defausse en misere : « on lache d'abord les cartes hautes des
    couleurs courtes » -- une haute carte protegee par des petites de sa
    couleur est moins urgente."""
    same = [r for r in _ranks(v.hand, suit_of(c)) if r < rank_of(c)]
    return rank_of(c) - 2 * len(same)


def _misere_declarer(v):
    """Declarant de misere (0 pli) ou de piccolo (exactement 1 pli)."""
    need_trick = v.kind == "piccolo" and v.tricks_won[v.seat] == 0
    later = v.still_to_play()
    if not v.current_trick:
        if need_trick:
            # « prendre son pli le plus tot possible » : une maitresse que
            # les autres doivent fournir
            takers = [c for c in v.legal if v.is_master(c)
                      and all(v.may_hold(o, suit_of(c)) for o in range(NUM_PLAYERS) if o != v.seat)]
            if takers:
                return _lowest(takers)
        # entame la plus sure : le plus de cartes non vues AU-DESSUS (quelqu'un
        # devra passer par-dessus), jamais une couleur ou personne ne fournit
        def risk(c):
            s = suit_of(c)
            above = sum(1 for u in v.unseen_in(s) if rank_of(u) > rank_of(c))
            nobody = not any(v.may_hold(o, s) for o in range(NUM_PLAYERS) if o != v.seat)
            return (nobody, -above, rank_of(c))
        return min(v.legal, key=risk)
    led = suit_of(v.current_trick[0][1])
    winner_seat = trick_winner(v.current_trick, None)
    win_card = dict(v.current_trick)[winner_seat]
    following = [c for c in v.legal if suit_of(c) == led]
    if following:
        over = [c for c in following if rank_of(c) > rank_of(win_card)]
        if need_trick and over:
            sure = [c for c in over if not later or v.is_master(c)]
            if sure:
                return _highest(sure)  # prendre avec la plus haute : c'est elle qui genera ensuite
        under = [c for c in following if rank_of(c) < rank_of(win_card)]
        if under:
            return _highest(under)  # passer juste sous le pli
        # force de depasser : dernier a jouer -> prendre avec la plus haute
        # (se debarrasser d'une carte dangereuse) ; sinon la plus petite
        return _highest(following) if not later else _lowest(following)
    options = v.legal
    if need_trick:  # garder sa carte prenante
        keep = [c for c in v.legal if not v.is_master(c)]
        options = keep or v.legal
    return max(options, key=lambda c: _misere_unload_key(v, c))


def _misere_defense(v):
    """Defense contre une misere : faire prendre des plis aux declarants.
    Piccolo : tant que le declarant n'a pas son pli, l'en empecher (le
    depasser) ; ensuite, comme une misere, lui en faire prendre un 2e.
    Misere etalee : sa main est visible -> attaquer la couleur ou sa plus
    petite carte est la plus haute (Ecole : « la moindre faiblesse sera
    exploitee »)."""
    declarers = [d for d in v.declarer_side]
    block = {d for d in declarers if v.kind == "piccolo" and v.tricks_won[d] == 0}
    if not v.current_trick:
        if block:
            masters = [c for c in v.legal if v.is_master(c)]
            if masters:
                return _highest(masters)
        best = None
        for c in v.legal:
            s = suit_of(c)
            for d in declarers:
                if d in block or not v.may_hold(d, s):
                    continue
                if d in v.revealed:
                    theirs = [x for x in v.revealed[d] if suit_of(x) == s]
                    if not theirs or rank_of(c) > min(rank_of(x) for x in theirs):
                        continue
                    score = (min(rank_of(x) for x in theirs), -rank_of(c))
                else:
                    score = (0, -rank_of(c))
                if best is None or score > best[0]:
                    best = (score, c)
        if best is not None:
            return best[1]
        options = [c for c in v.legal if any(v.may_hold(d, suit_of(c)) for d in declarers)]
        return _lowest(options or v.legal)
    led = suit_of(v.current_trick[0][1])
    winner_seat = trick_winner(v.current_trick, None)
    win_card = dict(v.current_trick)[winner_seat]
    following = [c for c in v.legal if suit_of(c) == led]
    if following:
        if winner_seat in block:
            over = [c for c in following if rank_of(c) > rank_of(win_card)]
            if over:
                return _lowest(over)
        if winner_seat in v.declarer_side:
            under = [c for c in following if rank_of(c) < rank_of(win_card)]
            return _highest(under) if under else _lowest(following)
        return _lowest(following)
    return _highest(v.legal)  # defausse : se debarrasser des hautes cartes


def _trick_state(v):
    led = suit_of(v.current_trick[0][1])
    winner_seat = trick_winner(v.current_trick, v.trump)
    win_card = dict(v.current_trick)[winner_seat]
    return led, winner_seat, win_card


def _wins_now(v, card):
    return trick_winner(v.current_trick + [(v.seat, card)], v.trump) == v.seat


def _safe_win(v, card, later):
    """`card` gagnerait le pli meme apres les adversaires restants (compte)."""
    led = suit_of(v.current_trick[0][1]) if v.current_trick else suit_of(card)
    opps = [s for s in later if s not in v.teammates(v.seat)]
    for s in opps:
        if suit_of(card) == v.trump:
            if led != v.trump and v.may_hold(s, led):
                continue  # doit fournir
            higher = [c for c in v.unseen_in(v.trump) if rank_of(c) > rank_of(card)]
            if higher and v.may_hold(s, v.trump):
                return False
        else:
            if v.may_hold(s, led):
                if not v.is_master(card):
                    return False
            elif v.may_ruff(s, led):
                return False
    return True


def _discard(v):
    """Defausse : jamais un atout si possible, jamais une carte maitresse ;
    la plus petite carte de la couleur la plus COURTE -> se mettre en renom."""
    side = [c for c in v.legal if suit_of(c) != v.trump] or v.legal
    non_master = [c for c in side if not v.is_master(c)] or side
    count = {s: sum(1 for c in v.hand if suit_of(c) == s) for s in range(NUM_SUITS)}
    return min(non_master, key=lambda c: (count[suit_of(c)], rank_of(c)))


def _follow(v):
    led, winner_seat, win_card = _trick_state(v)
    later = v.still_to_play()
    mates = v.teammates(v.seat)
    partner_winning = winner_seat in mates
    following = [c for c in v.legal if suit_of(c) == led]

    if partner_winning:
        # ne jamais surcouper ni depasser le partenaire ; le laisser tenir la main
        partner_safe = not any(s not in mates for s in later) or _partner_safe(v, win_card, led, later)
        if RUFF_IF_OPP_LATER and v.trump is not None and not following and suit_of(win_card) != v.trump and any(
                s not in mates and not v.void[s][v.trump] and v.unseen_in(v.trump) for s in later):
            partner_safe = False
        if following:
            if not partner_safe and not NO_OVERTAKE:
                masters = [c for c in following if _wins_now(v, c) and _safe_win(v, c, later)]
                if masters:
                    return _lowest(masters)
            return _lowest(following)
        # renom dans la couleur du partenaire qui n'est PAS sur de tenir le pli :
        # couper (signalement 2026-10-23 : « ne joue pas d'atout en renom »)
        ruffs = [] if partner_safe else [c for c in v.legal if suit_of(c) == v.trump and _wins_now(v, c)]
        if ruffs:
            safe = [c for c in ruffs if _safe_win(v, c, later)]
            return _lowest(safe) if safe else _lowest(ruffs)
        return _discard(v)

    winners = [c for c in v.legal if _wins_now(v, c)]
    if following:
        win_follow = [c for c in winners if suit_of(c) == led]
        if not win_follow:
            return _lowest(following)
        if not later or all(s in mates for s in later):
            return _lowest(win_follow)
        if SECOND_HAND_LOW and len(v.current_trick) == 1 and any(s in mates for s in later):
            return _lowest(following)
        safe = [c for c in win_follow if _safe_win(v, c, later)]
        if safe:
            return _lowest(safe)
        if len(v.current_trick) == 1:
            return _lowest(following)  # second : petit
        return _highest(win_follow)  # troisieme : fort, forcer l'adversaire
    # renom dans la couleur demandee
    ruffs = [c for c in winners if suit_of(c) == v.trump]
    if ruffs:
        # le partenaire joue apres moi et prendra surement : ne pas couper
        mates_later = [s for s in later if s in mates]
        if mates_later and _mate_likely_wins(v, led, mates_later, later):
            return _discard(v)
        safe = [c for c in ruffs if _safe_win(v, c, later)]
        return _lowest(safe) if safe else _lowest(ruffs)
    return _discard(v)


def _partner_safe(v, win_card, led, later):
    for s in later:
        if s in v.teammates(v.seat):
            continue
        if suit_of(win_card) == v.trump:
            if v.may_hold(s, v.trump) and (led == v.trump or not v.may_hold(s, led)):
                higher = [c for c in v.unseen_in(v.trump) if rank_of(c) > rank_of(win_card)]
                if higher:
                    return False
        else:
            if v.may_hold(s, led):
                if any(rank_of(c) > rank_of(win_card) for c in v.unseen_in(led)):
                    return False
            elif v.may_ruff(s, led):
                return False
    return True


def _mate_likely_wins(v, led, mates_later, later):
    """Le partenaire joue apres moi et est en renom (couperait) alors que
    l'adversaire suivant n'a plus d'atout, ou il tient la couleur."""
    for m in mates_later:
        if v.may_ruff(m, led):
            opps_after = [s for s in later if s not in v.teammates(v.seat) and later.index(s) > later.index(m)]
            if not any(v.may_ruff(s, led) for s in opps_after):
                return True
    return False


def _lead(v):
    mates = v.teammates(v.seat)
    opps = v.opponents()
    trump = v.trump
    my_trumps = [c for c in v.legal if suit_of(c) == trump]
    declaring = v.seat in v.declarer_side

    # 1. declarant : tirer les atouts adverses
    if declaring and my_trumps:
        outstanding = v.opponent_trumps_outstanding()
        opps_with_trumps = [s for s in opps if not v.void[s][trump]]
        if v.team_contract:
            # tirer tant qu'UN adversaire peut encore couper (signalement 2026-10-23 :
            # « laisse les adversaires utiliser leurs atouts en renom »)
            keep_drawing = outstanding > 0 and len(opps_with_trumps) >= 1
            # Ecole : « garder un atout de controle pour couper la longue
            # adverse » -- ne pas purger avec son dernier atout
            if len(my_trumps) == 1 and len(my_trumps) < len(v.legal) and not DRAW_WITH_LAST_TRUMP:
                keep_drawing = False
        else:  # solo : 3 adversaires, tirer tant que 2+ peuvent encore en avoir
            keep_drawing = outstanding > 0 and len(opps_with_trumps) >= 2
        if keep_drawing:
            masters = [c for c in my_trumps if v.is_master(c)]
            if masters:
                return _highest(masters)
            if TRUMP_LEAD_SEQUENCE:
                ranks = {rank_of(c): c for c in my_trumps}
                seq = [r for r in ranks if r >= JACK and r - 1 in ranks]
                if seq:
                    return ranks[max(seq)]
            return _lowest(my_trumps)

    # 2. jouer le renom du coequipier (il coupe), si l'adversaire apres lui ne surcoupe pas
    for m in mates:
        for s in range(NUM_SUITS):
            if s == trump or not v.may_ruff(m, s):
                continue
            cards = [c for c in v.legal if suit_of(c) == s]
            if not cards:
                continue
            order = [(v.seat + i) % NUM_PLAYERS for i in range(1, NUM_PLAYERS)]
            after_mate = order[order.index(m) + 1:]
            if not any(v.may_ruff(o, s) for o in after_mate if o in opps):
                return _lowest(cards)

    # 3. encaisser une carte maitresse dans une couleur ou aucun adversaire ne coupe
    def ruff_risk(s):
        if trump is None or s == trump:
            return False
        if any(v.may_ruff(o, s) for o in opps):
            return True
        # compte par l'enchere/les longueurs : peu de cartes restantes chez
        # les adversaires dans cette couleur -> risque de coupe
        remaining = len(v.unseen_in(s))
        for m in mates:
            if v.proposed_long(m, s):
                remaining -= 3
        return remaining < len(opps) and any(v.may_hold(o, trump) for o in opps)

    masters = [c for c in v.legal if suit_of(c) != trump and v.is_master(c) and not ruff_risk(suit_of(c))]
    if masters:
        return _highest(masters)

    # 4. se mettre en renom : singleton d'une couleur annexe si on a de
    #    l'atout -- mais jamais un PETIT singleton (Ecole du whist : « faux
    #    renom, ton partenaire va y gaspiller ses honneurs » ; « seul un
    #    honneur sec est tolere »)
    side = [c for c in v.legal if suit_of(c) != trump]
    count = {s: sum(1 for c in v.hand if suit_of(c) == s) for s in range(NUM_SUITS)}
    if my_trumps and not declaring:
        singletons = [c for c in side if count[suit_of(c)] == 1 and not v.is_master(c)
                      and rank_of(c) >= JACK and not ruff_risk(suit_of(c))]
        if singletons:
            return singletons[0]

    # 5. defaut : petite carte de la plus longue couleur annexe ; jamais dans
    #    la couleur annoncee par un adversaire (il y est long). Declarant :
    #    sans coupe adverse. Defense (Ecole) : « jouer la couleur longue de la
    #    defense pour forcer l'annonceur a couper » -- chaque coupe lui mange
    #    un atout ; et jamais l'atout (« purger, c'est lui rendre service »).
    if LEAD_PARTNER_SUIT and not declaring:
        theirs = [c for c in side if any(v.proposed_long(m, suit_of(c)) for m in mates)]
        if theirs:
            return _seq_top(theirs, SIDE_LEAD_SEQUENCE) or _lowest(theirs)
    safe_side = side if not declaring else ([c for c in side if not ruff_risk(suit_of(c))] or side)
    if safe_side:
        not_theirs = [c for c in safe_side if not any(v.proposed_long(o, suit_of(c)) for o in opps)] or safe_side
        longest = max(count[suit_of(c)] for c in not_theirs)
        pick = [c for c in not_theirs if count[suit_of(c)] == longest]
        return (SIDE_LEAD_SEQUENCE and _seq_top(pick, True)) or _lowest(pick)
    return _lowest(v.legal)


def _seq_top(cards, on):
    """Tete de sequence d'honneurs (>= 10 suivi de la carte juste en dessous),
    ou None."""
    if not on:
        return None
    by = {(suit_of(c), rank_of(c)): c for c in cards}
    tops = [c for c in cards if rank_of(c) >= JACK - 1 and (suit_of(c), rank_of(c) - 1) in by]
    return max(tops, key=rank_of) if tops else None
