"""Decideur HYBRIDE des jokers (demande utilisateur 2026-10-16, via
whist-4-it-02 : « un hybride fiable comme l'ExpertBot ») -- A PORTER A
L'IDENTIQUE en GDScript (roguelike/joker_expert.gd), verifie decision par
decision par godot_export/joker_expert_cases.json.

Une REGLE explicite par joker (quand le lancer, sur quelle cible) ; le
reseau jokers (scripts/train_jokers.py) ne departage que les cas que les
regles laissent OUVERTS. Couvre toutes les decisions a jokers de tous les
sieges :
  * bots (sieges 1-3) : tour de pli, jokers CASTABLE -> `bot_action` ;
  * joueur simule (siege 0, style "expert") : tour de pli (pli, sort,
    piege, aura) -> `player_trick_casts`, enchere -> `player_bid_casts`.
Le jeu de la carte reste l'ExpertBot. Deterministe.

Verdict d'une regle : OUI (lancer), NON (jamais ici), OUVERT_OUI / OUVERT_NON
(le reseau decide ; sans reseau -- empreinte perimee -- la valeur par defaut).
Bots : s'il y a des OUI, le premier dans BOT_PRIORITY ; sinon, s'il y a des
OUVERT, argmax des logits sur {aucun cast} + OUVERT (sans reseau : le
premier OUVERT_OUI dans BOT_PRIORITY, sinon aucun) ; sinon aucun cast.

Toutes les regles lisent une VUE (`view`) calculee depuis l'etat de la donne
avec les seules notions suivantes (memes definitions que joker_obs.py) :
camp = JokerRules.is_opponent ; meilleur actuel = rules.trick_winner(plis,
demandee, rules.trick_trump()) ; « gagnerait » = meme calcul avec la carte
ajoutee. Cartes : suit*13+rank (rank 0 = 2 ... 12 = As), couleurs ♥♦♣♠.
"""

from engine.cards import NUM_PLAYERS, suit_of, rank_of, legal_plays
from engine import jokers as J
from engine.misere import MISERE_MAX_TRICKS

JOKER_EXPERT_VERSION = 1

YES, NO, OPEN_YES, OPEN_NO = "oui", "non", "ouvert_oui", "ouvert_non"
OPEN = (OPEN_YES, OPEN_NO)

JACK, QUEEN, KING, ACE = 9, 10, 11, 12

# Ordre de priorite (premier = lance d'abord). Le plus cher / le plus
# decisif d'abord pour ne pas gaspiller l'energie sur un petit joker.
BOT_PRIORITY = ["inversion", "coup_de_vent", "coup_du_chapeau", "kermesse", "embuscade", "faux_pas", "main_lourde"]
PLAYER_PRIORITY = [
    # sorts / pieges / auras decisifs, puis cartes de pli, puis le reste
    "horloge_arretee", "valet_borgne", "inversion", "brouillard_de_la_senne", "remonte_pli", "pli_refile",
    "coup_de_vent",
    "manche_truquee", "gueule_de_bois", "dette_de_jeu", "lune_noire", "coup_du_chapeau", "kermesse",
    "treizieme_carte", "coeur_de_zinneke", "silence_du_cercle", "embuscade", "collet", "fausse_piste", "faux_pas",
    "croupier_endormi", "contre_espionnage", "fumee_de_pipe", "tournee_generale", "echange_de_manche",
    "faux_depart",
    "borgne", "double_valet", "cameleon", "atout_errant", "traitre", "carte_fantome", "zero", "miroir",
    "courant_d_air", "main_lourde", "pourboire", "regard_furtif", "livre_de_comptes",
]
BID_PRIORITY = ["main_de_fer", "pot_de_vin", "rachat", "bluff", "cartes_sur_table", "coup_d_oeil"]


# --- vue ---------------------------------------------------------------------

def contract_key(c):
    """Cle de contrat fine (stats, rentabilite) : emballage / solo / trou /
    piccolo / petite_misere / grande_misere / misere_etalee."""
    if c["type"] != "misere":
        return c["type"]
    return {"piccolo": "piccolo", "petite": "petite_misere", "grande": "grande_misere",
            "etalee": "misere_etalee"}.get(c.get("kind"), "misere")


def declarer_side(c):
    side = set(c.get("declarers") or [c["declarer"]])
    if c.get("partner") is not None and c.get("partner", -1) >= 0:
        side.add(c["partner"])
    return side


def view(hand, me):
    """Tout ce que lisent les regles, pour le siege `me` a son tour de pli
    (avant sa carte)."""
    g, rules, state, c = hand.game, hand.rules, hand.state, hand.contract
    trick = list(g.current_trick)
    led = suit_of(trick[0][1]) if trick else None
    eff = rules.trick_trump()
    my = list(g.hands[me])
    legal = legal_plays(my, led) if trick else list(my)
    side = declarer_side(c)
    misere = c["type"] == "misere"
    v = {
        "me": me, "trick_count": g.trick_count, "remaining": rules.hand_tricks() - g.trick_count,
        "position": len(trick), "led": led, "trump": g.trump_suit, "eff_trump": eff,
        "contract": contract_key(c), "misere": misere, "level": (MISERE_MAX_TRICKS[c["kind"]] if misere else (c.get("level") or getattr(g, "level", 0) or 0)),
        "declares": me in side, "misere_declarer": misere and me in (c.get("declarers") or []),
        "hand": my, "legal": legal, "energy": state.energy[me], "suspicion": state.suspicion,
        "joker_hand": list(state.joker_hand[me]), "piles": len(state.draw_pile[me]) + len(state.discard[me]),
        "tricks_won": list(g.tricks_won), "declarer_tricks": sum(g.tricks_won[s] for s in side),
        "my_tricks": g.tricks_won[me],
        "opponents": [s for s in range(NUM_PLAYERS) if rules.is_opponent(me, s)],
        "friends": [s for s in range(NUM_PLAYERS) if s != me and not rules.is_opponent(me, s)],
        "side": sorted(side),
        "armed": sorted(e.id for e in rules.effects if e.owner == me and not e.spent),
        "enemy_jokers": sum(len(state.joker_hand[s]) for s in range(NUM_PLAYERS) if rules.is_opponent(me, s)),
        "eye": getattr(hand, "eye", "") if me == J.PLAYER else "",
        "best_seat": None, "best_card": None, "best_friendly": False, "enemy_ruff_best": False,
        "can_win_now": False, "forced_win": False, "wins_with": {},
    }
    if trick:
        best = rules.trick_winner(trick, led, eff)
        bc = [cd for s, cd in trick if s == best][0]
        v["best_seat"], v["best_card"] = best, bc
        v["best_friendly"] = not rules.is_opponent(me, best)
        v["enemy_ruff_best"] = (rules.is_opponent(me, best) and eff is not None and led != eff and suit_of(bc) == eff)
        wins = [rules.trick_winner(trick + [(me, x)], led, eff) == me for x in legal]
        v["can_win_now"] = any(wins)
        v["forced_win"] = bool(wins) and all(wins)
        # cartes de joker hypothetiques (Cameleon V depuis 6c54, Double Valet V, Borgne As de la demandee,
        # Atout Errant 2 d'atout) : gagneraient-elles le pli tel quel ?
        hyp = {"cameleon": led * 13 + JACK, "double_valet": led * 13 + JACK, "borgne": led * 13 + ACE}
        if eff is not None:
            hyp["atout_errant"] = eff * 13 + 0
        v["wins_with"] = {k: rules.trick_winner(trick + [(me, x)], led, eff) == me for k, x in hyp.items()}
        if eff is not None and led != eff:  # Manche Truquee : ma meilleure carte legale comptee atout
            top = max(legal, key=rank_of)
            trumps = [rank_of(cd) for s, cd in trick if suit_of(cd) == eff]
            v["truquee_wins"] = all(rank_of(top) > r for r in trumps) and not v["best_friendly"]
        # Traitre (pli inverse : la plus faible de la demandee gagne, ma carte ne concourt pas)
        lows = [(rank_of(cd), s) for s, cd in trick if suit_of(cd) == led]
        v["inverted_winner"] = min(lows)[1] if lows else trick[0][0]
    return v


def _count(v, pred):
    return sum(1 for x in v["hand"] if pred(x))


def aces(v):
    return _count(v, lambda x: rank_of(x) == ACE)


def faces(v):
    return _count(v, lambda x: rank_of(x) in J.FACE_RANKS)


def trumps(v):
    return 0 if v["trump"] is None else _count(v, lambda x: suit_of(x) == v["trump"])


def honours(v):
    """As + roi + atouts maitres (D/V d'atout) : la force du siege."""
    t = v["trump"]
    return _count(v, lambda x: rank_of(x) >= KING or (t is not None and suit_of(x) == t and rank_of(x) >= JACK))


def want_trick(v):
    return not v["misere"]


def owned(v, jid):
    return jid in v["armed"]


def spare_energy(v, jid):
    return v["energy"] - J.jcost(jid)


def hopeless_declarer(v):
    """Camp declarant hors misere qui ne peut plus faire son contrat."""
    return not v["misere"] and v["declares"] and v["level"] - v["declarer_tricks"] > v["remaining"]


def contract_made(v):
    return not v["misere"] and v["level"] > 0 and v["declarer_tricks"] >= v["level"]


# --- regles : tour de pli -------------------------------------------------------
# chaque regle -> (verdict, id_regle) ; id_regle = "<joker>:<cas>"

# Cas fermes (OUI/NON) des bots rendus OUVERTS (le reseau decide) : diagnostic
# (scripts/audit_joker_*.py) ; vide en production.
SOFTEN = set()


def rule(v, jid):
    r = RULES.get(J.base_id(jid))
    if r is None:
        return NO, J.base_id(jid) + ":sans_regle"
    verdict, case = r(v)
    rid = "%s:%s" % (J.base_id(jid), case)
    if verdict in (YES, NO) and (rid in SOFTEN or "*" in SOFTEN):
        verdict = OPEN_YES if verdict == YES else OPEN_NO
    return verdict, rid


def _coup_de_vent(v):
    # l'atout adverse bat le pli alors que sans atout un ami ou moi le gagnerait
    if not v["enemy_ruff_best"] or v["misere"]:
        return NO, "pas_de_coupe_adverse"
    if v["position"] == 0:
        return NO, "entame"
    return YES, "annule_la_coupe"


def _main_lourde(v):
    if v["piles"] == 0 or spare_energy(v, "main_lourde") < 1:
        return NO, "rien_a_piocher_ou_plus_d_energie"
    if v["remaining"] < 4:
        return NO, "trop_tard"
    return OPEN_NO, "pioche"  # audit : -0.31 +- 0.27


def _embuscade(v):
    if v["eff_trump"] is None or v["misere"]:
        return NO, "sans_atout"
    if owned(v, "embuscade"):
        return NO, "deja_armee"  # audit : -1.17 +- 0.16
    if v["remaining"] < 4:
        return OPEN_YES, "tard"
    if v["declares"] and trumps(v) >= 5:
        return OPEN_NO, "je_tiens_l_atout"
    return OPEN_NO, "contre_les_coupes"  # audit : -1.20 +- 0.38 en moyenne, le reseau choisit le moment


def _kermesse(v):
    if v["misere"]:
        return NO, "misere"
    if v["remaining"] < 5:
        return NO, "tard"  # audit : -3.70
    if faces(v) >= 4:
        return OPEN_YES, "figures_en_main"
    # veto mesure a -16 pts/match contre le reseau seul (2026-10-16) : le reseau sait l'utiliser ici
    return OPEN_NO, "peu_de_figures"


def _inversion(v):
    if v["misere"]:
        return (NO, "declarant_misere") if v["misere_declarer"] else (OPEN_NO, "defense_misere")
    if v["declares"]:
        return NO, "je_declare"  # audit : -17.9
    if v["remaining"] < 6:
        return OPEN_NO, "tard"
    if v["level"] >= 8:
        return OPEN_YES, "gros_contrat_adverse"
    if honours(v) <= 1:
        return OPEN_YES, "main_basse"
    return OPEN_NO, "defense_ordinaire"


def _coup_du_chapeau(v):
    if v["remaining"] < 4:
        return OPEN_NO, "tard"
    if v["misere"]:
        return OPEN_YES, "misere"  # audit : +1.24 +- 0.13
    return OPEN_NO, "main"  # audit : negatif quelle que soit la main, le reseau choisit


def _faux_pas(v):
    if v["remaining"] < 3:
        return OPEN_YES, "tard"
    if v["remaining"] >= 9:
        return OPEN_NO, "tot"  # audit : -0.71 +- 0.15
    return OPEN_YES, "les_defausses_arrivent"


# --- joueur : cartes de pli ------------------------------------------------------

def _hyp_win(v, jid):
    return v["position"] > 0 and v["wins_with"].get(jid, False)


def _cameleon(v):
    if not want_trick(v) or v["best_friendly"]:
        return NO, "pas_besoin"
    return (YES, "dame_gagne_en_dernier") if v["position"] == 3 and _hyp_win(v, "cameleon") else (NO, "ne_gagne_pas")


def _double_valet(v):
    if not want_trick(v) or v["best_friendly"]:
        return NO, "pas_besoin"
    return (YES, "valet_gagne_en_dernier") if v["position"] == 3 and _hyp_win(v, "double_valet") else (NO, "ne_gagne_pas")


def _borgne(v):
    if not want_trick(v) or v["best_friendly"] or v["position"] == 0:
        return NO, "pas_besoin"
    if not _hyp_win(v, "borgne"):
        return NO, "coupe"
    if v["position"] == 3 or not any(suit_of(x) == v["led"] and rank_of(x) == ACE for x in v["hand"]):
        return YES, "as_de_la_demandee"
    return NO, "j_ai_l_as"


def _atout_errant(v):
    if not want_trick(v) or v["best_friendly"]:
        return NO, "pas_besoin"
    return (YES, "petit_atout_gagne_en_dernier") if v["position"] == 3 and _hyp_win(v, "atout_errant") else (NO, "ne_gagne_pas")


def _lose_card(v):
    """Zero / Miroir / Carte Fantome : le declarant de misere oblige de prendre."""
    return v["misere_declarer"] and v["position"] > 0 and v["forced_win"]


def _zero(v):
    return (YES, "evite_le_pli_de_misere") if _lose_card(v) else (NO, "pas_besoin")


def _miroir(v):
    return (YES, "evite_le_pli_de_misere") if _lose_card(v) else (NO, "pas_besoin")


def _carte_fantome(v):
    if _lose_card(v):
        return YES, "annule_le_pli_de_misere"
    if want_trick(v) and v["position"] == 3 and not v["best_friendly"] and not v["can_win_now"]:
        return YES, "annule_un_pli_adverse"
    return NO, "pas_besoin"


def _traitre(v):
    if v["position"] != 3:
        return NO, "pas_en_dernier"
    w = v["inverted_winner"]
    if v["misere"]:
        if v["misere_declarer"]:
            return (YES, "un_autre_prend") if v["forced_win"] and w != v["me"] else (NO, "pas_besoin")
        return (YES, "le_declarant_prend") if w in v["side"] and v["best_seat"] not in v["side"] else (NO, "pas_besoin")
    if not v["best_friendly"] and w in v["friends"] and not v["can_win_now"]:
        return YES, "renverse_vers_un_ami"
    return NO, "pas_besoin"


# --- joueur : sorts -----------------------------------------------------------------

def _never_info(v):
    return NO, "information_seulement"


def _courant_d_air(v):
    if len(v["hand"]) <= v["remaining"]:
        return NO, "pas_de_carte_en_trop"  # CourantDAir.on_play -> error "hand"
    # 6c54 : defausse la plus FORTE carte -> un joker de misere
    return (YES, "jette_ma_plus_forte") if v["misere_declarer"] else (NO, "garde_ma_plus_forte")


def _tournee_generale(v):
    return (YES, "soupcon_haut") if v["suspicion"] >= 6 else (NO, "soupcon_bas")


def _gueule_de_bois(v):
    # 6c54 : les adversaires jouent leur plus FORTE carte legale -> attaque et defense d'une misere
    if not v["misere"]:
        return NO, "hors_misere"
    return (YES, "force_les_hautes") if v["position"] <= 1 else (NO, "trop_tard_dans_le_pli")


def _echange_de_manche(v):
    if v["misere"]:
        return NO, "misere"
    if v["remaining"] < 4:
        return NO, "tard"
    return YES, "prend_une_carte_adverse"


def _remonte_pli(v):
    if v["misere"]:
        return NO, "misere"
    if not any(v["tricks_won"][s] > 0 for s in v["opponents"]):
        return NO, "aucun_pli_adverse"
    return YES, "reprend_un_pli"


def _pli_refile(v):
    """6c54 : mon dernier pli passe a l'adversaire choisi (joker de misere)."""
    if not v["misere"] or v["my_tricks"] <= 0:
        return NO, "hors_misere_ou_aucun_pli"
    return (YES, "rend_un_pli") if v["misere_declarer"] else (YES, "charge_le_declarant")


def _faux_depart(v):
    if v["position"] != 3 or v["best_friendly"] or v["can_win_now"] or v["remaining"] < 2:
        return NO, "pas_besoin"
    return YES, "garde_la_main_au_camp"


def _manche_truquee(v):
    if not want_trick(v) or v["position"] != 3 or v["can_win_now"]:
        return NO, "pas_besoin"
    return (YES, "ma_carte_coupe") if v.get("truquee_wins") else (NO, "ne_gagne_pas")


def _valet_borgne(v):
    if v["misere"]:
        if v["misere_declarer"] and v["my_tricks"] > v["level"]:
            return YES, "misere_chutee"
        return NO, "misere_en_cours"
    if hopeless_declarer(v):
        return YES, "contrat_perdu"
    if not v["declares"] and contract_made(v):
        return YES, "contrat_adverse_fait"
    return NO, "donne_jouable"


def _horloge_arretee(v):
    if v["misere"]:
        if v["misere_declarer"] and v["my_tricks"] <= v["level"] and not v["forced_win"]:
            return YES, "fige_la_misere"
        return NO, "misere"
    # 6c54 : les plis restants vont au camp qui en a le plus -> la defense doit mener
    defence = sum(v["tricks_won"]) - v["declarer_tricks"]
    if (not v["declares"] and v["level"] - v["declarer_tricks"] >= 2 and v["remaining"] >= 2
            and defence > v["declarer_tricks"]):
        return YES, "contrat_adverse_pas_fait"
    return NO, "pas_besoin"


def _pourboire(v):
    # pas de points : seulement l'energie qui serait perdue en fin de donne
    if v["remaining"] <= 2 and v["can_win_now"] and v["energy"] >= 1:
        return YES, "energie_restante"
    return NO, "garde_l_energie"


# --- joueur : pieges -----------------------------------------------------------------

def _fausse_piste(v):
    if owned(v, "fausse_piste") or v["enemy_jokers"] == 0:
        return NO, "inutile"
    return (YES, "contre_les_jokers") if v["remaining"] >= 6 else (NO, "tard")


def _collet(v):
    # 6c54 : ma prochaine carte ne peut pas gagner son pli -> la sortie d'une misere
    if not v["misere_declarer"] or owned(v, "collet"):
        return NO, "hors_ma_misere_ou_arme"
    return (YES, "evite_un_pli_force") if v["forced_win"] else (NO, "je_peux_passer_dessous")


def _contre_espionnage(v):
    if owned(v, "contre_espionnage") or not v["eye"]:
        return NO, "inutile"
    return (YES, "soupcon_haut") if v["suspicion"] >= 6 else (NO, "soupcon_bas")


def _croupier_endormi(v):
    if owned(v, "croupier_endormi") or v["remaining"] < 5:
        return NO, "tard_ou_arme"
    if v["misere_declarer"] or honours(v) <= 1:
        return YES, "je_vais_perdre_des_plis"
    return NO, "main_forte"


def _dette_de_jeu(v):
    if v["position"] != 3 or v["best_friendly"] or v["can_win_now"]:
        return NO, "pas_sur"
    return YES, "le_gagnant_paie"


# --- joueur : auras ------------------------------------------------------------------

def _fumee_de_pipe(v):
    suspects = sum(1 for j in v["joker_hand"] if J.jsuspicion(j) > 0)
    return (YES, "cache_le_soupcon") if v["suspicion"] >= 5 and suspects >= 1 else (NO, "inutile")


def _silence_du_cercle(v):
    return (YES, "bloque_les_adversaires") if v["enemy_jokers"] >= 2 and v["remaining"] >= 4 else (NO, "inutile")


def _lune_noire(v):
    if v["trump"] not in J.RED_SUITS or v["eff_trump"] not in J.RED_SUITS or v["misere"]:
        return NO, "atout_pas_rouge"
    if v["declares"]:
        return NO, "je_declare"
    return (YES, "coupe_l_atout_adverse") if trumps(v) <= 2 and v["remaining"] >= 5 else (NO, "j_ai_de_l_atout")


def _coeur_de_zinneke(v):
    if v["misere"] or v["remaining"] < 5:
        return NO, "misere_ou_tard"
    good = _count(v, lambda x: suit_of(x) == J.HEARTS and (rank_of(x) >= QUEEN or v["trump"] == J.HEARTS))
    return (YES, "coeurs_maitres") if good >= 2 else (NO, "peu_de_coeurs")


def _brouillard(v):
    # 6c54 : ma plus longue couleur est l'atout des plis que j'entame
    if v["misere"]:
        return NO, "misere"
    return (YES, "atout_a_mon_entame") if v["remaining"] >= 6 else (NO, "tard")


def _treizieme_carte(v):
    if v["misere"]:
        return NO, "misere"
    return (YES, "dernier_pli") if v["remaining"] >= 2 else (NO, "trop_tard")


RULES = {
    "coup_de_vent": _coup_de_vent, "main_lourde": _main_lourde, "embuscade": _embuscade, "kermesse": _kermesse,
    "inversion": _inversion, "coup_du_chapeau": _coup_du_chapeau, "faux_pas": _faux_pas,
    "cameleon": _cameleon, "double_valet": _double_valet, "borgne": _borgne, "atout_errant": _atout_errant,
    "zero": _zero, "miroir": _miroir, "carte_fantome": _carte_fantome, "traitre": _traitre,
    "coup_d_oeil": _never_info, "regard_furtif": _never_info, "livre_de_comptes": _never_info,
    "courant_d_air": _courant_d_air, "tournee_generale": _tournee_generale, "gueule_de_bois": _gueule_de_bois,
    "echange_de_manche": _echange_de_manche, "remonte_pli": _remonte_pli, "pli_refile": _pli_refile, "faux_depart": _faux_depart,
    "manche_truquee": _manche_truquee, "valet_borgne": _valet_borgne, "horloge_arretee": _horloge_arretee,
    "pourboire": _pourboire,
    "fausse_piste": _fausse_piste, "collet": _collet, "contre_espionnage": _contre_espionnage,
    "croupier_endormi": _croupier_endormi, "dette_de_jeu": _dette_de_jeu,
    "fumee_de_pipe": _fumee_de_pipe, "silence_du_cercle": _silence_du_cercle, "lune_noire": _lune_noire,
    "coeur_de_zinneke": _coeur_de_zinneke, "brouillard_de_la_senne": _brouillard,
    "treizieme_carte": _treizieme_carte,
}


def target(v, jid):
    """Cible des jokers qui en demandent une (-1 sinon)."""
    b = J.base_id(jid)
    if b == "dette_de_jeu":
        return v["best_seat"]
    if b == "brouillard_de_la_senne":
        return J.longest_suit(v["hand"])  # couleur (6c54)
    if b == "pli_refile":
        decl = [s for s in v["opponents"] if s in v["side"]]
        defs = [s for s in v["opponents"] if s not in v["side"]]
        return ((defs or decl) if v["misere_declarer"] else (decl or v["opponents"]))[0]
    if b == "faux_depart":
        return v["me"] if aces(v) >= 1 or not v["friends"] else v["friends"][0]
    if b in ("echange_de_manche", "coup_d_oeil", "regard_furtif"):
        decl = [s for s in v["opponents"] if s in v["side"]]
        return (decl or v["opponents"] or [s for s in range(NUM_PLAYERS) if s != v["me"]])[0]
    return -1


# --- bots ---------------------------------------------------------------------------

def bot_action(hand, seat, logits=None, trace=None):
    """-> action 0..ACTION_DIM-1 (0 = aucun cast). `logits` (reseau) ou None
    (regles seules). `trace` (dict) recoit verdicts et source."""
    mask = hand.legal_casts(seat)
    if not any(mask[1:]):
        return _done(trace, 0, "regle:rien_de_lancable", {})
    v = view(hand, seat)
    verdicts = {}
    for a in range(1, J.ACTION_DIM):
        if mask[a]:
            verdicts[J.CASTABLE[a - 1]] = rule(v, J.CASTABLE[a - 1])
    order = [j for j in BOT_PRIORITY if j in verdicts]
    firm = [j for j in order if verdicts[j][0] == YES]
    if firm:
        return _done(trace, J.CASTABLE.index(firm[0]) + 1, "regle:" + verdicts[firm[0]][1], verdicts)
    opened = [j for j in order if verdicts[j][0] in OPEN]
    if not opened:
        return _done(trace, 0, "regle:aucun_oui", verdicts)
    if logits is not None:
        cands = [0] + [J.CASTABLE.index(j) + 1 for j in opened]
        best = max(cands, key=lambda i: logits[i])  # premier en cas d'egalite (ordre : aucun, puis priorite)
        return _done(trace, best, "reseau", verdicts)
    yes = [j for j in opened if verdicts[j][0] == OPEN_YES]
    if yes:
        return _done(trace, J.CASTABLE.index(yes[0]) + 1, "regle:" + verdicts[yes[0]][1], verdicts)
    return _done(trace, 0, "regle:ouvert_non_sans_reseau", verdicts)


def _done(trace, action, source, verdicts):
    if trace is not None:
        trace.update({"action": action, "source": source,
                      "verdicts": {j: {"verdict": vv, "regle": r} for j, (vv, r) in verdicts.items()}})
    return action


# --- joueur (siege 0) --------------------------------------------------------------------

def player_trick_choice(hand, candidates):
    """Premier joker (PLAYER_PRIORITY) que les regles lancent a ce tour de
    pli parmi `candidates` (Player._candidates) -> (jid, cible, id_regle) ou
    None. OUVERT_* = valeur par defaut (pas de reseau cote joueur)."""
    v = view(hand, J.PLAYER)
    rank = {j: i for i, j in enumerate(PLAYER_PRIORITY)}
    for jid in sorted(dict.fromkeys(candidates), key=lambda j: (rank.get(J.base_id(j), 999), j)):
        verdict, rid = rule(v, jid)
        if verdict in (YES, OPEN_YES):
            return jid, target(v, jid), rid
    return None


def player_wants(hand, jid):
    """Verdict de la regle pour un joker du joueur (test de rentabilite : « bien joue »)."""
    verdict, rid = rule(view(hand, J.PLAYER), jid)
    return verdict in (YES, OPEN_YES), rid


# --- joueur : enchere -------------------------------------------------------------------

def bid_view(cards, state, me=J.PLAYER):
    from agents import expert_bot
    best = max(range(4), key=lambda s: (expert_bot.estimate_tricks(cards, s), -s))
    return {"hand": sorted(cards), "energy": state.energy[me], "suspicion": state.suspicion,
            "est_tricks": expert_bot.estimate_tricks(cards, best), "best_suit": best,
            "aces": sum(1 for x in cards if rank_of(x) == ACE),
            "joker_hand": list(state.joker_hand[me])}


def bid_rule(bv, jid):
    b = J.base_id(jid)
    strong = bv["est_tricks"] >= 6  # assez pour un solo / un emballage meneur
    if b == "main_de_fer":  # 6c54 : choisit l'atout de son emballage / solo
        return (YES, "main_de_fer:main_forte") if strong else (NO, "main_de_fer:main_moyenne")
    if b == "pot_de_vin":  # 6c54 : 1er pli de sa misere a 0
        return (YES, "pot_de_vin:misere_probable") if bv["est_tricks"] <= 2 else (NO, "pot_de_vin:pas_de_misere")
    if b == "rachat":
        return YES, "rachat:pli_de_plus"
    if b == "bluff":
        return (YES, "bluff:je_vais_declarer") if strong else (NO, "bluff:je_ne_declarerai_pas")
    if b == "cartes_sur_table":
        need = sum(J.jcost(j) for j in bv["joker_hand"] if J.jtype(j) != "enchere")
        return (YES, "cartes_sur_table:energie_utile") if need > bv["energy"] else (NO, "cartes_sur_table:energie_suffit")
    return NO, b + ":information_seulement"


def bid_target(jid, me=J.PLAYER, bv=None):
    if J.base_id(jid) == "main_de_fer":  # couleur (6c54)
        return bv["best_suit"] if bv is not None else -1
    return (me + 1) % NUM_PLAYERS if J.base_id(jid) == "coup_d_oeil" else -1


def player_bid_choice(cards, state, candidates):
    bv = bid_view(cards, state)
    rank = {j: i for i, j in enumerate(BID_PRIORITY)}
    for jid in sorted(dict.fromkeys(candidates), key=lambda j: (rank.get(J.base_id(j), 999), j)):
        verdict, rid = bid_rule(bv, jid)
        if verdict in (YES, OPEN_YES):
            return jid, bid_target(jid, bv=bv), rid
    return None
