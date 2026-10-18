"""Bareme de points -- port fidele de game/scoring.gd (projet Godot
whist-4-it), lui-meme verifie cellule par cellule contre
whist-phi.vercel.app (voir WHIST-RULES-REFERENCE.txt cote Godot). Ce
fichier DOIT rester en synchro avec l'original : toute divergence rend
l'IA incompatible avec le vrai jeu. Contrairement au reste de engine/, ce
n'est PAS une reinvention volontaire -- ce sont des regles de jeu reelles,
portees telles quelles.
"""

EMBALLAGE_CEILING_TRICKS = 13  # plis dans une donne, pas un niveau de contrat
EMBALLAGE_BASE = 5
EMBALLAGE_STEP = 2
EMBALLAGE_GRAND_SLAM_BONUS = 5

SOLO_CEILING_TRICKS = 8  # au-dela, un Solo ne marque pas plus (voir Abondance)
SOLO_BASE = 5
SOLO_SUCCESS_STEP = 2
SOLO_FAILURE_STEP = 1  # asymetrique : l'echec penalise moins fort que le succes ne recompense

ABONDANCE_MIN_LEVEL = 9
ABONDANCE_MAX_LEVEL = 12
SOLO_CHELEM_LEVEL = 13  # meme formule qu'Abondance (voir abondance_base)

TROU_GRAND_SLAM_TRICKS = 13
TROU_GRAND_SLAM_VALUE = 30
TROU_FLAT_VALUE = 16

MISERE_ORDER = ["petite", "piccolo", "grande", "etalee"]
MISERE_UNIT = {"petite": 6, "piccolo": 8, "grande": 12, "etalee": 16}
MISERE_MAX_TRICKS = {"petite": 0, "piccolo": 1, "grande": 0, "etalee": 0}
# Seule Petite Misere differe : chaque siege ecarte 1 carte avant de jouer,
# donc 12 plis au lieu de 13 (voir engine/misere.py:discard_for_kind).
MISERE_HAND_SIZE = {"petite": 12, "piccolo": 13, "grande": 13, "etalee": 13}
# (n declarants, k reussites) -> multiplicateur de MISERE_UNIT[kind] pour
# {gagnant, perdant, adversaire}. None = role absent pour ce (n, k).
MISERE_MULTIPLIER_TABLE = {
    (1, 0): {"winner": None, "loser": -3, "opponent": 1},
    (1, 1): {"winner": 3, "loser": None, "opponent": -1},
    (2, 0): {"winner": None, "loser": -2, "opponent": 2},
    (2, 1): {"winner": 4, "loser": -4, "opponent": 0},
    (2, 2): {"winner": 2, "loser": None, "opponent": -2},
    (3, 0): {"winner": None, "loser": -1, "opponent": 3},
    (3, 1): {"winner": 5, "loser": -3, "opponent": 1},
    (3, 2): {"winner": 3, "loser": -5, "opponent": -1},
    (3, 3): {"winner": 1, "loser": None, "opponent": -3},
    (4, 0): {"winner": None, "loser": 0, "opponent": None},
    (4, 1): {"winner": 6, "loser": -2, "opponent": None},
    (4, 2): {"winner": 4, "loser": -4, "opponent": None},
    (4, 3): {"winner": 2, "loser": -6, "opponent": None},
    (4, 4): {"winner": 0, "loser": None, "opponent": None},
}


def emballage_success(contract_min, tricks_taken):
    delta = tricks_taken - contract_min
    value = EMBALLAGE_BASE + EMBALLAGE_STEP * delta
    if tricks_taken >= EMBALLAGE_CEILING_TRICKS and contract_min < EMBALLAGE_CEILING_TRICKS:
        value += EMBALLAGE_GRAND_SLAM_BONUS
    return value


def emballage_failure(contract_min, tricks_taken):
    shortfall = contract_min - tricks_taken
    return -(EMBALLAGE_BASE + EMBALLAGE_STEP * shortfall)


def emballage_or_trou_team_value(contract_min, tricks_taken):
    """Valeur donnee EN ENTIER a CHAQUE membre du camp declarant (pas
    divisee en deux) ; l'autre camp recoit l'inverse, chacun en entier
    aussi -- voir team_trump.py:TeamTrumpDeal.team_value_scores."""
    if tricks_taken >= contract_min:
        return emballage_success(contract_min, tricks_taken)
    return emballage_failure(contract_min, tricks_taken)


def solo_success(contract_min, tricks_taken):
    delta = tricks_taken - contract_min
    capped_delta = min(delta, SOLO_CEILING_TRICKS - contract_min)
    return SOLO_BASE + SOLO_SUCCESS_STEP * capped_delta


def solo_failure(contract_min, tricks_taken):
    shortfall = contract_min - tricks_taken
    return -(SOLO_BASE + SOLO_FAILURE_STEP * shortfall)


def solo_base_value(contract_min, tricks_taken):
    if tricks_taken >= contract_min:
        return solo_success(contract_min, tricks_taken)
    return solo_failure(contract_min, tricks_taken)


def apply_solo_result(base_value):
    """1 contre 3 : le/la soliste marque 3x la valeur de base, chacun des 3
    adversaires en perd 1x -- toujours somme nulle."""
    return {"soloist": base_value * 3, "opponent": -base_value}


def abondance_base(level):
    return 5 * level - 35


def abondance_result(level, tricks_taken):
    """Plat (pas de gradation) : succes/echec uniquement, meme formule pour
    Abondance (9-12) et Solo Chelem (13). Reussite ssi tricks_taken>=level
    -- pour Solo Chelem (level=13) ca ne peut donc reussir qu'a exactement
    13 plis."""
    base = abondance_base(level)
    base_value = base if tricks_taken >= level else -base
    return apply_solo_result(base_value)


def trou_result(level, tricks_taken):
    if tricks_taken >= TROU_GRAND_SLAM_TRICKS:
        return TROU_GRAND_SLAM_VALUE
    return TROU_FLAT_VALUE if tricks_taken >= level else -TROU_FLAT_VALUE


def misere_result(kind, n, k):
    unit = MISERE_UNIT[kind]
    mult = MISERE_MULTIPLIER_TABLE[(n, k)]
    return {role: (m * unit if m is not None else 0) for role, m in mult.items()}
