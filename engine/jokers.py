"""Port Python des jokers du roguelike (whist-4-it/roguelike/) pour l'IA
jokers : la decision de CAST d'un bot elite/boss a son tour de pli. Le jeu
de la carte reste celui de l'ExpertBot.

Source de verite : whist-4-it/roguelike/{joker_catalog, joker_def,
joker_effects, joker_rules, match_state, rule_modifiers, rogue_match,
rogue_bosses}.gd. JOKER_SPEC_HASH ci-dessous = l'empreinte
(tools/joker_spec.mjs) des regles portees ici ; a chaque modification des
jokers cote jeu, porter le diff, mettre a jour le hash, reentrainer
(scripts/train_jokers.py) et reexporter.

v2 (2026-10-13, demande utilisateur « il faut s'entrainer contre des joueurs
qui jouent eventuellement des jokers ») : TOUT le catalogue est porte (49
jokers + versions ameliorees « id+ »), avec le Soupcon et l'Oeil, pour que
le joueur simule du siege 0 (engine/joker_match.py) lance des jokers tires
du catalogue. Les bots ne lancent toujours que les jokers `bot_casts`
(CASTABLE inchange -> ACTION_DIM inchange).

Ecarts connus (sans effet sur les decisions des bots) : Brimborion Dore
inerte (pas de donne a points doubles dans la simulation), pas de
recyclage cote joueur, pas de mises (seuil de l'Oeil = SUSPICION_MAX).
Cartes de pli : une carte de joker est un vrai entier carte (couleur/rang)
pour le moteur et l'ExpertBot, comme dans le jeu ; JokerRules.joker_cards
retient quel siege l'a posee dans le pli en cours.
"""

import random

from engine.cards import NUM_PLAYERS, suit_of, rank_of, trick_winner as classic_trick_winner

JOKER_SPEC_HASH = "68f81468d34a321a"
# Empreintes dont les REGLES sont identiques a JOKER_SPEC_HASH (changement de
# texte ou de cablage seulement) : un reseau entraine sous l'une d'elles est
# reexporte tel quel avec JOKER_SPEC_HASH. Une regle qui change -> vider la
# liste (sauf le hash courant) et reentrainer.
RULE_EQUIVALENT_HASHES = {JOKER_SPEC_HASH, "043f0376f53115aa"}
# 68f8 (whist-4-it, 18/10) : niveaux de difficulte supprimes, rogue_match.gd appelle server_add_ai_bot sans
# WhistExpertBot.DIFFICULTY (ExpertBot sur les sieges 1-3 comme avant) -> regles identiques a 043f, les
# checkpoints 043f du cloud (whist-train) restent valables.
# 043f (whist-4-it 7b320b0, mesure de rentabilite du 2026-10-18) : Kermesse camp du proprietaire seulement,
# Main de Fer apres l'annonce du proprietaire (emballer sa proposition reste permis), Treizieme Carte 2 derniers
# plis (sans effet dans sa misere), Rachat salle (sans Soupcon), Pacte dette 5, Faux Pas +3 au proprietaire,
# Voeu / Croupier / Chope : pioches au lieu de l'energie, Silence du Cercle cout 1 toute la donne, couts Gueule
# de Bois 1, Remonte-Pli 1, Collet 0, Echange de Manche et Ascete retires -> reentrainement. Fenetre de cast du
# joueur avant l'entame d'un bot : a l'ecran seulement (trick_window false sans affichage), non modelisee.
# 3e5d (whist-4-it 2cc6f15) : ExpertBot sur les sieges 1-3 a tous les actes (EXPERT_SEATS_BY_ACT supprime) ;
# le simulateur jouait deja ExpertBot partout -> ecart de l'acte I resorbe, regles portees identiques a 55e8.
# 55e8 (whist-4-it c6f620e) : renommage roguelite -> roguelike (chemins + classes) ; be8c (3fa4851) : Moine max_hands 5 -> 6
# (le simulateur joue toujours HANDS_PER_MATCH = 5, max_hands non modelise, obs inchangee) + EchangeDeManche
# _send_hand_to_seat (affichage) -> regles portees identiques a cab7, iter 29 reexportee sans reentrainement.
# cab7 (whist-4-it, couts 17/10 depuis joker_profit v2, +-1 par nuit ; remplace 895d) : cout ⚡ de traitre, dette_de_jeu,
# borgne, valet_borgne, silence_du_cercle, faux_depart 1->2, carte_fantome 0->1, manche_truquee 2->3 -> reentrainement.
# Deck de depart (cameleon, coup_de_vent, embuscade) plafonne a 1 ⚡ : le tuto les joue tous les 3 dans une donne a 3 ⚡.
# fd03 (whist-4-it af9d87d + 25a14fa) : boss au siege 2 (en face), complice du Fossoyeur = voisin
# de gauche (PLAYER + 1) -> regles changees, reentrainement. Brim Dore -> Brimborion Dore (nom seul).
# 1bbb (whist-4-it cf7c0ff) : JokerCatalog._translated() (validation des textes, validate() seulement), regles identiques a 43b8.
# 8392 (whist-4-it 51ff230) : renommage trump -> atout dans joker_effects.gd, regles identiques a 43b8.
# 43b8 (whist-4-it 2c63f60 + 6bf4a7c, revue du 2026-10-14) : Rachat, Pot-de-Vin, Inversion
# (aura, lancee par les bots), Cartes sur Table (main exposee), Silence du Cercle, Fossoyeur,
# Brim Dore + 5 nouveaux jokers -> CASTABLE change, reentrainement.
OBS_VERSION = 2  # 2 = enemy_traps rempli (joueur a jokers) ; le jeu ne remplit out[55]/[56] que si OBS_VERSION >= 2

# id -> (type, rarete, cout, bot_casts) -- joker_catalog.gd / joker_effects.gd
CATALOG = {
    "cameleon": ("pli", "comptoir", 1, False),
    "zero": ("pli", "comptoir", 0, False),
    "traitre": ("pli", "salle", 2, False),
    "coup_de_vent": ("sort", "comptoir", 1, True),
    "main_lourde": ("sort", "comptoir", 1, True),
    "embuscade": ("piege", "comptoir", 1, True),
    "kermesse": ("aura", "salle", 2, True),
    "coup_d_oeil": ("enchere", "comptoir", 1, False),
    "regard_furtif": ("sort", "arriere_salle", 1, False),
    "voeu_de_pauvrete": ("passif", "salle", 0, False),
    "miroir": ("pli", "comptoir", 1, False),
    "borgne": ("pli", "salle", 2, False),
    "carte_fantome": ("pli", "salle", 1, False),
    "double_valet": ("pli", "arriere_salle", 2, False),
    "atout_errant": ("pli", "arriere_salle", 2, False),
    "courant_d_air": ("sort", "comptoir", 0, False),
    "tournee_generale": ("sort", "comptoir", 1, False),
    "gueule_de_bois": ("sort", "salle", 1, False),
    "remonte_pli": ("sort", "salle", 1, False),
    "faux_depart": ("sort", "salle", 2, False),
    "inversion": ("aura", "arriere_salle", 3, True),
    "fausse_piste": ("piege", "comptoir", 1, False),
    "collet": ("piege", "salle", 0, False),
    "contre_espionnage": ("piege", "salle", 1, False),
    "croupier_endormi": ("piege", "salle", 1, False),
    "dette_de_jeu": ("piege", "arriere_salle", 2, False),
    "fumee_de_pipe": ("aura", "comptoir", 1, False),
    "silence_du_cercle": ("aura", "comptoir", 1, False),
    "lune_noire": ("aura", "salle", 2, False),
    "coeur_de_zinneke": ("aura", "salle", 2, False),
    "brouillard_de_la_senne": ("aura", "arriere_salle", 3, False),
    "cartes_sur_table": ("enchere", "comptoir", 0, False),
    "bluff": ("enchere", "salle", 1, False),
    "rachat": ("enchere", "salle", 2, False),
    "main_de_fer": ("enchere", "arriere_salle", 2, False),
    "pot_de_vin": ("enchere", "arriere_salle", 1, False),
    "fossoyeur_de_plis": ("passif", "salle", 0, False),
    "brimborion_dore": ("passif", "arriere_salle", 0, False),
    "manche_truquee": ("sort", "arriere_salle", 3, False),
    "coup_du_chapeau": ("aura", "salle", 2, True),
    "pourboire": ("sort", "comptoir", 1, False),
    "chope_vide": ("passif", "comptoir", 0, False),
    "faux_pas": ("piege", "salle", 1, True),
    "valet_borgne": ("sort", "cave", 2, False),
    "horloge_arretee": ("sort", "cave", 2, False),
    "treizieme_carte": ("aura", "cave", 1, False),
    "livre_de_comptes": ("sort", "cave", 1, False),
    "cartes_marquees": ("passif", "maudit", 0, False),
    "joueur_fantome": ("passif", "maudit", 0, False),
    "pacte_du_tenancier": ("passif", "maudit", 0, False),
    "heritage_empoisonne": ("passif", "maudit", 0, False),
}
STARTING_DECK = ["cameleon", "coup_de_vent", "embuscade"]  # JokerCatalog.STARTING_DECK
UPGRADE_SUFFIX = "+"  # JokerDef : cout -1 (jamais un cout 0 ni un maudit)
MAX_COPIES = 2
SUSPICION_BY_RARITY = {"arriere_salle": 2, "cave": 4}  # JokerDef.SUSPICION_BY_RARITY
FROM_DECK_IDS = ("brimborion_dore",)  # JokerDef.FROM_DECK_IDS : jamais pioches, agissent depuis le deck
BOT_TYPES = ("sort", "piege", "aura")  # RogueMatch.BOT_TYPES
# Actions de l'IA : 0 = aucun cast, 1..K = ces jokers (ids tries).
CASTABLE = sorted(i for i, (t, _r, _c, bc) in CATALOG.items() if bc and t in BOT_TYPES)
ACTION_DIM = 1 + len(CASTABLE)

DRAW_PER_HAND = 3  # MatchState
ENERGY_PER_HAND = 3
SUSPICION_MAX = 10
ELITE_DECK_SIZE = 4  # RogueRun
ELITE_SEATS = (1, 3)
ELITE_EYE = "points"
EYE_POINTS = 20  # RogueMatch
NO_ENERGY = 99
RARITIES_BY_ACT = {1: ("comptoir", "salle"), 2: ("comptoir", "salle", "arriere_salle"),
                   3: ("comptoir", "salle", "arriere_salle", "cave")}
BOSS_SEAT = 2  # en face du joueur (rapport 2026-10-15)
PLAYER = 0
BOSS_DECKS = {  # RogueBosses.DEFS
    "moine": ["coup_de_vent", "embuscade", "main_lourde", "coup_de_vent"],
    "baronne": ["embuscade", "kermesse", "coup_de_vent", "main_lourde", "embuscade"],
    "fossoyeur": ["embuscade", "kermesse", "coup_de_vent", "voeu_de_pauvrete", "main_lourde"],
    "grand_maitre": ["kermesse", "kermesse", "embuscade", "embuscade", "coup_de_vent", "coup_de_vent", "main_lourde"],
}
BOSS_SANCTIONS = {"moine": "points", "baronne": "confisque", "fossoyeur": "allumettes", "grand_maitre": "confisque"}
FACE_RANKS = (9, 10, 11)  # J, Q, K (engine/cards.py : rang 12 = As)
ACE = 12
HEARTS = 0  # WhistCard.SUITS : ♥ ♦ ♣ ♠ = 0..3
RED_SUITS = (0, 1)


def base_id(jid):
    return jid[:-1] if jid.endswith(UPGRADE_SUFFIX) else jid


def known(jid):
    return base_id(jid) in CATALOG and (jid == base_id(jid) or can_upgrade(base_id(jid)))


def jtype(jid):
    return CATALOG[base_id(jid)][0]


def jrarity(jid):
    return CATALOG[base_id(jid)][1]


def jcost(jid):
    cost = CATALOG[base_id(jid)][2]
    return cost - 1 if jid.endswith(UPGRADE_SUFFIX) else cost


def jsuspicion(jid):
    return SUSPICION_BY_RARITY.get(jrarity(jid), 0)


def acts_from_deck(jid):
    """JokerDef.acts_from_deck : maudits + FROM_DECK_IDS (MatchState.curses)."""
    return jrarity(jid) == "maudit" or base_id(jid) in FROM_DECK_IDS


def can_upgrade(jid):
    return not jid.endswith(UPGRADE_SUFFIX) and CATALOG[jid][2] > 0 and CATALOG[jid][1] != "maudit"


def elite_deck(act, rng):
    pool = sorted(i for i, (t, r, _c, bc) in CATALOG.items()
                  if t in BOT_TYPES and r in RARITIES_BY_ACT[act] and r != "cave" and bc)
    return [rng.choice(pool) for _ in range(ELITE_DECK_SIZE)]


def _declares(seat, contract):
    return (seat in (contract.get("declarers") or []) or seat == contract.get("declarer", -1)
            or seat == contract.get("partner", -1))


def _weakest(cards):
    """Godot : la carte de plus grand rank_index (la plus faible), premiere
    rencontree en cas d'egalite."""
    low = cards[0]
    for c in cards:
        if rank_of(c) < rank_of(low):
            low = c
    return low


def longest_suit(hand):
    counts = [sum(1 for c in hand if suit_of(c) == s) for s in range(4)]
    return counts.index(max(counts))


# --- effets (joker_effects.gd) ---------------------------------------------

TRICK, HAND, INSTANT = "trick", "hand", "instant"


class Effect:
    """JokerEffects.Base : TOUS les hooks du jeu, neutres par defaut. Un
    nouveau joker = une sous-classe qui surcharge ce qu'il change + une
    entree dans CATALOG et EFFECTS (registre generique, aucune reecriture).
    `play` = (siege, carte) ; `rules` = JokerRules."""
    scope = INSTANT

    def __init__(self, jid, owner):
        self.id = jid
        self.owner = owner
        self.spent = False

    def make_card(self, led_suit, rules):  # cartes de pli : la carte posee dans le pli
        return None

    def needs_target(self):
        return False

    def on_play(self, rules, target=-1):
        return {}

    def trump(self, trump, rules):
        return trump

    def competes(self, play, rules):
        return True

    def inverts(self):
        return False

    def trick_value(self, value, winner_play, rules):
        return value

    def voids_trick(self, rules):
        return False

    def on_trick_resolved(self, winner, plays, rules):
        pass

    def bot_wants(self, rules):
        return False

    def bot_casts(self):
        return False

    def pointless(self, rules):
        """Rien a changer dans le contrat de la donne -> injouable (whist-4-it
        bf79074, signalement 2026-10-12 : Coup de Vent en Grande Misere)."""
        return False

    def energy_bonus(self):
        return 0

    def adjust_deltas(self, deltas, contract, rules):
        return deltas

    def blocks(self, jid, seat, rules):
        """True = `seat` ne peut pas jouer `jid` maintenant."""
        return False

    def plays_as_trump(self, play, rules):
        """True = cette carte compte comme un atout pour ce pli (Manche Truquee)."""
        return False

    def on_hand_end(self, rules):
        """Donne comptee, avant la defausse de ses jokers (Chope Vide)."""
        pass

    def cancels(self, seat, jid, rules):
        return False

    def bot_card(self, seat, card, legal, rules):
        return card

    def forces_pass(self, seat, action, bidding):
        """bot_bid : True = l'annonce `action` ({"action": ...}) du bot `seat`
        devient un passe (Main de Fer, Pot-de-Vin)."""
        return False

    def next_leader(self, winner):
        return -1

    def tricks_in_hand(self, classic):
        return classic

    def hides_suspicion(self, seat):
        return False

    def absorbs_sanction(self):
        return False

    def on_contract(self, contract, rules):
        return False

    def cost_override(self, jid):
        return -1

    def draw_bonus(self):
        return 0

    def freezes_suspicion(self):
        return False


class TrickJoker(Effect):
    scope = TRICK

    def _own_card(self, play, rules):
        return play[0] == self.owner and rules.joker_cards.get(self.owner) == self.id


class Cameleon(TrickJoker):
    def make_card(self, led_suit, rules):
        return led_suit * 13 + 10  # Dame


class Zero(TrickJoker):
    def make_card(self, led_suit, rules):
        return led_suit * 13 + 0  # 2

    def competes(self, play, rules):
        return not self._own_card(play, rules)


class Traitor(Zero):
    def inverts(self):
        return True


class CoupDeVent(Effect):
    scope = TRICK

    def trump(self, trump, rules):
        return None

    def pointless(self, rules):
        return rules.trick_trump() is None

    def bot_casts(self):
        return True

    def bot_wants(self, rules):
        trump = rules.contract_trump()
        led = rules.led_suit()
        return trump is not None and led is not None and led != trump and any(
            rules.is_opponent(self.owner, s) and suit_of(c) == trump for s, c in rules.game.current_trick)


class MainLourde(Effect):
    def bot_casts(self):
        return True

    def bot_wants(self, rules):
        return True

    def on_play(self, rules, target=-1):
        rules.state.draw_jokers(self.owner, 1)
        return {}


class Embuscade(Effect):
    scope = HAND

    def pointless(self, rules):
        return rules.trick_trump() is None

    def bot_casts(self):
        return True

    def bot_wants(self, rules):
        return True

    def _is_enemy_ruff(self, play, rules):
        seat, card = play
        return (rules.is_opponent(self.owner, seat) and rules.current_trump is not None
                and rules.current_led != rules.current_trump and suit_of(card) == rules.current_trump)

    def competes(self, play, rules):
        return not self._is_enemy_ruff(play, rules)

    def on_trick_resolved(self, winner, plays, rules):
        if any(self._is_enemy_ruff(p, rules) for p in plays):
            self.spent = True


class Kermesse(Effect):
    scope = HAND

    def bot_casts(self):
        return True

    def bot_wants(self, rules):
        return True

    def trick_value(self, value, winner_play, rules):
        """043f : plis du camp du proprietaire seulement."""
        mine = not rules.is_opponent(self.owner, winner_play[0])
        return value + 1 if mine and rank_of(winner_play[1]) in FACE_RANKS else value


class CoupDOeil(Effect):
    """Information seulement (l'ExpertBot du siege 0 ne s'en sert pas)."""
    def needs_target(self):
        return True

    def on_play(self, rules, target=-1):
        if target < 0 or target == self.owner:
            return {"error": "target"}
        return {"seen": True}


class RegardFurtif(CoupDOeil):
    pass


class VoeuDePauvrete(Effect):
    """043f : un joker pioche tous les LOST plis perdus (plus d'energie)."""
    scope = HAND
    LOST = 3

    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.lost = 0

    def on_trick_resolved(self, winner, plays, rules):
        if winner == self.owner:
            return
        self.lost += 1
        if self.lost % self.LOST == 0:
            rules.state.draw_jokers(self.owner, 1)


class Curse(Effect):
    scope = HAND


class PacteDuTenancier(Curse):
    DEBT = 5  # 043f (10 avant)

    def energy_bonus(self):
        return 2

    def adjust_deltas(self, deltas, contract, rules):
        if _declares(self.owner, contract) and deltas[self.owner] < 0:
            deltas[self.owner] -= self.DEBT
        return deltas


class HeritageEmpoisonne(Curse):
    pass


class Miroir(TrickJoker):
    def make_card(self, led_suit, rules):
        plays = rules.game.current_trick if rules.game is not None else []
        return plays[-1][1] if plays else led_suit * 13 + 0


class Borgne(TrickJoker):
    def make_card(self, led_suit, rules):
        return led_suit * 13 + ACE

    def competes(self, play, rules):
        if not self._own_card(play, rules) or rules.current_trump is None or rules.current_led == rules.current_trump:
            return True
        return not any(suit_of(c) == rules.current_trump and s != self.owner for s, c in rules.current_plays)


class CarteFantome(Zero):
    def competes(self, play, rules):
        return True

    def voids_trick(self, rules):
        return True


class DoubleValet(TrickJoker):
    def make_card(self, led_suit, rules):
        return led_suit * 13 + 9  # Valet

    def trick_value(self, value, winner_play, rules):
        return value + 1 if winner_play[0] == self.owner else value


class AtoutErrant(TrickJoker):
    def make_card(self, led_suit, rules):
        trump = rules.trick_trump() if rules.game is not None else None
        return trump * 13 + 0 if trump is not None else led_suit * 13 + 9


class CourantDAir(Effect):
    def on_play(self, rules, target=-1):
        g = rules.game
        hand = g.hands[self.owner]
        if len(hand) <= rules.hand_tricks() - g.trick_count:
            return {"error": "hand"}
        weakest = _weakest(hand)
        hand.remove(weakest)
        rules.vanished.append(weakest)  # JokerRules.set_aside : sous le paquet ramasse (Match.finish)
        rules.state.draw_jokers(self.owner, 1)
        return {"discarded": weakest}


class TourneeGenerale(Effect):
    RELIEF = 2

    def on_play(self, rules, target=-1):
        rules.state.lower_suspicion(self.RELIEF)
        return {}


class GueuleDeBois(Effect):
    scope = TRICK

    def bot_card(self, seat, card, legal, rules):
        if not rules.is_opponent(self.owner, seat) or not legal:
            return card
        return _weakest(legal)


class RemontePli(Effect):
    def on_play(self, rules, target=-1):
        g = rules.game
        for w in reversed(rules.trick_winners):
            if rules.is_opponent(self.owner, w) and g.tricks_won[w] > 0:
                g.tricks_won[w] -= 1
                g.tricks_won[self.owner] += 1
                return {"from": w}
        return {"error": "none"}


class FauxDepart(Effect):
    scope = TRICK

    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.target = -1

    def needs_target(self):
        return True

    def on_play(self, rules, target=-1):
        if target < 0:
            return {"error": "target"}
        self.target = target
        return {}

    def next_leader(self, winner):
        return self.target


class Inversion(Effect):
    """Revue 2026-10-14 : une aura, chaque pli de la donne a la plus faible
    carte de la couleur demandee."""
    scope = HAND

    def inverts(self):
        return True

    def bot_casts(self):
        return True

    def bot_wants(self, rules):
        return (rules.contract or {}).get("type") == "misere"


class Trap(Effect):
    scope = HAND


class FauxPas(Trap):
    """Le premier adversaire qui ne fournit pas (hors carte de joker) cede
    POINTS au proprietaire (043f : il ne faisait que les perdre) ; une fois."""
    POINTS = 3

    def bot_casts(self):
        return True

    def bot_wants(self, rules):
        return True

    def on_trick_resolved(self, winner, plays, rules):
        if self.spent:
            return
        for seat, card in plays:
            if (rules.is_opponent(self.owner, seat) and seat not in rules.joker_cards
                    and suit_of(card) != rules.current_led):
                rules.add_score(seat, -self.POINTS)
                rules.add_score(self.owner, self.POINTS)
                self.spent = True
                return


class FaussePiste(Trap):
    def cancels(self, seat, jid, rules):
        if self.spent or jtype(jid) != "sort" or not rules.is_opponent(self.owner, seat):
            return False
        self.spent = True
        return True


class Collet(Trap):
    def _snared(self, play, rules):
        seat, card = play
        return (rules.is_opponent(self.owner, seat) and rank_of(card) == ACE
                and seat not in rules.joker_cards)

    def competes(self, play, rules):
        return self.spent or not self._snared(play, rules)

    def on_trick_resolved(self, winner, plays, rules):
        if any(self._snared(p, rules) for p in plays):
            self.spent = True


class ContreEspionnage(Trap):
    def absorbs_sanction(self):
        if self.spent:
            return False
        self.spent = True
        return True


class CroupierEndormi(Trap):
    STREAK = 3
    DRAWS = 2  # 043f : jokers pioches (+2 energie avant)

    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.lost = 0

    def on_trick_resolved(self, winner, plays, rules):
        self.lost = self.lost + 1 if winner != self.owner else 0
        if self.lost >= self.STREAK:
            rules.state.draw_jokers(self.owner, self.DRAWS)
            self.spent = True


class DetteDeJeu(Effect):
    scope = TRICK
    DEBT = 5

    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.target = -1

    def needs_target(self):
        return True

    def on_play(self, rules, target=-1):
        if target < 0 or not rules.is_opponent(self.owner, target):
            return {"error": "target"}
        self.target = target
        return {}

    def on_trick_resolved(self, winner, plays, rules):
        if winner == self.target:
            rules.add_score(self.target, -self.DEBT)
            rules.add_score(self.owner, self.DEBT)


class Aura(Effect):
    scope = HAND


class FumeeDePipe(Aura):
    def hides_suspicion(self, seat):
        return seat == self.owner


class SilenceDuCercle(Aura):
    """043f : toute la donne (3 plis avant)."""
    def blocks(self, jid, seat, rules):
        return rules.is_opponent(self.owner, seat)  # revue 2026-10-14 : adversaires seulement, tout type


class LuneNoire(Aura):
    def trump(self, trump, rules):
        return None if trump in RED_SUITS else trump

    def pointless(self, rules):
        return rules.trick_trump() not in RED_SUITS


class CoeurDeZinneke(Aura):
    BONUS = 2

    def on_trick_resolved(self, winner, plays, rules):
        for seat, card in plays:
            if seat == winner and winner == self.owner and suit_of(card) == HEARTS:
                rules.add_score(self.owner, self.BONUS)


class BrouillardDeLaSenne(Aura):
    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.start = 0

    def on_play(self, rules, target=-1):
        self.start = rules.game.trick_count
        return {}

    def trump(self, trump, rules):
        return (rules.game.trick_count - self.start) % 4


class CartesSurTable(Aura):
    """Revue 2026-10-14 : le contrat porte le siege expose ; les ExpertBots
    jouent en voyant sa main (WhistExpertBot._revealed, PublicView.revealed)."""
    def on_play(self, rules, target=-1):
        rules.state.gain_energy(self.owner, 1)
        self._expose(rules.contract, rules)
        return {"revealed": True}

    def on_contract(self, contract, rules):
        return self._expose(contract, rules)

    def _expose(self, contract, rules):
        if not contract or self.owner in contract.get("exposed", []):
            return False
        contract["exposed"] = list(contract.get("exposed", [])) + [self.owner]
        if rules.game is not None:
            rules.game.exposed = list(contract["exposed"])
        return True


class Bluff(Aura):
    def adjust_deltas(self, deltas, contract, rules):
        if _declares(self.owner, contract) and deltas[self.owner] < 0:
            deltas[self.owner] = int(deltas[self.owner] / 2)  # division entiere GDScript (vers 0)
        return deltas


class Rachat(Aura):
    """Rapport 2026-10-14 : le camp du proprietaire compte un pli de plus a
    la fin de la donne (pris au declarant, sinon a son partenaire, quand le
    proprietaire defend). Jamais en misere."""
    def adjust_deltas(self, deltas, contract, rules):
        ctype = contract.get("type", "")
        if ctype in ("", "misere") or rules.scorer is None:
            return deltas
        tricks = rules.game.tricks_won
        saved = list(tricks)
        base = rules.score_hand()
        side = [contract.get("declarer", -1), contract.get("partner", -1)]
        side = [-1 if x is None else x for x in side]
        if self.owner in side:
            tricks[self.owner] += 1
        else:
            giver = side[0] if side[0] >= 0 and tricks[side[0]] > 0 else side[1]
            if giver < 0 or tricks[giver] <= 0:
                return deltas
            tricks[giver] -= 1
            tricks[self.owner] += 1
        bought = rules.score_hand()
        tricks[:] = saved
        return [d + b - a for d, a, b in zip(deltas, base, bought)]


class MainDeFer(Aura):
    """043f : rien tant que son proprietaire n'a pas annonce (passe, attente et
    passe parole ne comptent pas) ; ensuite les bots passent, sauf pour
    emballer une proposition en attente du proprietaire."""
    NO_BID = (None, "pass", "wait", "pass_parole")

    def forces_pass(self, seat, action, bidding):
        if seat == self.owner or bidding.last_action[self.owner] in self.NO_BID:
            return False
        return not (action["action"] == "emballer" and bidding.pending_proposals.get(action["suit"], -1) == self.owner)


class PotDeVin(Aura):
    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.target = -1

    def needs_target(self):
        return True

    def on_play(self, rules, target=-1):
        if target < 0 or target == self.owner:
            return {"error": "target"}
        self.target = target
        return {}

    def forces_pass(self, seat, action, bidding):
        return seat == self.target


class FossoyeurDePlis(Aura):
    def on_trick_resolved(self, winner, plays, rules):
        if (rules.contract or {}).get("type") in ("emballage", "trou") and winner != self.owner and not rules.is_opponent(self.owner, winner):
            rules.add_score(self.owner, 1)


class BrimborionDore(Curse):
    pass  # agit depuis le deck ; donne a points doubles : absente de la simulation


class ValetBorgne(Aura):
    def on_play(self, rules, target=-1):
        return {"exhaust": True}

    def adjust_deltas(self, deltas, contract, rules):
        return [0 for _ in deltas]


class HorlogeArretee(Aura):
    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.stop_at = 13

    def on_play(self, rules, target=-1):
        self.stop_at = rules.game.trick_count + 1
        return {}

    def tricks_in_hand(self, classic):
        return min(classic, self.stop_at)


class TreiziemeCarte(Aura):
    """043f : les LAST derniers plis (1 avant), sans effet dans une misere de
    son proprietaire."""
    LAST = 2

    def pointless(self, rules):
        c = rules.contract or {}
        return c.get("type") == "misere" and self.owner in (c.get("declarers") or [])

    def competes(self, play, rules):
        if rules.game.trick_count < rules.hand_tricks() - self.LAST:
            return True
        return play[0] == self.owner or not any(s == self.owner for s, _c in rules.current_plays)


class LivreDeComptes(Effect):
    def on_play(self, rules, target=-1):
        rules.state.suspicion = SUSPICION_MAX
        return {"seen_all": True}


class CartesMarquees(Curse):
    def cost_override(self, jid):
        return 0 if jtype(jid) == "sort" else -1

    def freezes_suspicion(self):
        return True


class JoueurFantome(Curse):
    def draw_bonus(self):
        return 1

    def voids_trick(self, rules):
        return rules.game.trick_count % 4 == 3


class MancheTruquee(Effect):
    """Pour ce pli, la carte du proprietaire compte comme un atout de son rang."""
    scope = TRICK

    def pointless(self, rules):
        return rules.trick_trump() is None

    def plays_as_trump(self, play, rules):
        return play[0] == self.owner


class CoupDuChapeau(Aura):
    STREAK = 3
    BONUS = 6

    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.streak = 0

    def bot_casts(self):
        return True

    def bot_wants(self, rules):
        return True

    def on_trick_resolved(self, winner, plays, rules):
        self.streak = self.streak + 1 if winner == self.owner else 0
        if self.streak >= self.STREAK:
            rules.add_score(self.owner, self.BONUS)
            self.streak = 0


class Pourboire(Effect):
    scope = TRICK

    def on_trick_resolved(self, winner, plays, rules):
        if winner == self.owner:
            rules.state.win_sous(self.owner, 1)


class ChopeVide(Aura):
    """043f : un joker de plus pioche a la donne suivante (+2 energie avant)."""
    def on_hand_end(self, rules):
        if rules.state.played_this_hand[self.owner] == 0:
            rules.state.next_hand_draws[self.owner] += 1


class BaronneSees(Effect):
    """RogueBosses.BaronneSees (regle propre de la Baronne) : la premiere
    carte du joueur de chaque donne ne gagne jamais son pli."""
    scope = HAND

    def competes(self, play, rules):
        if play[0] != PLAYER:
            return True
        for seat, card in list(rules.game.play_log) + list(rules.current_plays):
            if seat == PLAYER:
                return card != play[1]
        return True


EFFECTS = {
    "cameleon": Cameleon, "zero": Zero, "traitre": Traitor, "coup_de_vent": CoupDeVent,
    "main_lourde": MainLourde, "embuscade": Embuscade, "kermesse": Kermesse, "coup_d_oeil": CoupDOeil,
    "regard_furtif": RegardFurtif, "voeu_de_pauvrete": VoeuDePauvrete, "miroir": Miroir, "borgne": Borgne,
    "carte_fantome": CarteFantome, "double_valet": DoubleValet, "atout_errant": AtoutErrant,
    "courant_d_air": CourantDAir, "tournee_generale": TourneeGenerale, "gueule_de_bois": GueuleDeBois,
    "remonte_pli": RemontePli, "faux_depart": FauxDepart,
    "inversion": Inversion, "fausse_piste": FaussePiste, "collet": Collet,
    "contre_espionnage": ContreEspionnage, "croupier_endormi": CroupierEndormi, "dette_de_jeu": DetteDeJeu,
    "fumee_de_pipe": FumeeDePipe, "silence_du_cercle": SilenceDuCercle, "lune_noire": LuneNoire,
    "coeur_de_zinneke": CoeurDeZinneke, "brouillard_de_la_senne": BrouillardDeLaSenne,
    "cartes_sur_table": CartesSurTable, "bluff": Bluff, "rachat": Rachat, "main_de_fer": MainDeFer,
    "pot_de_vin": PotDeVin, "fossoyeur_de_plis": FossoyeurDePlis,
    "brimborion_dore": BrimborionDore, "valet_borgne": ValetBorgne, "horloge_arretee": HorlogeArretee,
    "treizieme_carte": TreiziemeCarte, "livre_de_comptes": LivreDeComptes, "cartes_marquees": CartesMarquees,
    "joueur_fantome": JoueurFantome, "pacte_du_tenancier": PacteDuTenancier,
    "heritage_empoisonne": HeritageEmpoisonne, "manche_truquee": MancheTruquee,
    "coup_du_chapeau": CoupDuChapeau, "pourboire": Pourboire, "chope_vide": ChopeVide, "faux_pas": FauxPas,
}
assert set(EFFECTS) == set(CATALOG)


def make_effect(jid, owner):
    return EFFECTS[base_id(jid)](jid, owner)


# --- economie (match_state.gd) ---------------------------------------------

class MatchState:
    def __init__(self, decks, rng, suspicion=0):
        self.rng = rng
        self.draw_pile = [[j for j in d if not acts_from_deck(j)] for d in decks]
        self.curses = [[j for j in d if acts_from_deck(j)] for d in decks]
        self.joker_hand = [[] for _ in range(NUM_PLAYERS)]
        self.discard = [[] for _ in range(NUM_PLAYERS)]
        self.energy = [0] * NUM_PLAYERS
        self.suspicion = suspicion
        self.next_hand_malus = 0
        self.played_this_hand = [0] * NUM_PLAYERS  # Chope Vide
        self.next_hand_draws = [0] * NUM_PLAYERS  # Chope Vide
        self.sous_won = 0  # Pourboire (joueur seulement)
        self.confiscated = []
        self.rules = None
        for pile in self.draw_pile:
            self.rng.shuffle(pile)

    def arm_curses(self):
        self.rules.curses = [e for e in self.rules.curses if not isinstance(e, Curse)]
        for seat in range(NUM_PLAYERS):
            for jid in self.curses[seat]:
                self.rules.curses.append(make_effect(jid, seat))

    def start_hand(self):
        for seat in range(NUM_PLAYERS):
            bonus = sum(e.energy_bonus() for e in self.rules.curses if e.owner == seat) if self.rules else 0
            malus = self.next_hand_malus if seat == PLAYER else 0
            self.energy[seat] = max(0, ENERGY_PER_HAND + bonus - malus)
            self.draw_jokers(seat, DRAW_PER_HAND + self.next_hand_draws[seat] + (self.rules.draw_bonus(seat) if self.rules else 0))
        self.next_hand_malus = 0
        self.next_hand_draws = [0] * NUM_PLAYERS
        self.played_this_hand = [0] * NUM_PLAYERS

    def end_hand(self):
        for seat in range(NUM_PLAYERS):
            self.discard[seat].extend(self.joker_hand[seat])
            self.joker_hand[seat] = []

    def draw_jokers(self, seat, n):
        for _ in range(n):
            if not self.draw_pile[seat]:
                if not self.discard[seat]:
                    return
                self.draw_pile[seat] = self.discard[seat]
                self.discard[seat] = []
                self.rng.shuffle(self.draw_pile[seat])
            jid = self.draw_pile[seat].pop()
            self.joker_hand[seat].append(jid)
            if jtype(jid) == "passif" and self.rules is not None:
                self.rules.play(seat, jid)

    def win_sous(self, seat, n):
        if seat == PLAYER:
            self.sous_won += n

    def gain_energy(self, seat, n):
        self.energy[seat] += n

    def lower_suspicion(self, n):
        if self.rules is not None and self.rules.freezes_suspicion(PLAYER):
            return
        self.suspicion = max(0, self.suspicion - n)

    def cost_of(self, seat, jid):
        return self.rules.cost_of(seat, jid) if self.rules is not None else jcost(jid)

    def play_error(self, seat, jid):
        if not known(jid) or jid not in self.joker_hand[seat]:
            return "not_in_hand"
        if jtype(jid) == "passif":
            return "passive"
        # match_state.gd : "pointless" en phase PLAYING (ici : une donne liee)
        if self.rules is not None and self.rules.game is not None and make_effect(jid, seat).pointless(self.rules):
            return "pointless"
        if self.cost_of(seat, jid) > self.energy[seat]:
            return "energy"
        return ""

    def spend(self, seat, jid):
        self.energy[seat] -= self.cost_of(seat, jid)
        self.played_this_hand[seat] += 1
        self.joker_hand[seat].remove(jid)
        self.discard[seat].append(jid)
        if seat == PLAYER and not (self.rules is not None and self.rules.hides_suspicion(seat)):
            self.suspicion = min(SUSPICION_MAX, self.suspicion + jsuspicion(jid))


# --- regles (joker_rules.gd) ------------------------------------------------

def beats(ch, best, led, trump):
    """WhistTrick._beats : meme couleur -> rang ; sinon seul l'atout bat."""
    if suit_of(ch) == suit_of(best):
        return rank_of(ch) > rank_of(best)
    return trump is not None and suit_of(ch) == trump


class JokerRules:
    """JokerRules (joker_rules.gd) : les effets vivants d'une donne + les
    maledictions/regles de boss (`curses`, jamais videes). `game` = moteur
    de pli de la donne (TeamTrumpDeal / LoneDeclarerDeal), None pendant
    l'enchere. Chaque hook replie les effets vivants dans l'ordre de jeu,
    comme le jeu. `score_bonus` = points ajoutes au score du match pendant
    la donne (table.total_score cote jeu : Dette de Jeu, Coeur de Zinneke,
    Fossoyeur de Plis, sanction « points » de l'Oeil)."""

    def __init__(self, state, rng=None):
        self.state = state
        state.rules = self
        self.rng = rng or random.Random(0)
        self.effects = []
        self.curses = []
        self.game = None
        self.contract = None
        self.deal_hands = None
        self.current_led = None
        self.current_trump = None
        self.current_plays = []
        self.trick_winners = []
        self.joker_cards = {}  # siege -> joker de pli pose dans le pli en cours
        self.score_bonus = [0] * NUM_PLAYERS
        self.vanished = []  # cartes defaussees par Courant d'Air
        # Consigne du joueur a son camp {"suit", "high"} (joker_rules.gd
        # directive) : jamais posee a l'entrainement -> sans effet.
        self.directive = {}
        # scorer(game, contract) -> score de base de la donne (WhistTable._score_hand) ; une
        # fonction de module (jamais une fermeture : clone() = deepcopy garde la meme fonction)
        self.scorer = None

    def new_hand(self, game, contract):
        self.effects = []
        self.trick_winners = []
        self.joker_cards = {}
        self.score_bonus = [0] * NUM_PLAYERS
        self.vanished = []
        self.directive = {}
        self.bind(game, contract)

    def bind(self, game, contract):
        self.game = game
        self.contract = contract

    def score_hand(self):
        return self.scorer(self.game, self.contract)

    def live(self):
        return self.curses + self.effects

    def set_directive(self, suit, high=False):
        self.directive = {} if suit is None else {"suit": suit, "high": high}

    def teammates(self):
        return [s for s in (1, 2, 3) if not self.is_opponent(PLAYER, s)]

    def is_opponent(self, seat, other):
        """joker_rules.gd (whist-4-it a3d70cf) : les camps viennent du
        CONTRAT, pas de partner_of. Misere : chaque declarant seul, les
        defenseurs allies ; sinon camp = [declarer, partner] contre le reste
        (les defenseurs d'un emballage/trou/solo forment un camp) ; pas de
        contrat -> tout le monde est adversaire."""
        if other == seat:
            return False
        c = self.contract
        if not c:
            return True
        if "declarers" in c:
            return seat in c["declarers"] or other in c["declarers"]
        if "declarer" not in c:
            return True
        side = (c["declarer"], c.get("partner", -1))
        return (seat in side) != (other in side)

    def contract_trump(self):
        return self.game.trump_suit

    def led_suit(self):
        return suit_of(self.game.current_trick[0][1]) if self.game is not None and self.game.current_trick else None

    def trick_trump(self):
        trump = self.game.trump_suit
        for e in self.live():
            trump = e.trump(trump, self)
        return trump

    def hand_tricks(self):
        return self.tricks_in_hand(getattr(self.game, "num_tricks", 13))

    def add_score(self, seat, n):
        self.score_bonus[seat] += n

    def blocked(self, jid, seat):
        return any(e.blocks(jid, seat, self) for e in self.live())

    def play(self, seat, jid, target=-1):
        effect = make_effect(jid, seat)
        if self.blocked(jid, seat):
            return {"error": "silence"}
        led = self.led_suit()
        card = effect.make_card(led if led is not None else 0, self)
        if card is not None and led is None:
            return {"error": "lead"}
        if any(e.cancels(seat, jid, self) for e in self.live()):
            return {"cancelled": True}
        result = effect.on_play(self, target)
        if result.get("error"):
            return result
        if effect.scope != INSTANT:
            self.effects.append(effect)
        if card is not None:
            result["card"] = card
        return result

    def disarm(self, seat, jid):
        for e in self.effects:
            if e.owner == seat and e.id == jid:
                self.effects.remove(e)
                return

    def trick_winner(self, plays, led, trump):
        self.current_led = led
        self.current_trump = trump
        self.current_plays = list(plays)
        live = self.live()
        competing = [p for p in plays if all(e.competes(p, self) for e in live)]
        if not competing:
            return plays[0][0]
        best = competing[0]
        if any(e.inverts() for e in live):
            # joker_rules.gd : la plus faible de la couleur demandee gagne
            for p in competing:
                if suit_of(p[1]) == led and (suit_of(best[1]) != led or rank_of(p[1]) < rank_of(best[1])):
                    best = p
        else:
            def as_played(p):
                if trump is not None and any(e.plays_as_trump(p, self) for e in live):
                    return trump * 13 + rank_of(p[1])
                return p[1]
            for p in competing:
                if beats(as_played(p), as_played(best), led, trump):
                    best = p
        return best[0]

    def trick_value(self, winner, plays):
        winner_play = [p for p in plays if p[0] == winner][0]
        value = 1
        for e in self.live():
            value = e.trick_value(value, winner_play, self)
        if any(e.voids_trick(self) for e in self.live()):
            return 0
        return value

    def next_leader(self, winner):
        for e in self.live():
            if e.next_leader(winner) >= 0:
                return e.next_leader(winner)
        return winner

    def tricks_in_hand(self, classic):
        n = classic
        for e in self.live():
            n = min(n, e.tricks_in_hand(classic))
        return n

    def overrides_bot_card(self):
        return bool(self.directive) or any(type(e).bot_card is not Effect.bot_card for e in self.live())

    def bot_card(self, seat, card, legal):
        # La consigne d'abord : un joker qui force une carte l'emporte.
        if self.directive and seat in self.teammates():
            asked = [c for c in legal if suit_of(c) == self.directive["suit"]]
            if asked:
                card = (max if self.directive["high"] else min)(asked, key=rank_of)
        for e in self.live():
            card = e.bot_card(seat, card, legal, self)
        return card

    def forces_pass(self, seat, action, bidding):
        return any(e.forces_pass(seat, action, bidding) for e in self.live())

    def hides_suspicion(self, seat):
        return any(e.hides_suspicion(seat) for e in self.live())

    def absorbs_sanction(self):
        for e in self.live():
            if e.absorbs_sanction():
                return True
        return False

    def on_contract(self, contract):
        changed = False
        for e in self.live():
            changed = e.on_contract(contract, self) or changed
        return changed

    def cost_of(self, seat, jid):
        cost = jcost(jid)
        for e in self.curses:
            if e.owner == seat and e.cost_override(jid) >= 0:
                cost = min(cost, e.cost_override(jid))
        return cost

    def draw_bonus(self, seat):
        return sum(e.draw_bonus() for e in self.curses if e.owner == seat)

    def freezes_suspicion(self, seat):
        return any(e.owner == seat and e.freezes_suspicion() for e in self.curses)

    def on_trick_resolved(self, winner, plays):
        self.trick_winners.append(winner)
        for e in self.live():
            e.on_trick_resolved(winner, plays, self)
        self.effects = [e for e in self.effects if not e.spent and e.scope != TRICK]
        self.joker_cards = {}

    def adjust_deltas(self, deltas, contract):
        out = list(deltas)
        for e in self.live():
            out = e.adjust_deltas(out, contract, self)
        for e in self.live():
            e.on_hand_end(self)
        return out


# --- la donne sous jokers --------------------------------------------------

class JokerHand:
    """Une donne jouee sous JokerRules : delegue au moteur de pli et ne
    corrige que la resolution du pli (WhistTable._commit_play_only avec
    modifiers) : gagnant, valeur du pli (tricks_won += valeur), meneur
    suivant, fin de donne sur tricks_in_hand."""

    def __init__(self, game, contract, state, rules):
        self.game = game
        self.contract = contract
        self.state = state
        self.rules = rules
        self.eye = ""
        self.joker_play_idx = set()  # indices de game.play_order poses par des cartes de joker

    def step(self, seat, card):
        g = self.game
        if getattr(g, "pending_discards", None) or len(g.current_trick) < NUM_PLAYERS - 1:
            g.step(seat, card)
            return
        plays = list(g.current_trick) + [(seat, card)]
        led = suit_of(plays[0][1])
        trump = self.rules.trick_trump()
        winner = self.rules.trick_winner(plays, led, trump)
        value = self.rules.trick_value(winner, plays)
        g.step(seat, card)
        classic = g.turn if not g.done else classic_trick_winner(plays, g.trump_suit)
        g.tricks_won[classic] -= 1
        g.tricks_won[winner] += value
        leader = self.rules.next_leader(winner)
        self.rules.on_trick_resolved(winner, plays)
        if not g.done and g.trick_count >= self.rules.hand_tricks():
            g.done = True  # WhistTable._hand_over : la donne finit sur son nombre de plis
        if not g.done:
            g.turn = leader

    def step_joker_card(self, seat, card, jid):
        """Carte de joker posee dans le pli a la place d'une carte de la main
        (RogueMatch.play_joker -> table._commit_play_only) : la main ne
        diminue pas."""
        self.rules.joker_cards[seat] = jid
        self.joker_play_idx.add(len(self.game.play_order))
        self.game.hands[seat].append(card)  # le moteur la retire aussitot
        self.step(seat, card)

    def holders(self):
        return [s for s in range(NUM_PLAYERS) if s != PLAYER and (
            self.state.joker_hand[s] or self.state.draw_pile[s] or self.state.discard[s])]

    def legal_casts(self, seat):
        """Masque des actions (0 = aucun cast, toujours legal), comme
        RogueMatch._bot_turn : type de bot, play_error == "" et pas bloque."""
        mask = [True] + [False] * len(CASTABLE)
        if seat not in self.holders():
            return mask
        for i, jid in enumerate(CASTABLE):
            if self.state.play_error(seat, jid) == "" and not self.rules.blocked(jid, seat):
                mask[i + 1] = True
        return mask

    def cast(self, seat, action):
        if action == 0:
            return
        jid = CASTABLE[action - 1]
        if self.rules.play(seat, jid).get("error"):
            return
        self.state.spend(seat, jid)


def watch_eye(state, rules, sanction, jid):
    """RogueMatch._watch_eye : un joker suspect qui remplit la jauge est
    pris ; la sanction tombe, la jauge se vide."""
    if sanction == "" or state.suspicion < SUSPICION_MAX or jsuspicion(jid) == 0:
        return
    if rules.absorbs_sanction():
        sanction = "absorbed"
    if sanction == "points":
        rules.add_score(PLAYER, -EYE_POINTS)
    elif sanction == "confisque":
        if jid in state.discard[PLAYER]:
            state.discard[PLAYER].remove(jid)
        state.confiscated.append(jid)
    elif sanction == "allumettes":
        state.next_hand_malus = NO_ENERGY
    state.suspicion = 0


def heuristic_action(hand, seat):
    """RogueMatch._bot_turn actuel : le premier joker de la main (ordre de
    pioche) que le bot veut (bot_wants) et peut payer."""
    mask = hand.legal_casts(seat)
    for jid in hand.state.joker_hand[seat]:
        if jid not in CASTABLE or not mask[CASTABLE.index(jid) + 1]:
            continue
        if make_effect(jid, seat).bot_wants(hand.rules):
            return CASTABLE.index(jid) + 1
    return 0


def never_action(hand, seat):
    return 0
