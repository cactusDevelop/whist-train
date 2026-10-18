"""Machine a etats de la phase d'enchere -- role `encherisseur` (voir
DESIGN.md). Reproduit fidelement l'echelle de contrats et la legalite
tour par tour du jeu (regles du jeu, invariantes au reset), mais
integralement reecrit pour ce projet : l'ancien whist/bidding_v2.py n'est
jamais importe ni copie (archive uniquement, voir
archive/pre-reset-20260918). Verifie ligne a ligne contre le vrai
game/bidding.gd (projet Godot whist-4-it) le 2026-09-18.

Echelle des contrats, du plus faible au plus fort (LADDER, source de
verite pour savoir qui "domine" qui) -- verifiee contre
WHIST-RULES-REFERENCE.txt (projet Godot whist-4-it) :
  Emballage 8 < Solo 6 < Emballage 9 < Solo 7 < Emballage 10 <
  Petite Misere < Solo 8 < Emballage 11 < Emballage 12 < Piccolo <
  Emballage 13 < Solo 9 < Trou < Grande Misere < Solo 10 < Solo 11 <
  Misere Etalee < Solo 12 < Solo 13.

Solo/Abondance/Solo Chelem sont UN SEUL type de contrat interne ("solo")
et UNE SEULE chaine de relance 6->13 -- confirme contre game/bidding.gd
(raise_contract() ne fait aucune distinction, incremente `level` jusqu'a
13 ; le LADDER interne du jeu utilise litteralement les cles "solo9".."
solo13", jamais "abondance9"). Seule la legalite de COULEUR differe par
niveau (can_solo_suit ne s'applique qu'a 6-8, voir can_solo) et seule la
formule de SCORE differe par niveau (solo_base_value en dessous de 9,
abondance_result a partir de 9 -- voir engine/scoring.py et
engine/solitaire.py, qui dispatchent sur le niveau, jamais sur un type
separe).
"""

from engine.bid_summary import MAX_RANK, new_bid_summary, update_bid_summary
from engine.cards import NUM_PLAYERS, NUM_SUITS
from engine.scoring import MISERE_ORDER

LADDER = {
    "emballage-8": 1, "solo-6": 2, "emballage-9": 3, "solo-7": 4,
    "emballage-10": 5, "misere-petite": 6, "solo-8": 7, "emballage-11": 8,
    "emballage-12": 9, "misere-piccolo": 10, "emballage-13": 11, "solo-9": 12,
    "trou": 13, "misere-grande": 14, "solo-10": 15, "solo-11": 16,
    "misere-etalee": 17, "solo-12": 18, "solo-13": 19,
}

assert max(LADDER.values()) == MAX_RANK

EMBALLAGE_MIN_LEVEL, EMBALLAGE_MAX_LEVEL = 8, 13
SOLO_MIN_LEVEL, SOLO_PLAIN_MAX_LEVEL = 6, 8
ABONDANCE_MIN_LEVEL, ABONDANCE_MAX_LEVEL = 9, 12
SOLO_CHELEM_LEVEL, SOLO_MAX_LEVEL = 13, 13

# --- Actions discretes -------------------------------------------------

ACTION_WAIT = 0
ACTION_PASS = 1
ACTION_PASS_PAROLE = 2
ACTION_RAISE = 3
_PROPOSE_BASE = 4
_EMBALLER_BASE = _PROPOSE_BASE + NUM_SUITS  # 8
_SOLO_LEVELS = SOLO_PLAIN_MAX_LEVEL - SOLO_MIN_LEVEL + 1  # 3
_SOLO_BASE = _EMBALLER_BASE + NUM_SUITS  # 12
_ABONDANCE_LEVELS = ABONDANCE_MAX_LEVEL - ABONDANCE_MIN_LEVEL + 1  # 4
_ABONDANCE_BASE = _SOLO_BASE + NUM_SUITS * _SOLO_LEVELS  # 24
_SOLO_CHELEM_BASE = _ABONDANCE_BASE + NUM_SUITS * _ABONDANCE_LEVELS  # 40
_MISERE_BASE = _SOLO_CHELEM_BASE + NUM_SUITS  # 44
# ids 48-51 : ancien choix d'atout du bouche-trou AVANT l'enchere ; plus
# jamais legaux depuis la regle du 2026-10-01 (l'atout se fixe a l'entame,
# voir engine/team_trump.py:_fix_trou_contract), gardes pour ne pas decaler
# les ids suivants ni la tete des reseaux d'enchere existants.
_TROU_CHOICE_BASE = _MISERE_BASE + len(MISERE_ORDER)  # 48
# Saut du PROPOSEUR en passe parole (vraie regle, confirmee par l'utilisateur
# le 2026-09-27 ; whist-4-it 9c0dfb3 raise_contract_to) : id 52+i = saut au
# niveau 10+i. Le +1 reste ACTION_RAISE -> un saut n'est legal qu'a
# niveau_actuel + 2 ou plus (jamais deux actions pour le meme coup).
_RAISE_TO_BASE = _TROU_CHOICE_BASE + NUM_SUITS  # 52
RAISE_TO_LEVELS = tuple(range(EMBALLAGE_MIN_LEVEL + 2, EMBALLAGE_MAX_LEVEL + 1))  # 10..13
NUM_BID_ACTIONS = _RAISE_TO_BASE + len(RAISE_TO_LEVELS)  # 56
NUM_BID_ACTIONS_V1 = _RAISE_TO_BASE  # 52 : reseaux d'avant le saut, voir agents/evolution.py:widen_bid_net


def bid_action_to_dict(action_id):
    """Traduit un id d'action en description (action, suit/level/kind) --
    API publique destinee au futur pont Godot. `solo`/`abondance`/
    `solo_chelem` sont 3 POINTS D'ENTREE distincts (niveaux/legalite de
    couleur differents a la declaration, voir can_solo/can_abondance/
    can_solo_chelem) mais creent tous un contrat de type interne "solo"."""
    if action_id == ACTION_WAIT:
        return {"action": "wait"}
    if action_id == ACTION_PASS:
        return {"action": "pass"}
    if action_id == ACTION_PASS_PAROLE:
        return {"action": "pass_parole"}
    if action_id == ACTION_RAISE:
        return {"action": "raise"}
    if _PROPOSE_BASE <= action_id < _EMBALLER_BASE:
        return {"action": "propose", "suit": action_id - _PROPOSE_BASE}
    if _EMBALLER_BASE <= action_id < _SOLO_BASE:
        return {"action": "emballer", "suit": action_id - _EMBALLER_BASE}
    if _SOLO_BASE <= action_id < _ABONDANCE_BASE:
        offset = action_id - _SOLO_BASE
        return {"action": "solo", "suit": offset // _SOLO_LEVELS, "level": SOLO_MIN_LEVEL + offset % _SOLO_LEVELS}
    if _ABONDANCE_BASE <= action_id < _SOLO_CHELEM_BASE:
        offset = action_id - _ABONDANCE_BASE
        return {"action": "abondance", "suit": offset // _ABONDANCE_LEVELS,
                "level": ABONDANCE_MIN_LEVEL + offset % _ABONDANCE_LEVELS}
    if _SOLO_CHELEM_BASE <= action_id < _MISERE_BASE:
        return {"action": "solo_chelem", "suit": action_id - _SOLO_CHELEM_BASE}
    if _MISERE_BASE <= action_id < _TROU_CHOICE_BASE:
        return {"action": "misere", "kind": MISERE_ORDER[action_id - _MISERE_BASE]}
    if _TROU_CHOICE_BASE <= action_id < _RAISE_TO_BASE:
        return {"action": "trou_choice", "suit": action_id - _TROU_CHOICE_BASE}
    if _RAISE_TO_BASE <= action_id < NUM_BID_ACTIONS:
        return {"action": "raise", "level": RAISE_TO_LEVELS[action_id - _RAISE_TO_BASE]}
    raise ValueError("action id invalide: %r" % action_id)


def first_leader(dealer_index, contract):
    """Qui mene le premier pli -- verifie contre net/whist_table.gd
    (_first_leader()) : le siege a gauche du donneur, SAUF Abondance/Solo
    Chelem (contrat "solo" de niveau >= 9) ou c'est le declarant lui-meme
    qui entame, et Trou (regle du 2026-10-01) ou c'est le BOUCHE-TROU, dont
    la carte d'entame fixe l'atout et le niveau."""
    if contract["type"] == "solo" and contract["level"] >= ABONDANCE_MIN_LEVEL:
        return contract["declarer"]
    if contract["type"] == "trou":
        return contract["partner"]
    return (dealer_index + 1) % NUM_PLAYERS


class Bidding:
    def __init__(self, dealer_index):
        self.dealer_index = dealer_index
        self.active = [True] * NUM_PLAYERS
        self.pending_proposals = {}  # suit -> proposeur
        self.emballed_suits = set()
        self.contracts = []
        self.lie_contract_index = [-1] * NUM_PLAYERS
        self.solo_only_suit = [None] * NUM_PLAYERS
        self.own_suit = [None] * NUM_PLAYERS  # couleur du 1er propose/emballer de ce siege
        self.has_acted = [False] * NUM_PLAYERS
        # Genre de la derniere annonce de chaque siege (last_action[seat]["action"]
        # de game/bidding.gd, None = pas encore parle) : Main de Fer (043f).
        self.last_action = [None] * NUM_PLAYERS
        self.waiting_player = -1
        self.turn = (dealer_index + 1) % NUM_PLAYERS
        self.first_bidder = self.turn
        self.has_started = False
        self.finished = False
        self.general_pass = False
        self.locked_contract = None
        # Annonces PUBLIQUES de chaque siege (voir engine/bid_summary.py), mises
        # a jour par apply() -- seul point d'entree des actions d'enchere.
        self.bid_summary = new_bid_summary()

    def seed_trou(self, trou_info):
        """Trou/Trou Royal auto-declenche (main a 3-4 as, jamais un choix
        d'enchere -- voir engine/trou.py:detect_trou), AVANT le debut de
        l'enchere normale. Regle du 2026-10-01 (game/bidding.gd::
        seed_trou_contract) : le contrat est seede SANS atout ni niveau
        (fixes plus tard par l'entame du bouche-trou, voir
        engine/team_trump.py:_fix_trou_contract) ; il garde son rang 13 et
        reste surenchérissable. Plus aucune decision pre-enchere."""
        contract = {"type": "trou", "suit": None, "level": 0,
                    "declarer": trou_info["declarer"], "partner": trou_info["partner"],
                    "royal": trou_info["is_royal"], "anchor_card": trou_info["anchor_card"]}
        idx = len(self.contracts)
        self.contracts.append(contract)
        self.lie_contract_index[contract["declarer"]] = idx
        self.lie_contract_index[contract["partner"]] = idx
        # Le declarant du trou a "parle" : game/bidding.gd::seed_trou_contract()
        # remplit last_action[declarer] -> plus de propose/abondance/solo
        # chelem "1er tour" pour lui, et drapeau deja-agi dans l'observation.
        self.has_acted[contract["declarer"]] = True
        self.last_action[contract["declarer"]] = "trou"
        self.has_started = True
        update_bid_summary(self.bid_summary, self, contract["partner"], "trou")
        self._advance_to_next_pending_seat()

    # --- lecture de l'etat -----------------------------------------------

    def _can_act(self, seat):
        return not self.finished and seat == self.turn and self.active[seat]

    def _is_free_agent(self, seat):
        return self.active[seat] and self.lie_contract_index[seat] == -1

    def _live_contracts(self):
        out = []
        for c in self.contracts:
            if c["type"] == "misere":
                if c["declarers"]:
                    out.append(c)
            elif c["type"] in ("emballage", "trou"):
                if self.active[c["declarer"]] and self.active[c["partner"]]:
                    out.append(c)
            else:  # solo
                if self.active[c["declarer"]]:
                    out.append(c)
        return out

    def _ladder_key(self, c):
        if c["type"] == "trou":
            return "trou"
        if c["type"] == "misere":
            return "misere-%s" % c["kind"]
        return "%s-%d" % (c["type"], c["level"])

    def ladder_rank(self, contract_index):
        return LADDER[self._ladder_key(self.contracts[contract_index])]

    def _score(self, c):
        """rang*10 + priorite de couleur -- verifie contre
        game/bidding.gd::_contract_score(). Departage deux contrats du
        MEME rang (ex: deux Emballage8 dans des couleurs differentes) :
        Coeur(3) > Carreau(2) > Trefle(1) > Pique(0), 4 pour un contrat
        sans couleur (Misere). Un simple rang de LADDER SANS ce
        departage (comme la version precedente de cette fonction)
        provoque une domination mutuelle incorrecte entre deux contrats
        du meme rang -- bug reel trouve par
        tests/bid_cross_check_runner.gd cote Godot."""
        rung = LADDER[self._ladder_key(c)]
        suit = c.get("suit")
        suit_priority = 4 if suit is None else (3 - suit)
        return rung * 10 + suit_priority

    def _outranks_everything_live(self, candidate):
        score = self._score(candidate)
        return all(self._score(c) < score for c in self._live_contracts())

    def _is_dominated(self, contract_index):
        c = self.contracts[contract_index]
        live = self._live_contracts()
        if c not in live:
            return False
        score = self._score(c)
        return any(self._score(o) >= score for o in live if o is not c)

    def _sole_remaining_free_agent(self, seat):
        return all(not self._is_free_agent(s) for s in range(NUM_PLAYERS) if s != seat)

    def can_solo_suit(self, seat, suit):
        """Legalite de couleur pour un Solo PLAIN (6-8) FRAIS (pas une
        relance) : sa propre couleur (1er tour), ou dernier agent libre
        restant, ou (ayant deja agi une fois) une couleur deja emballee par
        quelqu'un d'autre. Ne s'applique PAS a Abondance/Solo Chelem
        (couleur libre, voir can_abondance/can_solo_chelem)."""
        if self.own_suit[seat] == suit:
            return True
        if self._sole_remaining_free_agent(seat):
            return True
        if self.own_suit[seat] is None:
            return False
        return suit in self.emballed_suits

    # --- legalite : predicats par action --------------------------------

    def can_wait(self, seat):
        return not self.has_started and seat == self.first_bidder and self._can_act(seat)

    def can_propose(self, seat, suit):
        if not self._can_act(seat) or seat == self.waiting_player:
            return False
        if self.has_acted[seat] or self.lie_contract_index[seat] != -1 or self.solo_only_suit[seat] is not None:
            return False
        if suit in self.emballed_suits:
            return False
        other = self.pending_proposals.get(suit)
        return other is None or other == seat

    def can_emballer(self, seat, suit):
        if not self._can_act(seat):
            return False
        if self.lie_contract_index[seat] != -1 or self.solo_only_suit[seat] is not None:
            return False
        proposer = self.pending_proposals.get(suit)
        if proposer is None or proposer == seat or self.lie_contract_index[proposer] != -1:
            return False
        return self.emballage_entry_level(suit) is not None

    def emballage_entry_level(self, suit):
        """Niveau auquel un NOUVEL emballage de `suit` serait cree : le plus
        petit L de 8..13 qui depasse tous les contrats vivants, ou None si
        aucun (ex. Trou vivant). Regle whist-phi « chaque enchere doit etre
        plus haute que l'enchere la plus elevee » -- emballer peut donc
        partir au-dessus de 8 (bug report 2026-09-29, miroir de
        game/bidding.gd::emballer / is_new_emballage_legal)."""
        for level in range(EMBALLAGE_MIN_LEVEL, EMBALLAGE_MAX_LEVEL + 1):
            if self._outranks_everything_live({"type": "emballage", "level": level, "suit": suit}):
                return level
        return None

    def _can_declare_solo_family(self, seat, suit, level):
        if not self._can_act(seat) or seat == self.waiting_player:
            return False
        if self.lie_contract_index[seat] != -1:
            return False
        restricted = self.solo_only_suit[seat]
        if restricted is not None and restricted != suit:
            return False
        if restricted is None and level <= SOLO_PLAIN_MAX_LEVEL and not self.can_solo_suit(seat, suit):
            return False
        return self._outranks_everything_live({"type": "solo", "level": level, "suit": suit})

    def can_solo(self, seat, suit, level):
        return SOLO_MIN_LEVEL <= level <= SOLO_PLAIN_MAX_LEVEL and self._can_declare_solo_family(seat, suit, level)

    def can_abondance(self, seat, suit, level):
        if not (ABONDANCE_MIN_LEVEL <= level <= ABONDANCE_MAX_LEVEL) or self.has_acted[seat]:
            return False
        return self._can_declare_solo_family(seat, suit, level)

    def can_solo_chelem(self, seat, suit):
        if self.has_acted[seat]:
            return False
        return self._can_declare_solo_family(seat, suit, SOLO_CHELEM_LEVEL)

    def can_misere(self, seat, kind):
        if not self._can_act(seat) or seat == self.waiting_player:
            return False
        if self.lie_contract_index[seat] != -1 or self.solo_only_suit[seat] is not None:
            return False
        if self._find_misere_entry(kind) is not None:
            return True  # rejoindre une famille deja ouverte : toujours legal
        return self._outranks_everything_live({"type": "misere", "kind": kind})

    def can_raise(self, seat):
        if not self._can_act(seat):
            return False
        ci = self.lie_contract_index[seat]
        if ci == -1:
            return False
        c = self.contracts[ci]
        if c["type"] == "emballage":
            if not (seat == c["spokesperson"] or (seat == c["declarer"] and c["parole_pending"])):
                return False
            return c["level"] < EMBALLAGE_MAX_LEVEL
        if c["type"] == "solo":
            return seat == c["declarer"] and c["level"] < SOLO_MAX_LEVEL
        if c["type"] == "misere":
            return seat in c["declarers"] and MISERE_ORDER.index(c["kind"]) < len(MISERE_ORDER) - 1
        return False  # trou ne se relance jamais

    def can_raise_to(self, seat, level):
        """Saut du PROPOSEUR pendant un passe parole (regle confirmee par
        l'utilisateur le 2026-09-27, whist-4-it 9c0dfb3) : une seule
        decision, mais a n'importe quel niveau entre level+1 et 13 --
        verifie contre game/bidding.gd::raise_contract_to(). L'emballeur
        (spokesperson) n'a toujours que raise_contract (+1)."""
        if not self._can_act(seat):
            return False
        ci = self.lie_contract_index[seat]
        if ci == -1:
            return False
        c = self.contracts[ci]
        if c["type"] != "emballage" or seat != c["declarer"] or not c["parole_pending"]:
            return False
        return c["level"] < level <= EMBALLAGE_MAX_LEVEL

    def can_pass_parole(self, seat):
        if not self._can_act(seat):
            return False
        ci = self.lie_contract_index[seat]
        if ci == -1:
            return False
        c = self.contracts[ci]
        return c["type"] == "emballage" and seat == c["spokesperson"] and not c["parole_passed"]

    def can_pass(self, seat):
        return self._can_act(seat)

    # --- actions : mutation ------------------------------------------------

    def wait(self, seat):
        if not self.can_wait(seat):
            return False
        self.waiting_player = seat
        self.has_acted[seat] = True
        self._advance()
        return True

    def propose(self, seat, suit):
        if not self.can_propose(seat, suit):
            return False
        self.pending_proposals[suit] = seat
        self.own_suit[seat] = suit
        self.has_acted[seat] = True
        self._advance()
        return True

    def emballer(self, seat, suit):
        if not self.can_emballer(seat, suit):
            return False
        level = self.emballage_entry_level(suit)
        proposer = self.pending_proposals.pop(suit)
        contract = {
            "type": "emballage", "suit": suit, "level": level,
            "declarer": proposer, "partner": seat, "spokesperson": seat,
            "parole_passed": False, "parole_pending": False, "closed": False,
        }
        idx = len(self.contracts)
        self.contracts.append(contract)
        self.lie_contract_index[proposer] = idx
        self.lie_contract_index[seat] = idx
        self.emballed_suits.add(suit)
        if self.own_suit[seat] is None:
            self.own_suit[seat] = suit
        self.has_acted[seat] = True
        self._advance()
        return True

    def _declare_solo_family(self, seat, suit, level):
        contract = {"type": "solo", "suit": suit, "level": level, "declarer": seat}
        idx = len(self.contracts)
        self.contracts.append(contract)
        self.lie_contract_index[seat] = idx
        self.solo_only_suit[seat] = None
        self.has_acted[seat] = True
        self._advance()

    def solo(self, seat, suit, level):
        if not self.can_solo(seat, suit, level):
            return False
        self._declare_solo_family(seat, suit, level)
        return True

    def abondance(self, seat, suit, level):
        if not self.can_abondance(seat, suit, level):
            return False
        self._declare_solo_family(seat, suit, level)
        return True

    def solo_chelem(self, seat, suit):
        if not self.can_solo_chelem(seat, suit):
            return False
        self._declare_solo_family(seat, suit, SOLO_CHELEM_LEVEL)
        return True

    def misere(self, seat, kind):
        if not self.can_misere(seat, kind):
            return False
        idx = self._find_or_create_misere_entry(kind)
        self.contracts[idx]["declarers"].append(seat)
        self.lie_contract_index[seat] = idx
        self.has_acted[seat] = True
        self._advance()
        return True

    def raise_contract(self, seat):
        if not self.can_raise(seat):
            return False
        self.has_acted[seat] = True  # miroir de last_action[player] (game/bidding.gd)
        c = self.contracts[self.lie_contract_index[seat]]
        if c["type"] == "emballage":
            c["level"] += 1
            if c["parole_pending"]:
                c["closed"] = True
            c["parole_pending"] = False
        elif c["type"] == "solo":
            c["level"] += 1
        else:  # misere
            next_kind = MISERE_ORDER[MISERE_ORDER.index(c["kind"]) + 1]
            c["declarers"].remove(seat)
            target = self._find_or_create_misere_entry(next_kind)
            self.contracts[target]["declarers"].append(seat)
            self.lie_contract_index[seat] = target
        self._advance()
        return True

    def raise_contract_to(self, seat, level):
        if not self.can_raise_to(seat, level):
            return False
        self.has_acted[seat] = True  # miroir de last_action[player] (game/bidding.gd)
        c = self.contracts[self.lie_contract_index[seat]]
        c["level"] = level
        c["parole_pending"] = False
        c["closed"] = True
        self._advance()
        return True

    def apply_raise_to(self, seat, level):
        """Point d'entree du saut hors ids d'action (l'espace d'actions des
        reseaux d'enchere n'a encore que ACTION_RAISE = +1) : applique et
        tient bid_summary a jour exactement comme apply()."""
        ok = self.raise_contract_to(seat, level)
        if ok:
            self.last_action[seat] = "raise"
            update_bid_summary(self.bid_summary, self, seat, "raise")
        return ok

    def pass_parole(self, seat):
        if not self.can_pass_parole(seat):
            return False
        self.has_acted[seat] = True  # miroir de last_action[player] (game/bidding.gd)
        c = self.contracts[self.lie_contract_index[seat]]
        c["parole_passed"] = True
        c["parole_pending"] = True
        self._advance()
        return True

    def pass_bid(self, seat):
        if not self.can_pass(seat):
            return False
        # has_acted = miroir EXACT de `last_action[player] != null` cote Godot
        # (game/bidding.gd remplit last_action pour CHAQUE action, passe,
        # relance et passe parole compris) -- ecart de l'observation d'enchere
        # trouve le 2026-09-27 par tests/ai_bridge_parity_v2_bid_runner.gd.
        self.has_acted[seat] = True
        self.active[seat] = False
        ci = self.lie_contract_index[seat]
        if ci != -1:
            c = self.contracts[ci]
            if c["type"] in ("emballage", "trou"):
                other = c["partner"] if seat == c["declarer"] else c["declarer"]
                self.lie_contract_index[other] = -1
                self.solo_only_suit[other] = c["suit"]
                if c["type"] == "emballage":
                    c["parole_pending"] = False
            elif c["type"] == "misere":
                c["declarers"].remove(seat)
        self._advance()
        return True

    # --- avancement / fin de l'enchere --------------------------------

    def _find_misere_entry(self, kind):
        for i, c in enumerate(self.contracts):
            if c["type"] == "misere" and c["kind"] == kind:
                return i
        return None

    def _find_or_create_misere_entry(self, kind):
        existing = self._find_misere_entry(kind)
        if existing is not None:
            return existing
        idx = len(self.contracts)
        self.contracts.append({"type": "misere", "kind": kind, "declarers": []})
        return idx

    def _seat_has_pending_decision(self, seat):
        if not self.active[seat]:
            return False
        ci = self.lie_contract_index[seat]
        if ci == -1:
            return True
        c = self.contracts[ci]
        if c["type"] in ("solo", "trou"):
            return seat == c["declarer"] and self._is_dominated(ci)
        if c["type"] == "misere":
            return seat in c["declarers"] and self._is_dominated(ci)
        # emballage
        if c.get("closed"):
            return False
        if seat == c["spokesperson"]:
            return not c["parole_pending"] and self._is_dominated(ci)
        if seat == c["declarer"]:
            return c["parole_pending"]
        return False

    def _advance_to_next_pending_seat(self):
        seat = self.turn
        for _ in range(NUM_PLAYERS):
            if self._seat_has_pending_decision(seat):
                self.turn = seat
                return True
            seat = (seat + 1) % NUM_PLAYERS
        return False

    def _fold_outbid_closed_emballages(self):
        # Passe Parole, whist-phi.vercel.app : le proposeur decide « une seule
        # fois » ; surencheri ensuite, le duo est dehors (miroir de
        # game/bidding.gd::_fold_outbid_closed_emballages, rapport 2026-10-06).
        for ci, c in enumerate(self.contracts):
            if c["type"] == "emballage" and c.get("closed") and self._is_dominated(ci):
                self.active[c["declarer"]] = False
                self.active[c["partner"]] = False

    def _advance(self):
        self.has_started = True
        self._fold_outbid_closed_emballages()
        self._check_finished()
        if not self.finished:
            self.turn = (self.turn + 1) % NUM_PLAYERS
            if not self._advance_to_next_pending_seat():
                self.finished = True

    def _party_count(self):
        n = len(self._live_contracts())
        n += sum(1 for s in range(NUM_PLAYERS) if self._is_free_agent(s))
        return n

    def _check_finished(self):
        parties = self._party_count()
        if parties == 0:
            self.finished = True
            self.general_pass = True
            return True
        if parties == 1:
            live = self._live_contracts()
            if len(live) == 1:
                self.finished = True
                self.locked_contract = live[0]
                return True
        return False

    # --- interface reseau -----------------------------------------------

    def legal_action_ids(self, seat):
        ids = []
        if self.can_wait(seat):
            ids.append(ACTION_WAIT)
        if self.can_pass(seat):
            ids.append(ACTION_PASS)
        if self.can_pass_parole(seat):
            ids.append(ACTION_PASS_PAROLE)
        if self.can_raise(seat):
            ids.append(ACTION_RAISE)
        for suit in range(NUM_SUITS):
            if self.can_propose(seat, suit):
                ids.append(_PROPOSE_BASE + suit)
            if self.can_emballer(seat, suit):
                ids.append(_EMBALLER_BASE + suit)
            for level in range(SOLO_MIN_LEVEL, SOLO_PLAIN_MAX_LEVEL + 1):
                if self.can_solo(seat, suit, level):
                    ids.append(_SOLO_BASE + suit * _SOLO_LEVELS + (level - SOLO_MIN_LEVEL))
            for level in range(ABONDANCE_MIN_LEVEL, ABONDANCE_MAX_LEVEL + 1):
                if self.can_abondance(seat, suit, level):
                    ids.append(_ABONDANCE_BASE + suit * _ABONDANCE_LEVELS + (level - ABONDANCE_MIN_LEVEL))
            if self.can_solo_chelem(seat, suit):
                ids.append(_SOLO_CHELEM_BASE + suit)
        for i, kind in enumerate(MISERE_ORDER):
            if self.can_misere(seat, kind):
                ids.append(_MISERE_BASE + i)
        ci = self.lie_contract_index[seat]
        if ci != -1:
            current = self.contracts[ci].get("level", 0)
            for i, level in enumerate(RAISE_TO_LEVELS):
                if level >= current + 2 and self.can_raise_to(seat, level):
                    ids.append(_RAISE_TO_BASE + i)
        return ids

    def apply(self, seat, action_id):
        info = bid_action_to_dict(action_id)
        if info["action"] == "raise" and "level" in info:
            level = info["level"]
            c = self.contracts[self.lie_contract_index[seat]] if self.lie_contract_index[seat] != -1 else None
            if c is None or level < c.get("level", 0) + 2:
                return False  # un saut de +1 passe par ACTION_RAISE
            return self.apply_raise_to(seat, level)
        ok = self._apply(seat, info)
        if ok:
            self.last_action[seat] = info["action"]
            update_bid_summary(self.bid_summary, self, seat, info["action"], info.get("suit"))
        return ok

    def _apply(self, seat, info):
        kind = info["action"]
        if kind == "wait":
            return self.wait(seat)
        if kind == "pass":
            return self.pass_bid(seat)
        if kind == "pass_parole":
            return self.pass_parole(seat)
        if kind == "raise":
            return self.raise_contract(seat)
        if kind == "propose":
            return self.propose(seat, info["suit"])
        if kind == "emballer":
            return self.emballer(seat, info["suit"])
        if kind == "solo":
            return self.solo(seat, info["suit"], info["level"])
        if kind == "abondance":
            return self.abondance(seat, info["suit"], info["level"])
        if kind == "solo_chelem":
            return self.solo_chelem(seat, info["suit"])
        if kind == "misere":
            return self.misere(seat, info["kind"])
        if kind == "trou_choice":
            return False  # ids 48-51 reserves, plus jamais legaux depuis le 2026-10-01
        raise ValueError("action inconnue: %r" % kind)


# --- observation --------------------------------------------------------

_LADDER_KEYS = sorted(LADDER, key=LADDER.get)  # ordre croissant de force

# main (52) + siege (4) + donneur (4) + actifs (4) + ma_couleur (4) +
# couleur_restreinte_solo (4) + propositions_en_attente (4) +
# couleurs_emballees (4) + occupation_de_l_echelle (19) + deja_agi (4) +
# nb_de_parties_normalise (1) + choix_de_trou_en_attente (1)
OBS_DIM = 52 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + len(_LADDER_KEYS) + 4 + 1 + 1


def observation(bidding, seat, hand):
    hand_mask = [1.0 if c in hand else 0.0 for c in range(52)]
    seat_onehot = [1.0 if i == seat else 0.0 for i in range(NUM_PLAYERS)]
    dealer_onehot = [1.0 if i == bidding.dealer_index else 0.0 for i in range(NUM_PLAYERS)]
    active_flags = [1.0 if a else 0.0 for a in bidding.active]
    own_suit_onehot = [1.0 if bidding.own_suit[seat] == s else 0.0 for s in range(NUM_SUITS)]
    restricted_onehot = [1.0 if bidding.solo_only_suit[seat] == s else 0.0 for s in range(NUM_SUITS)]
    pending_flags = [1.0 if s in bidding.pending_proposals else 0.0 for s in range(NUM_SUITS)]
    emballed_flags = [1.0 if s in bidding.emballed_suits else 0.0 for s in range(NUM_SUITS)]
    live_keys = {bidding._ladder_key(c) for c in bidding._live_contracts()}
    ladder_occupancy = [1.0 if key in live_keys else 0.0 for key in _LADDER_KEYS]
    has_acted_flags = [1.0 if a else 0.0 for a in bidding.has_acted]
    party_count = [bidding._party_count() / float(NUM_PLAYERS)]
    trou_choice_pending = [0.0]  # garde la dimension ; plus de choix pre-enchere depuis le 2026-10-01
    return (
        hand_mask + seat_onehot + dealer_onehot + active_flags + own_suit_onehot
        + restricted_onehot + pending_flags + emballed_flags + ladder_occupancy
        + has_acted_flags + party_count + trou_choice_pending
    )


def legal_action_mask(bidding, seat):
    ids = set(bidding.legal_action_ids(seat))
    return [i in ids for i in range(NUM_BID_ACTIONS)]


def _hand_has_suit(hand, suit):
    return any(c // 13 == suit for c in hand)


def legal_action_ids_with_hand(bidding, seat, hand):
    """Comme Bidding.legal_action_ids(), plus la contrainte "minimum une
    carte de l'atout" pour propose/emballer -- NON geree par la classe
    Bidding elle-meme (volontairement hand-agnostic, comme le vrai
    game/bidding.gd : "Card-holding ... is enforced one layer up, in
    WhistBot.apply_bid_action() -- this class stays hand-agnostic by
    design"), donc geree ICI, une couche au-dessus, pour que
    l'entrainement voie EXACTEMENT la meme surface de legalite que le
    pont Godot (game/whist_ai_features_v2_bid.gd:_hand_has_suit()) --
    sans quoi le reseau apprendrait a partir d'un espace d'actions plus
    large que celui qu'il verra jamais en jeu reel."""
    ids = bidding.legal_action_ids(seat)
    out = []
    for action_id in ids:
        info = bid_action_to_dict(action_id)
        if info["action"] in ("propose", "emballer") and not _hand_has_suit(hand, info["suit"]):
            continue
        out.append(action_id)
    return out


def legal_action_mask_with_hand(bidding, seat, hand):
    ids = set(legal_action_ids_with_hand(bidding, seat, hand))
    return [i in ids for i in range(NUM_BID_ACTIONS)]
