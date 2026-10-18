"""Moteur generique de pli avec atout pour un camp de 2 (declarant +
partenaire) contre 2 defenseurs — partage par Emballage et Trou (voir
DESIGN.md). Le contrat lui-meme (declarant/partenaire/atout/niveau) est
fourni par l'appelant ; ce module ne gere que le deroulement des 13 plis,
l'observation et le score.
"""

from engine.cards import NUM_CARDS, NUM_PLAYERS, legal_plays, suit_of, trick_winner
from engine.observation import OBS_DIM, build as build_observation

ACTION_DIM = NUM_CARDS


class TeamTrumpDeal:
    def __init__(self, hands, declarer, partner, trump_suit, level, leader,
                 declarer_role, partner_role, defender_role, contract_type="emballage", royal=False,
                 anchor_card=None, contract_record=None):
        self.hands = [list(h) for h in hands]
        self.declarer = declarer
        self.partner = partner
        self.trump_suit = trump_suit
        self.level = level  # nombre de plis requis par le camp declarant (8..13) ; 0 = Trou pas encore fixe
        # Trou (regle du 2026-10-01) : atout None + niveau 0 jusqu'a la 1re
        # carte du bouche-trou, qui les fixe -- voir _fix_trou_contract().
        self.anchor_card = anchor_card
        # Dict du contrat de l'enchere, complete a la fixation comme le fait
        # net/whist_table.gd::_fix_trou_atout (stats, evals, score reel).
        self.contract_record = contract_record
        self.declarer_role = declarer_role
        self.partner_role = partner_role
        self.defender_role = defender_role
        self.played = [False] * NUM_CARDS
        self.current_trick = []  # [(seat, card), ...]
        self.play_order = []  # cartes jouees dans l'ordre chronologique (deck whiste, voir engine/whisted.py)
        self.play_log = []  # [(siege, carte)] chronologique -- base de l'observation v3
        self.contract_type = contract_type  # "emballage" ou "trou"
        self.royal = royal
        self.set_public_context(dealer=(leader - 1) % NUM_PLAYERS)
        self.tricks_won = [0] * NUM_PLAYERS
        self.trick_count = 0
        self.turn = leader
        self.done = False

    def set_public_context(self, dealer, bid_summary=None, whisted=False, level_known=True):
        """Informations publiques hors du jeu de la carte lui-meme, visibles
        par tout client Godot : donneur, annonces d'enchere (None = pas
        d'enchere, Phase 1), donne whistee, niveau annonce (inconnu en
        Phase 1 : aucun contrat reel)."""
        self.dealer = dealer
        self.bid_summary = bid_summary
        self.whisted = whisted
        self.level_known = level_known

    def trou_pending(self):
        """Trou dont l'atout/niveau n'est pas encore fixe par l'entame."""
        return self.contract_type == "trou" and self.trump_suit is None

    def public_contract(self):
        return {
            "type": self.contract_type, "trump": self.trump_suit,
            "level": self.level if (self.level_known and not self.trou_pending()) else None, "kind": None,
            "declarers": [self.declarer], "partner": self.partner, "royal": self.royal,
        }

    def role_of(self, seat):
        if seat == self.declarer:
            return self.declarer_role
        if seat == self.partner:
            return self.partner_role
        return self.defender_role

    def observation(self, seat):
        return build_observation(
            seat=seat, hand=self.hands[seat], play_log=self.play_log, dealer=self.dealer,
            contract=self.public_contract(), bid_summary=self.bid_summary, whisted=self.whisted,
        )

    def legal_action_mask(self, seat):
        led_suit = suit_of(self.current_trick[0][1]) if self.current_trick else None
        legals = legal_plays(self.hands[seat], led_suit)
        mask = [False] * NUM_CARDS
        for c in legals:
            mask[c] = True
        return mask

    def step(self, seat, card):
        if seat != self.turn or self.done:
            raise ValueError("hors-tour ou donne terminee")
        if card not in self.hands[seat]:
            raise ValueError("carte pas en main")
        if self.trou_pending():
            self._fix_trou_contract(seat, card)
        self.hands[seat].remove(card)
        self.current_trick.append((seat, card))
        self.play_order.append(card)
        self.play_log.append((seat, card))
        self.played[card] = True
        if len(self.current_trick) == NUM_PLAYERS:
            winner = trick_winner(self.current_trick, self.trump_suit)
            self.tricks_won[winner] += 1
            self.current_trick = []
            self.trick_count += 1
            self.turn = winner
            if self.trick_count == 13:
                self.done = True
        else:
            self.turn = (self.turn + 1) % NUM_PLAYERS

    def _fix_trou_contract(self, seat, card):
        """Regle du 2026-10-01 : la 1re carte du bouche-trou fixe le contrat
        AVANT la resolution du pli -- carte d'ancrage -> 8 plis, toute autre
        carte -> 9 plis ; atout = couleur de la carte jouee dans les deux cas."""
        if seat != self.partner:
            raise ValueError("le Trou s'ouvre par l'entame du bouche-trou")
        self.trump_suit = suit_of(card)
        self.level = 8 if card == self.anchor_card else 9
        if self.contract_record is not None:
            self.contract_record["suit"] = self.trump_suit
            self.contract_record["level"] = self.level

    def declarer_side_tricks(self):
        return self.tricks_won[self.declarer] + self.tricks_won[self.partner]

    def team_value_scores(self, value):
        """Chaque membre du camp declarant recoit `value` EN ENTIER (pas
        divise en deux) ; chaque defenseur recoit -value EN ENTIER aussi --
        c'est la vraie regle (voir game/scoring.gd/net/whist_table.gd),
        pas une invention : reste a somme nulle car 2*value + 2*(-value) =
        0. Le bareme (`value`) vient de engine/scoring.py, jamais invente
        ici -- voir engine/emballage.py et engine/trou.py."""
        if not self.done:
            raise ValueError("donne pas terminee")
        declarer_side = {self.declarer, self.partner}
        scores = [0.0] * NUM_PLAYERS
        for s in range(NUM_PLAYERS):
            scores[s] = float(value) if s in declarer_side else float(-value)
        return scores
