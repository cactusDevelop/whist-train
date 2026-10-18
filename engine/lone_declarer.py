"""Moteur generique de pli pour un ou plusieurs declarants seuls (chacun
pour soi -- jamais une equipe) contre les autres sieges, avec ou sans
atout — partage par Solitaire (contrats individuels positifs, 1 seul
declarant, avec atout) et Misere (contrats individuels negatifs, 1 A 4
declarants simultanes possibles -- voir WHIST-RULES-REFERENCE.txt --, sans
atout). Voir DESIGN.md. Le score n'est PAS calcule ici (il differe trop
entre Solitaire et Misere) -- voir engine/solitaire.py et
engine/misere.py, tous deux bases sur engine/scoring.py.

Petite Misere seule a une phase d'ECART prealable ("chaque personne doit
ecarter une carte de son choix qui ne sera pas jouee", voir
WHIST-RULES-REFERENCE.txt) : les 4 sieges, dans l'ordre, choisissent une
carte de leur propre main a mettre de cote (vraie decision, jamais au
hasard) AVANT que le premier pli ne commence -- voir `needs_discard`.
"""

from engine.cards import NUM_CARDS, NUM_PLAYERS, legal_plays, suit_of, trick_winner
from engine.observation import OBS_DIM, build as build_observation

ACTION_DIM = NUM_CARDS


class LoneDeclarerDeal:
    def __init__(self, hands, declarers, trump_suit, num_tricks, leader,
                 declarer_role, defender_role, reveal_declarer_hands=False, needs_discard=False,
                 contract_type="solo", level=None, kind=None):
        self.hands = [list(h) for h in hands]
        self.declarers = list(declarers)  # 1 pour Solitaire, 1-4 pour Misere
        self._declarer_set = set(self.declarers)
        self.trump_suit = trump_suit  # None pour Misere (aucun atout)
        self.num_tricks = num_tricks  # 12 pour petite misere, 13 sinon
        self.declarer_role = declarer_role
        self.defender_role = defender_role
        self.reveal_declarer_hands = reveal_declarer_hands  # etalee uniquement
        self.played = [False] * NUM_CARDS
        self.current_trick = []
        self.play_order = []  # cartes jouees dans l'ordre chronologique (deck whiste, voir engine/whisted.py)
        self.play_log = []  # [(siege, carte)] chronologique -- base de l'observation v3
        self.contract_type = contract_type  # "solo" (Solitaire) ou "misere"
        self.level = level
        self.kind = kind
        self.set_public_context(dealer=(leader - 1) % NUM_PLAYERS)
        self.discards = {}  # siege -> carte ecartee (petite misere)
        self.tricks_won = [0] * NUM_PLAYERS
        self.trick_count = 0
        # Petite misere uniquement : les 4 sieges ecartent 1 carte chacun,
        # dans l'ordre des sieges, avant que `leader` n'entame le 1er pli.
        self.pending_discards = list(range(NUM_PLAYERS)) if needs_discard else []
        self._leader_after_discard = leader
        self.turn = self.pending_discards[0] if self.pending_discards else leader
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

    def public_contract(self):
        return {
            "type": self.contract_type, "trump": self.trump_suit,
            "level": self.level if self.level_known else None, "kind": self.kind,
            "declarers": list(self.declarers), "partner": None, "royal": False,
        }

    def role_of(self, seat):
        return self.declarer_role if seat in self._declarer_set else self.defender_role

    def observation(self, seat):
        revealed = {}
        if self.reveal_declarer_hands:
            revealed = {d: list(self.hands[d]) for d in self.declarers}
        return build_observation(
            seat=seat, hand=self.hands[seat], play_log=self.play_log, dealer=self.dealer,
            contract=self.public_contract(), bid_summary=self.bid_summary,
            revealed_hands=revealed, own_discard=self.discards.get(seat),
            is_discard_phase=bool(self.pending_discards), whisted=self.whisted,
        )

    def legal_action_mask(self, seat):
        if self.pending_discards:
            mask = [False] * NUM_CARDS
            for c in self.hands[seat]:
                mask[c] = True
            return mask
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
        if self.pending_discards:
            self.hands[seat].remove(card)  # ecartee : jamais jouee, jamais vue par personne
            self.discards[seat] = card
            self.pending_discards.pop(0)
            self.turn = self.pending_discards[0] if self.pending_discards else self._leader_after_discard
            return
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
            if self.trick_count == self.num_tricks:
                self.done = True
        else:
            self.turn = (self.turn + 1) % NUM_PLAYERS
