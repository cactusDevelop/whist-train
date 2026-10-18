"""Deck whiste : port fidele de whist-4-it (commit 089a837,
net/whist_table.gd::_deal_whisted/_gather_unplayed_hands/_finish_hand,
game/deck.gd::cut/deal_packets/shuffled_full_deck).

Un deck whiste n'est PAS un deck trie coupe une fois : c'est le paquet
RAMASSE de la donne precedente, coupe une fois, puis distribue par 4, 5
puis 4 a partir du siege a gauche du donneur. Pas de paquet precedent ->
52 cartes melangees uniformement, puis coupe. Reconstitution du paquet
(index 0 = premiere carte distribuee) :

- donne jouee : chaque carte dans l'ordre chronologique exact du jeu (pli
  1 a N, chaque pli dans l'ordre de jeu), puis les ecarts de Petite
  Misere dans l'ordre des sieges donneur+1, +2, +3, donneur (donneur de la
  donne qui vient de se jouer) ; le donneur tourne ensuite de 1.
- donne non jouee (passe generale, jeu blanc sur une donne whistee) : les
  mains entieres dans l'ordre donneur+1, +2, +3, donneur, chacune dans
  l'ordre de reception ; le donneur ne tourne PAS.
- apres un contrat Trou, la donne suivante est un Brimborion dans le vrai
  jeu (melange complet, jamais whiste) -- le Brimborion lui-meme n'existe
  pas dans ce moteur : on le remplace par une donne uniforme, dont les plis
  alimentent ensuite normalement le paquet suivant.

Carte = entier 0..51 (voir engine/cards.py), rank 9..12 = Valet..As.
"""

from engine.cards import HAND_SIZE, NUM_CARDS, NUM_PLAYERS, rank_of

WHISTED_PACKET_SIZES = (4, 5, 4)
FACE_RANKS = (9, 10, 11, 12)  # Valet, Dame, Roi, As -- "jeu blanc" de whist_table.gd


def shuffled_full_deck(rng):
    deck = list(range(NUM_CARDS))
    rng.shuffle(deck)
    return deck


def cut(pack, rng):
    """Point de coupe uniforme sur 1..51 (jamais 0 ni 52), comme
    rng.randi_range(1, n - 1) cote Godot."""
    point = rng.randint(1, len(pack) - 1)
    return pack[point:] + pack[:point]


def deal_packets(cards, start, packet_sizes=WHISTED_PACKET_SIZES):
    """Mains indexees par siege ABSOLU, dans l'ordre de reception (jamais
    triees -- le paquet d'une passe generale en depend)."""
    hands = [[] for _ in range(NUM_PLAYERS)]
    i = 0
    for size in packet_sizes:
        for offset in range(NUM_PLAYERS):
            seat = (start + offset) % NUM_PLAYERS
            hands[seat].extend(cards[i:i + size])
            i += size
    return hands


def deal_whisted(pack, dealer, rng):
    if pack is None or len(pack) != NUM_CARDS:
        pack = shuffled_full_deck(rng)
    return deal_packets(cut(pack, rng), (dealer + 1) % NUM_PLAYERS)


def deal_uniform(rng):
    """Meme loi que engine.cards.deal() mais mains dans l'ordre de reception
    (non triees), pour qu'une passe generale ramasse un paquet realiste."""
    deck = shuffled_full_deck(rng)
    return [deck[i * HAND_SIZE:(i + 1) * HAND_SIZE] for i in range(NUM_PLAYERS)]


def is_jeu_blanc(hands):
    """Vrai des qu'UNE main n'a aucune figure (As, Roi, Dame, Valet)."""
    return any(not any(rank_of(c) in FACE_RANKS for c in hand) for hand in hands)


def gather_unplayed_hands(hands, dealer):
    pack = []
    for offset in (1, 2, 3, 4):
        pack.extend(hands[(dealer + offset) % NUM_PLAYERS])
    return pack


def gather_played(game, dealer):
    pack = list(game.play_order)
    discards = getattr(game, "discards", {})
    for offset in (1, 2, 3, 4):
        seat = (dealer + offset) % NUM_PLAYERS
        if seat in discards:
            pack.append(discards[seat])
    return pack


class DeckChain:
    """Suite de donnes d'une meme "partie" d'entrainement : garde le paquet
    ramasse et le donneur d'une donne a l'autre. Chaque donne est whistee
    avec la probabilite `whisted_ratio` (50/50 par defaut, voir
    scripts/train.py), sinon uniforme -- dans les deux cas elle alimente le
    paquet de la suivante, comme dans le vrai jeu."""

    def __init__(self, whisted_ratio, rng):
        self.whisted_ratio = whisted_ratio
        self.pack = None
        self.dealer = rng.randrange(NUM_PLAYERS)
        self.next_is_brimborion = False
        self.whisted_deals = 0
        self.uniform_deals = 0

    def deal(self, rng):
        """-> (mains, donneur, whiste?)."""
        dealer = self.dealer
        whisted = not self.next_is_brimborion and rng.random() < self.whisted_ratio
        self.next_is_brimborion = False
        if not whisted:
            self.uniform_deals += 1
            return deal_uniform(rng), dealer, False
        self.whisted_deals += 1
        hands = deal_whisted(self.pack, dealer, rng)
        # Jeu blanc sur une donne whistee : on re-ramasse les mains non
        # jouees et on recoupe (jamais un melange complet), meme donneur.
        while is_jeu_blanc(hands):
            hands = deal_whisted(gather_unplayed_hands(hands, dealer), dealer, rng)
        return hands, dealer, True

    def finish_played(self, game, dealer, contract_type=None):
        self.pack = gather_played(game, dealer)
        assert sorted(self.pack) == list(range(NUM_CARDS)), "paquet ramasse incomplet"
        self.dealer = (dealer + 1) % NUM_PLAYERS
        if contract_type == "trou":
            self.next_is_brimborion = True

    def finish_unplayed(self, hands, dealer):
        self.pack = gather_unplayed_hands(hands, dealer)
        self.dealer = dealer
