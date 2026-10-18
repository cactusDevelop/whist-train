"""Encodage des cartes et regles de pli, communes a tous les roles de jeu.

Carte = entier 0..51, carte = suit * 13 + rank. Suits: 0=Coeur, 1=Carreau,
2=Trefle, 3=Pique. Rank: 0..12 pour 2..As (rank 12 = As).
"""

import random

NUM_PLAYERS = 4
NUM_SUITS = 4
NUM_CARDS = 52
HAND_SIZE = 13
ACE_RANK = 12


def suit_of(card):
    return card // 13


def rank_of(card):
    return card % 13


def deal(rng=None):
    """Retourne 4 mains de 13 cartes triees."""
    rng = rng or random
    deck = list(range(NUM_CARDS))
    rng.shuffle(deck)
    return [sorted(deck[i * HAND_SIZE:(i + 1) * HAND_SIZE]) for i in range(NUM_PLAYERS)]


def legal_plays(hand, led_suit):
    """Cartes jouables : suivre la couleur demandee si possible, sinon tout."""
    if led_suit is None:
        return list(hand)
    same_suit = [c for c in hand if suit_of(c) == led_suit]
    return same_suit if same_suit else list(hand)


def trick_winner(plays, trump_suit):
    """plays: liste de (seat, card) dans l'ordre joue. Retourne le seat gagnant."""
    led_suit = suit_of(plays[0][1])

    def strength(card):
        s, r = suit_of(card), rank_of(card)
        if s == trump_suit:
            return (2, r)
        if s == led_suit:
            return (1, r)
        return (0, r)

    return max(plays, key=lambda sc: strength(sc[1]))[0]
