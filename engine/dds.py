"""Solveur double-mort (engine/dds.c, charge par ctypes) : plis restants du
camp declarant quand tout le monde voit toutes les cartes et joue au mieux.
Professeur du jeu de la carte (rapport utilisateur 2026-10-23 : « ExpertBot
est horrible aussi ! Ce n'est pas une reference ») -- voir
agents/rollouts.py, policy "dds".

Contrats a atout seulement (emballage, trou, solo) : la misere et l'ecart
de petite misere n'ont pas de camp qui maximise ses plis.

Le .c est compile au premier appel (cc -O3) dans WHIST_DDS_DIR (defaut :
dossier temporaire), nomme par l'empreinte du source.
"""

import copy
import ctypes
import hashlib
import os
import subprocess
import tempfile

from engine.cards import NUM_PLAYERS

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dds.c")
_LIB = None
SUPPORTED = ("emballage", "trou", "solo")


def _lib():
    global _LIB
    if _LIB is None:
        with open(_SRC, "rb") as f:
            tag = hashlib.sha1(f.read()).hexdigest()[:12]
        out_dir = os.environ.get("WHIST_DDS_DIR") or os.path.join(tempfile.gettempdir(), "whist_dds")
        os.makedirs(out_dir, exist_ok=True)
        so = os.path.join(out_dir, "dds_%s.so" % tag)
        if not os.path.exists(so):
            tmp = "%s.%d.tmp" % (so, os.getpid())
            subprocess.check_call([os.environ.get("CC", "cc"), "-O3", "-shared", "-fPIC", "-o", tmp, _SRC])
            os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        lib.dds_solve.restype = ctypes.c_int
        lib.dds_solve.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                  ctypes.c_int]
        lib.dds_nodes.restype = ctypes.c_longlong
        lib.dds_set_limit.argtypes = [ctypes.c_longlong]
        _LIB = lib
    return _LIB


def supported(game):
    return getattr(game, "contract_type", None) in SUPPORTED and not getattr(game, "pending_discards", None)


def side_seats(game):
    """Camp declarant : declarant + partenaire (emballage, trou), le soliste seul."""
    if hasattr(game, "partner"):
        return (game.declarer, game.partner)
    return tuple(game.declarers)


def solve(game, new_gen=True, node_limit=0):
    """Plis restants (pli en cours compris) du camp declarant, jeu parfait a
    cartes ouvertes, depuis l'etat de `game` (ses mains). `new_gen=False`
    garde la table de transposition (meme atout, memes camps). `node_limit`
    > 0 : None si la recherche depasse ce nombre de noeuds."""
    if game.trump_suit is None and getattr(game, "trou_pending", lambda: False)():
        raise ValueError("Trou pas encore fixe : jouer l'entame d'abord")
    lib = _lib()
    hands = (ctypes.c_uint64 * NUM_PLAYERS)(*[sum(1 << c for c in h) for h in game.hands])
    trick = list(game.current_trick)
    leader = trick[0][0] if trick else game.turn
    cards = (ctypes.c_int * 4)(*([c for _s, c in trick] + [0] * (4 - len(trick))))
    seats = (ctypes.c_int * 4)(*([s for s, _c in trick] + [0] * (4 - len(trick))))
    mask = sum(1 << s for s in set(side_seats(game)))
    trump = -1 if game.trump_suit is None else game.trump_suit
    lib.dds_set_limit(node_limit)
    r = lib.dds_solve(hands, trump, mask, leader, len(trick), cards, seats, 1 if new_gen else 0)
    if r == -2:
        return None
    if r < 0:
        raise MemoryError("dds : table de transposition")
    return r


def nodes():
    return _lib().dds_nodes()


def finished_copy(game, side_more):
    """Copie TERMINEE de la donne ou le camp declarant gagne encore
    `side_more` plis (les autres a la defense) -- pour la fonction de score
    (les final_scores ne lisent que les plis du camp / du soliste)."""
    side = side_seats(game)
    g = copy.copy(game)
    g.tricks_won = list(game.tricks_won)
    remaining = 13 - game.trick_count
    defender = next(s for s in range(NUM_PLAYERS) if s not in side)
    g.tricks_won[side[0]] += side_more
    g.tricks_won[defender] += remaining - side_more
    g.trick_count = 13
    g.current_trick = []
    g.done = True
    return g
