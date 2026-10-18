"""Observation v3 du jeu de la carte, commune a TOUS les roles de jeu
(emballage, solitaire, les 4 miseres, trou, bouche_trou, defenseur).

Principe (demande explicite de l'utilisateur, 2026-09-24) : l'IA recoit
EXACTEMENT ce qu'un client Godot voit a la table (mode debug desactive),
et jamais rien des mains adverses. Tout est construit a partir de donnees
PUBLIQUES brutes -- sa main, le journal des cartes jouees (siege, carte)
dans l'ordre, le contrat, le resume des annonces d'enchere, les mains
etalees, son propre ecart -- donc le port GDScript
(game/whist_ai_features_v2_play.gd, whist-4-it) recoit les memes entrees
brutes et doit produire un vecteur identique au float pres (verifie par
tests/ai_bridge_parity_v2_play_runner.gd).

Sieges RELATIFS partout : r = (siege - moi) % 4 -> 0 = moi, 1 = joue juste
apres moi (a ma gauche), 2 = en face, 3 = joue juste avant moi.

Blocs, dans l'ordre :
  A  main                                           52
  B  cartes jouees cette donne, par siege relatif   4 x 52
  C  pli en cours, par siege relatif                4 x 52
  D  dernier pli termine, par siege relatif         4 x 52
     + gagnant du dernier pli (relatif)             4
  E  cartes restantes non vues (ni en main, ni
     jouees, ni etalees, ni mon ecart)              52
  F  mains etalees (misere etalee), par relatif     4 x 52
  G  mon ecart (petite misere) + phase d'ecart      52 + 1
  H  contrat : type (emballage, trou, solo,
     misere) 4 + sorte de misere 4 + niveau 6..13
     8 + niveau inconnu 1 + atout 4 + trou royal 1  22
  I  declarant(s) / partenaire, par relatif         4 + 4
  J  plis gagnes par siege relatif (one-hot 0..13)  4 x 14
  K  numero du pli (0..12) + ma position dans le
     pli (0 = j'entame) + donneur (relatif)         13 + 4 + 4
  L  enchere : a eu lieu 1 + par siege relatif
     (proposes 4, emballes 4, rang max 0..19 20,
     attendu, couche, parole passee, relance)       1 + 4 x 32
  M  donne whistee                                  1
  N  (v4, 2026-09-26) pli en cours, derive des blocs
     C/H/I pour que le reseau n'ait plus a le
     recalculer : qui tient le pli (relatif) 4 +
     un coequipier tient le pli 1 + cartes jouables
     de ma main qui prendraient le pli              4 + 1 + 52
     Ajoute A LA FIN : les poids existants gardent
     leurs positions (scripts/migrate_obs_v4.py).
  O  (v5, 2026-09-28) renonces connues : siege
     relatif r x couleur s = 1 si r n'a pas fourni
     la couleur demandee s a un pli (termine ou en
     cours) -> il n'en a plus. Index r * 4 + s.
     Information publique qu'un joueur voit a la
     table ; avant la v5 le reseau ne l'avait que
     pour le dernier pli et le pli en cours (bloc B
     = cartes par siege SANS la couleur demandee),
     alors que l'ExpertBot s'en sert partout
     (may_hold / may_ruff). Ajoute A LA FIN
     (scripts/migrate_obs_v5.py).                  4 x 4
"""

from engine.bid_summary import MAX_RANK
from engine.cards import NUM_CARDS, NUM_PLAYERS, NUM_SUITS, legal_plays, suit_of, trick_winner
from engine.scoring import MISERE_ORDER

CONTRACT_TYPES = ["emballage", "trou", "solo", "misere"]
MIN_LEVEL, MAX_LEVEL = 6, 13
MAX_TRICKS = 13
BID_SEAT_DIM = NUM_SUITS + NUM_SUITS + (MAX_RANK + 1) + 4

OBS_DIM = (
    NUM_CARDS                                   # A
    + 4 * NUM_CARDS                             # B
    + 4 * NUM_CARDS                             # C
    + 4 * NUM_CARDS + 4                         # D
    + NUM_CARDS                                 # E
    + 4 * NUM_CARDS                             # F
    + NUM_CARDS + 1                             # G
    + len(CONTRACT_TYPES) + len(MISERE_ORDER) + (MAX_LEVEL - MIN_LEVEL + 1) + 1 + NUM_SUITS + 1  # H
    + 4 + 4                                     # I
    + 4 * (MAX_TRICKS + 1)                      # J
    + MAX_TRICKS + 4 + 4                        # K
    + 1 + 4 * BID_SEAT_DIM                      # L
    + 1                                         # M
    + 4 + 1 + NUM_CARDS                         # N
    + NUM_PLAYERS * NUM_SUITS                   # O
)
OBS_DIM_V4 = OBS_DIM - NUM_PLAYERS * NUM_SUITS
OBS_DIM_V3 = OBS_DIM_V4 - (4 + 1 + NUM_CARDS)


def teammates(contract, seat):
    """Coequipiers de `seat` d'apres le contrat PUBLIC (jamais `seat`) :
    declarant(s) + partenaire contre les autres si partenaire ; sinon
    chaque declarant seul et les defenseurs ensemble."""
    declarers = set(contract["declarers"])
    partner = contract.get("partner")
    if partner is not None:
        side = declarers | {partner}
        mine = side if seat in side else set(range(NUM_PLAYERS)) - side
    else:
        mine = {seat} if seat in declarers else set(range(NUM_PLAYERS)) - declarers
    return mine - {seat}


def current_trick_features(seat, hand, current, contract, trump):
    """Bloc N -> liste de 57 flottants."""
    f = [0.0] * (4 + 1 + NUM_CARDS)
    if not current:
        return f
    w = trick_winner(current, trump)
    f[(w - seat) % NUM_PLAYERS] = 1.0
    if w in teammates(contract, seat):
        f[4] = 1.0
    for c in legal_plays(hand, suit_of(current[0][1])):
        if trick_winner(current + [(seat, c)], trump) == seat:
            f[5 + c] = 1.0
    return f


def void_features(seat, play_log):
    """Bloc O -> 16 flottants : [r * 4 + couleur] = 1 si le siege relatif r
    n'a pas fourni la couleur demandee a un pli (termine ou en cours)."""
    f = [0.0] * (NUM_PLAYERS * NUM_SUITS)
    for i in range(0, len(play_log), NUM_PLAYERS):
        chunk = play_log[i:i + NUM_PLAYERS]
        led = suit_of(chunk[0][1])
        for s, c in chunk[1:]:
            if suit_of(c) != led:
                f[((s - seat) % NUM_PLAYERS) * NUM_SUITS + led] = 1.0
    return f


def split_tricks(play_log, trump_suit):
    """play_log : [(siege, carte), ...] chronologique (cartes JOUEES
    uniquement, jamais les ecarts). -> (plis termines [(plays, gagnant)],
    pli en cours [(siege, carte)])."""
    done = []
    n_full = len(play_log) // NUM_PLAYERS
    for t in range(n_full):
        plays = play_log[t * NUM_PLAYERS:(t + 1) * NUM_PLAYERS]
        done.append((plays, trick_winner(plays, trump_suit)))
    return done, play_log[n_full * NUM_PLAYERS:]


def build(seat, hand, play_log, dealer, contract, bid_summary=None,
          revealed_hands=None, own_discard=None, is_discard_phase=False, whisted=False):
    """`contract` : {"type": "emballage"|"trou"|"solo"|"misere",
    "trump": 0..3|None, "level": 6..13|None (None = inconnu, Phase 1),
    "kind": sorte de misere|None, "declarers": [sieges], "partner": siege|None,
    "royal": bool}. `bid_summary` : engine/bid_summary.py, ou None si aucune
    enchere n'a eu lieu (Phase 1). `revealed_hands` : {siege: cartes encore
    en main} pour chaque main etalee."""
    rel = lambda s: (s - seat) % NUM_PLAYERS
    trump = contract.get("trump")
    done, current = split_tricks(play_log, trump)
    revealed_hands = revealed_hands or {}

    def cards_mask(cards):
        m = [0.0] * NUM_CARDS
        for c in cards:
            m[c] = 1.0
        return m

    def per_rel_seat(pairs):
        blocks = [[0.0] * NUM_CARDS for _ in range(NUM_PLAYERS)]
        for s, c in pairs:
            blocks[rel(s)][c] = 1.0
        return [v for b in blocks for v in b]

    def onehot(n, i):
        v = [0.0] * n
        if i is not None:
            v[i] = 1.0
        return v

    f = []
    f += cards_mask(hand)                                               # A
    f += per_rel_seat(play_log)                                         # B
    f += per_rel_seat(current)                                          # C
    if done:                                                            # D
        last_plays, last_winner = done[-1]
        f += per_rel_seat(last_plays) + onehot(4, rel(last_winner))
    else:
        f += [0.0] * (4 * NUM_CARDS + 4)
    seen = set(hand) | {c for _, c in play_log}                         # E
    for cards in revealed_hands.values():
        seen |= set(cards)
    if own_discard is not None:
        seen.add(own_discard)
    f += [0.0 if c in seen else 1.0 for c in range(NUM_CARDS)]
    f += per_rel_seat([(s, c) for s, cards in revealed_hands.items() for c in cards])  # F
    f += cards_mask([] if own_discard is None else [own_discard])       # G
    f += [1.0 if is_discard_phase else 0.0]
    ctype = contract["type"]                                            # H
    f += onehot(len(CONTRACT_TYPES), CONTRACT_TYPES.index(ctype))
    kind = contract.get("kind")
    f += onehot(len(MISERE_ORDER), None if kind is None else MISERE_ORDER.index(kind))
    level = contract.get("level")
    f += onehot(MAX_LEVEL - MIN_LEVEL + 1, None if level is None else level - MIN_LEVEL)
    f += [1.0 if (level is None and ctype != "misere") else 0.0]
    f += onehot(NUM_SUITS, trump)
    f += [1.0 if contract.get("royal") else 0.0]
    declarer_rel = [0.0] * 4                                            # I
    for d in contract["declarers"]:
        declarer_rel[rel(d)] = 1.0
    partner = contract.get("partner")
    f += declarer_rel + onehot(4, None if partner is None else rel(partner))
    tricks = [0] * NUM_PLAYERS                                          # J
    for _plays, w in done:
        tricks[w] += 1
    for r in range(NUM_PLAYERS):
        f += onehot(MAX_TRICKS + 1, tricks[(seat + r) % NUM_PLAYERS])
    f += onehot(MAX_TRICKS, min(len(done), MAX_TRICKS - 1))             # K
    f += onehot(4, len(current))
    f += onehot(4, rel(dealer))
    f += [0.0 if bid_summary is None else 1.0]                          # L
    for r in range(NUM_PLAYERS):
        if bid_summary is None:
            f += [0.0] * BID_SEAT_DIM
            continue
        e = bid_summary[(seat + r) % NUM_PLAYERS]
        f += [1.0 if x else 0.0 for x in e["proposed"]]
        f += [1.0 if x else 0.0 for x in e["emballed"]]
        f += onehot(MAX_RANK + 1, e["max_rank"])
        f += [1.0 if e[k] else 0.0 for k in ("waited", "passed", "pass_parole", "raised")]
    f += [1.0 if whisted else 0.0]                                      # M
    f += current_trick_features(seat, hand, current, contract, trump)   # N
    f += void_features(seat, play_log)                                  # O
    assert len(f) == OBS_DIM, (len(f), OBS_DIM)
    return f


# --- Version rapide (rollouts, agents/rollouts.py) --------------------------
# MEME vecteur que build(), ecrit directement dans une ligne numpy a partir
# de l'objet donne (plis gagnes, pli en cours deja suivis par le moteur :
# pas de recalcul de tous les plis passes). build() reste la REFERENCE
# (parite Godot) ; l'egalite exacte est verifiee par
# scripts/gates/check_fast_observation.py.

def _offsets():
    sizes = [("A", NUM_CARDS), ("B", 4 * NUM_CARDS), ("C", 4 * NUM_CARDS), ("D", 4 * NUM_CARDS),
             ("Dw", 4), ("E", NUM_CARDS), ("F", 4 * NUM_CARDS), ("G", NUM_CARDS), ("Gp", 1),
             ("Ht", len(CONTRACT_TYPES)), ("Hk", len(MISERE_ORDER)), ("Hl", MAX_LEVEL - MIN_LEVEL + 1),
             ("Hu", 1), ("Htr", NUM_SUITS), ("Hr", 1), ("Id", 4), ("Ip", 4), ("J", 4 * (MAX_TRICKS + 1)),
             ("Kt", MAX_TRICKS), ("Kp", 4), ("Kd", 4), ("L", 1), ("Ls", 4 * BID_SEAT_DIM), ("M", 1),
             ("N", 4 + 1 + NUM_CARDS), ("O", NUM_PLAYERS * NUM_SUITS)]
    out, pos = {}, 0
    for name, n in sizes:
        out[name] = pos
        pos += n
    assert pos == OBS_DIM
    return out


_OFF = _offsets()


def build_fast_into(out, game, seat):
    """Remplit `out` (ligne numpy float32 de OBS_DIM, deja a zero) avec
    exactement game.observation(seat)."""
    o = _OFF
    rel = lambda s: (s - seat) % NUM_PLAYERS
    hand = game.hands[seat]
    play_log = game.play_log
    for c in hand:
        out[o["A"] + c] = 1.0
    seen = set(hand)
    for s, c in play_log:
        out[o["B"] + rel(s) * NUM_CARDS + c] = 1.0
        seen.add(c)
    for s, c in game.current_trick:
        out[o["C"] + rel(s) * NUM_CARDS + c] = 1.0
    n_done = len(play_log) // NUM_PLAYERS
    if n_done:
        last = play_log[(n_done - 1) * NUM_PLAYERS:n_done * NUM_PLAYERS]
        for s, c in last:
            out[o["D"] + rel(s) * NUM_CARDS + c] = 1.0
        out[o["Dw"] + rel(trick_winner(last, game.trump_suit))] = 1.0
    revealed = getattr(game, "reveal_declarer_hands", False)
    if revealed:
        for d in game.declarers:
            for c in game.hands[d]:
                out[o["F"] + rel(d) * NUM_CARDS + c] = 1.0
                seen.add(c)
    own_discard = getattr(game, "discards", {}).get(seat)
    if own_discard is not None:
        seen.add(own_discard)
        out[o["G"] + own_discard] = 1.0
    for c in range(NUM_CARDS):
        if c not in seen:
            out[o["E"] + c] = 1.0
    if getattr(game, "pending_discards", None):
        out[o["Gp"]] = 1.0
    contract = game.public_contract()
    ctype = contract["type"]
    out[o["Ht"] + CONTRACT_TYPES.index(ctype)] = 1.0
    kind = contract.get("kind")
    if kind is not None:
        out[o["Hk"] + MISERE_ORDER.index(kind)] = 1.0
    level = contract.get("level")
    if level is not None:
        out[o["Hl"] + level - MIN_LEVEL] = 1.0
    elif ctype != "misere":
        out[o["Hu"]] = 1.0
    trump = contract.get("trump")
    if trump is not None:
        out[o["Htr"] + trump] = 1.0
    if contract.get("royal"):
        out[o["Hr"]] = 1.0
    for d in contract["declarers"]:
        out[o["Id"] + rel(d)] = 1.0
    partner = contract.get("partner")
    if partner is not None:
        out[o["Ip"] + rel(partner)] = 1.0
    for r in range(NUM_PLAYERS):
        out[o["J"] + r * (MAX_TRICKS + 1) + game.tricks_won[(seat + r) % NUM_PLAYERS]] = 1.0
    out[o["Kt"] + min(n_done, MAX_TRICKS - 1)] = 1.0
    out[o["Kp"] + len(game.current_trick)] = 1.0
    out[o["Kd"] + rel(game.dealer)] = 1.0
    bid_summary = game.bid_summary
    if bid_summary is not None:
        out[o["L"]] = 1.0
        for r in range(NUM_PLAYERS):
            e = bid_summary[(seat + r) % NUM_PLAYERS]
            base = o["Ls"] + r * BID_SEAT_DIM
            for i, x in enumerate(e["proposed"]):
                if x:
                    out[base + i] = 1.0
            for i, x in enumerate(e["emballed"]):
                if x:
                    out[base + NUM_SUITS + i] = 1.0
            if e["max_rank"] is not None:
                out[base + 2 * NUM_SUITS + e["max_rank"]] = 1.0
            for i, k in enumerate(("waited", "passed", "pass_parole", "raised")):
                if e[k]:
                    out[base + 2 * NUM_SUITS + MAX_RANK + 1 + i] = 1.0
    if game.whisted:
        out[o["M"]] = 1.0
    if game.current_trick:
        n = current_trick_features(seat, hand, list(game.current_trick), contract, game.trump_suit)
        for i, v in enumerate(n):
            if v:
                out[o["N"] + i] = 1.0
    for i, v in enumerate(void_features(seat, play_log)):
        if v:
            out[o["O"] + i] = 1.0
