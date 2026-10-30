"""Observation de l'IA jokers (v3) -- SPEC A PORTER A L'IDENTIQUE en GDScript
(voir godot_export/joker_obs_spec.md, verifiee par
godot_export/joker_parity_cases.json).

Relative au siege `me` qui decide (son tour de pli, avant sa carte).
Couleurs dans l'ordre WhistCard.SUITS : ♥ ♦ ♣ ♠ (engine/cards.py 0..3).
"Ami" / "adversaire" = JokerRules.is_opponent(me, x) : camps tires du
contrat (misere : chaque declarant seul, defenseurs allies ; sinon
[declarer, partner] contre le reste).
"""

from engine.cards import NUM_CARDS, NUM_PLAYERS, suit_of, rank_of, legal_plays
from engine.misere import MISERE_MAX_TRICKS
from engine.jokers import CASTABLE, FACE_RANKS, ACE, PLAYER, Collet, Embuscade, FaussePiste, Kermesse, VoeuDePauvrete

# (nom, taille) dans l'ordre ; OBS_DIM = somme.
LAYOUT = [
    ("tricks_done", 1),        # plis deja joues / 13
    ("position", 4),           # cartes deja dans le pli : 0,1,2,3 (one-hot)
    ("my_side_declares", 1),   # me dans les declarants (ou partner du contrat)
    ("contract_kind", 3),      # misere / solo (seul avec atout) / equipe (emballage, trou)
    ("contract_trump", 5),     # atout du contrat ♥♦♣♠ / sans
    ("effective_trump", 5),    # atout du pli apres effets vivants (JokerRules.trick_trump)
    ("led_suit", 5),           # couleur demandee ♥♦♣♠ / aucune (j'entame)
    ("best_friendly", 1),      # meilleur actuel (JokerRules.trick_winner) = moi ou ami
    ("best_opponent", 1),      # meilleur actuel = adversaire
    ("opponent_ruffed", 1),    # un adversaire a joue l'atout effectif, demande != atout effectif
    ("best_is_face", 1),       # carte du meilleur actuel K/D/V (pas un As)
    ("best_rank", 1),          # rang du meilleur actuel / 12 (2 = 0, As = 1)
    ("hand_size", 1),          # mes cartes / 13
    ("suit_counts", 4),        # mes cartes par couleur / 13
    ("my_trumps", 1),          # mes atouts du contrat / 13 (0 sans atout)
    ("my_faces", 1),           # mes K/D/V / 13
    ("my_aces", 1),            # mes As / 4
    ("can_follow", 1),         # j'ai la couleur demandee (0 si j'entame)
    ("can_win_now", 1),        # une de mes cartes legales gagnerait le pli tel quel (0 si j'entame)
    ("tricks_team", 1),        # plis (tricks_won) de moi + mes allies (non is_opponent) / 13
    ("tricks_others", 1),      # plis des autres / 13
    ("declarer_tricks", 1),    # plis du camp declarant / 13
    ("contract_level", 1),     # niveau du contrat / 13 (misere : plis max autorises / 13)
    ("energy", 1),             # ⚡ / 6 (plafonne a 1)
    ("joker_counts", len(CASTABLE)),  # exemplaires de chaque CASTABLE en main / 2
    ("piles", 2),              # pioche / 10, defausse / 10 (plafonnes a 1)
    ("embuscade", 3),          # embuscades vivantes non depensees : a moi / a un ami / a un adversaire (0/1)
    ("kermesse", 1),           # kermesses vivantes (tout proprietaire) / 2
    ("voeu", 1),               # vœu de pauvrete arme a moi
    ("enemy_traps", 2),        # pieges adverses armes non depenses : fausse_piste / collet (0/1) -- v2
    # v3 (30/10, demande user : tout ce que le joueur voit), AJOUTE A LA FIN (train_jokers.widen_obs)
    ("my_cards", NUM_CARDS),   # ma main exacte
    ("played", 4 * NUM_CARDS), # cartes jouees cette donne, par siege relatif (r = (siege - me) % 4)
    ("trick", 4 * NUM_CARDS),  # pli en cours, par siege relatif
    ("suspicion", 2),          # jauge de Soupcon du joueur / seuil de l'Oeil (plafonne a 1), seuil / 10
    ("eye", 4),                # sanction de l'Oeil : aucune / points / confisque / allumettes
    ("player_seat", 4),        # siege relatif du joueur
    ("joker_hand_sizes", 4),   # jokers en main par siege relatif / 10 (plafonne a 1)
]
EYE_SANCTIONS = ("", "points", "confisque", "allumettes")
OBS_DIM = sum(n for _k, n in LAYOUT)
OFFSETS = {}
_o = 0
for _k, _n in LAYOUT:
    OFFSETS[_k] = _o
    _o += _n


def _suit5(s):
    return 4 if s is None else s


def build(hand, me):
    """hand : engine.jokers.JokerHand."""
    g, rules, state = hand.game, hand.rules, hand.state
    c = hand.contract
    out = [0.0] * OBS_DIM

    def put(key, value, i=0):
        out[OFFSETS[key] + i] = float(value)

    put("tricks_done", g.trick_count / 13.0)
    trick = list(g.current_trick)
    put("position", 1, len(trick))
    declarers = set(c.get("declarers") or [c["declarer"]])
    if c.get("partner") is not None and c.get("partner", -1) >= 0:
        declarers.add(c["partner"])
    put("my_side_declares", me in declarers)
    kind = 0 if c["type"] == "misere" else (1 if c["type"] == "solo" else 2)
    put("contract_kind", 1, kind)
    put("contract_trump", 1, _suit5(g.trump_suit))
    eff = rules.trick_trump()
    put("effective_trump", 1, _suit5(eff))
    led = suit_of(trick[0][1]) if trick else None
    put("led_suit", 1, _suit5(led))
    my = g.hands[me]
    if trick:
        best_seat = rules.trick_winner(trick, led, eff)
        best_card = [cd for s, cd in trick if s == best_seat][0]
        put("best_friendly", not rules.is_opponent(me, best_seat))
        put("best_opponent", rules.is_opponent(me, best_seat))
        put("opponent_ruffed", eff is not None and led != eff and any(
            rules.is_opponent(me, s) and suit_of(cd) == eff for s, cd in trick))
        put("best_is_face", rank_of(best_card) in FACE_RANKS)
        put("best_rank", rank_of(best_card) / 12.0)
        put("can_follow", any(suit_of(x) == led for x in my))
        put("can_win_now", any(rules.trick_winner(trick + [(me, x)], led, eff) == me
                               for x in legal_plays(my, led)))
    put("hand_size", len(my) / 13.0)
    for s in range(4):
        put("suit_counts", sum(1 for x in my if suit_of(x) == s) / 13.0, s)
    put("my_trumps", (sum(1 for x in my if suit_of(x) == g.trump_suit) if g.trump_suit is not None else 0) / 13.0)
    put("my_faces", sum(1 for x in my if rank_of(x) in FACE_RANKS) / 13.0)
    put("my_aces", sum(1 for x in my if rank_of(x) == ACE) / 4.0)
    team = sum(g.tricks_won[x] for x in range(NUM_PLAYERS) if not rules.is_opponent(me, x))
    put("tricks_team", team / 13.0)
    put("tricks_others", (sum(g.tricks_won) - team) / 13.0)
    put("declarer_tricks", sum(g.tricks_won[s] for s in declarers) / 13.0)
    level = MISERE_MAX_TRICKS[c["kind"]] if c["type"] == "misere" else (c.get("level") or 0)
    put("contract_level", level / 13.0)
    put("energy", min(1.0, state.energy[me] / 6.0))
    for i, jid in enumerate(CASTABLE):
        put("joker_counts", state.joker_hand[me].count(jid) / 2.0, i)
    put("piles", min(1.0, len(state.draw_pile[me]) / 10.0), 0)
    put("piles", min(1.0, len(state.discard[me]) / 10.0), 1)
    kermesses = 0
    for e in rules.effects:
        if e.spent:
            continue
        if isinstance(e, Embuscade):
            idx = 0 if e.owner == me else (2 if rules.is_opponent(me, e.owner) else 1)
            put("embuscade", 1, idx)
        elif isinstance(e, Kermesse):
            kermesses += 1
        elif isinstance(e, VoeuDePauvrete) and e.owner == me:
            put("voeu", 1)
        elif isinstance(e, FaussePiste) and rules.is_opponent(me, e.owner):
            put("enemy_traps", 1, 0)
        elif isinstance(e, Collet) and rules.is_opponent(me, e.owner):
            put("enemy_traps", 1, 1)
    put("kermesse", min(1.0, kermesses / 2.0))
    for c_ in my:
        put("my_cards", 1, c_)
    for s, cd in g.play_log:
        put("played", 1, ((s - me) % NUM_PLAYERS) * NUM_CARDS + cd)
    for s, cd in trick:
        put("trick", 1, ((s - me) % NUM_PLAYERS) * NUM_CARDS + cd)
    put("suspicion", min(1.0, state.suspicion / state.eye_threshold), 0)
    put("suspicion", state.eye_threshold / 10.0, 1)
    put("eye", 1, EYE_SANCTIONS.index(getattr(hand, "eye", "")))
    put("player_seat", 1, (PLAYER - me) % NUM_PLAYERS)
    for s in range(NUM_PLAYERS):
        put("joker_hand_sizes", min(1.0, len(state.joker_hand[s]) / 10.0), (s - me) % NUM_PLAYERS)
    return out
