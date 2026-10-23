"""Valeur de CHAQUE carte legale par rollouts determinises (2026-09-26).

Pourquoi : PPO ne voit que la carte jouee et un retour bruite par toute la
suite de la donne ; une erreur dont le cout arrive 5 plis plus tard
(surcoupe du partenaire, atout gaspille...) n'est presque jamais attribuee
a la bonne carte. Ici, pour une decision donnee, on rejoue la fin de la
donne pour CHAQUE carte legale, dans plusieurs mondes possibles :
  - les cartes cachees (mains adverses/partenaire, ecarts inconnus de la
    petite misere) sont redistribuees au hasard, en respectant tout ce que
    le joueur sait : sa main, les cartes jouees, les mains etalees, son
    propre ecart, les coupes/defausses (siege qui n'a pas fourni = chicane
    dans la couleur demandee), le nombre de cartes de chacun ;
  - les MEMES mondes servent pour toutes les cartes (comparaison appariee :
    la difference entre deux cartes ne depend pas du tirage) ;
  - la suite est jouee par les reseaux ACTUELS en greedy (la vraie IA) ;
  - la valeur = la MEME fonction de score que la recompense PPO.
Resultat : Q(carte) pour chaque carte legale -> ensemble des MEILLEURES
cartes (a `marge` pres du meilleur Q). agents/ppo.py demande seulement au
reseau de mettre sa probabilite quelque part dans cet ensemble : les
egalites ne coutent rien (on ne pousse jamais vers un jeu au hasard entre
cartes equivalentes), seule une carte nettement moins bonne est corrigee.
Aucune regle de jeu inventee, aucun malus : on compare juste les cartes
entre elles sur le vrai score.

Limite connue : l'enchere et le deck whiste donnent des indices sur les
mains cachees que ce tirage uniforme ignore (approximation standard des
jeux de levees).
"""

import copy

import numpy as np
import torch

from agents.play_guards import ai_play_mask
from engine.cards import NUM_CARDS, NUM_PLAYERS, suit_of
from engine.observation import OBS_DIM, build_fast_into

_LIST_ATTRS = ("played", "current_trick", "play_order", "play_log", "tricks_won", "pending_discards")


def clone_with_hands(game, hands):
    """Copie independante de la donne (etat de jeu), avec d'autres mains."""
    g = copy.copy(game)
    g.hands = [list(h) for h in hands]
    for name in _LIST_ATTRS:
        if hasattr(game, name):
            setattr(g, name, list(getattr(game, name)))
    if hasattr(game, "discards"):
        g.discards = dict(game.discards)
    if getattr(game, "contract_record", None) is not None:
        g.contract_record = dict(game.contract_record)  # la fixation du Trou d'un rollout reste locale
    return g


def void_suits(play_log):
    """{siege: couleurs ou il n'a pas fourni} depuis les plis joues."""
    voids = {s: set() for s in range(NUM_PLAYERS)}
    for start in range(0, len(play_log), NUM_PLAYERS):
        trick = play_log[start:start + NUM_PLAYERS]
        led = suit_of(trick[0][1])
        for s, c in trick[1:]:
            if suit_of(c) != led:
                voids[s].add(led)
    return voids


def determinize(game, seat, rng, tries=30):
    """Mains possibles de tous les sieges, vues par `seat`. -> liste de 4 mains."""
    fixed = {seat: list(game.hands[seat])}
    if getattr(game, "reveal_declarer_hands", False):
        for d in game.declarers:
            fixed[d] = list(game.hands[d])
    known = {c for h in fixed.values() for c in h}
    known.update(c for _s, c in game.play_log)
    own_discard = getattr(game, "discards", {}).get(seat)
    if own_discard is not None:
        known.add(own_discard)
    hidden_seats = [s for s in range(NUM_PLAYERS) if s not in fixed]
    pool = [c for c in range(NUM_CARDS) if c not in known]
    need = {s: len(game.hands[s]) for s in hidden_seats}
    spare = len(pool) - sum(need.values())  # ecarts inconnus (petite misere), jamais joues
    voids = void_suits(game.play_log)

    for attempt in range(tries + 1):
        respect_voids = attempt < tries  # dernier recours : ignorer les chicanes
        cap = dict(need)
        spare_left = spare
        out = {s: [] for s in hidden_seats}
        cards = list(pool)
        rng.shuffle(cards)
        eligible = {c: [s for s in hidden_seats if not (respect_voids and suit_of(c) in voids[s])] for c in cards}
        cards.sort(key=lambda c: len(eligible[c]))  # cartes les plus contraintes d'abord
        ok = True
        for c in cards:
            choices = [s for s in eligible[c] if cap[s] > 0]
            weights = [cap[s] for s in choices]
            if spare_left > 0:
                choices.append(None)
                weights.append(spare_left)
            if not choices:
                ok = False
                break
            s = rng.choices(choices, weights=weights)[0]
            if s is None:
                spare_left -= 1
            else:
                out[s].append(c)
                cap[s] -= 1
        if ok:
            hands = [None] * NUM_PLAYERS
            for s, h in fixed.items():
                hands[s] = h
            for s in hidden_seats:
                hands[s] = sorted(out[s])
            return hands
    raise AssertionError("determinisation impossible")  # inatteignable : le dernier essai n'a pas de contrainte


def play_out_greedy_batched(games, nets):
    """Joue toutes les donnes jusqu'au bout, en greedy, un passage de reseau
    par role et par pas (toutes les donnes en parallele)."""
    active = [g for g in games if not g.done]
    with torch.no_grad():
        while active:
            by_role = {}
            for g in active:
                by_role.setdefault(g.role_of(g.turn), []).append(g)
            for role, group in by_role.items():
                obs = np.zeros((len(group), OBS_DIM), dtype=np.float32)
                mask = np.zeros((len(group), NUM_CARDS), dtype=bool)
                for row, g in enumerate(group):
                    build_fast_into(obs[row], g, g.turn)
                    mask[row] = ai_play_mask(g, g.turn)
                obs, mask = torch.from_numpy(obs), torch.from_numpy(mask)
                logits, _v = nets[role](obs)
                actions = logits.masked_fill(~mask, -1e9).argmax(dim=1).tolist()
                for g, a in zip(group, actions):
                    g.step(g.turn, a)
            active = [g for g in active if not g.done]


def play_out_expert(games):
    """Joue toutes les donnes jusqu'au bout avec l'ExpertBot aux 4 sieges."""
    from agents import expert_bot
    for g in games:
        while not g.done:
            g.step(g.turn, expert_bot.choose_card(g, g.turn))
    return games


# Pool de processus (multiprocessing.Pool) pour les rollouts "expert", installe
# par scripts/train.py (--rollout-workers, 2026-10-20) : la suite jouee par
# l'ExpertBot est du Python pur, sans reseau -> les mondes d'une decision se
# repartissent sur les coeurs (donnes picklables, ~700 octets chacune).
_POOL = None
_POOL_SIZE = 1


def set_pool(pool, size):
    global _POOL, _POOL_SIZE
    _POOL, _POOL_SIZE = pool, size


def _play_out_expert_parallel(games):
    if _POOL is None or _POOL_SIZE < 2 or len(games) < 2:
        return play_out_expert(games)
    step = -(-len(games) // _POOL_SIZE)
    chunks = [games[i:i + step] for i in range(0, len(games), step)]
    return [g for chunk in _POOL.map(play_out_expert, chunks) for g in chunk]


def action_values(game, seat, nets, score_fn, rng, samples, policy="nets"):
    """-> (cartes legales, Q moyen de chacune pour `seat`) ou None si une
    seule carte legale. `score_fn(donne_terminee)` -> score des 4 sieges.
    `policy` : "nets" (suite jouee par les reseaux) ou "expert" (suite jouee
    par l'ExpertBot : un pas d'amelioration de politique au-dessus du bot,
    PIMC -- voir scripts/train.py:ROLLOUT_POLICY_DEFAULT)."""
    legal = [c for c, ok in enumerate(game.legal_action_mask(seat)) if ok]
    if len(legal) < 2:
        return None
    if policy == "dds":
        q = dds_values(game, seat, legal, score_fn, rng, samples)
        return None if q is None else (legal, q)
    sims = []
    for _ in range(samples):
        hands = determinize(game, seat, rng)
        for i, card in enumerate(legal):
            g = clone_with_hands(game, hands)
            g.step(seat, card)
            sims.append((i, g))
    if policy == "expert":
        sims = list(zip([i for i, _g in sims], _play_out_expert_parallel([g for _i, g in sims])))
    else:
        play_out_greedy_batched([g for _i, g in sims], nets)
    q = [0.0] * len(legal)
    for i, g in sims:
        q[i] += score_fn(g)[seat] / samples
    return legal, q


DDS_WORLD_NODES = 15_000_000  # budget d'un monde (~1.5 s) : au-dela il est ecarte


def _dds_world(args):
    """Un monde : plis restants du camp declarant apres chaque carte legale
    (None si la carte termine la donne), ou None si le monde depasse son
    budget de noeuds. Picklable (pool de processus)."""
    from engine import dds
    game, seat, legal, hands = args
    out = []
    last_trump = object()
    left = DDS_WORLD_NODES
    for card in legal:
        g = clone_with_hands(game, hands)
        g.step(seat, card)
        if g.done:
            out.append(None)
            continue
        n0 = dds.nodes()
        r = dds.solve(g, new_gen=g.trump_suit != last_trump, node_limit=left)
        left -= dds.nodes() - n0
        if r is None or left <= 0:
            return None
        out.append(r)
        last_trump = g.trump_suit
    return out


def dds_values(game, seat, legal, score_fn, rng, samples):
    """Q de chaque carte legale : moyenne sur `samples` mondes du vrai score
    quand la suite est jouee PARFAITEMENT a cartes ouvertes (engine/dds.c)
    -- un professeur plus fort que n'importe quel bot a regles (2026-10-23).
    Contrats a atout seulement (dds.supported). Les mondes se repartissent
    sur le pool de processus (set_pool) s'il existe."""
    from engine import dds
    worlds = [(game, seat, legal, determinize(game, seat, rng)) for _ in range(samples)]
    if _POOL is not None and _POOL_SIZE > 1 and samples > 1:
        results = _POOL.map(_dds_world, worlds)
    else:
        results = [_dds_world(w) for w in worlds]
    kept = [(w[3], more) for w, more in zip(worlds, results) if more is not None]
    if len(kept) * 2 < samples:
        return None  # trop de mondes hors budget : pas de cible
    q = [0.0] * len(legal)
    for hands, more in kept:
        for i, card in enumerate(legal):
            g = clone_with_hands(game, hands)
            g.step(seat, card)
            if more[i] is not None:
                g = dds.finished_copy(g, more[i])
            q[i] += score_fn(g)[seat] / len(kept)
    return q


def best_set_target(legal, q, margin, scale=1.0):
    """Masque (52) des cartes dont Q/scale est a `margin` pres du meilleur,
    ou None si toutes les cartes legales en font partie (rien a apprendre)."""
    best = max(q)
    chosen = [c for c, v in zip(legal, q) if (best - v) / scale <= margin]
    if len(chosen) == len(legal):
        return None
    target = torch.zeros(NUM_CARDS)
    target[torch.tensor(chosen)] = 1.0
    return target
