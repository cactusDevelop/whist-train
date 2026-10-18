"""Genere un enregistrement de partie PERSONNALISE (contrat + sieges choisis
a la main, sans passer par l'enchere) au format JSON de replay attendu par
whist-4-it -- voir `game/replay_format.gd` (source de verite du schema,
docstring complete) et l'exemple `tests/fixtures/replay/sample_game.json`
dans ce projet, confirmes par la session whist-4-it le 2026-09-19. Ce format
est lisible par l'onglet "Replay" du menu debug (pause/reprendre/avancer/
reculer/scrubber).

Usage typique demande par l'utilisateur : "contrat Solo, je veux voir l'IA
Solo contre 3 IA sans contrat" :
    venv/Scripts/python.exe scripts/record_game.py --contract solitaire

Le(s) siege(s) declarant(s) jouent avec le reseau specialiste du contrat
choisi ; TOUS les autres sieges jouent avec `defenseur` -- jamais une
politique aleatoire ici (contrairement a scripts/eval_vs_random.py, qui sert
a MESURER un niveau) puisque le but est d'OBSERVER un vrai match entre
specialistes entraines. Politique GREEDY partout (agents/network.py:
act_greedy) -- comportement deterministe identique a l'inference Godot
reelle (game/whist_ai_policy.gd applique argmax, jamais d'echantillonnage).

Contrairement a scripts/train.py:build_game (qui utilise un niveau
PLACE-HOLDER, la vraie valeur de score n'important pas en Phase 1), ce
script utilise le NIVEAU REEL choisi (ou un defaut minimal) pour
Solitaire/Emballage : la partie enregistree est destinee a etre regardee,
son bareme de points doit donc etre reel (`engine/scoring.py`, deja
verifie contre Godot), pas un proxy.

Format de sortie : un objet {"format_version":1,"generated_by":"whist-ai",
"generated_at":..., "hands":[<deal>,...]}, un <deal> par partie enregistree
(--games N pour en enregistrer plusieurs a la suite dans le meme fichier,
score cumule d'une donne a l'autre comme une vraie partie). Cartes ecrites
comme l'entier brut `suit*13+rank` de `engine/cards.py` (rank 0=2..12=As) --
accepte tel quel par `WhistReplayFormat.card_from_json()` cote Godot (meme
convention que `game/card.gd:ai_id()`, confirme par la session whist-4-it),
aucune conversion necessaire. Un seul evenement "contract_locked" par donne
(pas de sequence "bid" simulee : Phase 1 n'a pas d'enchere, voir DESIGN.md).

Usage :
  venv/Scripts/python.exe scripts/record_game.py --contract solitaire --level 6
  venv/Scripts/python.exe scripts/record_game.py --contract emballage --declarer 0 --partner 2 --trump coeur
  venv/Scripts/python.exe scripts/record_game.py --contract petite_misere --declarers 0,2 --games 5
"""

import argparse
import json
import os
import random
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, ".")

from agents.play_guards import ai_play_mask
from agents import specialists
from agents.specialists import PHASE1_ROLES
from engine.bidding import (
    EMBALLAGE_MAX_LEVEL, EMBALLAGE_MIN_LEVEL,
    SOLO_MAX_LEVEL, SOLO_MIN_LEVEL, first_leader,
)
from engine.cards import deal
from scripts.godot_paths import find_godot_project
from engine.emballage import new_emballage_deal
from engine.emballage import final_scores as emballage_final_scores
from engine.misere import new_misere_deal
from engine.misere import final_scores as misere_final_scores
from engine.solitaire import new_solitaire_deal
from engine.solitaire import final_scores as solitaire_final_scores

DEFENSEUR = "defenseur"
MISERE_KIND_BY_FAMILY = {
    "piccolo": "piccolo", "petite_misere": "petite",
    "grande_misere": "grande", "misere_etalee": "etalee",
}
FAMILIES = ["solitaire", "emballage"] + list(MISERE_KIND_BY_FAMILY)

# Enregistrement automatique "un adversaire par famille" (scripts/
# record_training_samples.py, et le hook periodique de scripts/train.py) :
# dossier ou envoyer directement le resultat DANS le projet whist-4-it --
# demande explicite de l'utilisateur (2026-09-19), contrairement a la
# convention plus prudente de scripts/export_to_godot.py (copie locale,
# jamais ecrite directement dans whist-4-it). Jamais tests/fixtures/replay/ :
# ce dossier appartient aux fixtures de regression COMMITEES de whist-4-it,
# pas a des exports whist-ai. Un fichier par famille, nom FIXE (voir
# sample_filename()) : un nouvel envoi remplace l'ancien sans toucher aux
# autres fichiers du dossier.
# Projet detecte automatiquement (voir scripts/godot_sync.py:find_godot_project),
# None si introuvable (headless) : rien n'est envoye.
_GODOT_PROJECT = find_godot_project()
DEFAULT_GODOT_RECORDINGS_DIR = os.path.join(_GODOT_PROJECT, "recordings") if _GODOT_PROJECT else None

# Namespace "tout par defaut" reutilise par build_family_sample_games() --
# build_deal() sait deja tirer un declarant/partenaire/atout aleatoire et un
# niveau plancher quand ces champs sont None (voir CLI plus bas), inutile de
# dupliquer cette logique pour l'usage programmatique (scripts/train.py).
NO_OVERRIDES = SimpleNamespace(declarer=None, partner=None, declarers=None, trump=None, level=None)

# Meme ordre que WhistCard.SUITS cote whist-4-it (verifie avec la session
# whist-4-it) -- coincide avec l'index de couleur `engine/cards.py`
# (0=Coeur..3=Pique) utilise partout ailleurs dans ce projet.
SUIT_SYMBOLS = ["♥", "♦", "♣", "♠"]  # Coeur, Carreau, Trefle, Pique
SUIT_NAMES = {
    "coeur": 0, "coeurs": 0, "heart": 0, "hearts": 0, SUIT_SYMBOLS[0]: 0,
    "carreau": 1, "carreaux": 1, "diamond": 1, "diamonds": 1, SUIT_SYMBOLS[1]: 1,
    "trefle": 2, "trefles": 2, "club": 2, "clubs": 2, SUIT_SYMBOLS[2]: 2,
    "pique": 3, "piques": 3, "spade": 3, "spades": 3, SUIT_SYMBOLS[3]: 3,
}


def parse_suit(text):
    key = text.strip().lower()
    if key in SUIT_NAMES:
        return SUIT_NAMES[key]
    if key.isdigit() and 0 <= int(key) <= 3:
        return int(key)
    raise argparse.ArgumentTypeError(
        "couleur inconnue: %r (coeur/carreau/trefle/pique ou 0-3)" % text)


def parse_seats(text):
    seats = [int(s) for s in text.split(",")]
    if not seats or any(s < 0 or s > 3 for s in seats) or len(set(seats)) != len(seats):
        raise argparse.ArgumentTypeError("sieges invalides: %r (0-3, uniques)" % text)
    return seats


class GreedyPolicy:
    """Meme convention que scripts/eval_vs_random.py -- politique GREEDY
    (agents/network.py:act_greedy), comportement identique a l'inference
    Godot reelle, jamais l'echantillonnage utilise en entrainement."""

    def __init__(self, net):
        self.net = net

    def act(self, obs, mask):
        return self.net.act_greedy(obs, mask)


def load_all_nets():
    """Les 7 roles Phase 1 depuis checkpoints_v2/, enveloppes en politique
    GREEDY. Charger les 7 (plutot que les 2 reellement en jeu pour un seul
    contrat) est volontairement simple -- ces reseaux sont petits (voir
    agents/network.py, MLP 256) et build_family_sample_games() en a de toute
    facon besoin pour les 6 familles a la suite."""
    nets = {}
    for role in PHASE1_ROLES:
        net, _optimizer = specialists.load_or_create(role)
        net.eval()
        nets[role] = GreedyPolicy(net)
    return nets


def build_deal(family, hands, dealer, args, rng):
    """Construit la donne + l'evenement contract_locked correspondant, sans
    passer par l'enchere (contrat directement impose -- voir DESIGN.md,
    Phase 1). Retourne (game, contract_event, score_fn) ou score_fn(game)
    donne le VRAI bareme de points (jamais un proxy en plis)."""
    if family == "solitaire":
        if args.partner is not None or args.declarers is not None:
            raise SystemExit("--partner/--declarers ne s'appliquent pas a solitaire")
        declarer = args.declarer if args.declarer is not None else rng.randrange(4)
        trump = args.trump if args.trump is not None else rng.randrange(4)
        level = args.level if args.level is not None else SOLO_MIN_LEVEL
        if not (SOLO_MIN_LEVEL <= level <= SOLO_MAX_LEVEL):
            raise SystemExit("--level solitaire doit etre entre %d et %d" % (SOLO_MIN_LEVEL, SOLO_MAX_LEVEL))
        contract = {"type": "solo", "suit": SUIT_SYMBOLS[trump], "level": level, "declarer": declarer}
        leader = first_leader(dealer, {"type": "solo", "level": level, "declarer": declarer})
        game = new_solitaire_deal(hands, declarer, trump, level, leader)
        return game, contract, (lambda g: solitaire_final_scores(g, level))

    if family == "emballage":
        if args.declarers is not None:
            raise SystemExit("--declarers ne s'applique pas a emballage (utiliser --declarer/--partner)")
        declarer = args.declarer if args.declarer is not None else rng.randrange(4)
        remaining = [s for s in range(4) if s != declarer]
        partner = args.partner if args.partner is not None else rng.choice(remaining)
        if partner == declarer:
            raise SystemExit("--partner doit differer de --declarer")
        trump = args.trump if args.trump is not None else rng.randrange(4)
        level = args.level if args.level is not None else EMBALLAGE_MIN_LEVEL
        if not (EMBALLAGE_MIN_LEVEL <= level <= EMBALLAGE_MAX_LEVEL):
            raise SystemExit("--level emballage doit etre entre %d et %d" % (EMBALLAGE_MIN_LEVEL, EMBALLAGE_MAX_LEVEL))
        contract = {"type": "emballage", "suit": SUIT_SYMBOLS[trump], "level": level,
                    "declarer": declarer, "partner": partner}
        leader = (dealer + 1) % 4
        game = new_emballage_deal(hands, declarer, partner, trump, level, leader)
        return game, contract, emballage_final_scores

    # familles misere : piccolo/petite_misere/grande_misere/misere_etalee
    if args.partner is not None:
        raise SystemExit("--partner ne s'applique pas aux familles misere")
    kind = MISERE_KIND_BY_FAMILY[family]
    if args.declarers is not None:
        declarers = args.declarers
    elif args.declarer is not None:
        declarers = [args.declarer]
    else:
        declarers = [rng.randrange(4)]
    contract = {"type": "misere", "kind": kind, "declarers": declarers}
    leader = (dealer + 1) % 4
    game = new_misere_deal(hands, declarers, kind, leader)
    return game, contract, (lambda g: misere_final_scores(g, kind))


def play_and_record(game, contract_event, nets):
    """Rejoue la donne jusqu'au bout (politique GREEDY partout) en
    enregistrant chaque evenement dans l'ordre reel -- ecart eventuel
    (petite misere), carte jouee, pli gagne. `game.turn` vaut deja le
    gagnant juste apres qu'un pli se complete (voir engine/lone_declarer.py/
    team_trump.py:step), reutilise tel quel plutot que recalcule."""
    events = [{"type": "contract_locked", "contract": contract_event}]
    while not game.done:
        seat = game.turn
        is_discard = bool(getattr(game, "pending_discards", None))
        obs = torch.tensor(game.observation(seat), dtype=torch.float32)
        mask = torch.tensor(ai_play_mask(game, seat), dtype=torch.bool)
        role = game.role_of(seat)
        action = nets[role].act(obs, mask)
        trick_completes = (not is_discard) and len(game.current_trick) == 3
        game.step(seat, action)
        if is_discard:
            events.append({"type": "discard", "seat": seat, "card": action})
        else:
            events.append({"type": "card_played", "seat": seat, "card": action})
            if trick_completes:
                events.append({"type": "trick_won", "seat": game.turn})
    return events


def record_hand(family, args, nets, rng, dealer, cumulative_totals):
    hands = deal(rng)
    game, contract_event, score_fn = build_deal(family, hands, dealer, args, rng)
    # Meme contexte public que la Phase 1 (aucune enchere, niveau jamais montre au reseau).
    game.set_public_context(dealer=dealer, bid_summary=None, whisted=False, level_known=False)
    events = play_and_record(game, contract_event, nets)
    deltas = [int(round(v)) for v in score_fn(game)]
    for seat in range(4):
        cumulative_totals[seat] += deltas[seat]
    events.append({
        "type": "hand_finished",
        "tricks_won": list(game.tricks_won),
        "deltas": deltas,
        "total_score": list(cumulative_totals),
    })
    return {"dealer": dealer, "deal": hands, "events": events}, deltas


SAMPLE_HANDS_PER_FAMILY = 4  # "une partie de 4 donnes" -- demande explicite utilisateur (2026-09-19)


def sample_filename(family):
    """Nom de fichier FIXE par famille (jamais horodate) -- demande
    explicite de l'utilisateur : renvoyer le meme nom a chaque enregistrement
    remplace l'ancienne partie de cette famille sur whist-4-it, SANS toucher
    aux autres fichiers deja presents dans le dossier de destination."""
    return "whist_ai_training_sample_%s.json" % family


def build_family_sample_games(nets, rng, generated_by="whist-ai-training", hands_per_game=SAMPLE_HANDS_PER_FAMILY):
    """Retourne {family: <fichier replay>} -- UNE partie INDEPENDANTE par
    famille de declarant (agents/specialists.py:PHASE1_ROLES), chacune de
    `hands_per_game` donnes (donneur different a chaque donne, score cumule
    SEULEMENT a l'interieur de la partie de cette famille, jamais entre
    familles) -- demande explicite de l'utilisateur (2026-09-19) : "une
    partie de 4 donnes pour enregistrer chacun des differents
    entrainements". Contrat minimal de la famille (niveau plancher, sieges
    aleatoires) via NO_OVERRIDES a chaque donne. `nets` doit couvrir les 7
    roles (voir load_all_nets() pour un usage autonome depuis
    checkpoints_v2/, ou les nets d'entrainement en cours -- voir
    scripts/train.py)."""
    games = {}
    for family in FAMILIES:
        data = {
            "format_version": 1,
            "generated_by": generated_by,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "hands": [],
        }
        cumulative_totals = [0, 0, 0, 0]
        for _ in range(hands_per_game):
            dealer = rng.randrange(4)
            hand_record, _deltas = record_hand(family, NO_OVERRIDES, nets, rng, dealer, cumulative_totals)
            data["hands"].append(hand_record)
        games[family] = data
    return games


def send_family_samples_to_godot(games_by_family, godot_dir=DEFAULT_GODOT_RECORDINGS_DIR):
    """Ecrit DIRECTEMENT dans le projet whist-4-it (jamais dans
    tests/fixtures/, voir DEFAULT_GODOT_RECORDINGS_DIR), un fichier par
    famille sous son nom FIXE (sample_filename()) -- remplace l'ancienne
    partie de whist-ai pour cette famille si elle existe, ne touche jamais
    aux AUTRES fichiers du dossier (demande explicite de l'utilisateur).
    Retourne {family: chemin_ecrit | None} -- jamais fatal (None sur echec
    d'ecriture) pour ne jamais faire planter un entrainement en cours pour
    un probleme d'export cosmetique."""
    if godot_dir is None:
        return {family: None for family in games_by_family}
    try:
        os.makedirs(godot_dir, exist_ok=True)
    except OSError:
        return {family: None for family in games_by_family}
    results = {}
    for family, data in games_by_family.items():
        path = os.path.join(godot_dir, sample_filename(family))
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            results[family] = path
        except OSError:
            results[family] = None
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contract", required=True, choices=FAMILIES, help="contrat a imposer (pas d'enchere)")
    parser.add_argument("--declarer", type=int, default=None, help="siege declarant (0-3), aleatoire si omis")
    parser.add_argument("--partner", type=int, default=None, help="siege partenaire (emballage uniquement)")
    parser.add_argument("--declarers", type=parse_seats, default=None,
                         help="sieges declarants separes par des virgules, ex 0,2 (familles misere uniquement)")
    parser.add_argument("--trump", type=parse_suit, default=None,
                         help="atout: coeur/carreau/trefle/pique ou 0-3 (solitaire/emballage uniquement)")
    parser.add_argument("--level", type=int, default=None,
                         help="niveau du contrat (solitaire 6-13, emballage 8-13), minimum de la famille si omis")
    parser.add_argument("--dealer", type=int, default=None, choices=range(4), help="siege donneur (0-3)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--games", type=int, default=1, help="nombre de donnes a enregistrer dans le meme fichier")
    parser.add_argument("--out", default=None, help="chemin du fichier JSON de sortie")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    nets = load_all_nets()

    out_path = args.out or os.path.join(
        "recordings", "%s_%s.json" % (args.contract, time.strftime("%Y%m%d_%H%M%S")))

    data = {
        "format_version": 1,
        "generated_by": "whist-ai",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hands": [],
    }
    cumulative_totals = [0, 0, 0, 0]
    base_dealer = args.dealer
    for i in range(args.games):
        dealer = (base_dealer + i) % 4 if base_dealer is not None else rng.randrange(4)
        hand_record, deltas = record_hand(args.contract, args, nets, rng, dealer, cumulative_totals)
        data["hands"].append(hand_record)
        print("donne %d/%d : donneur=%d, deltas=%s, cumul=%s" % (
            i + 1, args.games, dealer, deltas, cumulative_totals), flush=True)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Enregistrement ecrit dans %s (%d donne(s), format replay whist-4-it v%d)." % (
        out_path, len(data["hands"]), data["format_version"]), flush=True)


if __name__ == "__main__":
    main()
