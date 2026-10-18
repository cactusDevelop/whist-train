"""Evaluation contre RandomBot, CONTRATS CHOISIS PAR L'ENCHERISSEUR
(demande explicite de l'utilisateur, 2026-09-24 : "il faut bien sur
laisser encherisseur decider de chaque contrat a jouer pour tester ceux-ci
contre randombot").

Pour chaque donne : une vraie enchere (engine/bidding.py) ou les 4 sieges
sont le meilleur individu de la lignee `encherisseur` (celui exporte vers
Godot, voir agents/evolution.py:best_per_lineage), greedy, sur une suite de
donnes 50% whistees comme en jeu reel. Passe generale -> donne suivante.
Le contrat verrouille est ensuite joue DEUX fois sur les memes mains :
  - test DECLARANT : le camp declarant joue avec ses specialistes (greedy,
    comme dans Godot), les autres sieges sont des RandomBot (une carte
    legale au hasard) ;
  - test DEFENSE : `defenseur` (greedy) sur les sieges de defense, le camp
    declarant est joue par des RandomBot.
Reussite = le vrai critere du contrat (plis du camp >= niveau pour
solo/emballage/trou ; exactement la cible pour chaque declarant de misere --
chaque declarant compte pour un essai) ; pour la defense, reussite = le
contrat echoue.

Rapport par contrat reel (solo-6, emballage-9, misere-piccolo, trou...),
seuil d'alerte FLAG_THRESHOLD (95%, "perd a plus de 5%").

Mesure CONTINUE en plus du taux de reussite (2026-09-25, demande utilisateur
"c'est le seul indicateur si l'ia devient reellement meilleure au jeu de la
carte") : l'encherisseur choisit surtout des contrats hors de portee
(solo-11/12/13) dont la reussite reste a 0% quoi qu'il arrive -- un taux
binaire ne peut alors montrer aucun progres. `ecart` = distance au critere
de reussite, par essai : plis manquants pour solo/emballage/trou
(max(0, niveau - plis du camp)), |plis - cible| pour chaque declarant de
misere. 0 = contrat rempli. C'est le juge du test DECLARANT (plus bas =
mieux ; sa recompense d'entrainement est en plis, donc alignee). Pas celui
du test DEFENSE : `defenseur` est entraine au bareme de points, et pour
abondance/solo chelem/misere prendre des plis au-dela de l'echec ne
rapporte rien -- sa tendance reste jugee au taux de reussite (l'ecart y est
seulement affiche). Chaque rapport compare a l'evaluation precedente du
meme fichier.

Graine FIXE (EVAL_SEED) quand train.py evalue : memes donnes et memes
tirages RandomBot a chaque evaluation, donc deux evaluations successives ne
different que par les reseaux (et l'encherisseur s'il a evolue).

Usage autonome : venv/Scripts/python.exe scripts/eval_vs_random.py [--contracts N] [--seed N]
"""

import argparse
import json
import sys
import time

import torch

sys.path.insert(0, ".")

from agents.play_guards import ai_play_mask
from agents import evolution, specialists
from agents.specialists import PLAY_ROLES
from engine.scoring import MISERE_MAX_TRICKS
from engine.whisted import DeckChain

EVAL_GAMES_DEFAULT = 1000  # contrats joues (chacun en test declarant ET defense)
# ~25 min d'entrainement Phase 1 (~14 donnes/s) pour ~80 s d'evaluation.
# Avant le 2026-09-25 : 1 000 000, jamais atteint par une nuit (~300k).
EVAL_EVERY_DEFAULT = 20_000
EVAL_SEED = 20260925
EVAL_FILE_DEFAULT = "eval_vs_random.jsonl"
FLAG_THRESHOLD = 0.95  # "perd a plus de 5%" -- demande explicite utilisateur
EVAL_BIDDER_LINEAGE = "encherisseur"


class RandomBot:
    """Bot de reference "qui ne reflechit pas" : une carte legale au hasard."""

    def __init__(self, rng):
        self.rng = rng

    def act(self, obs, mask):
        legal = mask.nonzero(as_tuple=True)[0].tolist()
        return self.rng.choice(legal), None, None


class GreedyPolicy:
    """Reseau entraine en politique deterministe (argmax, comme Godot)."""

    def __init__(self, net):
        self.net = net

    def act(self, obs, mask):
        return self.net.act_greedy(obs, mask), None, None


def play_out(game, seat_policies):
    while not game.done:
        seat = game.turn
        obs = torch.tensor(game.observation(seat), dtype=torch.float32)
        mask = torch.tensor(ai_play_mask(game, seat), dtype=torch.bool)
        action, _logp, _value = seat_policies[seat].act(obs, mask)
        game.step(seat, action)
    return game


def declarer_successes(contract, game, declarer_seats):
    """-> liste de booleens, un par essai (1 pour un camp, 1 par declarant
    de misere)."""
    if contract["type"] == "misere":
        target = MISERE_MAX_TRICKS[contract["kind"]]
        return [game.tricks_won[s] == target for s in sorted(declarer_seats)]
    return [sum(game.tricks_won[s] for s in declarer_seats) >= contract["level"]]


def declarer_gaps(contract, game, declarer_seats):
    """-> un ecart au critere de reussite par essai (meme decoupage que
    declarer_successes) ; 0 = reussi."""
    if contract["type"] == "misere":
        target = MISERE_MAX_TRICKS[contract["kind"]]
        return [abs(game.tricks_won[s] - target) for s in sorted(declarer_seats)]
    return [max(0, contract["level"] - sum(game.tricks_won[s] for s in declarer_seats))]


def eval_bidder(population):
    best = evolution.best_per_lineage(population)
    if EVAL_BIDDER_LINEAGE not in best:
        raise SystemExit("lignee %s absente de la population" % EVAL_BIDDER_LINEAGE)
    return best[EVAL_BIDDER_LINEAGE]


def run_eval(nets, rng, num_contracts=EVAL_GAMES_DEFAULT, population=None):
    """Retourne (records, meta) -- un record par (test, contrat reel)."""
    # Import differe : scripts.train importe ce module au demarrage.
    from scripts.train import (MEASURE_WHISTED_RATIO, declarer_role_and_seats, describe_contract,
                               dispatch_play_phase2, run_bidding_phase)
    if population is None:
        population, _generation = evolution.load_population(
            __import__("scripts.train", fromlist=["POPULATION_FILE_DEFAULT"]).POPULATION_FILE_DEFAULT)
    bidder = eval_bidder(population)
    chain = DeckChain(MEASURE_WHISTED_RATIO, rng)
    tallies = {}  # (test, contrat) -> [reussites, essais, somme des ecarts]
    general_passes = 0
    played = 0
    while played < num_contracts:
        hands, dealer, whisted = chain.deal(rng)
        bidding, _log = run_bidding_phase({s: evolution.policy_of(bidder) for s in range(4)}, hands, dealer)  # garde-fous + abondance, comme en jeu
        if bidding.general_pass:
            general_passes += 1
            chain.finish_unplayed(hands, dealer)
            continue
        contract = bidding.locked_contract
        role, declarer_seats = declarer_role_and_seats(contract)
        key = describe_contract(contract)

        game = dispatch_play_phase2(contract, hands, dealer, bidding.bid_summary, whisted)
        policies = [GreedyPolicy(nets[game.role_of(s)]) if s in declarer_seats else RandomBot(rng) for s in range(4)]
        play_out(game, policies)
        oks = declarer_successes(contract, game, declarer_seats)
        t = tallies.setdefault(("declarant", key), [0, 0, 0])
        t[0] += sum(oks)
        t[1] += len(oks)
        t[2] += sum(declarer_gaps(contract, game, declarer_seats))

        game_def = dispatch_play_phase2(contract, hands, dealer, bidding.bid_summary, whisted)
        policies = [RandomBot(rng) if s in declarer_seats else GreedyPolicy(nets["defenseur"]) for s in range(4)]
        play_out(game_def, policies)
        oks = declarer_successes(contract, game_def, declarer_seats)
        t = tallies.setdefault(("defenseur", key), [0, 0, 0])
        t[0] += sum(1 for ok in oks if not ok)
        t[1] += len(oks)
        t[2] += sum(declarer_gaps(contract, game_def, declarer_seats))

        chain.finish_played(game, dealer, contract["type"])
        played += 1

    records = []
    for (test, key), (ok, n, gap) in sorted(tallies.items()):
        rate = ok / n
        records.append({"test": test, "contract": key, "games": n, "success_rate": round(rate, 4),
                        "gap_mean": round(gap / n, 4), "flagged": rate < FLAG_THRESHOLD})
    meta = {"bidder_lineage": EVAL_BIDDER_LINEAGE, "bidder_id": bidder.id, "contracts": played,
            "general_passes": general_passes}
    return records, meta


def global_gaps(records):
    """-> {test: ecart moyen pondere par le nombre d'essais}."""
    out = {}
    for test in ("declarant", "defenseur"):
        rows = [r for r in records if r["test"] == test and "gap_mean" in r]
        n = sum(r["games"] for r in rows)
        if n:
            out[test] = sum(r["gap_mean"] * r["games"] for r in rows) / n
    return out


def global_rates(records):
    """-> {test: taux de reussite global}."""
    out = {}
    for test in ("declarant", "defenseur"):
        rows = [r for r in records if r["test"] == test]
        n = sum(r["games"] for r in rows)
        if n:
            out[test] = sum(round(r["success_rate"] * r["games"]) for r in rows) / n
    return out


def _trend(now, before, higher_is_better):
    if now == before:
        return "egal"
    return "mieux" if (now > before) == higher_is_better else "moins bien"


def last_eval_record(eval_file):
    try:
        with open(eval_file, encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        return json.loads(lines[-1]) if lines else None
    except (OSError, ValueError):
        return None


def emit_eval_report(records, meta, total_episodes, elapsed, eval_file):
    previous = last_eval_record(eval_file)
    prev_gaps = global_gaps(previous["results"]) if previous else {}
    prev_rates = global_rates(previous["results"]) if previous else {}
    gaps = global_gaps(records)
    rates = global_rates(records)
    print("\n=== Evaluation vs RandomBot, contrats choisis par %s #%d (total=%d donnes, %.0fs) ===" % (
        meta["bidder_lineage"], meta["bidder_id"], total_episodes, elapsed), flush=True)
    print("  %d contrats joues, %d passes generales ignorees" % (meta["contracts"], meta["general_passes"]),
          flush=True)
    for test in ("declarant", "defenseur"):
        rows = [r for r in records if r["test"] == test]
        ok = sum(round(r["success_rate"] * r["games"]) for r in rows)
        n = sum(r["games"] for r in rows)
        if test == "declarant":
            trend = " (precedente %.3f : %s)" % (prev_gaps[test], _trend(gaps[test], prev_gaps[test], False))                 if test in gaps and test in prev_gaps else ""
            print("  -- declarant : %.1f%% global (%d essais), ecart moyen %.3f pli%s  [juge : ecart, plus bas = mieux]"
                  % (100.0 * ok / n if n else 0.0, n, gaps.get(test, 0.0), trend), flush=True)
        else:
            trend = " (precedente %.1f%% : %s)" % (100 * prev_rates[test], _trend(rates[test], prev_rates[test], True))                 if test in rates and test in prev_rates else ""
            print("  -- defenseur : %.1f%% global%s (%d essais), ecart moyen %.3f pli  [juge : reussite]" % (
                100.0 * ok / n if n else 0.0, trend, n, gaps.get(test, 0.0)), flush=True)
        for r in rows:
            marker = "  !!ALERTE!!" if r["flagged"] else ""
            print("     %-18s reussite=%6.1f%%  ecart=%5.2f  (%d)%s" % (
                r["contract"], r["success_rate"] * 100, r["gap_mean"], r["games"], marker), flush=True)
    record = {"total_donnes": total_episodes, "elapsed_s": round(elapsed, 1),
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "meta": meta,
              "gap_mean": {test: round(g, 4) for test, g in gaps.items()}, "results": records}
    with open(eval_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("  (ajoute a %s)\n" % eval_file, flush=True)
    return record


def run_and_report(nets, rng, total_episodes, elapsed, num_games, eval_file, population=None):
    records, meta = run_eval(nets, rng, num_contracts=num_games, population=population)
    return emit_eval_report(records, meta, total_episodes, elapsed, eval_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", type=int, default=EVAL_GAMES_DEFAULT,
                        help="contrats joues (chacun en test declarant ET en test defense)")
    parser.add_argument("--seed", type=int, default=EVAL_SEED,
                        help="defaut EVAL_SEED, la meme que train.py : resultats comparables")
    parser.add_argument("--eval-file", default=EVAL_FILE_DEFAULT)
    args = parser.parse_args()

    import random
    rng = random.Random(args.seed)
    nets = {}
    for role in PLAY_ROLES:
        net, _optimizer = specialists.load_or_create(role)
        nets[role] = net
    start = time.time()
    with torch.no_grad():
        run_and_report(nets, rng, total_episodes=0, elapsed=0.0, num_games=args.contracts, eval_file=args.eval_file)
    print("Termine en %.1fs." % (time.time() - start), flush=True)


if __name__ == "__main__":
    main()
