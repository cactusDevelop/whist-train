"""IA LIBRE du jeu de la carte (demande utilisateur 2026-11-01 : « une IA
completement debridee dans la population »). A cote de scripts/train.py, sur
son propre dossier checkpoints_free/ : ne touche ni aux checkpoints_v2 ni a la
championne.

Ce qui est retire par rapport a train.py : imitation d'ExpertBot, professeur
DDS, rollouts, cibles de garde-fous, coups interdits (masque = cartes legales,
a l'entrainement comme a la mesure), recompense en plis. Reste : PPO, les
vraies regles, les VRAIS POINTS de la donne.

Population : deux niches qui ne se melangent jamais, pour qu'une lignee partie
de zero ne soit pas effacee par celles qui partent de la championne.
  - chaude : part de la championne du jeu de la carte ;
  - zero   : reseaux neufs.
Chaque cycle, chaque membre joue en parallele : moitie contre lui-meme (les 4
sieges apprennent), sinon un camp contre un pair de sa niche ou la championne
(seul ce camp apprend). Puis tous sont mesures contre la championne sur les
memes donnes, et dans chaque niche le dernier est remplace par une copie du
premier, reglages perturbes (lr, entropie), seulement si l'ecart depasse le
bruit. L'encherisseur est fixe (greedy) : on ne juge que la carte.

REFERENCE (correctif 2026-11-02) : la mesure du 02/11 a montre que 30 cycles
de portes "acceptees" contre l'ancienne championne ppo n'avaient rien apporte
(best vs ExpertBot -2.7 aux cycles 23, 41 et 53). Desormais :
  - la reference est best.pt (la meilleure IA libre), pas la championne ppo :
    selection et porte mesurent contre elle, cartes legales des deux cotes ;
  - la porte se joue sur des donnes NEUVES a chaque cycle (la selection sur
    des donnes fixes gonflait les gains), z = 2 par camp ; acceptee -> le
    membre devient best.pt, la barre monte a chaque acceptation ;
  - adversaires d'entrainement hors miroir : ExpertBot, best ou un pair ;
  - fin de bloc : BILAN best vs ExpertBot sur des donnes fixes (celles de
    free_match), le vrai indicateur de progres, ajoute a free_gate.jsonl.

DEFENSE CIBLEE (2026-11-05, accord user) : le BILAN vs ExpertBot plafonne
(-0.57, defense -0.97 = le gros du retard). Option --defense-rollout-prob :
une decision de DEFENSE d'un siege qui apprend, contrat a atout (dds), hors
1er pli, recoit avec cette probabilite une cible de politique = cartes a
--defense-rollout-margin plis pres de la meilleure, Q = vrais points moyens
sur --defense-rollout-samples mondes joues parfaitement (agents/rollouts.py,
comme train.py). Budget : jamais plus de --defense-rollout-share du temps du
membre en dds (sinon la decision passe sans cible). Mesure inchangee.

FOCUS (2026-11-06, accord user, diag scripts/emballage_free_diag.py) : en
emballage, best perdait ses points (defense -3.0) en prenant le pli avec une
carte trop haute et sur ses entames ; le declarant n'y tirait pas atout. Pour
les contrats de --rollout-focus-roles, la cible dds ne vise QUE les entames et
les prises a plusieurs cartes gagnantes, des DEUX camps, avec la probabilite
--rollout-focus-prob. Aucune contrainte de jeu : cartes legales, la cible
montre seulement la meilleure carte au dds.

Usage (cloud seulement) : ci/run_ppo_free.sh, mode ppo_free de train.yml.
"""

import argparse
import json
import multiprocessing
import os
import random
import sys
import time
from datetime import datetime, timezone

import torch

sys.path.insert(0, ".")

from agents import bid_ppo as bid_ppo_mod
from agents import expert_bot
from agents import evolution, ppo, rollouts, specialists
from agents.network import PolicyValueNet, widen_adam_state
from agents.play_guards import ai_play_mask
from agents.specialists import PLAY_ROLES, safe_save
from engine.bidding import Bidding
from engine.cards import trick_winner
from engine.trou import detect_trou
from engine.whisted import DeckChain
from scripts import champion_play
from scripts.duel_expert import _bid
from scripts.train import (MEASURE_WHISTED_RATIO, POINTS_PER_TRICK, WHISTED_RATIO_DEFAULT, compute_scores_real,
                           declarer_role_and_seats, dispatch_play_phase2, dds_supported, real_points_by_seat)

ROOT = "checkpoints_free"
GATE_FILE = os.path.join(ROOT, "free_gate.jsonl")
LOT_DEALS = 400  # donnes entre deux mises a jour PPO
MIN_DECISIONS = 256  # un role rare attend d'en avoir assez
RETURN_SCALE = 20.0  # les valeurs du reseau sont en points / 20
GAE_LAMBDA = 0.95
SELF_PLAY_PROB = 0.5
OPPONENTS = ("expert", "best", "peer")  # hors miroir, tirage uniforme (peer -> best sans pair)
BEST_PATH = os.path.join(ROOT, "best.pt")
GATE_SEED = 91_000_000  # + 1000 * cycle : donnes neuves a chaque porte
BILAN_SEED = 20261101  # = free_match.SEED : bilans comparables d'un bloc a l'autre
START_HP = {"chaude": {"lr": 1e-4, "entropy": 0.02}, "zero": {"lr": 3e-4, "entropy": 0.03}}
SELECTION_Z = 2.0

DEFENSE = {"prob": 0.0, "samples": 8, "margin_tricks": 0.5, "share": 0.5, "roles": None, "focus": set(), "focus_prob": 0.0}  # roles None = tous  # --defense-rollout-*, herite par fork

BIDDER = None  # herite par fork ; voir load_bidder
CHAMPION = None


def member_path(name):
    return os.path.join(ROOT, name + ".pt")


def new_nets():
    return {role: PolicyValueNet(*specialists.IMPLEMENTED[role]) for role in PLAY_ROLES}


def load_nets(state):
    nets = new_nets()
    for role in PLAY_ROLES:
        nets[role].load_state_dict(state[role])
        nets[role].eval()
    return nets


def create_member(name, niche):
    nets = new_nets()
    if niche == "chaude":
        for role in PLAY_ROLES:
            nets[role].load_state_dict(CHAMPION[role].state_dict())
    safe_save({"name": name, "niche": niche, "hp": dict(START_HP[niche]), "deals": 0, "history": [],
               "nets": {r: n.state_dict() for r, n in nets.items()}, "opts": None}, member_path(name))


def load_bidder():
    population, _generation = evolution.load_population(
        os.path.join(specialists.CHECKPOINT_DIR, "encherisseur_population.pt"))
    net, _opt, indiv_id, _seed = bid_ppo_mod.load_or_seed(population, specialists.CHECKPOINT_DIR)
    population = bid_ppo_mod.ppo_population(population, net, indiv_id)
    return evolution.policy_of(evolution.best_per_lineage(population)[evolution.EXPORTED_LINEAGES[0]])


def load_reference():
    """best.pt (meilleure IA libre) ; a defaut, la championne ppo."""
    if os.path.exists(BEST_PATH):
        return load_nets(torch.load(BEST_PATH, map_location="cpu", weights_only=False))
    return CHAMPION


def init_worker():
    torch.set_num_threads(1)


def next_contract(chain, rng):
    """Donne + vraie enchere greedy ; redonne sur passe generale."""
    while True:
        hands, dealer, whisted = chain.deal(rng)
        bidding = Bidding(dealer)
        trou = detect_trou(hands)
        if trou is not None:
            bidding.seed_trou(trou)
        with torch.no_grad():
            _bid(bidding, hands, {0, 1, 2, 3}, BIDDER)
        if not bidding.general_pass:
            return hands, dealer, whisted, bidding
        chain.finish_unplayed(hands, dealer)


def new_buffer():
    return {role: {"obs": [], "mask": [], "action": [], "logp": [], "ret": [], "adv": [], "target": []}
            for role in PLAY_ROLES}


def focus_decision(game, seat):
    """Entame, ou prise avec au moins deux cartes legales gagnantes (voir FOCUS)."""
    trick = game.current_trick
    if not trick:
        return True
    legal = [c for c, ok in enumerate(game.legal_action_mask(seat)) if ok]
    return sum(trick_winner(trick + [(seat, c)], game.trump_suit) == seat for c in legal) >= 2


def defense_target(game, seat, score_fn, rng, stats, t_start, role=None, declarer=False):
    """Cible dds d'une decision de defense (voir DEFENSE CIBLEE) ou None.
    --defense-rollout-roles (06/11) : limite aux contrats listes (role de declarer_role_and_seats).
    Contrats de --rollout-focus-roles : voir FOCUS (les deux camps)."""
    focus = role in DEFENSE["focus"]
    if not focus and (declarer or (DEFENSE["roles"] is not None and role not in DEFENSE["roles"])):
        return None
    prob = DEFENSE["focus_prob"] if focus else DEFENSE["prob"]
    if prob <= 0 or len(game.play_log) < 4 or not dds_supported(game) or rng.random() >= prob:
        return None
    if focus and not focus_decision(game, seat):
        return None
    if stats["seconds"] > DEFENSE["share"] * (time.time() - t_start):
        stats["skipped"] += 1
        return None
    legal = [c for c, ok in enumerate(game.legal_action_mask(seat)) if ok]
    if len(legal) < 2:
        return None
    t0 = time.time()
    q = rollouts.dds_values(game, seat, legal, score_fn, rng, DEFENSE["samples"])
    stats["seconds"] += time.time() - t0
    stats["count"] += 1
    if q is None:
        return None
    target = rollouts.best_set_target(legal, q, DEFENSE["margin_tricks"] * POINTS_PER_TRICK)
    stats["corrective"] += target is not None
    stats["focus"] = stats.get("focus", 0) + focus
    return target


def play_training_deal(chain, rng, seat_nets, learners, buffers, stats=None, t_start=None):
    hands, dealer, whisted, bidding = next_contract(chain, rng)
    contract = bidding.locked_contract
    role, decl = declarer_role_and_seats(contract)
    score_fn = lambda g: compute_scores_real(role, g, decl, contract)
    game = dispatch_play_phase2(contract, [list(h) for h in hands], dealer, bidding.bid_summary, whisted)
    steps = {s: [] for s in learners}
    with torch.no_grad():
        while not game.done:
            seat = game.turn
            who = game.role_of(seat)
            if seat_nets[seat] == "expert":
                game.step(seat, expert_bot.choose_card(game, seat))
                continue
            obs = torch.tensor(game.observation(seat), dtype=torch.float32)
            mask = torch.tensor(game.legal_action_mask(seat), dtype=torch.bool)
            action, logp, value = seat_nets[seat][who].act(obs, mask)
            if seat in steps:
                target = None
                if stats is not None and (seat not in decl or role in DEFENSE["focus"]):
                    target = defense_target(game, seat, score_fn, rng, stats, t_start, role, seat in decl)
                steps[seat].append((who, obs, mask, action, logp, value.item() * RETURN_SCALE, target))
            game.step(seat, action)
    points = real_points_by_seat(role, game, contract)
    chain.finish_played(game, dealer, contract["type"])
    for seat, seq in steps.items():
        gae, next_value = 0.0, None
        for who, obs, mask, action, logp, value, target in reversed(seq):
            # recompense unique en fin de donne : les vrais points du siege
            delta = (points[seat] if next_value is None else next_value) - value
            gae = delta + GAE_LAMBDA * gae
            next_value = value
            buf = buffers[who]
            buf["obs"].append(obs)
            buf["mask"].append(mask)
            buf["action"].append(action)
            buf["logp"].append(logp)
            buf["ret"].append(gae + value)
            buf["adv"].append(gae)
            buf["target"].append(target)
    return points


def learn(nets, opts, buffers, entropy):
    report = {}
    for role in PLAY_ROLES:
        buf = buffers[role]
        if len(buf["obs"]) < MIN_DECISIONS:
            continue
        targets = buf["target"] if any(t is not None for t in buf["target"]) else None
        s = ppo.update(nets[role], opts[role], buf["obs"], buf["mask"], buf["action"], buf["logp"], buf["ret"],
                       buf["adv"], return_scale=RETURN_SCALE, entropy_coef=entropy, targets=targets)
        report[role] = (s["entropy"], s["value_loss"])
        for key in buf:
            buf[key] = []
    return report


def train_member(args):
    """Tache d'un processus : joue et apprend jusqu'a l'echeance, puis sauvegarde."""
    name, peer, seconds, seed = args
    rng = random.Random(seed)
    torch.manual_seed(seed)
    member = torch.load(member_path(name), map_location="cpu", weights_only=False)
    nets = load_nets(member["nets"])
    opts = {r: torch.optim.Adam(n.parameters(), lr=member["hp"]["lr"]) for r, n in nets.items()}
    if member["opts"] is not None:
        for role, state in member["opts"].items():
            opts[role].load_state_dict(state)
            widen_adam_state(opts[role])
            for group in opts[role].param_groups:
                group["lr"] = member["hp"]["lr"]
    peer_nets = load_nets(torch.load(member_path(peer), map_location="cpu", weights_only=False)["nets"]) if peer else None
    best_nets = load_reference()
    chain = DeckChain(WHISTED_RATIO_DEFAULT, rng)
    buffers = new_buffer()
    t0 = time.time()
    deals, updates = 0, {}
    stats = {"count": 0, "corrective": 0, "skipped": 0, "seconds": 0.0, "focus": 0}
    while time.time() - t0 < seconds:
        for _ in range(LOT_DEALS):
            if rng.random() < SELF_PLAY_PROB:
                seat_nets, learners = [nets] * 4, {0, 1, 2, 3}
            else:
                pick = rng.choice(OPPONENTS)
                opp = "expert" if pick == "expert" else (peer_nets if pick == "peer" and peer_nets else best_nets)
                first = rng.randrange(2)  # un camp apprend : 0-2 ou 1-3, les enchainements de sieges tournent
                learners = {first, first + 2}
                seat_nets = [nets if s in learners else opp for s in range(4)]
            play_training_deal(chain, rng, seat_nets, learners, buffers, stats, t0)
            deals += 1
        for net in nets.values():
            net.train()
        updates.update(learn(nets, opts, buffers, member["hp"]["entropy"]))
        for net in nets.values():
            net.eval()
    member["nets"] = {r: n.state_dict() for r, n in nets.items()}
    member["opts"] = {r: o.state_dict() for r, o in opts.items()}
    member["deals"] += deals
    safe_save(member, member_path(name))
    return name, deals, time.time() - t0, updates, stats


def eval_chunk(args):
    """Comme champion_play.camp_rows : base championne, membre au camp
    declarant, membre en defense. Membre = cartes legales ; championne = avec
    ses garde-fous, comme l'IA exportee."""
    source, deals, seed, base_name = args
    state = torch.load(source, map_location="cpu", weights_only=False)
    free = load_nets(state["nets"] if "nets" in state else state)
    base_nets = load_reference() if base_name == "best" else None
    rng = random.Random(seed)
    chain = DeckChain(MEASURE_WHISTED_RATIO, rng)
    rows = {}

    def play(contract, role, hands, dealer, bidding, whisted, free_seats):
        game = dispatch_play_phase2(contract, [list(h) for h in hands], dealer, bidding.bid_summary, whisted)
        while not game.done:
            seat = game.turn
            if seat not in free_seats and base_name == "expert":
                game.step(seat, expert_bot.choose_card(game, seat))
                continue
            obs = torch.tensor(game.observation(seat), dtype=torch.float32)
            if seat in free_seats:
                mask, net = game.legal_action_mask(seat), free[game.role_of(seat)]
            elif base_nets is CHAMPION:
                mask, net = ai_play_mask(game, seat), CHAMPION[game.role_of(seat)]
            else:
                mask, net = game.legal_action_mask(seat), base_nets[game.role_of(seat)]
            game.step(seat, net.act_greedy(obs, torch.tensor(mask, dtype=torch.bool)))
        return real_points_by_seat(role, game, contract)

    with torch.no_grad():
        for _ in range(deals):
            hands, dealer, whisted = chain.deal(rng)
            chain.finish_unplayed(hands, dealer)
            bidding = Bidding(dealer)
            trou = detect_trou(hands)
            if trou is not None:
                bidding.seed_trou(trou)
            _bid(bidding, hands, {0, 1, 2, 3}, BIDDER)
            if bidding.general_pass:
                continue
            contract = bidding.locked_contract
            role, decl = declarer_role_and_seats(contract)
            defense = {s for s in range(4) if s not in decl}
            base = play(contract, role, hands, dealer, bidding, whisted, set())
            with_decl = play(contract, role, hands, dealer, bidding, whisted, set(decl))
            with_def = play(contract, role, hands, dealer, bidding, whisted, defense)
            r = rows.setdefault(role, {"decl": [], "def": []})
            r["decl"].append(sum(with_decl[s] for s in decl) - sum(base[s] for s in decl))
            r["def"].append(sum(with_def[s] for s in defense) - sum(base[s] for s in defense))
    return rows


def measure(pool, source, deals, seed, procs, base="best"):
    """-> mesure combinee sur `procs` paquets de donnes (memes graines = memes donnes).
    base : "best" (best.pt, a defaut la championne ppo) ou "expert" (ExpertBot)."""
    parts = pool.map(eval_chunk, [(source, deals // procs, seed + 10 * i, base) for i in range(procs)])
    rows = {}
    for part in parts:
        for role, r in part.items():
            into = rows.setdefault(role, {"decl": [], "def": []})
            into["decl"] += r["decl"]
            into["def"] += r["def"]
    decl = [x for r in rows.values() for x in r["decl"]]
    defense = [x for r in rows.values() for x in r["def"]]
    m_tot, se_tot = champion_play.mean_se([a + b for a, b in zip(decl, defense)])
    m_decl, se_decl = champion_play.mean_se(decl)
    m_def, se_def = champion_play.mean_se(defense)
    by_role = {role: round(champion_play.mean_se([a + b for a, b in zip(r["decl"], r["def"])])[0], 2)
               for role, r in sorted(rows.items())}
    return {"deals": len(decl), "gain": m_tot, "se": se_tot, "decl": m_decl, "decl_se": se_decl,
            "def": m_def, "def_se": se_def, "by_role": by_role}


def show(r):
    return "%+.2f +- %.2f (declarant %+.2f, defense %+.2f, %d donnes) %s" % (
        r["gain"], SELECTION_Z * r["se"], r["decl"], r["def"], r["deals"], r["by_role"])


def select(niche_scores, rng):
    """Dans une niche : le dernier copie le premier si l'ecart depasse le bruit."""
    ranked = sorted(niche_scores, key=lambda x: -x[1]["gain"])
    (best, b), (worst, w) = ranked[0], ranked[-1]
    if best == worst or b["gain"] - w["gain"] <= SELECTION_Z * (b["se"] ** 2 + w["se"] ** 2) ** 0.5:
        return None
    src = torch.load(member_path(best), map_location="cpu", weights_only=False)
    dst = torch.load(member_path(worst), map_location="cpu", weights_only=False)
    src["name"] = worst
    src["hp"] = {k: v * rng.choice((0.7, 1.4)) for k, v in src["hp"].items()}
    src["history"] = dst["history"]
    safe_save(src, member_path(worst))
    return "%s remplace par une copie de %s, reglages %s" % (worst, best, src["hp"])


def main():
    global BIDDER, CHAMPION
    parser = argparse.ArgumentParser()
    parser.add_argument("--until", required=True, help="heure d'arret UTC, AAAA-MM-JJTHH:MM")
    parser.add_argument("--cycle-seconds", type=int, default=900)
    parser.add_argument("--select-deals", type=int, default=2000)
    parser.add_argument("--gate-deals", type=int, default=10000)
    parser.add_argument("--per-niche", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--defense-rollout-prob", type=float, default=0.0, help="voir DEFENSE CIBLEE")
    parser.add_argument("--defense-rollout-samples", type=int, default=8)
    parser.add_argument("--defense-rollout-margin", type=float, default=0.5, help="marge en plis")
    parser.add_argument("--defense-rollout-share", type=float, default=0.5, help="part max du temps en dds")
    parser.add_argument("--defense-rollout-roles", default="", help="contrats cibles, ex. emballage,solitaire (vide = tous)")
    parser.add_argument("--rollout-focus-roles", default="", help="voir FOCUS, ex. emballage")
    parser.add_argument("--rollout-focus-prob", type=float, default=0.0)
    args = parser.parse_args()
    DEFENSE.update(prob=args.defense_rollout_prob, samples=args.defense_rollout_samples,
                   margin_tricks=args.defense_rollout_margin, share=args.defense_rollout_share,
                   roles=set(args.defense_rollout_roles.split(",")) if args.defense_rollout_roles else None,
                   focus=set(args.rollout_focus_roles.split(",")) if args.rollout_focus_roles else set(),
                   focus_prob=args.rollout_focus_prob)
    until = datetime.strptime(args.until, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc).timestamp()
    os.makedirs(ROOT, exist_ok=True)

    CHAMPION, meta = champion_play.load_champion()
    if CHAMPION is None:
        raise SystemExit("pas de championne dans checkpoints_v2/champion_play")
    BIDDER = load_bidder()
    print("Championne : %s ; encherisseur fixe charge." % meta, flush=True)

    names = [("%s_%d" % (niche, i), niche) for niche in START_HP for i in range(args.per_niche)]
    for name, niche in names:
        if not os.path.exists(member_path(name)):
            create_member(name, niche)
            print("membre cree : %s" % name, flush=True)

    rng = random.Random(args.seed)
    t_start = time.time()
    state_path = os.path.join(ROOT, "free_state.json")
    state = json.load(open(state_path, encoding="utf-8")) if os.path.exists(state_path) else {"cycle": 0, "best_gain": None}
    pool = multiprocessing.get_context("fork").Pool(args.workers, init_worker)
    gate_seconds = 0.0
    while True:
        # dernier cycle : garder de quoi mesurer avant l'heure d'arret
        left = until - time.time()
        seconds = min(args.cycle_seconds, left - gate_seconds * 2.5)  # mesures du cycle + bilan final
        if seconds < min(60, args.cycle_seconds):
            break
        state["cycle"] += 1
        tasks = []
        for name, niche in names:
            peers = [n for n, k in names if k == niche and n != name]
            tasks.append((name, rng.choice(peers) if peers else None, seconds, rng.randrange(1 << 30)))
        t0 = time.time()
        for name, deals, took, updates, rs in pool.map(train_member, tasks, chunksize=1):
            ent = sum(e for e, _v in updates.values()) / max(1, len(updates))
            print("[cycle %d] %-9s %6d donnes en %4.0fs (%.1f/s) entropie moy. %.3f" % (
                state["cycle"], name, deals, took, deals / max(took, 1e-9), ent), flush=True)
            if DEFENSE["prob"] > 0 or DEFENSE["focus_prob"] > 0:
                print("    defense dds : %d decisions (dont %d focus), %d corrigees, %d sautees (budget), %.0f s" % (
                    rs["count"], rs["focus"], rs["corrective"], rs["skipped"], rs["seconds"]), flush=True)

        t0 = time.time()
        scores = {}
        for name, niche in names:
            scores[name] = measure(pool, member_path(name), args.select_deals, 77_000 * state["cycle"], args.workers)
            print("  %-9s vs best : %s" % (name, show(scores[name])), flush=True)
        for niche in START_HP:
            note = select([(n, scores[n]) for n, k in names if k == niche], rng)
            if note:
                print("  selection %s : %s" % (niche, note), flush=True)

        # porte par camp sur le meilleur de toute la population, contre best, donnes neuves
        top = max(scores, key=lambda n: scores[n]["gain"])
        r = measure(pool, member_path(top), args.gate_deals, GATE_SEED + 1000 * state["cycle"], args.workers)
        z = champion_play.GATE_Z
        accepted = (r["gain"] - z * r["se"] > 0 and r["decl"] + z * r["decl_se"] >= 0 and r["def"] + z * r["def_se"] >= 0)
        print("=== Porte %s vs best, cycle %d (donnes neuves) : %s -> %s ===" % (
            top, state["cycle"], show(r), "ACCEPTEE, nouveau best" if accepted else "refusee"), flush=True)
        if accepted:
            safe_save(torch.load(member_path(top), map_location="cpu", weights_only=False)["nets"], BEST_PATH)
            state["best_cycle"] = state["cycle"]
        with open(GATE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"cycle": state["cycle"], "top": top, "accepted": accepted, "gate": r,
                                "scores": scores, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f)
        gate_seconds = time.time() - t0
    # bilan du bloc : best contre ExpertBot, toujours sur les memes donnes
    if os.path.exists(BEST_PATH):
        b = measure(pool, BEST_PATH, args.gate_deals, BILAN_SEED, args.workers, base="expert")
        print("=== BILAN best (cycle %s) a la place d'ExpertBot : %s ===" % (state.get("best_cycle", "?"), show(b)),
              flush=True)
        with open(GATE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"cycle": state["cycle"], "bilan_vs_expert": b, "best_cycle": state.get("best_cycle"),
                                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    pool.close()
    pool.join()
    print("Arret propre apres %d cycle(s), %.0f s." % (state["cycle"], time.time() - t_start), flush=True)


if __name__ == "__main__":
    main()
