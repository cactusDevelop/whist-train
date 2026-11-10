"""DEFENSEUR D'EMBALLAGE distille de PIMC (9e conseil 10/11, accord user « go »).

Constat : PIMC (dds sur mondes) bat le nouvel ExpertBot en defense d'emballage
(~+0.6 +- 0.17 pt/siege, runs 42860501612 + 42860513702) mais les regles
simples qui visent ces ecarts perdent nettement (A/B 24-25, z -5 a -15) : la
bonne carte depend du contexte. On apprend donc un reseau a imiter PIMC, branche
en jeu comme les miseres (routage emballage:def), ExpertBot partout ailleurs.

Trois etapes (cloud seulement) :
  gen     : donnes a encheres ExpertBot ; en emballage, a chaque decision de
            DEFENSE (>= 2 cartes legales, dds.supported) on enregistre
            (observation, cartes legales, Q PIMC de chaque carte, carte ExpertBot).
            Les defenseurs suivent ExpertBot, ou l'eleve (--student, DAgger)
            avec la probabilite --student-prob par donne.
  train   : depart = reseau defenseur de best.pt (IA libre) ; perte = entropie
            croisee vers softmax(Q / tau) sur les cartes legales. Rapport
            hors echantillon : accord avec l'argmax PIMC, et REGRET moyen
            (Q max - Q de la carte choisie) de l'eleve vs celui d'ExpertBot
            = estimation hors ligne du gain par decision.
  measure : duplicate, memes donnes : base = 4 ExpertBot ; variante = les 2
            defenseurs d'emballage jouent l'eleve (argmax, cartes legales,
            comme en jeu). Gain par siege defenseur, par profil adverse
            (--others, expert_bot.PROFILES) ; aussi en contribution par donne.

Critere pre-enregistre (conseil) : gain > 3 SE vs ExpertBot ET meme signe sur
les profils, aucune perte > 3 SE, puis confirmation sur graines neuves avant
tout export Godot.

Usage : python -u scripts/distill_def_pimc.py gen --minutes 150 --workers 4 --out data_0.pt
        python -u scripts/distill_def_pimc.py train --data data_*.pt --out student.pt
        python -u scripts/distill_def_pimc.py measure --student student.pt --deals 100000
"""

import argparse
import glob
import math
import multiprocessing as mp
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from agents import expert_bot, specialists
from agents.network import PolicyValueNet
from agents.rollouts import dds_values
from engine import dds
from engine.cards import NUM_PLAYERS
from engine.whisted import DeckChain
from scripts.expert_bot_match import run_bot_bidding
from scripts.pimc_vs_expert import new_game, sit_key
from scripts.train import MEASURE_WHISTED_RATIO, declarer_role_and_seats, real_points_by_seat

ROLE = "defenseur"
BEST = "checkpoints_free/best.pt"
TARGET_TYPES = ("emballage",)


def load_net(path, key=ROLE):
    net = PolicyValueNet(*specialists.IMPLEMENTED[ROLE])
    state = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(state[key] if key in state else state)
    net.eval()
    return net


def student_card(net, game, seat):
    obs = torch.tensor(game.observation(seat), dtype=torch.float32)
    return net.act_greedy(obs, torch.tensor(game.legal_action_mask(seat), dtype=torch.bool))


# ---------------------------------------------------------------- gen

def gen_worker(args):
    seed, samples, deadline, student_path, student_prob, others = args
    torch.set_num_threads(1)
    expert_bot.set_bid_params(dict(expert_bot.PROFILES[others]))
    net = load_net(student_path) if student_path else None
    rng = random.Random(seed)
    play_rng = random.Random(seed + 1)
    chain = DeckChain(MEASURE_WHISTED_RATIO, rng)
    rows = []  # (obs, mask, q[52], expert_card, sit)
    deals = 0
    while time.time() < deadline:
        hands, dealer, whisted = chain.deal(rng)
        bidding = run_bot_bidding(hands, dealer)
        if bidding.general_pass:
            chain.finish_unplayed(hands, dealer)
            continue
        contract = bidding.locked_contract
        role, _decl = declarer_role_and_seats(contract)
        game = new_game(contract, hands, dealer, bidding, whisted)
        target = contract["type"] in TARGET_TYPES and game.trump_suit is not None and dds.supported(game)
        defense = set(range(NUM_PLAYERS)) - set(dds.side_seats(game)) if target else set()
        use_student = net is not None and play_rng.random() < student_prob

        def score_fn(g):
            return real_points_by_seat(role, g, contract)
        with torch.no_grad():
            while not game.done:
                seat = game.turn
                card = expert_bot.choose_card(game, seat)
                if seat in defense and time.time() < deadline:
                    legal = [c for c, ok in enumerate(game.legal_action_mask(seat)) if ok]
                    if len(legal) > 1 and dds.supported(game):
                        q = dds_values(game, seat, legal, score_fn, play_rng, samples)
                        if q is not None:
                            qq = [0.0] * 52
                            for c, v in zip(legal, q):
                                qq[c] = v
                            rows.append((list(game.observation(seat)), list(game.legal_action_mask(seat)), qq,
                                         card, sit_key(game, seat, card)))
                    if use_student:
                        card = student_card(net, game, seat)
                game.step(seat, card)
        chain.finish_played(game, dealer, contract["type"])
        deals += 1
    return rows, deals


def cmd_gen(a):
    t0 = time.time()
    deadline = t0 + a.minutes * 60
    with mp.Pool(a.workers) as pool:
        parts = pool.map(gen_worker, [(a.seed * 1000 + w, a.samples, deadline, a.student, a.student_prob, a.others)
                                      for w in range(a.workers)])
    rows = [r for p, _ in parts for r in p]
    deals = sum(d for _, d in parts)
    data = {"obs": torch.tensor([r[0] for r in rows], dtype=torch.float32),
            "mask": torch.tensor([r[1] for r in rows], dtype=torch.bool),
            "q": torch.tensor([r[2] for r in rows], dtype=torch.float32),
            "expert": torch.tensor([r[3] for r in rows], dtype=torch.long),
            "sit": [r[4] for r in rows],
            "meta": {"samples": a.samples, "seed": a.seed, "student": a.student, "student_prob": a.student_prob,
                     "others": a.others, "deals": deals}}
    torch.save(data, a.out)
    print("GEN %d decisions de defense (%s) sur %d donnes en %.0f s -> %s" % (
        len(rows), ",".join(TARGET_TYPES), deals, time.time() - t0, a.out), flush=True)


# ---------------------------------------------------------------- train

def load_data(patterns):
    files = sorted({f for p in patterns for f in glob.glob(p)})
    parts = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    parts = [p for p in parts if len(p["sit"])]
    out = {k: torch.cat([p[k] for p in parts]) for k in ("obs", "mask", "q", "expert")}
    out["sit"] = [s for p in parts for s in p["sit"]]
    print("DATA %d decisions, %d fichiers" % (len(out["sit"]), len(files)), flush=True)
    return out


def masked_q(d):
    return d["q"].masked_fill(~d["mask"], -1e9)


def regret(d, choice):
    q = masked_q(d)
    return (q.max(1).values - q.gather(1, choice.unsqueeze(1)).squeeze(1))


def report(net, d, label):
    with torch.no_grad():
        logits, _ = net(d["obs"])
        pick = logits.masked_fill(~d["mask"], -1e9).argmax(1)
    q = masked_q(d)
    best = q.max(1).values
    r_net, r_exp = regret(d, pick), regret(d, d["expert"])
    n = len(r_net)
    diff = r_exp - r_net  # > 0 : l'eleve perd moins de Q que l'ExpertBot
    se = diff.std().item() / math.sqrt(max(1, n))
    agree = (q.gather(1, pick.unsqueeze(1)).squeeze(1) >= best - 1e-6).float().mean().item()
    agree_e = (q.gather(1, d["expert"].unsqueeze(1)).squeeze(1) >= best - 1e-6).float().mean().item()
    print("%-6s n %6d | carte PIMC-optimale : eleve %.3f expert %.3f | regret/decision : eleve %.3f expert %.3f"
          " | gain estime %+.3f +- %.3f pts/decision" % (label, n, agree, agree_e, r_net.mean().item(),
                                                         r_exp.mean().item(), diff.mean().item(), se), flush=True)
    return diff.mean().item()


def cmd_train(a):
    torch.manual_seed(a.seed)
    d = load_data(a.data)
    n = len(d["sit"])
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(a.seed))
    nv = max(1, int(n * a.val))
    vi, ti = perm[:nv], perm[nv:]
    sub = lambda idx: {**{k: d[k][idx] for k in ("obs", "mask", "q", "expert")}}
    val, tr = sub(vi), sub(ti)
    net = load_net(a.init)
    report(net, val, "init")
    target = F.softmax(masked_q(tr) / a.tau, dim=1)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    best_gain, best_state = report(net, val, "ep 0"), {k: v.clone() for k, v in net.state_dict().items()}
    for ep in range(1, a.epochs + 1):
        net.train()
        order = torch.randperm(len(ti))
        for i in range(0, len(order), a.batch):
            b = order[i:i + a.batch]
            logits, _ = net(tr["obs"][b])
            logp = F.log_softmax(logits.masked_fill(~tr["mask"][b], -1e9), dim=1)
            loss = -(target[b] * logp).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        g = report(net, val, "ep %d" % ep)
        if g > best_gain:
            best_gain, best_state = g, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    torch.save({ROLE: net.state_dict(), "meta": {"tau": a.tau, "lr": a.lr, "epochs": a.epochs, "n": n,
                                                  "val_gain": best_gain}}, a.out)
    print("TRAIN -> %s (meilleur gain hors echantillon %+.3f pts/decision)" % (a.out, best_gain), flush=True)
    # par situation (hors echantillon) : ou l'eleve gagne / perd du Q face a l'ExpertBot
    with torch.no_grad():
        logits, _ = net(val["obs"])
        pick = logits.masked_fill(~val["mask"], -1e9).argmax(1)
    diff = regret(val, val["expert"]) - regret(val, pick)
    sits = [d["sit"][i] for i in vi.tolist()]
    agg = {}
    for s, x in zip(sits, diff.tolist()):
        agg.setdefault(s, []).append(x)
    print("=== hors echantillon, par situation (gain de Q eleve - expert, pts/decision) ===")
    for s in sorted(agg, key=lambda k: -abs(sum(agg[k])))[:25]:
        xs = agg[s]
        m = sum(xs) / len(xs)
        print("%-62s n %5d %+.3f cumul %+.1f" % (s, len(xs), m, sum(xs)))


# ---------------------------------------------------------------- measure

def measure_worker(args):
    seed, deals, student_path, others = args
    torch.set_num_threads(1)
    expert_bot.set_bid_params(dict(expert_bot.PROFILES[others]))
    net = load_net(student_path)
    rng = random.Random(seed)
    chain = DeckChain(MEASURE_WHISTED_RATIO, rng)
    gains, played = [], 0
    with torch.no_grad():
        for _ in range(deals):
            hands, dealer, whisted = chain.deal(rng)
            bidding = run_bot_bidding(hands, dealer)
            if bidding.general_pass:
                chain.finish_unplayed(hands, dealer)
                continue
            played += 1
            contract = bidding.locked_contract
            role, _decl = declarer_role_and_seats(contract)
            base = new_game(contract, hands, dealer, bidding, whisted)
            while not base.done:
                base.step(base.turn, expert_bot.choose_card(base, base.turn))
            chain.finish_played(base, dealer, contract["type"])
            if contract["type"] not in TARGET_TYPES:
                continue
            defense = set(range(NUM_PLAYERS)) - set(dds.side_seats(base))
            g = new_game(contract, hands, dealer, bidding, whisted)
            while not g.done:
                s = g.turn
                g.step(s, student_card(net, g, s) if s in defense else expert_bot.choose_card(g, s))
            pb, pv = real_points_by_seat(role, base, contract), real_points_by_seat(role, g, contract)
            gains.append(sum(pv[s] - pb[s] for s in defense) / len(defense))
    return gains, played


def cmd_measure(a):
    for others in a.others.split(","):
        t0 = time.time()
        with mp.Pool(a.workers) as pool:
            parts = pool.map(measure_worker, [(a.seed * 1000 + w, a.deals // a.workers, a.student, others)
                                              for w in range(a.workers)])
        gains = [x for p, _ in parts for x in p]
        played = sum(n for _, n in parts)
        n = len(gains)
        m = sum(gains) / max(1, n)
        se = math.sqrt(sum((x - m) ** 2 for x in gains) / max(1, n - 1) / max(1, n))
        changed = sum(1 for x in gains if abs(x) > 1e-9) / max(1, n)
        print("MEASURE %-16s emballage def : n %6d gain %+.3f +- %.3f pts/siege (z %+.1f), donnes changees %.1f%% ;"
              " contribution %+.4f pts/donne/siege defenseur (%d donnes jouees) %.0f s" % (
                  others, n, m, se, m / se if se else 0.0, 100 * changed, m * n / max(1, played), played,
                  time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen")
    g.add_argument("--minutes", type=float, default=60)
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--samples", type=int, default=8)
    g.add_argument("--seed", type=int, default=20261110)
    g.add_argument("--student", default="")
    g.add_argument("--student-prob", type=float, default=0.5)
    g.add_argument("--others", default="expert")
    g.add_argument("--out", required=True)
    t = sp.add_parser("train")
    t.add_argument("--data", nargs="+", required=True)
    t.add_argument("--init", default=BEST)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--tau", type=float, default=1.0, help="temperature en points")
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--val", type=float, default=0.15)
    t.add_argument("--seed", type=int, default=0)
    m = sp.add_parser("measure")
    m.add_argument("--student", required=True)
    m.add_argument("--deals", type=int, default=100000)
    m.add_argument("--workers", type=int, default=4)
    m.add_argument("--seed", type=int, default=31261110)
    m.add_argument("--others", default="expert")
    a = ap.parse_args()
    {"gen": cmd_gen, "train": cmd_train, "measure": cmd_measure}[a.cmd](a)


if __name__ == "__main__":
    main()
