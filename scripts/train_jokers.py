"""Entrainement de l'IA jokers v1 (decision de cast des bots elites/boss,
voir engine/jokers.py) par ITERATION DE POLITIQUE A ROLLOUTS :

  a chaque tour de pli d'un bot a jokers, chaque action legale (aucun cast
  ou un joker) est rejouee jusqu'a la fin de la donne depuis une copie de
  l'etat (memes mains, meme graine de pioche), les decisions suivantes
  etant prises par la politique courante ; Q(a) = recompense de la donne
  (engine/joker_match.hand_reward). Le reseau apprend A(a) = Q(a) - Q(0)
  par regression (sortie de politique = avantage estime) ; le jeu prend
  l'argmax masque, comme les autres reseaux v2.

Les decisions futures des rollouts sont prises par le CHAMPION
(checkpoints_jokers/jokers.pt, ce qui s'exporte ; sans champion : regles
seules). Porte (2026-10-21) : chaque iteration, le candidat est joue en
duplicate contre le champion sur des matchs NEUFS ; il ne devient champion
que s'il le bat de plus de GATE_Z erreurs standard -> le champion ne peut
que progresser (voir gate()).

Usage : venv/Scripts/python.exe scripts/train_jokers.py [--iters 8] [--hands 4000] [--workers 4]
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from engine import joker_match as jm
from engine import jokers as J
from engine import joker_obs as O
from scripts import joker_profit as P

CHECKPOINT_DIR = "checkpoints_jokers"
CHECKPOINT = os.path.join(CHECKPOINT_DIR, "jokers.pt")
LOG_FILE = "jokers_train.jsonl"
ADV_SCALE = 10.0
ADV_CLIP = 5.0
EPSILON = 0.15
EVAL_SEED = 20261011
SPARE_DEALS = 40  # donnes de reserve par match d'eval (voir eval_worker)
EVAL_MATCHES = 400


# --- politique numpy (meme calcul que Godot : tanh-MLP, argmax masque) ------

def net_to_numpy(net):
    sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in net.state_dict().items()}
    return (sd["trunk.0.weight"], sd["trunk.0.bias"], sd["trunk.2.weight"], sd["trunk.2.bias"],
            sd["policy_head.weight"], sd["policy_head.bias"])


def logits_np(weights, obs):
    w0, b0, w2, b2, wp, bp = weights
    h = np.tanh(w0 @ obs + b0)
    h = np.tanh(w2 @ h + b2)
    return wp @ h + bp


def net_policy(weights):
    def policy(hand, seat):
        mask = hand.legal_casts(seat)
        if not any(mask[1:]):
            return 0
        lg = logits_np(weights, np.asarray(O.build(hand, seat)))
        return max((i for i in range(J.ACTION_DIM) if mask[i]), key=lambda i: lg[i])
    return policy


def hybrid_policy(weights):
    """agents/joker_expert.bot_action : regles, le reseau ne departage que les
    cas ouverts (weights None = regles seules)."""
    from agents import joker_expert as X

    def policy(hand, seat):
        mask = hand.legal_casts(seat)
        if not any(mask[1:]):
            return 0
        lg = None if weights is None else logits_np(weights, np.asarray(O.build(hand, seat)))
        return X.bot_action(hand, seat, lg)
    return policy


# "hybrid" (2026-10-16, defaut) : agents/joker_expert + reseau ; "net" : reseau seul
# (iteration 0 sans reseau = heuristique bot_wants) ; "rules" : regles seules ;
# "heuristic" : bot_wants.
POLICY_MODE = "hybrid"


def make_policy(weights, mode=None):
    mode = mode or POLICY_MODE
    if mode == "heuristic" or (mode == "net" and weights is None):
        return J.heuristic_action
    if mode == "net":
        return net_policy(weights)
    # "hybrid:joker:cas,..." (diagnostic) : ces cas fermes deviennent ouverts
    # (agents/joker_expert.SOFTEN, global du processus -> fixe a chaque appel)
    from agents import joker_expert as X
    mode, _sep, soft = mode.partition(":")
    X.SOFTEN = set(soft.split(",")) if soft else set()
    return hybrid_policy(None if mode == "rules" else weights)


# --- generation (workers) -----------------------------------------------------

# Terme inter-donnes moyen par joker (cast - pas de cast), mesure par
# scripts/audit_joker_horizon.py (iter 29, 3e5d, 19811 decisions ; seuls les
# termes a > 2 sigma). --horizon corrected : ajoute a Q(joker) tant qu'il reste
# des donnes au match -> cible par donne (peu bruitee) + cout/gain moyen futur.
CROSS_HAND = {"coup_de_vent": -0.872, "kermesse": 0.689, "inversion": 1.233}


def gen_worker(args):
    seed, n_hands, weights = args[:3]
    horizon = args[3] if len(args) > 3 else "hand"
    rng = random.Random(seed)
    future = make_policy(weights)
    samples = []
    ctx = {}
    profit = P.Agg()  # test de rentabilite, cote bots : les memes rollouts
    match = None

    def deciding(hand, seat):
        mask = hand.legal_casts(seat)
        if not any(mask[1:]):
            return 0
        q = [None] * J.ACTION_DIM
        qm = [None] * J.ACTION_DIM
        for a in range(J.ACTION_DIM):
            if mask[a]:
                if horizon == "match":
                    q[a], qm[a] = jm.match_rollout(match, hand, seat, a, future, ctx["left"], ctx["deals"])
                else:
                    pts = jm.play_out(jm.clone(hand), future, first_action=a)
                    q[a] = jm.hand_reward(pts, seat)
        if horizon == "corrected":
            qm = [x if x is None or a == 0 or not ctx["left"] else x + CROSS_HAND.get(J.CASTABLE[a - 1], 0.0)
                  for a, x in enumerate(q)]
        target = q if horizon == "hand" else qm
        samples.append((O.build(hand, seat), mask, [0.0 if x is None else (x - target[0]) for x in target]))
        chosen = future(hand, seat)
        P.add_bot_records(profit, P.bot_records(hand, seat, mask, q, chosen, match.config))
        if rng.random() < EPSILON:
            return rng.choice([a for a in range(J.ACTION_DIM) if mask[a]])
        return chosen

    done = 0
    while done < n_hands:
        match = jm.Match(jm.random_config(rng), rng.randrange(1 << 30))
        for i in range(jm.HANDS_PER_MATCH):
            hand = match.next_hand()
            ctx["left"] = jm.HANDS_PER_MATCH - 1 - i
            if horizon == "match":
                ctx["deals"] = jm.spare_deals(match, rng.randrange(1 << 30))
            jm.play_out(hand, deciding)
            match.finish(hand)
            done += 1
    return samples, profit.data


def eval_worker(args):
    seeds, weights = args[0], args[1]
    mode = args[2] if len(args) > 2 else None
    return policies_worker((seeds, [(weights, mode)]))


def policies_worker(args):
    """Duplicate : chaque politique (weights, mode) de la liste, puis bot_wants
    et « jamais », sur les memes matchs -> (graine, config, [(r, p0), ...])
    dans cet ordre."""
    seeds, specs = args
    out = []
    for seed in seeds:
        res = []
        rng = random.Random(seed)
        cfg, mseed = jm.random_config(rng), rng.randrange(1 << 30)
        rec = jm.Match(cfg, mseed)  # donnes de reference (chaine whistee, sans cast)
        for _ in range(jm.HANDS_PER_MATCH):
            hand = rec.next_hand()
            jm.play_out(hand, J.never_action)
            rec.finish(hand)
        # v2 : le joueur du siege 0 lance des jokers d'enchere (passe forcee) selon
        # l'etat, donc les passes generales different entre politiques et une
        # replique peut consommer plus de donnes que la reference. Reserve commune.
        for _ in range(SPARE_DEALS):
            hands, dealer, whisted = rec.chain.deal(rec.rng)
            rec.recorded.append(([list(h) for h in hands], dealer, whisted))
            rec.chain.finish_unplayed(hands, dealer)
        for policy in [make_policy(w, m) for w, m in specs] + [J.heuristic_action, J.never_action]:
            match = jm.Match(cfg, mseed, deals=rec.recorded)
            r = p0 = 0.0
            for _ in range(jm.HANDS_PER_MATCH):
                hand = match.next_hand()
                pts = jm.play_out(hand, policy)
                r += sum(jm.hand_reward(pts, s) for s in hand.holders())
                p0 += pts[jm.PLAYER]
                match.finish(hand)
            res.append((r, p0))
        out.append((seed, match.config, res))
    return out


def evaluate(pool, weights, workers, matches=EVAL_MATCHES):
    seeds = [EVAL_SEED + i for i in range(matches)]
    chunks = [seeds[i::workers] for i in range(workers)]
    rows = [r for part in pool.map(eval_worker, [(c, weights) for c in chunks]) for r in part]
    d_heur = [res[0][0] - res[1][0] for _s, _c, res in rows]
    d_never = [res[0][0] - res[2][0] for _s, _c, res in rows]
    d_p0 = [res[0][1] - res[1][1] for _s, _c, res in rows]

    def stat(xs):
        m = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
        return round(m, 3), round(sd / math.sqrt(len(xs)), 3)

    by_kind = {}
    for _s, cfg, res in rows:
        k = cfg.get("boss") or "elite_acte%d" % cfg["act"]
        by_kind.setdefault(k, []).append(res[0][0] - res[1][0])
    return {"vs_heuristic": stat(d_heur), "vs_never": stat(d_never), "player_points_vs_heuristic": stat(d_p0),
            "by_kind_vs_heuristic": {k: stat(v) for k, v in sorted(by_kind.items()) if len(v) > 1}}


def _stat(xs):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return round(m, 3), round(sd / math.sqrt(len(xs)), 3)


# --- porte du champion (2026-10-21) ----------------------------------------------
# Le champion (checkpoints_jokers/jokers.pt, ce qui s'exporte) n'est remplace que
# par un candidat qui le bat en duplicate sur des matchs NEUFS (graines jamais
# reutilisees : pas de biais de selection), avec la politique livree (hybride),
# de plus de GATE_Z erreurs standard. Sinon le champion reste et genere les
# donnes ; apres GATE_PATIENCE refus d'affilee le reseau repart du champion.
GATE_SEED = 30000000  # 30M..50M : hors EVAL_SEED, PROFIT_SEED (77M), OOS_SEED (91M)
GATE_STRIDE = 10007  # graines de la porte de l'iteration it : GATE_SEED + it * GATE_STRIDE + i
GATE_MATCHES = 800
GATE_Z = 2.0
GATE_PATIENCE = 4


def gate(pool, cand, champ, workers, matches, it):
    seeds = [GATE_SEED + it * GATE_STRIDE + i for i in range(matches)]
    chunks = [seeds[i::workers] for i in range(workers)]
    rows = [r for part in pool.map(policies_worker, [(c, [(cand, None), (champ, None)]) for c in chunks])
            for r in part]
    by_kind = {}
    for _s, cfg, res in rows:
        by_kind.setdefault(cfg.get("boss") or "elite_acte%d" % cfg["act"], []).append(res[0][0] - res[2][0])
    ev = {"vs_champion": _stat([res[0][0] - res[1][0] for _s, _c, res in rows]),
          "vs_heuristic": _stat([res[0][0] - res[2][0] for _s, _c, res in rows]),
          "champion_vs_heuristic": _stat([res[1][0] - res[2][0] for _s, _c, res in rows]),
          "vs_never": _stat([res[0][0] - res[3][0] for _s, _c, res in rows]),
          "player_points_vs_heuristic": _stat([res[0][1] - res[2][1] for _s, _c, res in rows]),
          "by_kind_vs_heuristic": {k: _stat(v) for k, v in sorted(by_kind.items()) if len(v) > 1},
          "gate_seeds": [seeds[0], len(seeds)]}
    m, se = ev["vs_champion"]
    ev["accepted"] = m > 0 and m > GATE_Z * se
    return ev


# --- apprentissage --------------------------------------------------------------

def fit(net, data, epochs=6, lr=1e-3, batch=256):
    import torch
    obs = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32)
    mask = torch.tensor(np.array([d[1] for d in data]), dtype=torch.bool)
    adv = torch.tensor(np.array([d[2] for d in data]), dtype=torch.float32)
    adv = (adv / ADV_SCALE).clamp(-ADV_CLIP, ADV_CLIP)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(data)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            logits, _v = net(obs[idx])
            m = mask[idx].float()
            loss = (((logits - adv[idx]) ** 2) * m).sum() / m.sum()
            # l'avantage de « aucun cast » vaut 0 par definition
            loss = loss + (logits[:, 0] ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
    net.eval()
    return total / n


LAST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "jokers_last.pt")
STOP_FILE = ".stop_jokers"
REPLAY_ITERS = 3
CLEAN_STOP = "Arret propre IA jokers."


def _payload(net, it, ev, best):
    return {"net": net.state_dict(), "obs_dim": O.OBS_DIM, "action_dim": J.ACTION_DIM,
            "spec_hash": J.JOKER_SPEC_HASH, "castable": J.CASTABLE, "eval": ev, "iter": it, "best": best}


def _compatible(ckpt):
    return (ckpt.get("spec_hash") in J.RULE_EQUIVALENT_HASHES and ckpt.get("castable") == J.CASTABLE
            and ckpt.get("obs_dim") == O.OBS_DIM)


def main():
    import datetime
    import torch
    from agents.network import PolicyValueNet
    from agents.checkpoint_io import safe_save

    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--until", default=None, help="arret propre a cette heure locale (AAAA-MM-JJTHH:MM)")
    parser.add_argument("--resume", action="store_true", help="repart de jokers_last.pt (sinon jokers.pt)")
    parser.add_argument("--hands", type=int, default=4000, help="donnes generees par iteration")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-matches", type=int, default=EVAL_MATCHES, help="(compare_jokers_oos, audits)")
    parser.add_argument("--gate-matches", type=int, default=GATE_MATCHES,
                        help="matchs neufs de la porte du champion par iteration")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=6, help="passes sur les donnees a chaque iteration (0 = reseau inchange)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--horizon", choices=("hand", "match", "corrected"), default="hand",
                        help="cible des rollouts : donne en cours, donne + donnes restantes du match "
                             "(donnes fixes communes, scripts/audit_joker_horizon.py), ou donne + CROSS_HAND")
    parser.add_argument("--checkpoint-dir", default=None, help="autre dossier de checkpoints (essais)")
    parser.add_argument("--init", default=None,
                        help="demarrage a chaud : poids d'un checkpoint d'autres regles (meme obs et memes CASTABLE), "
                             "iterations et meilleur repartent de zero")
    parser.add_argument("--profit-matches", type=int, default=300,
                        help="matchs du test de rentabilite (cote joueur) par iteration, 0 = sans")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.checkpoint_dir:
        global CHECKPOINT, LAST_CHECKPOINT, LOG_FILE
        CHECKPOINT = os.path.join(args.checkpoint_dir, "jokers.pt")
        LAST_CHECKPOINT = os.path.join(args.checkpoint_dir, "jokers_last.pt")
        LOG_FILE = os.path.join(args.checkpoint_dir, "jokers_train.jsonl")
    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    deadline = datetime.datetime.fromisoformat(args.until) if args.until else None

    import copy
    net = PolicyValueNet(O.OBS_DIM, J.ACTION_DIM)
    start_it = 0
    # Champion = ce qui s'exporte (CHECKPOINT) ; genere les donnes. None = regles seules.
    champ = {"sd": None, "w": None, "iter": None, "eval": None}

    def set_champion(sd, it, ev):
        champ.update(sd=copy.deepcopy(sd), iter=it, eval=ev)
        n2 = PolicyValueNet(O.OBS_DIM, J.ACTION_DIM)
        n2.load_state_dict(sd)
        champ["w"] = net_to_numpy(n2)

    def champion_payload():
        return {"net": champ["sd"], "obs_dim": O.OBS_DIM, "action_dim": J.ACTION_DIM,
                "spec_hash": J.JOKER_SPEC_HASH, "castable": J.CASTABLE, "eval": champ["eval"],
                "iter": champ["iter"], "best": None, "champion": True}

    if args.resume:
        for path in (CHECKPOINT, LAST_CHECKPOINT):
            if os.path.exists(path):
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                if not _compatible(ckpt):
                    raise SystemExit("%s entraine sur d'autres regles/obs" % path)
        if os.path.exists(CHECKPOINT):
            b = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
            set_champion(b["net"], b["iter"], b["eval"])
            start_it = b["iter"] + 1
        if os.path.exists(LAST_CHECKPOINT):
            ckpt = torch.load(LAST_CHECKPOINT, map_location="cpu", weights_only=False)
            start_it = max(start_it, ckpt["iter"] + 1)
            # avant la porte (pas de cle "gate") : le dernier n'a rien prouve -> on repart du champion
            if "gate" in ckpt.get("eval", {}) or champ["sd"] is None:
                net.load_state_dict(ckpt["net"])
            else:
                net.load_state_dict(champ["sd"])
            if champ["sd"] is None:
                set_champion(ckpt["net"], ckpt["iter"], ckpt["eval"])
        elif champ["sd"] is not None:
            net.load_state_dict(champ["sd"])
        print("reprise : champion iteration %s, depart iteration %d" % (champ["iter"], start_it), flush=True)
    if args.init and champ["sd"] is None:
        ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
        if ckpt.get("castable") != J.CASTABLE or ckpt.get("obs_dim") != O.OBS_DIM:
            raise SystemExit("%s : autre obs ou autres CASTABLE" % args.init)
        net.load_state_dict(ckpt["net"])
        set_champion(ckpt["net"], -1, {"vs_heuristic": None, "init": args.init, "init_spec_hash": ckpt.get("spec_hash"),
                                       "init_iter": ckpt["iter"]})
        safe_save(champion_payload(), CHECKPOINT)
        print("demarrage a chaud depuis %s (%s, iteration %d) = champion initial" % (
            args.init, ckpt.get("spec_hash"), ckpt["iter"]), flush=True)
    data = []
    refused = 0
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        it = start_it
        while True:
            if os.path.exists(STOP_FILE):
                os.remove(STOP_FILE)
                print("fichier d'arret %s" % STOP_FILE, flush=True)
                break
            if deadline is None and it >= start_it + args.iters:
                break
            if deadline is not None and datetime.datetime.now() >= deadline:
                print("heure d'arret atteinte (%s)" % args.until, flush=True)
                break
            per = args.hands // args.workers
            jobs = [(args.seed * 1000003 + it * 101 + w, per, champ["w"], args.horizon) for w in range(args.workers)]
            parts = pool.map(gen_worker, jobs)
            new = [s for part, _p in parts for s in part]
            profit = P.Agg()
            for _s, pdata in parts:
                profit.merge(P.Agg(pdata))
            sizes = getattr(main, "_sizes", [])
            sizes = (sizes + [len(new)])[-REPLAY_ITERS:]
            main._sizes = sizes
            data = (data + new)[-sum(sizes):]  # les REPLAY_ITERS dernieres iterations
            loss = fit(net, data, epochs=args.epochs, lr=args.lr) if args.epochs > 0 else 0.0
            weights = net_to_numpy(net)
            ev = gate(pool, weights, champ["w"], args.workers, args.gate_matches, it)
            ev["gate"] = {"champion_iter": champ["iter"], "accepted": ev.pop("accepted"), "z": GATE_Z}
            if ev["gate"]["accepted"]:
                set_champion(net.state_dict(), it, ev)
                safe_save(champion_payload(), CHECKPOINT)
                refused = 0
            else:
                refused += 1
            if args.profit_matches > 0:
                seeds = [P.PROFIT_SEED + it * 10007 + i for i in range(args.profit_matches)]
                for pdata in pool.map(P.profit_worker, [(seeds[w::args.workers], champ["w"]) for w in range(args.workers)]):
                    profit.merge(P.Agg(pdata))
                # sans test de rentabilite : ni etat ni godot_export/joker_profit_stats.json
                ev["profit_verdicts"] = P.record_iteration(it, profit, args.profit_matches)
            rec = {"iter": it, "samples": len(new), "train_samples": len(data), "loss": round(loss, 4),
                   "elapsed_s": round(time.time() - t0),
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "spec_hash": J.JOKER_SPEC_HASH,
                   "policy": POLICY_MODE, "champion_iter": champ["iter"], **ev}
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if refused >= GATE_PATIENCE and champ["sd"] is not None:
                net.load_state_dict(champ["sd"])  # le reseau derive sans battre le champion : on repart de lui
                refused = 0
                print("%d refus d'affilee : reseau remis au champion (iteration %s)" % (GATE_PATIENCE, champ["iter"]),
                      flush=True)
            safe_save(_payload(net, it, ev, None), LAST_CHECKPOINT)
            it += 1
    print("champion : iteration %s, eval %s" % (champ["iter"], (champ["eval"] or {}).get("vs_heuristic")))
    print(CLEAN_STOP, flush=True)


if __name__ == "__main__":
    main()
