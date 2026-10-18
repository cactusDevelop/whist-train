"""Amorce le reseau de la lignee hybride `encherisseur_expert` (voir
agents/hybrid_bidder.py) par DISTILLATION de l'enchere de l'ExpertBot :
apprentissage supervise (entropie croisee) du coup du bot, sur les seules
decisions que le reseau prendra reellement (actions permises par les
garde-fous, hors decisions imposees par les regles).

Etats visites : chaque siege d'une donne est joue soit par le bot, soit par
un individu de la population actuelle (--population), pour que le reseau voie
aussi les situations creees par les autres lignees (qui surencherissent),
pas seulement celles d'une table de bots -- meme idee que DAgger.

Usage : venv/Scripts/python.exe scripts/distill_hybrid_bidder.py --population <copie> --out <seed.pt>
"""

import argparse
import random
import sys
import time

import torch

sys.path.insert(0, ".")

from agents import evolution, expert_bot
from agents.hybrid_bidder import HybridBidder, bot_choice_as_id, rule_layer
from engine.bidding import Bidding, legal_action_ids_with_hand
from engine.bidding import observation as bid_observation
from engine.trou import detect_trou
from engine.whisted import DeckChain

BOT_SEAT_PROB = 0.5


def new_bidding(hands, dealer):
    b = Bidding(dealer)
    t = detect_trou(hands)
    if t is not None:
        b.seed_trou(t)
    return b


def collect(population, deals, rng):
    from scripts.train import WHISTED_RATIO_DEFAULT
    chain = DeckChain(WHISTED_RATIO_DEFAULT, rng)
    # tableaux COMPACTS pre-alloues (machine a 4 Go partagee avec
    # l'entrainement : une premiere version en listes de tenseurs a ete tuee
    # faute de memoire le 2026-09-27)
    cap = deals * 12
    obs = torch.zeros(cap, evolution.OBS_DIM, dtype=torch.float16)
    masks = torch.zeros(cap, evolution.ACTION_DIM, dtype=torch.bool)
    labels = torch.zeros(cap, dtype=torch.long)
    n = 0
    skipped = 0
    for _ in range(deals):
        hands, dealer, _whisted = chain.deal(rng)
        b = new_bidding(hands, dealer)
        players = {s: None if rng.random() < BOT_SEAT_PROB else evolution.policy_of(rng.choice(population))
                   for s in range(4)}
        while not b.finished:
            seat = b.turn
            hand = hands[seat]
            bot_id = bot_choice_as_id(b, seat, expert_bot.choose_bid(b, seat, hand))
            force, permitted = rule_layer(b, seat, hand)
            if force is None and len(permitted) > 1:
                if bot_id in permitted and n < cap:
                    obs[n] = torch.tensor(bid_observation(b, seat, hand), dtype=torch.float16)
                    masks[n, permitted] = True
                    labels[n] = bot_id
                    n += 1
                elif bot_id not in permitted:
                    skipped += 1
            p = players[seat]
            if p is None:
                action = bot_id
            elif getattr(p, "hybrid", False):
                action = p.decide(b, seat, hand)
            else:
                o = torch.tensor(bid_observation(b, seat, hand), dtype=torch.float32)
                legal = legal_action_ids_with_hand(b, seat, hand)
                m = torch.zeros(evolution.ACTION_DIM, dtype=torch.bool)
                m[legal] = True
                action = p.act_greedy(o, m)
            assert b.apply(seat, action), (seat, action)
        chain.finish_unplayed(hands, dealer)
    return obs[:n].float(), masks[:n].clone(), labels[:n].clone(), skipped


def train(net, obs, masks, labels, epochs, rng_seed):
    g = torch.Generator().manual_seed(rng_seed)
    n = len(labels)
    perm = torch.randperm(n, generator=g)
    n_val = n // 10
    val, tr = perm[:n_val], perm[n_val:]
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for epoch in range(epochs):
        order = tr[torch.randperm(len(tr), generator=g)]
        for i in range(0, len(order), 512):
            idx = order[i:i + 512]
            logits, _v = net(obs[idx])
            logits = logits.masked_fill(~masks[idx], -1e9)
            loss = torch.nn.functional.cross_entropy(logits, labels[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            logits, _v = net(obs[val])
            acc = (logits.masked_fill(~masks[val], -1e9).argmax(-1) == labels[val]).float().mean().item()
        print("  epoque %2d : perte %.4f, accord validation %.2f%%" % (epoch + 1, loss.item(), 100 * acc), flush=True)
    return acc


def agreement_with_bot(net, deals, rng):
    """Tables de 4 bots : part des decisions ou l'hybride (regles + reseau)
    ferait exactement le coup du bot, decisions imposees comprises."""
    hybrid = HybridBidder(net)
    same = total = 0
    for _ in range(deals):
        from engine.cards import deal
        hands = deal(rng)
        b = new_bidding(hands, rng.randrange(4))
        while not b.finished:
            seat = b.turn
            bot_id = bot_choice_as_id(b, seat, expert_bot.choose_bid(b, seat, hands[seat]))
            same += hybrid.decide(b, seat, hands[seat]) == bot_id
            total += 1
            assert b.apply(seat, bot_id)
    return same / total, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", required=True, help="copie de checkpoints_v2/encherisseur_population.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--deals", type=int, default=60000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=27)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)  # laisser le CPU a l'entrainement en cours
    population, generation = evolution.load_population(args.population)
    population = [ind for ind in population if ind.lineage != evolution.HYBRID_LINEAGE]
    t0 = time.time()
    with torch.no_grad():
        obs, masks, labels, skipped = collect(population, args.deals, rng)
    print("%d decisions etiquetees (%d coups du bot hors garde-fous ignores), population gen %d, %.0fs" % (
        len(labels), skipped, generation, time.time() - t0), flush=True)
    net = evolution.new_net()
    acc = train(net, obs, masks, labels, args.epochs, args.seed)
    agree, n = agreement_with_bot(net, 3000, random.Random(args.seed + 1))
    print("Accord hybride/bot sur %d decisions de tables de bots : %.2f%% (validation reseau %.2f%%)" % (
        n, 100 * agree, 100 * acc), flush=True)
    torch.save({"state_dict": net.state_dict(), "val_accuracy": acc, "bot_agreement": agree}, args.out)
    print("-> %s" % args.out)


if __name__ == "__main__":
    main()
