"""Question utilisateur 22/11 : "les bots n'emballent jamais le joueur dans l'acte I ?"

Un humain simule (profil joueur_*) et 3 bots, memes donnes pour 4 tables :
  acte1   : ExpertBot ACT1_BID_PARAMS (Acte I entier, et combats hors boss des actes II-III, rogue_run.gd _watch) ;
  boss    : ExpertBot par defaut, sans routage (boss des actes II-III) ;
  avant   : table classique, routage d'avant a50 ;
  apres   : table classique, routage avec net_emballer_ok.
Compte : donnes ou l'humain propose, ou un bot l'emballe, emballages finis avec lui et leur reussite.
Non simule : jokers, boss, specialite des elites (BID_SLACK), derapage du jeu de la carte. Mesure seule, cloud.

Usage : python -u scripts/emb_human_freq.py <donnes par profil> <graine>
"""

import multiprocessing as mp
import random
import sys

sys.path.insert(0, ".")

from agents import bid_route, expert_bot
from engine.bidding import bid_action_to_dict
from engine.whisted import DeckChain
from scripts import expert_bid_ab as ab
from scripts.eval_vs_random import declarer_successes
from scripts.train import declarer_role_and_seats

TABLES = ("acte1", "boss", "avant", "apres")
PROFILES = ("joueur_0927", "joueur_1027")
state = {}
_apply = expert_bot.apply_choice
_ok = bid_route.net_emballer_ok


def apply_choice(bidding, seat, choice):
    if not isinstance(choice, tuple):
        info = bid_action_to_dict(choice)
        if info["action"] == "propose" and seat == state["human"]:
            state["proposed"] = True
        elif info["action"] == "emballer" and bidding.pending_proposals.get(info["suit"]) == state["human"]:
            state["joined"] = True
    return _apply(bidding, seat, choice)


def work(args):
    seed, n, profile = args
    rng, human_rng = random.Random(seed), random.Random(seed + 500)
    chain = DeckChain(ab.game_ai.WHISTED_RATIO, rng)
    out = {t: [0, 0, 0, 0] for t in TABLES}
    for _ in range(n):
        hands, dealer, whisted = chain.deal(rng)
        hands = [list(h) for h in hands]
        human = human_rng.randrange(4)
        for name in TABLES:
            ingame = name in ("avant", "apres")
            bots = expert_bot.ACT1_BID_PARAMS if name == "acte1" else {}
            params = [dict(expert_bot.PROFILES[profile] if s == human else bots) for s in range(4)]
            bid_route.net_emballer_ok = (lambda hand, suit: True) if name == "avant" else _ok
            state.update(human=human, proposed=False, joined=False)
            _pts, c, game = ab.play_deal(hands, dealer, whisted, params, ingame, human)
            a = out[name]
            a[0] += state["proposed"]
            a[1] += state["proposed"] and state["joined"]
            if c is not None and c["type"] == "emballage" and c["declarer"] == human:
                oks = declarer_successes(c, game, declarer_role_and_seats(c)[1])
                a[2] += 1
                a[3] += sum(oks) == len(oks)
        # la derniere table jouee est "apres" : le paquet suit le jeu tel qu'il sera
        if game is None:
            chain.finish_unplayed(hands, dealer)
        else:
            chain.finish_played(game, dealer)
    return out


def main():
    deals, seed = int(sys.argv[1]), int(sys.argv[2])
    ab.load_game_ai()
    expert_bot.apply_choice = apply_choice
    per = deals // 4
    for p, profile in enumerate(PROFILES):
        with mp.get_context("fork").Pool(4) as pool:
            parts = pool.map(work, [((seed + p) * 1000 + w, per, profile) for w in range(4)])
        for name in TABLES:
            proposed, joined, final, ok = (sum(part[name][i] for part in parts) for i in range(4))
            print("FREQ %s %-6s donnes %d propose %d emballe %d (%.1f%% des propositions) finis %d reussis %.1f%%" % (
                profile, name, per * 4, proposed, joined, 100.0 * joined / max(1, proposed), final,
                100.0 * ok / max(1, final)), flush=True)
    print("FREQ_DONE")


if __name__ == "__main__":
    main()
