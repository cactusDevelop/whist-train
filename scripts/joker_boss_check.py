"""Mesure pre-enregistree (conseil 24/11) : l'IA jokers livree perd-elle contre l'heuristique sur UN boss ?

Hypothese fixee AVANT le run (une seule, pas de choix de boss apres coup) : sur les matchs fossoyeur,
hybride (regles + reseau champion) < heuristique bot_wants. Acceptee si moyenne < 0 avec z <= -2.5 ;
alors seulement un routage par boss (heuristique contre fossoyeur) est propose a whist-4-it, qui decide.

Graines NEUVES : 98M + (graine % 1000) * 10007 (hors EVAL 20M, GATE 30M..50M, CONFIRM 60M, PROFIT 77M,
OOS 91M, moments 95M), filtrees sur jm.random_config(...)["boss"] == --boss. Memes matchs (duplicate) pour :
hybride champion, regles seules (cas ouverts = defaut des regles), heuristique, jamais. Detail par siege de bot
(1 = voisin du joueur, 2 = boss en face, 3 = autre voisin) pour voir de quel cote se fait l'ecart.

    python scripts/joker_boss_check.py CHAMPION.pt --boss fossoyeur --minutes 90 --seed <run id>
Sortie : joker_boss_check.json / .md (mis a jour a chaque lot). Mesure seule, ne touche a aucun checkpoint.
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECK_SEED = 98000000
BATCH = 200
Z_ACCEPT = -2.5


def boss_seeds(start, boss, n):
    from engine import joker_match as jm
    out, s = [], start
    while len(out) < n:
        if jm.random_config(random.Random(s)).get("boss") == boss:
            out.append(s)
        s += 1
    return out, s


def main():
    import torch
    from agents.network import PolicyValueNet
    from engine import joker_obs as O, jokers as J
    from scripts import train_jokers as T

    ap = argparse.ArgumentParser()
    ap.add_argument("champ")
    ap.add_argument("--boss", default="fossoyeur")
    ap.add_argument("--minutes", type=float, default=90)
    ap.add_argument("--matches", type=int, default=0, help="arret apres ce nombre de matchs (0 = au temps)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="joker_boss_check")
    args = ap.parse_args()
    torch.set_num_threads(1)

    ck = torch.load(args.champ, map_location="cpu", weights_only=False)
    if ck.get("castable") != J.CASTABLE or ck.get("obs_dim") != O.OBS_DIM:
        raise SystemExit("%s : CASTABLE / obs incompatibles avec les regles actuelles" % args.champ)
    net = PolicyValueNet(O.OBS_DIM, J.ACTION_DIM)
    net.load_state_dict(ck["net"])
    champ = T.net_to_numpy(net)
    info = {"path": args.champ, "iter": ck.get("iter"), "spec_hash": ck.get("spec_hash")}

    base = CHECK_SEED + (args.seed % 1000) * 10007
    nxt = base
    d = {"hybride_vs_heuristique": [], "regles_vs_heuristique": [], "hybride_vs_regles": [],
         "hybride_vs_jamais": [], "heuristique_vs_jamais": []}
    seat = {s: [] for s in (1, 2, 3)}  # hybride - heuristique, par siege de bot
    end, done = time.time() + args.minutes * 60, 0
    with mp.Pool(args.workers) as pool:
        while time.time() < end and (args.matches == 0 or done < args.matches):
            seeds, nxt = boss_seeds(nxt, args.boss, BATCH)
            for part in pool.map(T.policies_worker, [(seeds[i::args.workers], [(champ, None), (None, "rules")])
                                                     for i in range(args.workers)]):
                for _s, _cfg, res in part:
                    hyb, rul, heu, nev = res
                    d["hybride_vs_heuristique"].append(hyb[0] - heu[0])
                    d["regles_vs_heuristique"].append(rul[0] - heu[0])
                    d["hybride_vs_regles"].append(hyb[0] - rul[0])
                    d["hybride_vs_jamais"].append(hyb[0] - nev[0])
                    d["heuristique_vs_jamais"].append(heu[0] - nev[0])
                    for s in seat:
                        seat[s].append(hyb[3].get(s, 0.0) - heu[3].get(s, 0.0))
            done += len(seeds)
            out = {"champion": info, "spec_hash": J.JOKER_SPEC_HASH, "boss": args.boss,
                   "seeds": [base, nxt], "matches": done, "z_accept": Z_ACCEPT,
                   **{k: T._stat(v) for k, v in d.items()},
                   "hybride_vs_heuristique_par_siege": {str(s): T._stat(v) for s, v in seat.items()},
                   "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
            m, se = out["hybride_vs_heuristique"]
            out["z"] = round(m / se, 2) if se else None
            out["hypothese_acceptee"] = bool(out["z"] is not None and out["z"] <= Z_ACCEPT)
            with open(args.out + ".json", "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=1)
            lines = ["# Mesure pre-enregistree : hybride iter %s vs heuristique, boss %s (regles %s)\n"
                     % (info["iter"], args.boss, J.JOKER_SPEC_HASH),
                     "%d matchs neufs (graines %d..%d), duplicate.\n" % (done, base, nxt),
                     "Hypothese : hybride < heuristique, acceptee si z <= %.1f -> **%s** (z %s)\n"
                     % (Z_ACCEPT, "ACCEPTEE" if out["hypothese_acceptee"] else "non acceptee", out["z"])]
            lines += ["- %s : %+.2f (se %.2f)" % (k.replace("_", " "), *out[k]) for k in d]
            lines += ["- hybride - heuristique, siege %s : %+.2f (se %.2f)" % (s, *v)
                      for s, v in out["hybride_vs_heuristique_par_siege"].items()]
            with open(args.out + ".md", "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            print("matchs %d : hybride - heuristique %+.2f +- %.2f (z %s)" % (done, m, se, out["z"]), flush=True)


if __name__ == "__main__":
    main()
