"""Contre-mesure d'une acceptation de porte (23/11, iter 17 3fed : 1 acceptation sur 22, z 2.27).

Meme duel que la porte de train_jokers (candidat contre champion, politique livree, memes
matchs), sur des graines NEUVES : 60M + (graine % 1M) * 10007, hors EVAL (20M), GATE (30M..50M),
PROFIT (77M), OOS (91M), moments (95M). Mesure seule, ne touche a aucun checkpoint.

    python scripts/joker_confirm.py CANDIDAT.pt CHAMPION.pt --minutes 80 --seed <run id>
Sortie : joker_confirm.json / joker_confirm.md (mis a jour a chaque lot).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIRM_SEED = 60000000
BATCH = 200


def main():
    import torch
    from agents.network import PolicyValueNet
    from engine import joker_obs as O, jokers as J
    from scripts import train_jokers as T

    ap = argparse.ArgumentParser()
    ap.add_argument("cand")
    ap.add_argument("champ")
    ap.add_argument("--minutes", type=float, default=80)
    ap.add_argument("--matches", type=int, default=0, help="arret apres ce nombre de matchs (0 = au temps)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="joker_confirm")
    args = ap.parse_args()
    torch.set_num_threads(1)

    def load(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("castable") != J.CASTABLE or ck.get("obs_dim") != O.OBS_DIM:
            raise SystemExit("%s : CASTABLE / obs incompatibles avec les regles actuelles" % path)
        net = PolicyValueNet(O.OBS_DIM, J.ACTION_DIM)
        net.load_state_dict(ck["net"])
        return T.net_to_numpy(net), {"path": path, "iter": ck.get("iter"), "spec_hash": ck.get("spec_hash")}

    cand, cand_info = load(args.cand)
    champ, champ_info = load(args.champ)
    base = CONFIRM_SEED + (args.seed % 1000) * 10007
    d_champ, d_heur_c, d_heur_k = [], [], []
    end, done = time.time() + args.minutes * 60, 0
    with mp.Pool(args.workers) as pool:
        while time.time() < end and (args.matches == 0 or done < args.matches):
            seeds = [base + done + i for i in range(BATCH)]
            for part in pool.map(T.policies_worker, [(seeds[i::args.workers], [(cand, None), (champ, None)])
                                                     for i in range(args.workers)]):
                for _s, _cfg, res in part:
                    d_champ.append(res[0][0] - res[1][0])
                    d_heur_c.append(res[0][0] - res[2][0])
                    d_heur_k.append(res[1][0] - res[2][0])
            done += BATCH
            out = {"candidat": cand_info, "champion": champ_info, "spec_hash": J.JOKER_SPEC_HASH,
                   "seeds": [base, done], "matches": done,
                   "vs_champion": T._stat(d_champ), "candidat_vs_heuristic": T._stat(d_heur_c),
                   "champion_vs_heuristic": T._stat(d_heur_k),
                   "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
            m, se = out["vs_champion"]
            out["z"] = round(m / se, 2) if se else None
            with open(args.out + ".json", "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=1)
            with open(args.out + ".md", "w", encoding="utf-8") as f:
                f.write("# Contre-mesure candidat iter %s contre champion iter %s (regles %s)\n\n"
                        "%d matchs neufs (graines %d..), politique livree.\n\n"
                        "- candidat - champion : %+.2f (se %.2f, z %s)\n"
                        "- candidat vs heuristique : %+.2f (se %.2f)\n"
                        "- champion vs heuristique : %+.2f (se %.2f)\n" % (
                            cand_info["iter"], champ_info["iter"], J.JOKER_SPEC_HASH, done, base, m, se, out["z"],
                            *out["candidat_vs_heuristic"], *out["champion_vs_heuristic"]))
            print("matchs %d : vs champion %+.2f +- %.2f (z %s)" % (done, m, se, out["z"]), flush=True)


if __name__ == "__main__":
    main()
