"""Compare des checkpoints IA jokers hors echantillon (graines != EVAL_SEED).

Le meilleur de l'entrainement est choisi au max d'une eval a graines fixes :
biais de selection. Ici, memes matchs pour chaque checkpoint (comparaison appariee).

    venv/Scripts/python.exe scripts/compare_jokers_oos.py [--matches 800] [ckpt ...]
"""
import argparse
import math
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OOS_SEED = 91000000


def main():
    import torch
    from agents.network import PolicyValueNet
    from engine import joker_obs as O, jokers as J
    from scripts import train_jokers as T

    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="*", default=[T.CHECKPOINT, T.LAST_CHECKPOINT])
    ap.add_argument("--matches", type=int, default=800)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--modes", default="net", help="politiques par checkpoint separees par ';', ex. net;hybrid ; + 'rules' une fois ; "
                    "hybrid:joker:cas,... = ces cas de regle ouverts")
    ap.add_argument("--any-rules", action="store_true",
                    help="mesure aussi les checkpoints d'autres regles (meme obs, memes CASTABLE), joues sous les regles actuelles")
    args = ap.parse_args()
    torch.set_num_threads(1)
    seeds = [OOS_SEED + i for i in range(args.matches)]
    chunks = [seeds[i::args.workers] for i in range(args.workers)]
    per = {}
    def run(label, weights, mode, info):
        rows = [r for part in pool.map(T.eval_worker, [(c, weights, mode) for c in chunks]) for r in part]
        per[label] = {s: res[0][0] - res[1][0] for s, _c, res in rows}
        xs = list(per[label].values())
        m = sum(xs) / len(xs)
        se = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))
        print("%s %s : vs heuristique hors echantillon %+.2f +- %.2f" % (label, info, m, se), flush=True)

    modes = args.modes.split(";") if ";" in args.modes or ":" in args.modes else args.modes.split(",")
    with mp.Pool(args.workers) as pool:
        if "rules" in modes:
            run("regles", None, "rules", "(regles seules)")
        for path in args.ckpts:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            if not (T._compatible(ck) or (args.any_rules and ck.get("castable") == J.CASTABLE
                                          and ck.get("obs_dim") == O.OBS_DIM)):
                print("%s : incompatible, ignore" % path)
                continue
            net = PolicyValueNet(O.OBS_DIM, J.ACTION_DIM)
            net.load_state_dict(ck["net"])
            for mode in modes:
                if mode != "rules":
                    run("%s[%s]" % (path, mode), T.net_to_numpy(net), mode,
                        "(iter %s, %s, eval interne %s)" % (ck["iter"], ck.get("spec_hash"), (ck["eval"] or {}).get("vs_heuristic")))
    paths = list(per)
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            d = [per[paths[i]][s] - per[paths[j]][s] for s in seeds]
            m = sum(d) / len(d)
            se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))
            print("%s - %s : %+.2f +- %.2f (apparie)" % (paths[i], paths[j], m, se))


if __name__ == "__main__":
    main()
