"""Enchere entrainee par PPO (2026-09-28, accord utilisateur, "piste 3").

Pourquoi l'evolution est abandonnee pour `encherisseur` : mesure du 28/09 sur
la population (12 individus x 400 donnes, duplicate vs ExpertBot) -- ecart
type d'une donne 7.5 pts, ecart reel entre individus < 0.3 pt/donne,
correlation du classement entre deux lots de 200 donnes -0.23. Pour separer
deux individus il faudrait ~2500 donnes chacun par generation (~3 h) : la
selection est aveugle aux fautes rares et couteuses (emballage-12/13 a 0-14 %,
solo >= 10 a 0-6 %, ~1 % des donnes x -30 pts ~ 0.3 pt/donne) et les mutations
derivaient librement.

Ici chaque DECISION d'enchere recoit son propre merite : retour = vrais
points du siege a la fin de la donne (engine/*.final_scores, meme bareme que
Godot), avantage GAE sur les decisions d'enchere du siege avec la tete de
valeur comme reference. Une annonce a 13 qui chute est punie directement sur
cette decision, au lieu d'etre noyee dans 200 donnes de bruit. Aucune regle de
jeu imposee, aucun faconnage : l'IA trouve seule jusqu'ou monter.

Adversaires : la moitie des donnes en self-play (4 sieges = ce reseau,
echantillonne), l'autre moitie un camp ExpertBot (ancre fixe, empeche une
derive "entre soi"). Le jeu de la carte reste celui des 9 specialistes PPO.

Point de depart : le meilleur individu `encherisseur` de la population
(celui qui etait exporte), jamais un reseau neuf. Le fichier de population
garde sa forme (1 individu `encherisseur` = ce reseau, + la lignee hybride
figee) pour que export, duel, Phase 3 et verifications restent inchanges ;
l'ancienne population est sauvegardee a cote avant la premiere ecriture."""

import os
import shutil

import torch

from agents import evolution
from agents.checkpoint_io import safe_save
from agents.expert_bot import emballer_min, estimate_tricks
from engine.bidding import bid_action_to_dict

RETURN_SCALE = 10.0  # vrais points (jusqu'a ~100) -> meme echelle que `defenseur`
GAE_LAMBDA = 0.95
LR = 1e-4  # reseau deja forme (distille puis evolue) : pas trop brutal
EXPERT_OPP_PROB = 0.5
# Tables 100 % IA jouees en GREEDY (argmax, comme dans Godot) : accord
# utilisateur 2026-09-29. En echantillonne, une derive du comportement greedy
# (surenchere IA-IA jusqu'au solo-13 a -90 : 5/1000 donnes en Phase 3 le 29/09
# a 04:13) ne recoit presque pas de gradient, l'exploration cassant la spirale
# avant qu'elle ait lieu. Sur ces tables, l'action jouee est l'argmax et son
# log-prob sous la politique courante est garde pour PPO (le ratio PPO reste
# borne par le clip) : un coup greedy qui coute cher est directement puni.
GREEDY_TABLE_PROB = 0.25  # part des tables sans camp ExpertBot
CHECKPOINT_NAME = "encherisseur_ppo.pt"

# Garde d'emballage haut (2026-09-29, controle de 22:04). Depuis 51b9151,
# emballer par-dessus une misere piccolo entre directement a 13 ; avant, ce
# coup etait illegal et le reseau ne l'a jamais vu. Mesure : face a un camp
# ExpertBot qui annonce piccolo, le reseau GREEDY emballe a 13 dans 69 cas
# sur 92 (0 % de reussite en entrainement). Le PPO seul ne corrige pas (~20
# chutes par 8000 donnes, frequence stable sur 120 000 donnes). Imitation
# ciblee de l'ExpertBot, comme pour misere/trou : sur une decision ou un
# emballage d'entree >= EMBALLER_GUARD_LEVEL est legal alors que la main n'a
# pas les plis de expert_bot.emballer_min, la cible de distillation est
# "toute action legale sauf ces emballages". Aucune regle de jeu changee.
EMBALLER_GUARD_LEVEL = 11
# Historique du poids : 1.0 (2a32559) -> le reseau a generalise en "ne plus
# emballer du tout" (greedy face a l'ExpertBot, emballer possible a 8 : 81 %
# -> 7.6 % en 52 000 donnes) ; 0.01 (7a91fea) -> l'emballage normal remonte
# (36 %) mais l'emballage-13 revient (25 cas sur 157). Une garde d'un seul
# cote pousse l'action "emballer" en bloc dans un sens ou dans l'autre.
# D'ou l'ANCRE (30/09 01:34) : sur les decisions ou un emballage d'entree
# <= 10 est legal, cible = le choix greedy du reseau d'AVANT la garde (appris
# par PPO sur 128 000 donnes, 73-81 % de reussite a 8). Le reseau doit
# distinguer les deux contextes au lieu de basculer. La perte d'imitation est
# une moyenne ponderee : ancre poids 1, garde poids EMBALLER_GUARD_WEIGHT.
EMBALLER_GUARD_COEF = 1.0
EMBALLER_GUARD_WEIGHT = 20.0
ANCHOR_NAME = "encherisseur_ppo.pt.avant_garde_2a32559"
_EMBALLER_IDS = [i for i in range(evolution.ACTION_DIM) if bid_action_to_dict(i)["action"] == "emballer"]


def load_anchor(checkpoint_dir):
    """Reseau d'avant la garde (fige), ou None s'il est absent (ancre coupee)."""
    path = os.path.join(checkpoint_dir, ANCHOR_NAME)
    if not os.path.exists(path):
        return None
    net = evolution.new_net()
    net.load_state_dict(torch.load(path, map_location="cpu")["net"])
    net.eval()
    return net


def emballer_guard_target(bidding, hand, mask, obs=None, anchor=None):
    """-> (tenseur cible (1 = action acceptable), poids) ou (None, 0.0) si la
    decision n'est pas concernee. Garde : emballage d'entree >=
    EMBALLER_GUARD_LEVEL sans les plis de emballer_min -> tout sauf ces
    emballages. Ancre (si `anchor`) : emballage d'entree <= 10 legal -> le
    choix greedy de l'ancre. Voir EMBALLER_GUARD_LEVEL et l'ANCRE ci-dessus."""
    bad = []
    low = False
    for action_id in _EMBALLER_IDS:
        if not mask[action_id]:
            continue
        suit = bid_action_to_dict(action_id)["suit"]
        level = bidding.emballage_entry_level(suit)
        if level >= EMBALLER_GUARD_LEVEL:
            if estimate_tricks(hand, suit) < emballer_min(bidding, suit):
                bad.append(action_id)
        else:
            low = True
    if bad:
        target = mask.clone().float()
        target[bad] = 0.0
        if target.sum() > 0:
            return target, EMBALLER_GUARD_WEIGHT
        return None, 0.0
    if low and anchor is not None:
        with torch.no_grad():
            choice = int(anchor.act_greedy(obs, mask))
        target = torch.zeros(len(mask))
        target[choice] = 1.0
        return target, 1.0
    return None, 0.0


def checkpoint_path(checkpoint_dir):
    return os.path.join(checkpoint_dir, CHECKPOINT_NAME)


def load_or_seed(population, checkpoint_dir):
    """-> (net, optimizer, indiv_id, seeded_from). Reprend le checkpoint PPO
    s'il existe, sinon part du meilleur `encherisseur` de la population."""
    net = evolution.new_net()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    path = checkpoint_path(checkpoint_dir)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        net.load_state_dict(ckpt["net"])
        optimizer.load_state_dict(ckpt["optimizer"])
        return net, optimizer, ckpt["id"], ckpt.get("seeded_from")
    best = evolution.best_per_lineage(population)[evolution.EXPORTED_LINEAGES[0]]
    net.load_state_dict(best.net.state_dict())
    return net, optimizer, best.id, best.id


def save(checkpoint_dir, net, optimizer, indiv_id, seeded_from):
    safe_save({"net": net.state_dict(), "optimizer": optimizer.state_dict(), "id": indiv_id,
               "seeded_from": seeded_from}, checkpoint_path(checkpoint_dir))


def ppo_population(population, net, indiv_id):
    """Population vue par le reste du code : ce reseau comme SEUL individu
    `encherisseur` (meme objet : toujours a jour), + les hybrides figes."""
    ind = evolution.Individual(net, evolution.EXPORTED_LINEAGES[0], indiv_id=indiv_id)
    return [ind] + [i for i in population if i.lineage != evolution.EXPORTED_LINEAGES[0]]


def backup_population_once(population_file, generation):
    backup = "%s.avant_ppo_gen%d" % (population_file, generation)
    if os.path.exists(population_file) and not os.path.exists(backup):
        shutil.copyfile(population_file, backup)
        return backup
    return None


def new_buffer():
    return {"obs": [], "mask": [], "action": [], "logp": [], "ret": [], "adv": [], "target": [], "weight": []}


def record_seat(buffer, steps, final_points):
    """steps : [(obs, mask, action, logp, value_brute, cible, poids)] d'UN siege, dans
    l'ordre ; recompense nulle sauf la derniere (points finaux du siege)."""
    n = len(steps)
    if not n:
        return
    values = [float(step[4]) * RETURN_SCALE for step in steps]
    advantages = [0.0] * n
    running = 0.0
    for t in reversed(range(n)):
        reward = final_points if t == n - 1 else 0.0
        v_next = 0.0 if t == n - 1 else values[t + 1]
        running = reward + v_next - values[t] + GAE_LAMBDA * running
        advantages[t] = running
    for (obs, mask, action, logp, _v, target, weight), adv in zip(steps, advantages):
        buffer["obs"].append(obs)
        buffer["mask"].append(mask)
        buffer["action"].append(action)
        buffer["logp"].append(logp)
        buffer["ret"].append(final_points)
        buffer["adv"].append(adv)
        buffer["target"].append(target)
        buffer["weight"].append(weight)
