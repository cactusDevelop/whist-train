"""Population evolutive pour le role d'enchere (`encherisseur`, plus la
lignee hybride `encherisseur_expert`) -- Phase 2, voir DESIGN.md/TRAINING.md. PAS de PPO
ici : aucun gradient, aucun optimizer. C'est un algorithme genetique
classique (mutation gaussienne + selection elitiste sur fitness reelle),
demande explicite de l'utilisateur ("un vrai entrainement d'IA classique") --
contraste volontaire avec le reste du projet (100% PPO).

Deux LIGNEES de depart (`LINEAGES`), pas deux reseaux figes : chaque
lignee engendre une grande population d'enfants par mutation, tous
s'affrontent entre eux (melanges, pas cloisonnes par lignee) dans de
vraies donnes Phase 2, et les meilleurs (fitness reelle moyenne) survivent
a chaque generation.

**2026-09-27 : lignee `encherisseur_amical` SUPPRIMEE** (decision
utilisateur). Sa seule difference etait un faconnage de fitness (+15 par
proposition/emballage, meme si le contrat chutait) qui masquait ses
propres contrats a -41 points en moyenne ; elle a aussi ete retiree de
Godot. `encherisseur` a ete re-amorcee par distillation de l'enchere de
l'ExpertBot (scripts/reseed_encherisseur.py) : bloquee a -35 par donne
(solo-13 a 0 % de reussite), elle ne progressait plus malgre la garantie
de survie ci-dessous.

**Correction 2026-09-20 (le jour meme, apres un premier entrainement
reel)** : `select_elites` GARANTIT desormais un survivant par lignee a
CHAQUE generation (le meilleur individu de chaque lignee, meme si son
classement global seul ne l'aurait pas qualifie) -- demande explicite et
directe de l'utilisateur apres avoir observe `encherisseur` s'eteindre
completement des la generation 5 d'un premier run de 15 minutes ("il faut
garder le meilleur des deux à chaque fois, pour pas que encherisseur
disparaisse"). Un run PRECEDENT a cette correction peut avoir sauvegarde
une population ou une lignee est deja totalement absente -- voir
`repair_missing_lineages`, appele au rechargement (scripts/train.py)."""

import itertools

import torch

from agents.checkpoint_io import safe_save
from agents.network import PolicyValueNet
from engine.bidding import ACTION_RAISE, NUM_BID_ACTIONS, NUM_BID_ACTIONS_V1, RAISE_TO_LEVELS
from engine.bidding import OBS_DIM as BID_OBS_DIM

OBS_DIM = BID_OBS_DIM
ACTION_DIM = NUM_BID_ACTIONS

# 3e lignee (2026-09-27, demande utilisateur) : HYBRIDE reseau + regles de
# l'ExpertBot (agents/hybrid_bidder.py) -- le reseau evolue comme les autres,
# la couche de regles est fixe. Pas encore exportee vers Godot (la couche de
# regles doit d'abord y etre portee avec une parite testee) : les exports,
# la Phase 3 et les mesures n'utilisent que EXPORTED_LINEAGES.
HYBRID_LINEAGE = "encherisseur_expert"
LINEAGES = ("encherisseur", HYBRID_LINEAGE)
EXPORTED_LINEAGES = ("encherisseur",)

POP_SIZE_DEFAULT = 40
ELITE_COUNT_DEFAULT = 8
MUTATION_SIGMA_DEFAULT = 0.02  # 0.05 jusqu'au 2026-09-27 : trop fort pour un reseau deja distille

# --- Selection sous bruit (2026-09-28, demande utilisateur "corrige la selection
# de l'enchere") ------------------------------------------------------------
# Constat de la nuit du 27 au 28/09 : la fitness d'UNE generation (200 donnes
# communes) est surtout du bruit -- 12 individus notes sur 2 jeux de 200 donnes :
# correlation 0.17 ; seules 21 % des elites le restaient a la generation
# suivante, le n°1 a change 155 fois en 166 generations. Chaque generation, un
# enfant chanceux remplacait une elite -> les mutations s'enchainaient au hasard
# (marche aleatoire) : abondances d'entree sur mauvaises mains, sauts du passe
# parole a 12/13 (lignes de sortie rarement jouees qui derivent sans pression).
# Correction, sans aucune regle de jeu :
#  1. MEMOIRE : chaque individu cumule ses scores de duplicate sur toutes les
#     generations (plafond EVIDENCE_CAP_DEALS : l'environnement de jeu evolue,
#     une vieille preuve s'efface) ;
#  2. CONFIRMATION : le score qui sert a PRESELECTIONNER un enfant est biaise
#     (on garde le plus chanceux) -> jamais compte ; les meilleurs enfants
#     (CHALLENGERS) sont renotes avec les elites sur des donnes NEUVES et seul
#     ce score independant entre dans leur memoire ;
#  3. CLASSEMENT PRUDENT : borne basse moyenne - SELECTION_Z x erreur-type ->
#     une elite eprouvee n'est remplacee que par un enfant nettement meilleur.
EVIDENCE_CAP_DEALS = 1000
SELECTION_Z = 0.5  # simulation 2026-09-28 (bruit mesure, mutations nuisibles rares) : Z=0 +11.6, 0.5 +15.7, 1.0 +5.5, ancienne selection +14.2
CHALLENGERS_DEFAULT = 4
POOLED_SD_FALLBACK = 10.0

_id_counter = itertools.count()


class Individual:
    """Un membre de la population : un reseau + sa lignee d'origine + les
    compteurs de fitness de la generation EN COURS (remis a zero a chaque
    nouvelle generation, voir spawn_generation)."""

    def __init__(self, net, lineage, indiv_id=None, last_known_fitness=float("-inf")):
        if lineage not in LINEAGES:
            raise ValueError("lignee inconnue: %r (attendu %r)" % (lineage, LINEAGES))
        self.net = net
        self.lineage = lineage
        self.id = indiv_id if indiv_id is not None else next(_id_counter)
        self.fitness_sum = 0.0
        self.fitness_sumsq = 0.0
        self.games = 0
        # memoire sur plusieurs generations (voir "Selection sous bruit")
        self.life_n = 0.0
        self.life_sum = 0.0
        self.life_sumsq = 0.0
        # Derniere fitness CONNUE (generation precedente, avant la remise a
        # zero de fitness_sum/games pour le tournoi EN COURS) -- jamais
        # utilisee pour la selection elle-meme (select_elites reste sur
        # mean_fitness(), la fitness DE CETTE generation), seulement pour
        # savoir "qui exporter" avant que le tournoi de la nouvelle
        # generation n'ait ete rejoue (voir snapshot_fitness/
        # best_per_lineage) -- notamment juste apres un load_population().
        self.last_known_fitness = last_known_fitness

    def record_game(self, score):
        self.fitness_sum += score
        self.fitness_sumsq += score * score
        self.games += 1

    def reset_generation(self):
        self.fitness_sum = 0.0
        self.fitness_sumsq = 0.0
        self.games = 0

    def bank_generation(self):
        """Verse les scores du tournoi en cours dans la memoire (a n'appeler
        que pour un score NON biaise par la selection, voir CONFIRMATION)."""
        self.life_n += self.games
        self.life_sum += self.fitness_sum
        self.life_sumsq += self.fitness_sumsq
        if self.life_n > EVIDENCE_CAP_DEALS:
            f = EVIDENCE_CAP_DEALS / self.life_n
            self.life_n, self.life_sum, self.life_sumsq = EVIDENCE_CAP_DEALS, self.life_sum * f, self.life_sumsq * f

    def life_mean(self):
        return self.life_sum / self.life_n if self.life_n else float("-inf")

    def selection_score(self, pooled_sd):
        if not self.life_n:
            return float("-inf")
        return self.life_mean() - SELECTION_Z * pooled_sd / self.life_n ** 0.5

    def mean_fitness(self):
        return self.fitness_sum / self.games if self.games else float("-inf")

    def snapshot_fitness(self):
        """Fitness a utiliser pour un CONSTAT (export, affichage) : la
        fitness EN COURS si ce tournoi a deja joue des parties, sinon la
        derniere connue -- jamais -inf pour un individu qui a deja une
        histoire, meme juste apres un rechargement depuis disque."""
        if self.life_n:
            return self.life_mean()
        return self.mean_fitness() if self.games else self.last_known_fitness


def pooled_sd(population):
    """Ecart-type d'un score de donne, mis en commun sur les memoires."""
    n = s = ss = 0.0
    k = 0
    for ind in population:
        if ind.life_n >= 2:
            n += ind.life_n
            ss += ind.life_sumsq - ind.life_sum ** 2 / ind.life_n
            k += 1
    if n - k <= 0:
        return POOLED_SD_FALLBACK
    return max(ss / (n - k), 0.0) ** 0.5 or POOLED_SD_FALLBACK


def selection_key(population):
    sd = pooled_sd(population)
    return lambda ind: ind.selection_score(sd)


def new_net():
    return PolicyValueNet(OBS_DIM, ACTION_DIM)


def policy_of(individual):
    """Ce qui decide pour un individu a l'enchere : le reseau seul, ou
    reseau + regles pour la lignee hybride (agents/hybrid_bidder.py), reseau +
    regle d'abondance pour la lignee exportee (agents/abondance_rule.py)."""
    if individual.lineage == HYBRID_LINEAGE:
        from agents.hybrid_bidder import HybridBidder
        return HybridBidder(individual.net)
    if individual.lineage in EXPORTED_LINEAGES:  # reseau + regle d'abondance, comme dans le jeu
        from agents.abondance_rule import RuledBidder
        return RuledBidder(individual.net)
    return individual.net


# Ecart de logit des nouvelles sorties "saut" par rapport a "relance +1" : un
# reseau migre prefere toujours +1 au saut (comportement quasi inchange) mais
# la mutation peut faire emerger le saut.
JUMP_BIAS_OFFSET = 2.0


def widen_bid_state_dict(state_dict):
    """Migration SANS PERTE d'un reseau d'enchere a 52 sorties (avant le saut
    du passe parole) vers 56 : sorties 0..51 identiques, nouvelles sorties
    52+i (saut au niveau 10+i) = copie de la ligne ACTION_RAISE, biais
    abaisse de JUMP_BIAS_OFFSET*(1+i). Idempotent. -> (state_dict, migre?)"""
    w = state_dict["policy_head.weight"]
    if w.shape[0] == NUM_BID_ACTIONS:
        return state_dict, False
    assert w.shape[0] == NUM_BID_ACTIONS_V1, tuple(w.shape)
    b = state_dict["policy_head.bias"]
    n_new = len(RAISE_TO_LEVELS)
    new_w = w[ACTION_RAISE].unsqueeze(0).repeat(n_new, 1)
    new_b = torch.stack([b[ACTION_RAISE] - JUMP_BIAS_OFFSET * (1 + i) for i in range(n_new)])
    out = dict(state_dict)
    out["policy_head.weight"] = torch.cat([w, new_w], dim=0)
    out["policy_head.bias"] = torch.cat([b, new_b], dim=0)
    return out, True


def clone_net(net):
    clone = new_net()
    clone.load_state_dict(net.state_dict())
    return clone


def mutate_net(net, sigma):
    """Enfant = copie du parent + bruit gaussien independant sur CHAQUE
    parametre (poids ET biais des 2 couches trunk + tete politique/valeur).
    sigma=0 produirait un clone bit-a-bit identique -- jamais utilise en
    pratique (sigma > 0 toujours, voir POP_SIZE_DEFAULT/train.py CLI)."""
    child = clone_net(net)
    with torch.no_grad():
        for param in child.parameters():
            param.add_(torch.randn_like(param) * sigma)
    return child


def seed_population(seed_nets, pop_size, sigma):
    """Genere la toute premiere generation (generation 0) a partir des 2
    reseaux-graines (un par lignee, voir LINEAGES) : chaque graine survit
    elle-meme intacte comme individu (garantit qu'une lignee jamais mutee
    reste evaluable des la generation 0) et engendre le reste de son quota
    par mutation. `seed_nets` : dict lignee -> PolicyValueNet."""
    if set(seed_nets) != set(LINEAGES):
        raise ValueError("seed_nets doit avoir exactement les lignees %r, recu %r" % (LINEAGES, set(seed_nets)))
    elites = [(seed_nets[lineage], lineage) for lineage in LINEAGES]
    return spawn_generation(elites, pop_size, sigma)


def select_elites(population, elite_count, key=None):
    """Classement PAR fitness moyenne decroissante de cette generation,
    MAIS garantit toujours au moins 1 elite par lignee (le meilleur
    individu de CHAQUE lignee presente dans `population`, meme si son
    classement global seul ne l'aurait pas qualifie) -- voir la correction
    du 2026-09-20 dans la docstring de module : une lignee ne doit plus
    JAMAIS pouvoir s'eteindre tant qu'elle a au moins un individu dans la
    population passee en entree. Les slots restants sont remplis par le
    reste du classement global, toutes lignees confondues. Un individu qui
    n'a joue aucune partie cette generation (games=0) est trie en dernier
    (mean_fitness() = -inf). `key` (optionnel) : critere de classement a la
    place de mean_fitness() -- selection_key(population) depuis 2026-09-28."""
    key = key or (lambda ind: ind.mean_fitness())
    if elite_count < len(LINEAGES):
        raise ValueError(
            "elite_count (%d) doit etre >= au nombre de lignees (%d) pour garantir leur survie" % (
                elite_count, len(LINEAGES)))

    guaranteed = []
    guaranteed_ids = set()
    for lineage in LINEAGES:
        candidates = [ind for ind in population if ind.lineage == lineage]
        if not candidates:
            continue  # lignee deja absente de CETTE population, voir repair_missing_lineages
        best = max(candidates, key=key)
        guaranteed.append(best)
        guaranteed_ids.add(best.id)

    ranked = sorted(population, key=key, reverse=True)
    rest = [ind for ind in ranked if ind.id not in guaranteed_ids]
    return guaranteed + rest[:elite_count - len(guaranteed)]


def repair_missing_lineages(population):
    """Filet de securite pour une population sauvegardee AVANT la
    correction du 2026-09-20 (voir docstring de module), ou une lignee
    peut etre totalement absente : reinjecte un individu NEUF (poids
    vierges) pour chaque lignee manquante, a la place du PIRE individu de
    la population (jamais un ajout -- la taille de la population ne
    change pas). Retourne la liste des lignees reparees (vide si rien a
    faire -- le cas normal une fois select_elites corrige)."""
    present = {ind.lineage for ind in population}
    missing = [lineage for lineage in LINEAGES if lineage not in present]
    if not missing:
        return []
    worst_first = sorted(population, key=lambda ind: ind.snapshot_fitness())
    for lineage, victim in zip(missing, worst_first):
        population[population.index(victim)] = Individual(new_net(), lineage)
    return missing


def spawn_generation(elites, pop_size, sigma):
    """Nouvelle generation de taille EXACTE pop_size : chaque elite passe
    inchangee (fitness remise a zero -- nouvelle generation, nouveau
    tournoi), le reste de la population est reparti en enfants mutes des
    elites en tournant (round-robin), pour repartir la reproduction
    equitablement entre elites plutot que de sur-representer la premiere.
    `elites` : liste de (net, lineage) OU de Individual (les deux formes
    sont acceptees, pour reutiliser directement la sortie de
    select_elites() ou du premier appel de seed_population()). Un elite
    passe en Individual garde son id EXACT (meme identite d'une generation
    a l'autre, voir Individual.id) ; un elite passe en tuple (net, lineage)
    -- generation 0 uniquement, aucune identite prealable -- recoit un id
    neuf."""
    seeds = [
        (e.net, e.lineage, e.id, e.snapshot_fitness()) if isinstance(e, Individual) else (e[0], e[1], None, float("-inf"))
        for e in elites
    ]
    memories = [(e.life_n, e.life_sum, e.life_sumsq) if isinstance(e, Individual) else (0.0, 0.0, 0.0)
                for e in elites]
    if not seeds:
        raise ValueError("spawn_generation necessite au moins un elite")
    if pop_size < len(seeds):
        raise ValueError("pop_size (%d) < nombre d'elites (%d)" % (pop_size, len(seeds)))

    population = [
        Individual(clone_net(net), lineage, indiv_id=indiv_id, last_known_fitness=fitness)
        for net, lineage, indiv_id, fitness in seeds
    ]
    for ind, (n, s, ss) in zip(population, memories):  # une elite garde sa memoire
        ind.life_n, ind.life_sum, ind.life_sumsq = n, s, ss
    n_children = pop_size - len(seeds)
    for i in range(n_children):
        parent_net, parent_lineage, _parent_id, _parent_fitness = seeds[i % len(seeds)]
        child_net = mutate_net(parent_net, sigma)
        population.append(Individual(child_net, parent_lineage))
    return population


def best_per_lineage(population):
    """Meilleur individu par lignee, base sur snapshot_fitness() (la
    derniere fitness CONNUE, pas seulement celle du tournoi en cours --
    voir Individual.snapshot_fitness) -- utilise pour l'export Godot (2
    slots seulement, voir scripts/export_to_godot.py), y compris juste
    apres un load_population() ou aucun tournoi n'a encore ete rejoue.
    Depuis la correction du 2026-09-20 (select_elites garantit un
    survivant par lignee), une lignee absente de `population` ne devrait
    plus arriver qu'en rechargeant un fichier ANTERIEUR a cette
    correction (voir repair_missing_lineages) -- dans ce cas, cette
    lignee est absente du dict retourne, l'appelant doit gerer ce cas
    explicitement plutot que de supposer les 2 lignees toujours
    presentes."""
    best = {}
    if any(ind.life_n for ind in population):  # depuis 2026-09-28 : meme critere que la selection
        key = selection_key(population)
    else:
        key = lambda ind: ind.snapshot_fitness()
    for ind in population:
        current = best.get(ind.lineage)
        if current is None or key(ind) > key(current):
            best[ind.lineage] = ind
    return best


def save_population(path, population, generation):
    safe_save({
        "generation": generation,
        "individuals": [
            {
                "state_dict": ind.net.state_dict(),
                "lineage": ind.lineage,
                "id": ind.id,
                "mean_fitness": ind.snapshot_fitness(),
                "life": [ind.life_n, ind.life_sum, ind.life_sumsq],
            }
            for ind in population
        ],
    }, path)


def load_population(path):
    """Recharge une population sauvegardee -- la fitness DU TOURNOI N'EST
    PAS reportee (fitness_sum/games remis a zero, un individu recharge
    doit rejouer un tournoi complet avant que mean_fitness() ne redevienne
    valide) ; la DERNIERE fitness connue de la sauvegarde est gardee dans
    Individual.last_known_fitness (voir snapshot_fitness -- c'est ce que
    best_per_lineage/l'export utilisent, meme juste apres un rechargement,
    voir Individual.id, jamais recalcule -- garde la meme identite d'un
    run a l'autre)."""
    ckpt = torch.load(path, map_location="cpu")
    population = []
    max_id = -1
    dropped = {}
    for entry in ckpt["individuals"]:
        max_id = max(max_id, entry["id"])
        if entry["lineage"] not in LINEAGES:  # lignee supprimee (encherisseur_amical, 2026-09-27)
            dropped[entry["lineage"]] = dropped.get(entry["lineage"], 0) + 1
            continue
        net = new_net()
        state_dict, _migrated = widen_bid_state_dict(entry["state_dict"])
        net.load_state_dict(state_dict)
        ind = Individual(net, entry["lineage"], indiv_id=entry["id"],
                          last_known_fitness=entry["mean_fitness"])
        ind.life_n, ind.life_sum, ind.life_sumsq = entry.get("life", (0.0, 0.0, 0.0))
        population.append(ind)
    if dropped:
        print("  population %s : individus de lignees supprimees ignores : %r" % (path, dropped), flush=True)
    global _id_counter
    _id_counter = itertools.count(max_id + 1)
    return population, ckpt["generation"]


def select_under_noise(population, evaluate, elite_count, challengers=CHALLENGERS_DEFAULT):
    """Selection en deux passes (voir "Selection sous bruit" en tete de module).
    `evaluate(individus, passe)` joue le duplicate sur des donnes communes
    (passe 0 : tout le monde ; passe 1 : donnes NEUVES) et appelle
    record_game. -> elites (select_elites avec selection_key).
    Passe 0 : les veterans (memoire non vide = elites deja retenues) versent
    leur score (non biaise : pas choisis dessus) ; il sert aussi a
    PRESELECTIONNER les meilleurs enfants, score qui n'entre pas en memoire.
    Passe 1 : veterans + challengers renotes sur des donnes neuves -> memoire."""
    for ind in population:
        ind.reset_generation()
    evaluate(population, 0)
    veterans = [ind for ind in population if ind.life_n]
    newcomers = sorted((ind for ind in population if not ind.life_n), key=lambda i: i.mean_fitness(), reverse=True)
    n_new = challengers if veterans else elite_count + challengers  # demarrage sans memoire
    for lineage in LINEAGES:  # au moins un candidat par lignee
        if not any(i.lineage == lineage for i in veterans + newcomers[:n_new]):
            n_new += 1
            first = next((i for i in newcomers if i.lineage == lineage), None)
            if first is not None:
                newcomers.remove(first)
                newcomers.insert(0, first)
    for ind in veterans:
        ind.bank_generation()
    confirm = veterans + newcomers[:n_new]
    for ind in confirm:
        ind.reset_generation()
    evaluate(confirm, 1)
    for ind in confirm:
        ind.bank_generation()
    return select_elites(confirm, elite_count, key=selection_key(population))
