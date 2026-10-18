"""Registre des roles specialistes de JEU DE CARTE (PPO). Chaque role a son
propre reseau + optimizer + checkpoint, jamais partages entre roles.

Phase 1 (2026-09-19, voir DESIGN.md) : reset cible au niveau ROSTER +
BOUCLE D'ENTRAINEMENT (le moteur de regles engine/ reste celui, deja
verifie contre Godot, du reset precedent). 7 roles de jeu de carte, contrat
deja connu (placeholder, pas d'enchere reelle).

Phase 2 (2026-09-20, voir DESIGN.md/TRAINING.md) : ajoute `trou`/
`bouche_trou` a IMPLEMENTED -- memes dims que les 7 roles de jeu (le moteur
`engine/trou.py` reutilise `engine/team_trump.py`, identique a `emballage`).
Leur PREMIERE creation (aucun checkpoint sur disque) ne part PAS d'un init
aleatoire mais d'une COPIE des poids actuels de `emballage` (demande
explicite de l'utilisateur : "tu peux dupliquer le reseau de emballage...
deux fois et laisser Trou et Bouche-Trou se diriger naturellement vers leur
specialite") -- Trou/Bouche-trou partagent la meme structure "declarant +
partenaire force, avec atout, contre 2 defenseurs" que Emballage, donc ce
point de depart est un bien meilleur prior qu'un reseau vierge, pour un
role qui ne s'entrainera qu'organiquement (mains a 3-4 as, rare).

`encherisseur` (phase d'enchere) n'est PLUS un
role PPO a checkpoint unique : ce sont les LIGNEES d'une POPULATION
evolutive (mutation + selection sur score reel, aucun gradient) -- voir
agents/evolution.py, jamais IMPLEMENTED ici. `load_or_create` continue de
lever NotImplementedError pour ces deux noms, pour qu'aucun code PPO ne
puisse les charger par accident."""

import os

import torch

from agents.checkpoint_io import safe_save
from agents.network import PolicyValueNet
from engine.cards import NUM_CARDS
from engine.observation import OBS_DIM as PLAY_OBS_DIM

CHECKPOINT_DIR = "checkpoints_v2"

# Roles de jeu de carte Phase 1 (contrat connu, jamais d'enchere) :
# - solitaire   : solo/abondance/solo chelem, seul, maximise SES plis.
# - emballage   : declarant+partenaire (meme reseau), maximisent les plis
#                 de LEUR camp.
# - piccolo     : doit prendre EXACTEMENT 1 pli (ni plus, ni moins).
# - petite_misere : ecarte 1 carte puis doit prendre 0 pli.
# - grande_misere : doit prendre 0 pli (13 cartes, pas d'ecart).
# - misere_etalee : doit prendre 0 pli, main revelee a tous des le debut.
# - defenseur   : LE SEUL role a affronter toutes les familles ci-dessus
#                 (jamais deux roles de declarant l'un contre l'autre --
#                 structurellement vrai, une donne ne verrouille qu'UNE
#                 famille) ; maximise les plis de SON camp de defense (2
#                 adversaires en emballage/trou, 3 partout ailleurs).
PHASE1_ROLES = [
    "solitaire",
    "emballage",
    "piccolo",
    "petite_misere",
    "grande_misere",
    "misere_etalee",
    "defenseur",
]

# Roles de jeu de carte ajoutes en Phase 2 -- n'apparaissent QUE dans une
# vraie donne (bidding reel), jamais dans la branche Phase 1 forcee (Trou
# n'a pas de sens sans enchere reelle, voir engine/trou.py:detect_trou).
PHASE2_EXTRA_PLAY_ROLES = ["trou", "bouche_trou"]

# Tous les roles de jeu de carte a checkpoint PPO -- utilise par
# scripts/train.py pour charger/sauvegarder le roster complet de jeu.
PLAY_ROLES = PHASE1_ROLES + PHASE2_EXTRA_PLAY_ROLES

# role de jeu -> role source dont copier les poids a la toute premiere
# creation (aucun fichier .pt existant), au lieu d'un init aleatoire.
WARM_START_FROM = {"trou": "emballage", "bouche_trou": "emballage"}

# Roles d'enchere : population evolutive (agents/evolution.py), jamais un
# checkpoint PPO unique -- voir docstring de module. Listes ici uniquement
# pour la documentation du roster complet du projet (ROLES) et pour que
# load_or_create continue de les refuser explicitement.
EVOLUTION_ROLES = ["encherisseur"]

ROLES = PLAY_ROLES + EVOLUTION_ROLES

PLAY_ACTION_DIM = NUM_CARDS
IMPLEMENTED = {role: (PLAY_OBS_DIM, PLAY_ACTION_DIM) for role in PLAY_ROLES}


def checkpoint_path(role):
    return os.path.join(CHECKPOINT_DIR, "%s.pt" % role)


def load_or_create(role):
    if role not in IMPLEMENTED:
        if role in EVOLUTION_ROLES:
            raise NotImplementedError(
                "role '%s' n'est pas un role PPO -- c'est une lignee de la population "
                "evolutive geree par agents/evolution.py, jamais un checkpoint unique "
                "charge via specialists.load_or_create" % role
            )
        raise NotImplementedError(
            "role '%s' pas encore implemente (moteur de jeu manquant, voir DESIGN.md)" % role
        )
    obs_dim, action_dim = IMPLEMENTED[role]
    net = PolicyValueNet(obs_dim, action_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    path = checkpoint_path(role)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        net.load_state_dict(ckpt["net"])
        optimizer.load_state_dict(ckpt["optimizer"])
    elif role in WARM_START_FROM:
        source_role = WARM_START_FROM[role]
        source_path = checkpoint_path(source_role)
        if os.path.exists(source_path):
            source_ckpt = torch.load(source_path, map_location="cpu")
            net.load_state_dict(source_ckpt["net"])
            # optimizer JAMAIS copie depuis la source : cette lignee poursuit
            # un entrainement avec un objectif propre, un optimizer neuf
            # (momentum Adam vierge) est le comportement correct.
    return net, optimizer


def save(role, net, optimizer):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    safe_save({"net": net.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint_path(role))
