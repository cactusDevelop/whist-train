"""Evaluateur OMNISCIENT du jeu de la carte (2026-10-25, demande utilisateur :
« une IA qui connait toutes les cartes »).

Pourquoi : l'avantage PPO d'une decision = ce qui s'est passe ensuite moins
ce que la tete de valeur ATTENDAIT. Cette tete ne voit que la main du joueur,
donc elle confond « belle donne » et « bon coup » : la chance des cartes
cachees reste entiere dans l'avantage et noie les decisions a ±1 pli
(couper en renonce, tirer les atouts...). Ici, un reseau a part voit
l'observation du joueur PLUS les 4 mains restantes : il sait ce que la donne
vaut vraiment, l'avantage ne garde presque que l'effet du coup.

Seulement a l'entrainement (acteur-critique asymetrique) : la politique
exportee vers Godot reste aveugle et inchangee (meme reseau, memes
entrees) ; ces poids ne sont jamais exportes.

Un evaluateur par role (memes roles que agents/specialists.py), predit le
retour au pli divise par credit.return_scale, comme la tete de valeur
(agents/ppo.py). Tant qu'il n'est pas chaud (lots d'apprentissage ET perte
moyenne sous celle de la tete de valeur aveugle), train.py garde la tete de
valeur de l'acteur (scripts/train.py:oracle_in_gae).
"""

import json
import os

import torch
import torch.nn as nn

from agents.checkpoint_io import safe_save
from engine.cards import NUM_CARDS, NUM_PLAYERS
from engine.observation import OBS_DIM

ORACLE_DIM = NUM_PLAYERS * NUM_CARDS  # mains restantes, siege relatif x carte
HIDDEN = 256
LR = 3e-4
EPOCHS = 4
MINIBATCH_SIZE = 512
STATE_FILE = "oracle_state.json"  # {"lots": {role: n}, "ema": {role: [perte aveugle, perte omnisciente]}}


class OracleCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim + ORACLE_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, x):
        return self.body(x).squeeze(-1)


def oracle_features(game, seat):
    """4 x 52 : cartes encore en main de chaque siege, relatif au joueur
    (0 = lui-meme). Les ecarts caches de la petite misere sont hors jeu."""
    f = torch.zeros(ORACLE_DIM)
    for s in range(NUM_PLAYERS):
        hand = game.hands[s]
        if hand:
            f[((s - seat) % NUM_PLAYERS) * NUM_CARDS + torch.tensor(hand)] = 1.0
    return f


def path(checkpoint_dir, role):
    return os.path.join(checkpoint_dir, "oracle_%s.pt" % role)


def load_or_create(checkpoint_dir, role):
    critic = OracleCritic()
    optimizer = torch.optim.Adam(critic.parameters(), lr=LR)
    p = path(checkpoint_dir, role)
    if os.path.exists(p):
        ckpt = torch.load(p, map_location="cpu")
        critic.load_state_dict(ckpt["net"])
        optimizer.load_state_dict(ckpt["optimizer"])
    return critic, optimizer


def save(checkpoint_dir, role, critic, optimizer):
    os.makedirs(checkpoint_dir, exist_ok=True)
    safe_save({"net": critic.state_dict(), "optimizer": optimizer.state_dict()}, path(checkpoint_dir, role))


def load_state(checkpoint_dir):
    p = os.path.join(checkpoint_dir, STATE_FILE)
    if os.path.exists(p):
        with open(p) as fh:
            state = json.load(fh)
        state.setdefault("lots", {})
        state.setdefault("ema", {})
        return state
    return {"lots": {}, "ema": {}}


def save_state(checkpoint_dir, state):
    p = os.path.join(checkpoint_dir, STATE_FILE)
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh)
    os.replace(tmp, p)


def value(critic, x, scale):
    with torch.no_grad():
        return critic(x.unsqueeze(0)).item() * scale


def update(critic, optimizer, inputs, returns, scale, epochs=EPOCHS, minibatch_size=MINIBATCH_SIZE):
    """Regression MSE sur le retour au pli / scale. -> perte moyenne de la
    derniere epoque."""
    x = torch.stack(inputs)
    y = torch.tensor(returns, dtype=torch.float32) / scale
    n = x.shape[0]
    total, count = 0.0, 0
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, minibatch_size):
            idx = perm[start:start + minibatch_size]
            loss = ((critic(x[idx]) - y[idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            optimizer.step()
            if epoch == epochs - 1:
                total += loss.item() * len(idx)
                count += len(idx)
    return total / max(count, 1)
