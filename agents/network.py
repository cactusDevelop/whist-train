"""Reseau acteur-critique MLP partage, une instance par role specialiste
(voir DESIGN.md — chaque role a ses propres poids, jamais melanges).

Activation Tanh (pas ReLU) : impose par le moteur d'inference GDScript
existant cote Godot (game/whist_ai_policy.gd, deja teste et reutilise tel
quel pour chacun des 7 roles) qui applique tanh() en dur sur les 2 couches
cachees. Utiliser autre chose ici romprait silencieusement l'export --
memes poids, sortie differente."""

import torch
import torch.nn as nn

from engine.observation import ALLY_DIM, OBS_DIM, OBS_DIM_V5

HIDDEN = 256
NEG_INF = -1e9


def _insert_ally_columns(t):
    """Insere ALLY_DIM colonnes nulles a la position OBS_DIM_V5 (bloc P de
    l'observation v6) : un reseau v5 elargi calcule EXACTEMENT la meme chose."""
    pad = torch.zeros(t.shape[0], ALLY_DIM, dtype=t.dtype, device=t.device)
    return torch.cat([t[:, :OBS_DIM_V5], pad, t[:, OBS_DIM_V5:]], dim=1)


# Reseaux de JEU (entree = observation) et evaluateur omniscient (observation
# + 4 mains restantes, agents/oracle_critic.py) ; jamais les reseaux
# d'enchere ou de jokers (autres observations).
_PLAY_INPUTS = (OBS_DIM, OBS_DIM + 4 * 52)


def _needs_ally_widen(old_in, new_in):
    return new_in in _PLAY_INPUTS and old_in == new_in - ALLY_DIM


def obs_widen(state_dict, key, new_in):
    """Migration v5 -> v6 au chargement (premiere couche seulement)."""
    w = state_dict.get(key)
    if w is not None and w.dim() == 2 and _needs_ally_widen(w.shape[1], new_in):
        state_dict[key] = _insert_ally_columns(w)


def widen_adam_state(optimizer):
    """Apres optimizer.load_state_dict d'un checkpoint v5 : elargit les
    moments Adam de la premiere couche (colonnes nulles, comme les poids)."""
    for group in optimizer.param_groups:
        for p in group["params"]:
            st = optimizer.state.get(p, {})
            for k in ("exp_avg", "exp_avg_sq"):
                t = st.get(k)
                if t is not None and t.dim() == 2 and p.dim() == 2 and t.shape[0] == p.shape[0] \
                        and _needs_ally_widen(t.shape[1], p.shape[1]):
                    st[k] = _insert_ally_columns(t)


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
        )
        self.policy_head = nn.Linear(HIDDEN, action_dim)
        self.value_head = nn.Linear(HIDDEN, 1)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        obs_widen(state_dict, prefix + "trunk.0.weight", self.trunk[0].in_features)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, obs):
        z = self.trunk(obs)
        return self.policy_head(z), self.value_head(z).squeeze(-1)

    def act(self, obs, legal_mask):
        """obs: [obs_dim] float tensor. legal_mask: [action_dim] bool tensor.
        Retourne (action, log_prob, value)."""
        logits, value = self.forward(obs.unsqueeze(0))
        logits = logits.squeeze(0)
        logits = logits.masked_fill(~legal_mask, NEG_INF)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze(0)

    def act_greedy_logp(self, obs, legal_mask):
        """Comme act(), mais l'action est l'ARGMAX (comportement exporte) ;
        retourne aussi son log-prob sous la politique courante, pour que PPO
        puisse apprendre de ces decisions (agents/bid_ppo.py:GREEDY_TABLE_PROB)."""
        logits, value = self.forward(obs.unsqueeze(0))
        logits = logits.squeeze(0).masked_fill(~legal_mask, NEG_INF)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits)
        return action.item(), dist.log_prob(action), value.squeeze(0)

    def act_greedy(self, obs, legal_mask):
        """Politique DETERMINISTE (argmax sur les logits legaux) -- c'est le
        comportement REEL une fois exporte vers Godot (game/whist_ai_policy.gd
        applique argmax, jamais d'echantillonnage, voir scripts/verify_export.py),
        donc c'est la mesure fidele du niveau de jeu d'un reseau (utilise par
        scripts/eval_vs_random.py). Pas de grad necessaire (evaluation seule)."""
        with torch.no_grad():
            logits, _value = self.forward(obs.unsqueeze(0))
            logits = logits.squeeze(0).masked_fill(~legal_mask, NEG_INF)
            return int(torch.argmax(logits).item())
