"""Reseau acteur-critique MLP partage, une instance par role specialiste
(voir DESIGN.md — chaque role a ses propres poids, jamais melanges).

Activation Tanh (pas ReLU) : impose par le moteur d'inference GDScript
existant cote Godot (game/whist_ai_policy.gd, deja teste et reutilise tel
quel pour chacun des 7 roles) qui applique tanh() en dur sur les 2 couches
cachees. Utiliser autre chose ici romprait silencieusement l'export --
memes poids, sortie differente."""

import torch
import torch.nn as nn

HIDDEN = 256
NEG_INF = -1e9


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
        )
        self.policy_head = nn.Linear(HIDDEN, action_dim)
        self.value_head = nn.Linear(HIDDEN, 1)

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
