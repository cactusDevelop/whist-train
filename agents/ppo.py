"""Mise a jour PPO du jeu de la carte : avantages GAE et retours au pli
calcules en amont (agents/credit.py), ici seulement l'optimisation --
mini-lots melanges, plusieurs epoques, clipping du ratio ET du gradient.

La tete de valeur predit le retour DIVISE par `return_scale` (1 pour les
roles recompenses en plis, ~10 pour `defenseur` recompense en points, voir
agents/credit.py:RETURN_SCALE) : garde la perte de valeur a une echelle
comparable pour tous les roles sans toucher a la politique (seule la
politique est exportee vers Godot).

Cibles des rollouts (agents/rollouts.py, 2026-09-26) : pour les decisions
analysees carte par carte ou une carte est nettement moins bonne que les
autres, perte -log(probabilite totale des meilleures cartes) ajoutee a PPO
(poids `distill_coef`, moyennee sur les seules decisions qui ont une
cible). Nulle si le reseau joue deja une des meilleures cartes."""

import torch

CLIP_EPS = 0.2
EPOCHS = 4
MINIBATCH_SIZE = 512
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
DISTILL_COEF = 1.0


def update(net, optimizer, obs_batch, action_mask_batch, action_batch, old_logp_batch,
           return_batch, advantage_batch, return_scale=1.0, epochs=EPOCHS, minibatch_size=MINIBATCH_SIZE,
           targets=None, distill_coef=DISTILL_COEF, target_weights=None):
    obs = torch.stack(obs_batch)
    masks = torch.stack(action_mask_batch)
    actions = torch.tensor(action_batch, dtype=torch.long)
    old_logp = torch.stack(old_logp_batch).detach()
    value_targets = torch.tensor(return_batch, dtype=torch.float32) / return_scale
    advantages = torch.tensor(advantage_batch, dtype=torch.float32)
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    n_targets = sum(1 for t in targets if t is not None) if targets else 0
    if n_targets:
        policy_targets = torch.stack([torch.zeros(masks.shape[1]) if t is None else t for t in targets])
        has_target = torch.tensor([t is not None for t in targets], dtype=torch.float32)
        if target_weights is not None:  # moyenne ponderee de la perte d'imitation (enchere, garde + ancre)
            has_target = has_target * torch.tensor(target_weights, dtype=torch.float32)

    n = obs.shape[0]
    stats = {}
    distill_value = 0.0
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, minibatch_size):
            idx = perm[start:start + minibatch_size]
            logits, values = net(obs[idx])
            logits = logits.masked_fill(~masks[idx], -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(actions[idx])
            ratio = torch.exp(logp - old_logp[idx])
            adv = advantages[idx]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((values - value_targets[idx]) ** 2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
            if n_targets:
                w = has_target[idx]
                if w.sum() > 0:
                    log_probs = torch.log_softmax(logits, dim=-1)
                    in_best = (policy_targets[idx] > 0) & masks[idx]
                    in_best[w == 0] = masks[idx][w == 0]  # sans cible : ensemble = tout, perte nulle
                    nll = -torch.logsumexp(log_probs.masked_fill(~in_best, -1e9), dim=-1)
                    distill_loss = (nll * w).sum() / w.sum()
                    loss = loss + distill_coef * distill_loss
                    distill_value = distill_loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            stats = {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy.item(),
                "mean_return": (value_targets.mean() * return_scale).item(),
                "targets": n_targets,
                "distill_loss": distill_value,
            }
    return stats
