"""Attribution du merite AU PLI pour le jeu de la carte.

Avant (jusqu'au 2026-09-24) : le score final de la donne etait copie tel
quel sur les 13 decisions d'un siege -- une carte jouee au 1er pli etait
creditee des plis gagnes au 12e, bruit enorme, aucun apprentissage fin
(surcoupe du partenaire, gaspillage d'atout...).

Maintenant : on note `potentiels[k]` = la valeur de chaque siege SI la
donne s'arretait apres k plis (MEME fonction de score que la recompense
finale -- plis pour les declarants, points pour la defense, voir
scripts/train.py:compute_scores). Une decision prise pendant le pli k
recoit :
  - retour  G = valeur finale - potentiels[k][siege]  (ce qui s'est
    passe A PARTIR de ce pli ; somme telescopique exacte, rien d'invente)
  - avantage GAE(gamma=1, lambda) sur les recompenses par pli
    r_t = potentiels[k_{t+1}] - potentiels[k_t] (derniere : finale -
    potentiels[k_t]), avec la tete de valeur du reseau comme reference.
La recompense totale d'une donne reste EXACTEMENT celle d'avant.
"""

GAE_LAMBDA = 0.95

# Echelle de la tete de valeur (voir agents/ppo.py). Depuis le 2026-10-02 tous
# les roles sont notes en vrais points en Phase 2 (jusqu'a ~75), plus seulement
# `defenseur` ; seuls solitaire/emballage Phase 1 restent en plis (0..13).
RETURN_SCALE_DEFAULT = 10.0


def return_scale(role):
    return RETURN_SCALE_DEFAULT


def tricks_after_each(play_log_tricks, num_players=4):
    """play_log_tricks : liste des gagnants de chaque pli termine, dans
    l'ordre. -> [plis par siege apres 0, 1, ..., N plis]."""
    counts = [0] * num_players
    out = [list(counts)]
    for w in play_log_tricks:
        counts[w] += 1
        out.append(list(counts))
    return out


def seat_credit(steps, potentials, final, gae_lambda=GAE_LAMBDA):
    """steps : [(trick_index, value_estimee_brute)] des decisions d'UN siege,
    dans l'ordre. potentials : [valeur de ce siege apres k plis]. final :
    valeur finale de ce siege. -> (retours, avantages), un par decision."""
    n = len(steps)
    rewards = []
    for t, (k, _v) in enumerate(steps):
        nxt = final if t == n - 1 else potentials[steps[t + 1][0]]
        rewards.append(nxt - potentials[k])
    returns = [final - potentials[k] for k, _v in steps]
    advantages = [0.0] * n
    running = 0.0
    for t in reversed(range(n)):
        v_next = 0.0 if t == n - 1 else steps[t + 1][1]
        delta = rewards[t] + v_next - steps[t][1]
        running = delta + gae_lambda * running
        advantages[t] = running
    return returns, advantages
