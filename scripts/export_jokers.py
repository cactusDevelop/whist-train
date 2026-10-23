"""Export de l'IA jokers vers Godot (voir DESIGN.md, « IA jokers du
roguelike ») :

  godot_export/whist_ai_weights_data_v2_jokers.gd  -- meme format que les
      autres roles v2 (OBS_DIM, HIDDEN, ACTION_DIM, BASE64_WEIGHTS fp16,
      tanh-MLP) + JOKER_SPEC_HASH + CASTABLE (ordre des actions 1..K) ;
  godot_export/joker_parity_cases.json  -- etats reels de matchs simules
      (decisions de bots a jokers) : entrees brutes, observation attendue,
      masque, logits fp16 et action argmax attendue ;
  godot_export/joker_obs_spec.md  -- la spec de l'observation.

--godot-sync copie le .gd dans whist-4-it/game/ (prevenir la session
whist-4-it AVANT, voir la memoire « warn whist-4-it before export »).

Usage : venv/Scripts/python.exe scripts/export_jokers.py [--cases 60] [--godot-sync]
"""

import argparse
import base64
import json
import os
import random
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from agents.network import HIDDEN, PolicyValueNet
from engine import joker_match as jm
from engine import jokers as J
from engine import joker_obs as O
from scripts import joker_profit as P
from scripts.export_to_godot import fp16_rounded
from scripts.godot_paths import find_godot_project
from scripts.train_jokers import CHECKPOINT, logits_np, net_to_numpy

OUT_DIR = "godot_export"
GD_NAME = "whist_ai_weights_data_v2_jokers.gd"
EXPERT_CASES = "joker_expert_cases.json"
GODOT_PROJECT = find_godot_project()  # WHIST_4_IT_DIR ou dossier voisin
GODOT_GAME_DIR = os.path.join(GODOT_PROJECT or os.path.join("..", "whist-4-it"), "game")

TEMPLATE = '''class_name WhistAIWeightsDataV2Jokers
extends RefCounted
## IA jokers du roguelike (decision de cast d'un bot elite/boss a son tour
## de pli), exportee par whist-ai/scripts/export_jokers.py -- NE JAMAIS
## MODIFIER A LA MAIN. Observation : whist-ai/godot_export/joker_obs_spec.md.
## Action 0 = aucun cast, 1..K = CASTABLE[i-1], argmax des logits masques
## (masque : RogueMatch._bot_turn -- type de bot, play_error == "", pas
## bloque). Decideur HYBRIDE (whist-ai agents/joker_expert.py, version
## JOKER_EXPERT_VERSION, fixtures godot_export/joker_expert_cases.json) : les
## regles tranchent, ce reseau ne departage que les cas OUVERTS. Si
## JOKER_SPEC_HASH != tools/joker_spec.mjs : le jeu repasse sur bot_wants (reseau perime).

const JOKER_SPEC_HASH := "{spec_hash}"
const JOKER_EXPERT_VERSION := {expert_version}
const CASTABLE := {castable}
const OBS_DIM := {obs_dim}
const OBS_VERSION := {obs_version}
const HIDDEN := {hidden}
const ACTION_DIM := {action_dim}

const BASE64_WEIGHTS := "{weights}"
'''


def load_net():
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if ckpt["spec_hash"] not in J.RULE_EQUIVALENT_HASHES or ckpt["castable"] != J.CASTABLE or ckpt["obs_dim"] != O.OBS_DIM:
        raise SystemExit("checkpoint entraine sur d'autres regles/obs : %s %s %s" % (
            ckpt["spec_hash"], ckpt["castable"], ckpt["obs_dim"]))
    net = PolicyValueNet(O.OBS_DIM, J.ACTION_DIM)
    net.load_state_dict(ckpt["net"])
    net.eval()
    return net, ckpt


def write_weights(net, path):
    blob = b"".join(t.detach().numpy().astype("<f2").tobytes() for t in (
        net.trunk[0].weight, net.trunk[0].bias, net.trunk[2].weight, net.trunk[2].bias,
        net.policy_head.weight, net.policy_head.bias))
    from agents.joker_expert import JOKER_EXPERT_VERSION
    src = TEMPLATE.format(spec_hash=J.JOKER_SPEC_HASH, expert_version=JOKER_EXPERT_VERSION, castable=json.dumps(J.CASTABLE), obs_dim=O.OBS_DIM, obs_version=J.OBS_VERSION,
                          hidden=HIDDEN, action_dim=J.ACTION_DIM, weights=base64.b64encode(blob).decode("ascii"))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)


def raw_state(hand, seat):
    """Entrees brutes dont le port GDScript reconstruit l'observation.
    Cartes : entier suit*13+rank (rank 0 = 2, 12 = As ; couleurs ♥♦♣♠),
    comme les autres fixtures (ai_id)."""
    g, rules, state, c = hand.game, hand.rules, hand.state, hand.contract
    return {
        "seat": seat,
        "contract": {k: c.get(k) for k in ("type", "kind", "suit", "level", "declarer", "partner", "declarers", "exposed")},
        "contract_trump": g.trump_suit,
        "hand": sorted(g.hands[seat]),
        "current_trick": [[s, cd] for s, cd in g.current_trick],
        "tricks_done": g.trick_count,
        "tricks_won": list(g.tricks_won),
        "energy": state.energy[seat],
        "joker_hand": list(state.joker_hand[seat]),
        "draw_pile_size": len(state.draw_pile[seat]),
        "discard_size": len(state.discard[seat]),
        # v2 : dans l'ordre de rules.live() (curses puis effects) ; "data" =
        # save_data() du jeu ; id "boss:baronne" = RogueBosses.rule_effect("baronne").
        "live_effects": [{"id": "boss:baronne" if isinstance(e, J.BaronneSees) else e.id, "owner": e.owner,
                          "spent": e.spent, "curse": e in rules.curses, "data": effect_data(e)} for e in rules.live()],
        # v2 : sieges dont la carte du pli en cours est une carte de joker (JokerCard)
        "joker_cards": {str(s): j for s, j in rules.joker_cards.items()},
        "classic_tricks": getattr(g, "num_tricks", 13),
        "play_log": [[s, cd] for s, cd in g.play_log],
    }


# attribut Python -> cle de save_data() (joker_effects.gd)
EFFECT_DATA = {"target": "target", "start": "from", "left": "left", "used": "used", "lost": "lost", "stop_at": "stop_at",
               "streak": "streak"}


def effect_data(e):
    return {key: getattr(e, attr) for attr, key in EFFECT_DATA.items() if hasattr(e, attr)}


def dump_cases(weights, n_cases, seed=20261011):
    rng = random.Random(seed)
    cases = []
    policy_net = None

    def deciding(hand, seat):
        mask = hand.legal_casts(seat)
        if not any(mask[1:]):
            return 0
        obs = O.build(hand, seat)
        lg = logits_np(weights, np.asarray(obs))
        action = max((i for i in range(J.ACTION_DIM) if mask[i]), key=lambda i: lg[i])
        if len(cases) < n_cases and rng.random() < 0.35:
            cases.append({"state": raw_state(hand, seat), "obs_dim": O.OBS_DIM, "expected_obs": obs,
                          "expected_mask": mask, "expected_logits": [float(x) for x in lg],
                          "expected_action": action})
        return action

    while len(cases) < n_cases:
        match = jm.Match(jm.random_config(rng), rng.randrange(1 << 30))
        for _ in range(jm.HANDS_PER_MATCH):
            hand = match.next_hand()
            jm.play_out(hand, deciding)
            match.finish(hand)
    return cases


def _expert_state(hand, seat):
    """raw_state + ce que lisent en plus les regles de agents/joker_expert.py."""
    st = raw_state(hand, seat)
    st.update({"suspicion": hand.state.suspicion, "eye": getattr(hand, "eye", ""),
               "joker_hand_sizes": [len(h) for h in hand.state.joker_hand],
               "hand_tricks": hand.rules.hand_tricks(), "trou_level": getattr(hand.game, "level", None)})
    return st


def _jsonable(v):
    return {k: (sorted(x) if isinstance(x, set) else x) for k, x in v.items()}


def dump_expert_cases(weights, n_bot, n_player, n_bid, seed=20261016):
    """Fixtures de agents/joker_expert.py (decision par decision) :
    bot : etat, masque, verdicts par joker, logits fp16, action + source avec reseau
          ET action + source regles seules (empreinte perimee) ;
    player_trick : candidats (Player._candidates), choix (jid, cible, regle) ou null ;
    player_bid : main, energie, Soupcon, jokers, candidats, choix."""
    from agents import joker_expert as X
    rng = random.Random(seed)
    out = {"bot": [], "player_trick": [], "player_bid": []}
    hyb = __import__("scripts.train_jokers", fromlist=["x"]).make_policy(weights, "hybrid")

    def deciding(hand, seat):
        mask = hand.legal_casts(seat)
        if any(mask[1:]) and len(out["bot"]) < n_bot and rng.random() < 0.3:
            lg = logits_np(weights, np.asarray(O.build(hand, seat)))
            tr, tr0 = {}, {}
            a = X.bot_action(hand, seat, lg, tr)
            a0 = X.bot_action(hand, seat, None, tr0)
            out["bot"].append({"state": _expert_state(hand, seat), "view": _jsonable(X.view(hand, seat)),
                               "expected_mask": mask, "expected_logits": [float(x) for x in lg],
                               "verdicts": tr["verdicts"], "expected_action": a, "source": tr["source"],
                               "expected_action_rules_only": a0, "source_rules_only": tr0["source"]})
        return hyb(hand, seat)

    orig_trick, orig_bid = X.player_trick_choice, X.player_bid_choice

    def trick_choice(hand, cands):
        got = orig_trick(hand, cands)
        if len(out["player_trick"]) < n_player and (got is not None or rng.random() < 0.15):
            out["player_trick"].append({"state": _expert_state(hand, J.PLAYER), "view": _jsonable(X.view(hand, J.PLAYER)),
                                        "candidates": list(cands),
                                        "verdicts": {j: list(X.rule(X.view(hand, J.PLAYER), j)) for j in dict.fromkeys(cands)},
                                        "expected": list(got) if got else None,
                                        "source": "regle:" + got[2] if got else "regle:aucun"})
        return got

    def bid_choice(cards, state, cands):
        got = orig_bid(cards, state, cands)
        if len(out["player_bid"]) < n_bid:
            out["player_bid"].append({"hand": sorted(cards), "energy": state.energy[J.PLAYER], "suspicion": state.suspicion,
                                      "joker_hand": list(state.joker_hand[J.PLAYER]), "candidates": list(cands),
                                      "bid_view": X.bid_view(cards, state),
                                      "expected": list(got) if got else None,
                                      "source": "regle:" + got[2] if got else "regle:aucun"})
        return got

    X.player_trick_choice, X.player_bid_choice = trick_choice, bid_choice
    try:
        while len(out["bot"]) < n_bot or len(out["player_trick"]) < n_player:
            cfg = jm.random_config(rng)
            cfg["style"] = "expert"
            match = jm.Match(cfg, rng.randrange(1 << 30))
            for _ in range(jm.HANDS_PER_MATCH):
                hand = match.next_hand()
                jm.play_out(hand, deciding)
                match.finish(hand)
    finally:
        X.player_trick_choice, X.player_bid_choice = orig_trick, orig_bid
    return {"joker_expert_version": X.JOKER_EXPERT_VERSION, "joker_spec_hash": J.JOKER_SPEC_HASH,
            "bot_priority": X.BOT_PRIORITY, "player_priority": X.PLAYER_PRIORITY, "bid_priority": X.BID_PRIORITY,
            "cases": out}


def write_spec(path):
    lines = ["# Observation de l'IA jokers v2 (JOKER_SPEC_HASH %s)" % J.JOKER_SPEC_HASH, "",
             "Genere par whist-ai/scripts/export_jokers.py depuis engine/joker_obs.py (la reference).",
             "OBS_DIM = %d, ACTION_DIM = %d, actions : 0 = aucun cast, %s." % (
                 O.OBS_DIM, J.ACTION_DIM, ", ".join("%d = %s" % (i + 1, j) for i, j in enumerate(J.CASTABLE))),
             "",
             "Calculee pour le bot `me` a son tour de pli (avant sa carte, dans RogueMatch._bot_turn).",
             "Couleurs dans l'ordre WhistCard.SUITS ♥ ♦ ♣ ♠ ; one-hot a 5 = 4 couleurs + « aucune/sans ».",
             "« Ami/adversaire » = JokerRules.is_opponent(me, x) (camps tires du contrat, whist-4-it a3d70cf).",
             "Meilleur actuel / « gagnerait » = rules.trick_winner(plays, led_suit, rules.trick_trump(contract)).",
             "Figures = K, Q, J (pas l'As). Rang /12 : 2 -> 0, As -> 1.",
             "",
             "| index | nom | taille | definition |", "|---|---|---|---|"]
    doc = {}
    src = open("engine/joker_obs.py", encoding="utf-8").read()
    for name, _n in O.LAYOUT:
        for line in src.splitlines():
            if line.strip().startswith('("%s"' % name):
                doc[name] = line.split("#", 1)[1].strip() if "#" in line else ""
    for name, n in O.LAYOUT:
        lines.append("| %d | %s | %d | %s |" % (O.OFFSETS[name], name, n, doc.get(name, "")))
    lines += ["", "Details :",
              "- contract_kind : [misere, solo (contrat \"solo\", abondance comprise), equipe (emballage/trou)].",
              "- my_side_declares : me dans contract.declarers (misere), == declarer, ou == partner.",
              "- contract_level : misere -> plis max autorises (piccolo 1, sinon 0) ; trou non fixe -> 0.",
              "- tricks_team : somme de tricks_won[x] pour x tel que not is_opponent(me, x) (me compris) ;",
              "  tricks_others : somme(tricks_won) - tricks_team ; declarer_tricks : somme sur le camp declarant.",
              "  tricks_won compte la VALEUR des plis (Kermesse : 2), comme WhistTable.",
              "- can_follow / can_win_now / best_* / opponent_ruffed : 0 quand j'entame.",
              "- energy : min(1, ⚡/6) ; piles : min(1, taille/10).",
              "- embuscade : effets vivants non depenses (rules.effects + curses), index 0 si owner == me,",
              "  2 si is_opponent(me, owner), 1 sinon. kermesse : min(1, nb vivantes/2), tout proprietaire.",
              "- enemy_traps (v2) : effets vivants non depenses, FaussePiste -> index 0, Collet -> index 1,",
              "  si is_opponent(me, owner) (en pratique : les pieges du joueur).",
              "",
              "43b8 (2026-10-14) : CASTABLE gagne coup_du_chapeau, faux_pas, inversion (joker_counts suit",
              "CASTABLE) ; contract.exposed (Cartes sur Table) dans les fixtures ; data.streak (Coup du Chapeau).",
              "v2 (2026-10-13) : le joueur simule lance des jokers de tout le catalogue a l'entrainement.",
              "Fixtures : live_effects porte curse + data (save_data), joker_cards = sieges dont la carte",
              "du pli en cours est une JokerCard, play_log = cartes deja jouees de la donne (BaronneSees).",
              "Jeu de la carte ET enchere par l'ExpertBot partout, comme le jeu depuis 3e5d (sieges 1-3 ExpertBot",
              "a tous les actes) ; seule la regle propre de la Baronne est portee.",
              ""]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=60)
    parser.add_argument("--godot-sync", action="store_true")
    parser.add_argument("--expert-cases", type=int, default=120)
    args = parser.parse_args()
    net, ckpt = load_net()
    rounded = fp16_rounded(net)
    gd = os.path.join(OUT_DIR, GD_NAME)
    write_weights(net, gd)
    cases = dump_cases(net_to_numpy(rounded), args.cases)
    with open(os.path.join(OUT_DIR, "joker_parity_cases.json"), "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False)
    write_spec(os.path.join(OUT_DIR, "joker_obs_spec.md"))
    expert = dump_expert_cases(net_to_numpy(rounded), args.expert_cases, args.expert_cases, args.expert_cases // 2)
    with open(os.path.join(OUT_DIR, EXPERT_CASES), "w", encoding="utf-8") as f:
        json.dump(expert, f, ensure_ascii=False)
    print("fixtures joker_expert v%d : %s" % (expert["joker_expert_version"],
                                              {k: len(v) for k, v in expert["cases"].items()}))
    print("export : %s (iter %d, eval %s), %d cas de parite" % (gd, ckpt["iter"], ckpt["eval"]["vs_heuristic"], len(cases)))
    if args.godot_sync:
        shutil.copy(gd, os.path.join(GODOT_GAME_DIR, GD_NAME))
        print("copie dans %s" % GODOT_GAME_DIR)
        if os.path.exists(P.STATE_FILE):  # test de rentabilite (scripts/joker_profit.py), code a jour
            print("rentabilite :", P.rerender())
        if os.path.exists(P.GODOT_FILE):
            shutil.copy(P.GODOT_FILE, os.path.join(GODOT_GAME_DIR, os.path.basename(P.GODOT_FILE)))
            print("rentabilite des jokers copiee : %s" % os.path.basename(P.GODOT_FILE))


if __name__ == "__main__":
    main()
