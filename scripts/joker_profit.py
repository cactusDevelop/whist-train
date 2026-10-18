"""Test de rentabilite des jokers, lance a chaque iteration de
scripts/train_jokers.py (demande utilisateur 2026-10-14 : « garde en logs
quels jokers sont inutiles et lesquels sont utiles en creant un test dans
l'entrainement qui regarde leur rentabilite en fonction du cout et de la
situation »).

Methode : CONTREFACTUEL APPARIE. A chaque occasion de jouer un joker, la
meme situation (copie exacte de l'etat, memes mains, memes graines) est
rejouee avec et sans ce joker ; gain = difference de points.

  joueur (siege 0, tout le catalogue) -- matchs dedies (profit_worker) :
    * tour de pli : chaque joker lancable (Player._candidates) contre
      « rien a ce tour » ; la suite est jouee par les politiques habituelles ;
    * enchere : chaque joker d'enchere lancable au 1er tour d'enchere ;
    * passifs pioches : la donne avec le passif contre la donne sans lui.
    Gain = points du joueur sur la donne + la donne suivante (memes donnes
    dans les deux branches : effets differes compris, Chope Vide, pioches,
    malus de l'Oeil), et l'ecart au meilleur adversaire.
  bots a jokers (CASTABLE) -- gratuit : les rollouts de generation de
    train_jokers.gen_worker donnent deja Q(joker) - Q(aucun cast) a chaque
    decision (recompense jm.hand_reward : ecart au joueur, sur la donne).

Situations : contrat, role (declarant/defenseur), adversaire (elite de
l'acte / boss), pli (ou enchere), energie avant le joker.

Sorties :
  joker_profit.jsonl                 -- une ligne par iteration (stats de
                                        l'iteration + cumul) ;
  joker_profit_summary.md            -- classement lisible, cumul ;
  godot_export/joker_profit_stats.json -- format pour l'ecran du jeu
                                        (copie dans whist-4-it/game/ a
                                        l'export --godot-sync).
"""

import copy
import json
import math
import os
import random
import time

from engine import joker_match as jm
from engine import jokers as J

LOG_FILE = "joker_profit.jsonl"
SUMMARY_FILE = "joker_profit_summary.md"
STATE_FILE = os.path.join("checkpoints_jokers", "joker_profit_state.json")
GODOT_FILE = os.path.join("godot_export", "joker_profit_stats.json")
PROFIT_SEED = 77000000
SPARE_DEALS = 12
MIN_N = 30  # occasions mesurees avant un verdict
SOUPCON_PER_ENERGY = 2  # 2 de Soupcon ~ 1 ⚡ (cout effectif = ⚡ + Soupcon / 2)
# v2 (2026-10-16) : "contrat" = cle fine (agents/joker_expert.contract_key : emballage, solo,
# trou, piccolo, petite_misere, grande_misere, misere_etalee), "famille" = emballage/solo/trou/misere,
# "regle" = cas de la regle expert du joueur a cette occasion.
DIMS = ("contrat", "famille", "role", "adversaire", "pli", "energie", "regle")
FORMAT = 2
UNMEASURED = {  # jokers dont l'effet n'existe pas du tout dans la simulation
    "brimborion_dore": "donne a points doubles absente de la simulation",
    "heritage_empoisonne": "effet de run (capsules), hors match",
}
# Effet hors points (information, sous) : mesures (deck avec / sans), jamais a_supprimer.
OFF_POINTS = {
    "coup_d_oeil": "information seulement (l'ExpertBot du joueur ne s'en sert pas)",
    "regard_furtif": "information seulement",
    "livre_de_comptes": "information seulement",
    "pourboire": "gain en sous (run), pas en points",
}
# Mesures par CONTREFACTUEL DE DECK (match entier avec / sans le joker dans le deck).
DECK_CF = sorted(i for i in [i for i, v in J.CATALOG.items() if v[1] == "maudit"] + list(OFF_POINTS)
                 if i not in UNMEASURED)
ADMIN_KEPT_FILE = "admin_kept_jokers.json"  # ids gardes par un admin (whist-4-it) : plus jamais a_supprimer


# --- agregats -------------------------------------------------------------------

def _cell():
    return {"n": 0, "s": 0.0, "q": 0.0, "pos": 0, "neg": 0, "gap": 0.0}


def _add(cell, gain, gap):
    cell["n"] += 1
    cell["s"] += gain
    cell["q"] += gain * gain
    cell["pos"] += gain > 0.5
    cell["neg"] += gain < -0.5
    cell["gap"] += gap


def _merge(dst, src):
    for k in ("n", "s", "q", "pos", "neg", "gap"):
        dst[k] += src[k]


class Agg:
    """{side: {jid: {"all": cell, "by": {dim: {valeur: cell}}, "offers": n, "picks": n}}}"""

    def __init__(self, data=None):
        self.data = data or {"joueur": {}, "bot": {}}

    def _entry(self, side, jid):
        jid = J.base_id(jid)  # versions ameliorees « id+ » cumulees avec la base
        e = self.data[side].setdefault(jid, {"all": _cell(), "by": {d: {} for d in DIMS}, "offers": 0, "picks": 0})
        e.setdefault("picked", _cell())  # gain quand l'IA (bots) / la regle expert (joueur) le choisit
        e.setdefault("byp", {d: {} for d in DIMS})  # idem, par situation
        for d in DIMS:
            e["by"].setdefault(d, {})
            e["byp"].setdefault(d, {})
        return e

    def add(self, side, jid, gain, gap, situ, picked=True):
        e = self._entry(side, jid)
        _add(e["all"], gain, gap)
        if picked:
            _add(e["picked"], gain, gap)
        for d in DIMS:
            _add(e["by"][d].setdefault(str(situ.get(d, "-")), _cell()), gain, gap)
            if picked:
                _add(e["byp"][d].setdefault(str(situ.get(d, "-")), _cell()), gain, gap)

    def offer(self, side, jid, picked):
        e = self._entry(side, jid)
        e["offers"] += 1
        e["picks"] += bool(picked)

    def merge(self, other):
        for side, jokers in other.data.items():
            for jid, src in jokers.items():
                e = self._entry(side, jid)
                _merge(e["all"], src["all"])
                if "picked" in src:
                    _merge(e["picked"], src["picked"])
                e["offers"] += src["offers"]
                e["picks"] += src["picks"]
                for key in ("by", "byp"):
                    for d in DIMS:
                        for v, c in src.get(key, {}).get(d, {}).items():
                            _merge(e[key][d].setdefault(v, _cell()), c)


def _stat(cell):
    n = cell["n"]
    if n == 0:
        return None, None
    m = cell["s"] / n
    var = max(0.0, cell["q"] / n - m * m) * n / max(1, n - 1)
    return m, math.sqrt(var / n)


# --- situations -------------------------------------------------------------------

def _pli_bucket(trick_no):
    return "enchere" if trick_no is None else ("1-4" if trick_no < 4 else ("5-9" if trick_no < 9 else "10-13"))


def _energy_bucket(e):
    return "0-1" if e <= 1 else ("2-3" if e <= 3 else "4+")


def situation(hand, seat, trick_no, energy, config, regle="-"):
    from agents.joker_expert import contract_key
    c = hand.contract
    return {"contrat": contract_key(c), "famille": c["type"], "role": "declarant" if J._declares(seat, c) else "defenseur",
            "adversaire": config.get("boss") or "elite_acte%d" % config["act"],
            "pli": _pli_bucket(trick_no), "energie": _energy_bucket(energy), "regle": regle}


# --- joueur : matchs dedies -----------------------------------------------------------

def _spare(match, rng):
    chain = copy.deepcopy(match.chain)
    out = []
    for _ in range(SPARE_DEALS):
        hands, dealer, whisted = chain.deal(rng)
        out.append(([list(h) for h in hands], dealer, whisted))
        chain.finish_unplayed(hands, dealer)
    return out


def _gain(pts):
    return pts[jm.PLAYER], pts[jm.PLAYER] - max(pts[1:])


def _finish_branch(m, h, policy, spare, more):
    """Joue la fin de la donne de la branche, puis la donne suivante du match
    (memes donnes de reserve dans toutes les branches)."""
    pts = list(jm.play_out(h, policy))
    if more:
        m.deals = list(spare)
        h2 = m.next_hand()
        pts = [a + b for a, b in zip(pts, jm.play_out(h2, policy))]
    return pts


def _deck_cf(cfg, mseed, policy, rng, agg):
    """Contrefactuel de deck : le match entier avec / sans un joker DECK_CF dans
    le deck du joueur (memes graines). Gain par donne, appariee par rang de donne,
    situation = contrat de la donne de la branche « avec »."""
    jid = rng.choice(DECK_CF)
    deck = [d for d in cfg["deck"] if J.base_id(d) != jid]
    with_cfg, without_cfg = dict(cfg, deck=deck + [jid]), dict(cfg, deck=deck)
    per = []
    for c in (with_cfg, without_cfg):
        m = jm.Match(c, mseed)
        rows = []
        for _ in range(jm.HANDS_PER_MATCH):
            h = m.next_hand()
            situ = situation(h, jm.PLAYER, None, h.state.energy[jm.PLAYER], c, "deck")
            rows.append((_gain(jm.play_out(h, policy)), situ))
            m.finish(h)
        per.append(rows)
    for ((own_w, gap_w), situ), ((own_o, gap_o), _s) in zip(per[0], per[1]):
        agg.offer("joueur", jid, True)
        agg.add("joueur", jid, own_w - own_o, gap_w - gap_o, situ)


def profit_worker(args):
    seeds, weights = args[0], args[1]
    from scripts.train_jokers import make_policy
    from agents import joker_expert as X
    policy = make_policy(weights)
    agg = Agg()
    for seed in seeds:
        rng = random.Random(seed)
        cfg = jm.random_config(rng)
        cfg["style"] = "expert"  # v2 : le joueur joue ses jokers selon les regles (« bien joue »)
        mseed = rng.randrange(1 << 30)
        _deck_cf(cfg, mseed, policy, random.Random(seed * 7 + 1), agg)
        match = jm.Match(cfg, mseed)
        for k in range(jm.HANDS_PER_MATCH):
            more = k + 1 < jm.HANDS_PER_MATCH
            spare = _spare(match, random.Random(rng.random()))
            # enchere : jokers lancables au 1er tour d'enchere du joueur
            if cfg["kind"] == "elite":
                base_m = copy.deepcopy(match)
                base_m.player.force_bid, base_m.player.bid_probe = "", []
                base_h = base_m.next_hand()
                cands = base_m.player.bid_probe[0] if base_m.player.bid_probe else []
                if cands:
                    energy = base_h.state.energy[jm.PLAYER]  # (pas de joker paye : energie du debut de donne)
                    cards = base_m.rules.deal_hands[jm.PLAYER] if base_m.rules.deal_hands else []
                    bv = X.bid_view(cards, base_h.state) if cards else None
                    b_own, b_gap = _gain(_finish_branch(base_m, base_h, policy, spare, more))
                    for jid in dict.fromkeys(cands):
                        if J.base_id(jid) in DECK_CF:
                            continue  # mesure par contrefactuel de deck
                        m = copy.deepcopy(match)
                        m.player.force_bid = jid
                        h = m.next_hand()
                        if not m.player.forced_ok:
                            continue
                        verdict, rid = X.bid_rule(bv, jid) if bv else (X.NO, "-")
                        picked = verdict in (X.YES, X.OPEN_YES)
                        situ = situation(base_h, jm.PLAYER, None, energy, cfg, rid)
                        own, gap = _gain(_finish_branch(m, h, policy, spare, more))
                        agg.offer("joueur", jid, picked)
                        agg.add("joueur", jid, own - b_own, gap - b_gap, situ, picked)
            hand = match.next_hand()
            # passifs pioches : la donne avec contre sans
            passives = [i for i, e in enumerate(hand.rules.effects)
                        if e.owner == jm.PLAYER and J.jtype(e.id) == "passif"]
            if passives:
                m0, h0 = copy.deepcopy((match, hand))
                situ = situation(h0, jm.PLAYER, 0, h0.state.energy[jm.PLAYER], cfg, "passif")
                b_own, b_gap = _gain(_finish_branch(m0, h0, policy, spare, more))
                for i in passives:
                    m, h = copy.deepcopy((match, hand))
                    jid = h.rules.effects[i].id
                    del h.rules.effects[i]
                    own, gap = _gain(_finish_branch(m, h, policy, spare, more))
                    agg.offer("joueur", jid, True)
                    agg.add("joueur", jid, b_own - own, b_gap - gap, situ)

            def before_player(h_now):
                cands = h_now.player._candidates(h_now.state, h_now.rules, h_now.boss, False)
                if not cands:
                    return
                energy = h_now.state.energy[jm.PLAYER]
                v = X.view(h_now, jm.PLAYER)
                m0, h0 = copy.deepcopy((match, h_now))
                h0.player.force_trick = ""
                b_own, b_gap = _gain(_finish_branch(m0, h0, policy, spare, more))
                for jid in dict.fromkeys(cands):
                    if J.base_id(jid) in DECK_CF:
                        continue  # mesure par contrefactuel de deck
                    verdict, rid = X.rule(v, jid)
                    picked = verdict in (X.YES, X.OPEN_YES)
                    situ = situation(h_now, jm.PLAYER, h_now.game.trick_count, energy, cfg, rid)
                    m, h = copy.deepcopy((match, h_now))
                    h.player.force_trick = jid
                    pts = _finish_branch(m, h, policy, spare, more)
                    if not h.player.forced_ok:
                        continue
                    own, gap = _gain(pts)
                    agg.offer("joueur", jid, picked)
                    agg.add("joueur", jid, own - b_own, gap - b_gap, situ, picked)

            jm.play_out(hand, policy, before_player=before_player)
            match.finish(hand)
    return agg.data


def bot_records(hand, seat, mask, q, chosen, config):
    """Depuis gen_worker : une decision d'un bot a jokers -> enregistrements
    (jid, gain, situation, pris) pour chaque joker legal."""
    situ = situation(hand, seat, hand.game.trick_count, hand.state.energy[seat], config)
    return [(J.CASTABLE[a - 1], q[a] - q[0], situ, chosen == a) for a in range(1, J.ACTION_DIM) if mask[a]]


def add_bot_records(agg, records):
    for jid, gain, situ, picked in records:
        agg.offer("bot", jid, picked)
        agg.add("bot", jid, gain, 0.0, situ, picked)


# --- verdicts -------------------------------------------------------------------------------

# Couts en vigueur lors d'une mesure faite sous une autre empreinte (seulement les jokers dont le
# cout a change depuis) : tant que measured_spec_hash != spec_hash, verdicts, strong_in et a_supprimer
# sont juges avec les couts de la mesure (un gain mesure a 1 ⚡ n'est pas divise par le nouveau cout).
COSTS_BY_HASH = {
    "fd031a536f6d5e12": {"traitre": 1, "dette_de_jeu": 1, "borgne": 1, "valet_borgne": 1, "carte_fantome": 0,
                         "manche_truquee": 2, "silence_du_cercle": 1, "faux_depart": 1},
}
MEASURE_COSTS = {}  # rempli par rerender() pour une mesure sous une autre empreinte


def measure_cost(jid):
    return MEASURE_COSTS.get(jid, J.jcost(jid))


def effective_cost(jid, side):
    soupcon = J.jsuspicion(jid) if side == "joueur" else 0
    return measure_cost(jid) + soupcon / SOUPCON_PER_ENERGY


def judged_cell(side, e):
    """Bots : quand l'IA le choisit ; joueur (v2) : quand la regle expert le lance
    (« bien joue ») ; toutes les occasions tant que ce sous-ensemble est < MIN_N."""
    return e["picked"] if e.get("picked", {}).get("n", 0) >= MIN_N else e["all"]


def _judged_by(e):
    return (e.get("byp") or e["by"]) if e.get("picked", {}).get("n", 0) >= MIN_N else e["by"]


def verdict(side, jid, e):
    """utile / marginal / inutile / trop fort / jamais pris / trop peu de donnees."""
    if jid in UNMEASURED:
        return "non mesurable", UNMEASURED[jid]
    if e is None or e["all"]["n"] == 0:
        return "jamais pris", "aucune occasion de le jouer dans les matchs simules"
    cell = judged_cell(side, e)
    n = cell["n"]
    if n < MIN_N:
        return "trop peu de donnees", "%d occasions" % n
    m, se = _stat(cell)
    per = m / max(1.0, effective_cost(jid, side))
    neg = cell["neg"] / n
    pos = cell["pos"] / n
    if per >= 6 or (per >= 3 and neg <= 0.05 and pos >= 0.5):
        return "trop fort", "%+.1f pts par ⚡ effectif, perd dans %d %% des cas" % (per, round(100 * neg))
    if m + 2 * se <= 0.5:
        return "inutile", "jamais rentable : %+.2f ± %.2f pts" % (m, se)
    if m - 2 * se > 0 and per >= 1.0:
        return "utile", "%+.1f pts par ⚡ effectif" % per
    return "marginal", "%+.2f ± %.2f pts, %+.1f par ⚡ effectif" % (m, se, per)


MISERE_KEYS = ("piccolo", "petite_misere", "grande_misere", "misere_etalee", "misere")


def _contract_cells(e):
    """[(contrat, cellule)] mesurees (n >= MIN_N) : contrats fins, plus la famille
    « misere » entiere quand aucune misere fine n'atteint MIN_N."""
    by = _judged_by(e)
    cells = [(c, cell) for c, cell in sorted(by.get("contrat", {}).items()) if cell["n"] >= MIN_N]
    mis = by.get("famille", {}).get("misere")
    if mis and mis["n"] >= MIN_N and not any(c in MISERE_KEYS for c, _x in cells):
        cells.append(("misere", mis))
    return cells


def contract_rule(side, jid, e):
    """Regle utilisateur (2026-10-16) : un joker doit etre FORT dans au moins un type
    de contrat (gain > 0 a 2 sigma et >= 1 pt par ⚡ effectif sur les donnes de ce
    contrat, quand il est bien joue), sinon il est range « a supprimer ».
    -> (statut, contrats forts) ; statut fort / vraiment inutile / a confirmer
    (aucun contrat fort mais pas assez de donnees) ; None hors joueur ou non mesurable."""
    if side != "joueur" or jid in UNMEASURED or e is None or e["all"]["n"] == 0:
        return None, []
    strong = []
    cells = _contract_cells(e)
    for c, cell in cells:
        m, se = _stat(cell)
        if m - 2 * se > 0 and m / max(1.0, effective_cost(jid, side)) >= 1.0:
            strong.append(c)
    if strong:
        return "fort", strong
    # joueur : « vraiment inutile » seulement s'il a ete BIEN JOUE (regle expert) au moins MIN_N fois
    return ("vraiment inutile" if e.get("picked", {}).get("n", 0) >= MIN_N and cells else "a confirmer"), []


# --- note et cout conseille (fin de nuit) ------------------------------------------------
# note = 100 * (0.5 force + 0.2 polyvalence + 0.2 regularite + 0.1 soupcon), chaque composante 0..1 :
#   force       = meilleure borne basse (gain - 2 sigma, pts, bien joue) parmi les contrats mesures / FORCE_REF,
#                 borne 0..1 (v2.1 : un gain non significatif ne donne plus de force)
#   polyvalence = contrats ou le gain est > 0 a 2 sigma / contrats mesures (n >= MIN_N)
#   regularite  = 0.5 (1 - part des occasions perdantes) + 0.5 borne(1 + gain du pire contrat / FORCE_REF)
#   soupcon     = 1 - Soupcon du joker / 4
# Gains BRUTS (pas divises par le cout) : la note ne depend pas du cout actuel.
# Cout conseille : note < 20 -> 0, < 45 -> 1, < 70 -> 2, sinon 3 ; plancher 1 (un joker ameliore
# a -1 garde un sens) sauf note < 10 ; maudits 0 ; Cave bornee a 1-2 ; non mesurable et effets
# hors points (information, sous) et passifs (jamais payes, MatchState.play_error -> "passive") : cout actuel garde ;
# deck de depart (STARTING_DECK) plafonne a 1 : le tuto joue ses 3 jokers dans une donne a 3 ⚡.
FORCE_REF = 4.0
NOTE_WEIGHTS = {"force": 0.5, "polyvalence": 0.2, "regularite": 0.2, "soupcon": 0.1}
NOTE_FORMULA = ("note = 100 x (0.5 force + 0.2 polyvalence + 0.2 regularite + 0.1 soupcon) ; force = meilleure borne basse "
                "(gain bien joue - 2 sigma) par contrat / %.0f pts (borne 0-1) ; polyvalence = contrats a gain > 0 (2 sigma) / "
                "contrats mesures ; regularite = 0.5 (1 - part perdante) + 0.5 borne(1 + pire contrat / %.0f) ; "
                "soupcon = 1 - Soupcon/4. Cout conseille : note < 20 -> 0, < 45 -> 1, < 70 -> 2, sinon 3 ; "
                "plancher 1 sauf note < 10 ; maudits 0 ; Cave 1-2 ; non mesurable / hors points / passifs : cout actuel ; deck de depart <= 1."
                % (FORCE_REF, FORCE_REF))


def _clip(x):
    return max(0.0, min(1.0, x))


def note_of(jid, e):
    if e is None or e["all"]["n"] == 0 or jid in UNMEASURED:
        return None
    judged = judged_cell("joueur", e)
    cells = _contract_cells(e) or ([("tout", judged)] if judged["n"] >= MIN_N else [])
    if not cells:
        return None
    means = [_stat(c) for _k, c in cells]
    comp = {"force": _clip(max(m - 2 * se for m, se in means) / FORCE_REF),
            "polyvalence": sum(1 for m, se in means if m - 2 * se > 0) / len(means),
            "regularite": 0.5 * (1 - judged["neg"] / max(1, judged["n"])) + 0.5 * _clip(1 + min(m for m, _s in means) / FORCE_REF),
            "soupcon": _clip(1 - J.jsuspicion(jid) / 4.0)}
    comp = {k: round(v, 3) for k, v in comp.items()}
    return round(100 * sum(NOTE_WEIGHTS[k] * comp[k] for k in comp)), comp


def recommended_cost(jid, note):
    if J.jrarity(jid) == "maudit":
        return 0
    if note is None or jid in OFF_POINTS or J.jtype(jid) == "passif":
        return J.jcost(jid)
    cost = 0 if note < 20 else 1 if note < 45 else 2 if note < 70 else 3
    if note >= 10:
        cost = max(1, cost)
    if J.jrarity(jid) == "cave":
        cost = max(1, min(2, cost))
    if jid in J.STARTING_DECK:
        cost = min(1, cost)
    return cost


def admin_kept():
    try:
        with open(ADMIN_KEPT_FILE, encoding="utf-8") as f:
            return set(json.load(f))
    except (OSError, ValueError):
        return set()


def _best_worst(e):
    cands = []
    for d in DIMS:
        if d == "regle":
            continue
        for v, c in _judged_by(e).get(d, {}).items():
            if c["n"] >= max(10, MIN_N // 2):
                cands.append((c["s"] / c["n"], c["n"], d, v))
    if not cands:
        return None, None
    cands.sort()
    fmt = lambda t: {"dim": t[2], "valeur": t[3], "gain": round(t[0], 2), "n": t[1]}
    return fmt(cands[-1]), fmt(cands[0])


def table(agg):
    rows = []
    kept_ids = admin_kept()
    for side in ("joueur", "bot"):
        ids = sorted(J.CATALOG) if side == "joueur" else list(J.CASTABLE)
        for jid in ids:
            e = agg.data[side].get(jid)
            v, why = verdict(side, jid, e)
            rule, strong = contract_rule(side, jid, e)
            row = {"id": jid, "side": side, "type": J.jtype(jid), "rarity": J.jrarity(jid), "cost": J.jcost(jid),
                   "suspicion": J.jsuspicion(jid) if side == "joueur" else 0, "verdict": v, "why": why,
                   "contract_rule": rule, "strong_in": strong}
            if side == "joueur":
                nt = note_of(jid, e)
                kept = jid in kept_ids
                row.update({"note": nt[0] if nt else None, "note_parts": nt[1] if nt else None,
                            "recommended_cost": recommended_cost(jid, nt[0] if nt else None),
                            "measure": "deck" if jid in DECK_CF else ("non mesurable" if jid in UNMEASURED else "occasion"),
                            "admin_kept": kept,
                            "a_supprimer": (rule == "vraiment inutile" and not kept and jid not in OFF_POINTS
                                            and J.jrarity(jid) != "maudit")})
                if jid in OFF_POINTS:
                    row["why"] = row["why"] + " ; effet hors points : " + OFF_POINTS[jid]
            if e is not None and e["all"]["n"]:
                m, se = _stat(judged_cell(side, e))  # bots : quand l'IA le choisit
                n = judged_cell(side, e)["n"]
                best, worst = _best_worst(e)
                row.update({"n": n, "gain": round(m, 3), "se": round(se, 3),
                            "gain_per_energy": round(m / max(1.0, effective_cost(jid, side)), 3),
                            "gap": round(e["all"]["gap"] / n, 3), "win_rate": round(e["all"]["pos"] / n, 3),
                            "loss_rate": round(e["all"]["neg"] / n, 3),
                            "pick_rate": round(e["picks"] / e["offers"], 3) if side == "bot" and e["offers"] else None,
                            "gain_all_offers": round(_stat(e["all"])[0], 3) if side == "bot" else None,
                            "picked_n": e.get("picked", {}).get("n", 0),
                            "best": best, "worst": worst,
                            "by": {d: {v2: {"n": c["n"], "gain": round(c["s"] / c["n"], 2)}
                                       for v2, c in sorted(_judged_by(e).get(d, {}).items())} for d in DIMS}})
            rows.append(row)
    return rows


ORDER = ("trop fort", "utile", "marginal", "inutile", "jamais pris", "trop peu de donnees", "non mesurable")


def write_summary(rows, meta):
    lines = ["# Rentabilite des jokers (%s)" % meta["generated"], "",
             "Empreinte %s, iterations %s, %d matchs joueur dedies. Gain = points (contrefactuel apparie, "
             "meme situation avec/sans le joker). Joueur : ses points sur la donne + la suivante. Bots : "
             "ecart au joueur sur la donne, quand l'IA choisit le joker. Cout effectif = ⚡ + Soupcon/%d." % (
                 meta["spec_hash"], meta["iterations"], meta["profit_matches"], SOUPCON_PER_ENERGY),
             "",
             "v2 (2026-10-16) : le joueur joue ses jokers selon agents/joker_expert.py ; un joker est juge "
             "« bien joue » (quand la regle le lance) des qu'il l'a ete %d fois. Maudits et jokers hors points : "
             "contrefactuel de deck (match entier avec / sans)." % MIN_N, "",
             "**Note et cout conseille** : " + NOTE_FORMULA, ""]
    if meta.get("measured_spec_hash"):
        lines[2:2] = ["**Mesure faite sous l'empreinte %s (anciens couts), reaffichee sous %s en attendant "
                      "une nouvelle mesure ; verdicts, fort en et a supprimer jugés aux couts de la mesure, "
                      "note et cout conseille v2.1.**" % (meta["measured_spec_hash"], meta["spec_hash"]), ""]
    sup = [r for r in rows if r.get("a_supprimer")]
    lines += ["## A supprimer (%d) -- decision d'un admin" % len(sup), ""]
    lines += ["- %s : %s (note %s)" % (r["id"], r["why"], r.get("note")) for r in sup] or ["- aucun"]
    lines += ["", "## Notes et couts conseilles (joueur)", "",
              "| joker | note | force | polyv. | regul. | soupcon | ⚡ actuel | ⚡ conseille | fort en | mesure |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted((r for r in rows if r["side"] == "joueur"), key=lambda r: -(r.get("note") or -1)):
        np_ = r.get("note_parts") or {}
        lines.append("| %s | %s | %s | %s | %s | %s | %d | %d | %s | %s |" % (
            r["id"], r.get("note") if r.get("note") is not None else "-",
            np_.get("force", "-"), np_.get("polyvalence", "-"), np_.get("regularite", "-"), np_.get("soupcon", "-"),
            r["cost"], r.get("recommended_cost", r["cost"]), ", ".join(r["strong_in"]) or "-", r.get("measure", "")))
    lines.append("")
    jr = [r for r in rows if r["side"] == "joueur" and r.get("contract_rule")]
    lines += ["## Regle : fort dans au moins un type de contrat", "",
              "Fort = gain > 0 a 2 sigma et >= 1 pt par ⚡ effectif sur les donnes de ce contrat (n >= %d)." % MIN_N, ""]
    for status in ("vraiment inutile", "a confirmer", "fort"):
        ids = [r["id"] + (" (%s)" % ", ".join(r["strong_in"]) if r["strong_in"] else "")
               for r in jr if r["contract_rule"] == status]
        lines.append("- **%s** (%d) : %s" % (status, len(ids), ", ".join(ids) or "-"))
    lines.append("")
    for side, title in (("joueur", "Joueur (tout le catalogue)"), ("bot", "Bots a jokers (decision de l'IA)")):
        lines += ["## " + title, "", "| verdict | joker | ⚡ | Soupcon | n | gain | ± | /⚡ eff. | %s | meilleure situation | pire situation |" % (
            "pris par l'IA" if side == "bot" else "ecart"), "|---|---|---|---|---|---|---|---|---|---|---|"]
        side_rows = sorted((r for r in rows if r["side"] == side), key=lambda r: (ORDER.index(r["verdict"]), -(r.get("gain_per_energy") or 0)))
        for r in side_rows:
            sit = lambda s: "%s=%s (%+.1f, n=%d)" % (s["dim"], s["valeur"], s["gain"], s["n"]) if s else ""
            extra = ("%d %% (toutes occasions %+.2f)" % (round(100 * r["pick_rate"]), r["gain_all_offers"])
                     if r.get("pick_rate") is not None else "") if side == "bot" else (
                "%+.1f" % r["gap"] if "gap" in r else "")
            lines.append("| %s | %s | %d | %d | %s | %s | %s | %s | %s | %s | %s |" % (
                r["verdict"], r["id"], r["cost"], r["suspicion"], r.get("n", 0),
                "%+.2f" % r["gain"] if "gain" in r else "", "%.2f" % r["se"] if "se" in r else "",
                "%+.2f" % r["gain_per_energy"] if "gain_per_energy" in r else "", extra,
                sit(r.get("best")), sit(r.get("worst"))))
        lines.append("")
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            st = json.load(f)
        if st.get("spec_hash") in J.RULE_EQUIVALENT_HASHES:  # regles identiques : la rentabilite reste valable
            st["spec_hash"] = J.JOKER_SPEC_HASH
            return st
    return {"spec_hash": J.JOKER_SPEC_HASH, "iterations": [], "profit_matches": 0, "agg": None}


def render(st, rows):
    meta = {"spec_hash": J.JOKER_SPEC_HASH, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "iterations": "%d-%d" % (min(st["iterations"]), max(st["iterations"])),
            "profit_matches": st["profit_matches"]}
    if st["spec_hash"] != J.JOKER_SPEC_HASH:  # mesure sous d'autres couts, reaffichee faute de mieux
        meta["measured_spec_hash"] = st["spec_hash"]
    write_summary(rows, meta)
    out = {"format": FORMAT, "note_formula": NOTE_FORMULA, **meta, "dims": list(DIMS), "verdicts": list(ORDER), "jokers": rows}
    with open(GODOT_FILE, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return meta


def rerender():
    """Reecrit le resume et le fichier du jeu depuis l'etat cumule (sans iteration) :
    venv/Scripts/python.exe -c "from scripts import joker_profit as P; print(P.rerender())" """
    st = load_state()
    if st["agg"] is None and os.path.exists(STATE_FILE):
        # regles changees (couts) sans nouvelle mesure : on reaffiche la derniere mesure, marquee
        # measured_spec_hash ; le prochain entrainement repart d'un etat vide (load_state)
        with open(STATE_FILE, encoding="utf-8") as f:
            st = json.load(f)
        if st["spec_hash"] not in COSTS_BY_HASH:
            raise SystemExit("couts de la mesure %s inconnus (COSTS_BY_HASH)" % st["spec_hash"])
        MEASURE_COSTS.update(COSTS_BY_HASH[st["spec_hash"]])
    try:
        rows = table(Agg(st["agg"]))
    finally:
        MEASURE_COSTS.clear()
    render(st, rows)
    return {v: sum(1 for r in rows if r["verdict"] == v) for v in ORDER}


def record_iteration(it, iter_agg, n_matches):
    """Cumule l'iteration, ecrit les trois sorties."""
    st = load_state()
    cum = Agg(st["agg"])
    cum.merge(iter_agg)
    st["agg"] = cum.data
    if it not in st["iterations"]:
        st["iterations"].append(it)
    st["profit_matches"] += n_matches
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f)
    os.replace(tmp, STATE_FILE)
    meta = {"spec_hash": J.JOKER_SPEC_HASH, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "iterations": "%d-%d" % (min(st["iterations"]), max(st["iterations"])),
            "profit_matches": st["profit_matches"]}
    rows = table(cum)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"iter": it, **meta, "iteration": table(iter_agg), "cumul": rows}, ensure_ascii=False) + "\n")
    write_summary(rows, meta)
    out = {"format": FORMAT, "note_formula": NOTE_FORMULA, **meta, "dims": list(DIMS), "verdicts": list(ORDER), "jokers": rows}
    with open(GODOT_FILE, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return {v: sum(1 for r in rows if r["verdict"] == v) for v in ORDER}
