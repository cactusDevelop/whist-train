"""Rapport d'equilibrage des jokers par lot de regles (09/11/2026, conseil « facon TCG »).
Lit joker_moments.json (scripts/joker_moments.py) et, si donne, celui de la version
precedente ; ecrit joker_report.md. Aucune simulation ici : lecture seule.

1. FIABILITE (avant toute note) : la valeur d'un joker vient de la mesure « oracle »
   (jouer maintenant contre la meilleure alternative, decide puis mesure sur des rejouees
   differentes) ; le joueur simule n'intervient que par sa regle de lancer. Un joker est
     - trop peu d'occasions : < MIN_USEFUL occasions utiles -> rien n'est conclu ;
     - mal joue par le bot : la regle rate la plupart des occasions utiles (bien_vu < 0.5)
       ou lance surtout hors occasion utile (precision < 0.5) -> la valeur oracle reste
       lisible, mais l'experience en jeu (et les autres mesures) est a tester a la main ;
     - fiable.
2. MATRICE contrat x role x adversaire : seulement les cellules sures (n utiles >= MIN_CELL,
   borne basse force - 2 se > 0). Etiquettes : dominant (fort dans >= 60 % des cellules
   mesurees, >= 3), pilier (fort dans 1-2 cellules), jamais utile (occasions >= 100 et
   < 5 % utiles), sans cellule sure.
3. DIFF contre la version precedente (memes graines de match : SEED de joker_moments) :
   ecarts de force et de taux d'occasions utiles avec |z| >= 2, jokers nouveaux / disparus.

Ce que ce rapport ne voit pas : le ressenti, le vrai meta des humains, les combos de
plusieurs jokers (voir pair_jokers), le draft (le bot prend presque toujours la 1re option).

Usage : python scripts/joker_report.py [--cur joker_moments.json] [--prev prev_joker_moments.json]
"""

import argparse
import json
import math
import os

MIN_USEFUL = 30
MIN_CELL = 20
STRONG_SHARE = 0.6
NEVER_OCC = 100
NEVER_RATE = 0.05


def _lb(ms):
    if not ms or ms.get("mean") is None or ms.get("se") is None:
        return None
    return ms["mean"] - 2 * ms["se"]


def reliability(r):
    if (r.get("useful") or 0) < MIN_USEFUL:
        return "trop peu d'occasions"
    bv, pr = r.get("bien_vu"), r.get("precision")
    if (bv is not None and bv < 0.5) or (pr is not None and r.get("rule_casts", 0) >= MIN_USEFUL and pr < 0.5):
        return "mal joue par le bot"
    return "fiable"


def cells(r):
    out = []
    for key, c in (r.get("by_ctx") or {}).items():
        f = c.get("force") or {}
        if c.get("useful", 0) >= MIN_CELL and f.get("se") is not None:
            out.append((key, c, f["mean"], f["mean"] - 2 * f["se"]))
    return out


def label(r):
    if (r.get("occasions") or 0) >= NEVER_OCC and (r.get("useful_rate") or 0) < NEVER_RATE:
        return "jamais utile"
    cs = cells(r)
    strong = [c for c in cs if c[3] > 0]
    if len(cs) >= 3 and len(strong) >= STRONG_SHARE * len(cs):
        return "dominant"
    if 1 <= len(strong) <= 2:
        return "pilier"
    if strong:
        return "polyvalent"
    return "sans cellule sure"


def fmt(x, nd=2):
    return "-" if x is None else ("%.*f" % (nd, x) if isinstance(x, float) else str(x))


def diff(cur, prev):
    pr = {r["id"]: r for r in prev["rows"]}
    cr = {r["id"]: r for r in cur["rows"]}
    lines = []
    for jid in sorted(cr):
        a, b = cr[jid], pr.get(jid)
        if b is None:
            lines.append("- **%s** : nouveau (force %s, utiles %s)" % (jid, fmt(a.get("force")), fmt(a.get("useful_rate"))))
            continue
        notes = []
        if a.get("force") is not None and b.get("force") is not None and a.get("force_se") and b.get("force_se"):
            d = a["force"] - b["force"]
            z = d / math.sqrt(a["force_se"] ** 2 + b["force_se"] ** 2)
            if abs(z) >= 2:
                notes.append("force %+.2f (%s -> %s, z %+.1f)" % (d, fmt(b["force"]), fmt(a["force"]), z))
        if a.get("occasions") and b.get("occasions") and a.get("useful_rate") is not None and b.get("useful_rate") is not None:
            p1, n1, p2, n2 = a["useful_rate"], a["occasions"], b["useful_rate"], b["occasions"]
            se = math.sqrt(max(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2, 1e-12))
            if abs(p1 - p2) / se >= 2:
                notes.append("occasions utiles %s -> %s" % (fmt(p2), fmt(p1)))
        if a.get("cost") != b.get("cost"):
            notes.append("cout %s -> %s" % (b.get("cost"), a.get("cost")))
        if notes:
            lines.append("- **%s** : %s" % (jid, " ; ".join(notes)))
    for jid in sorted(set(pr) - set(cr)):
        lines.append("- **%s** : disparu" % jid)
    return lines or ["- aucun ecart sur (|z| < 2 partout)"]


def build(cur, prev=None):
    rows = cur["rows"]
    out = ["# Rapport d'equilibrage des jokers", "",
           "Empreinte **%s**, %d matchs (genere %s). Valeur d'un ⚡ : %s pts, d'une pioche : %s pts." % (
               cur.get("spec_hash"), cur.get("matches", 0), cur.get("generated"),
               fmt((cur.get("value") or {}).get("energie", {}).get("mean")),
               fmt((cur.get("value") or {}).get("pioche", {}).get("mean"))), "",
           "Force = points gagnes jusqu'a la fin du match quand le lancer est utile (oracle : jouer maintenant "
           "contre la meilleure alternative). Independante de la regle du bot ; « bien vu » et « precision » jugent la regle.", ""]
    if prev is not None:
        out += ["## 1. Ce que le dernier lot a change (contre %s, %d matchs)" % (prev.get("spec_hash"), prev.get("matches", 0)), ""]
        out += diff(cur, prev) + [""]
    groups = {}
    for r in rows:
        groups.setdefault(reliability(r), []).append(r)
    out += ["## 2. Fiabilite de la mesure", "",
            "| joker | cout | occasions | utiles | force (se) | bien vu | precision | gain perdu par la regle | statut |",
            "|---|---|---|---|---|---|---|---|---|"]
    for st in ("mal joue par le bot", "trop peu d'occasions", "fiable"):
        for r in sorted(groups.get(st, []), key=lambda r: -(r.get("force") or -99)):
            missed = (r.get("missed") or {}).get("mean")
            out.append("| %s | %s | %s | %s | %s (%s) | %s | %s | %s | %s |" % (
                r["id"], r.get("cost"), r.get("occasions"), fmt(r.get("useful_rate")), fmt(r.get("force")),
                fmt(r.get("force_se")), fmt(r.get("bien_vu")), fmt(r.get("precision")), fmt(missed), st))
    out += ["", "## 3. Ou chaque joker est fort (cellules sures seulement)", "",
            "Cellule = contrat | role | adversaire, n utiles >= %d, force - 2 se > 0." % MIN_CELL, ""]
    by_label = {}
    for r in rows:
        if reliability(r) != "trop peu d'occasions":
            by_label.setdefault(label(r), []).append(r)
    for lab in ("dominant", "polyvalent", "pilier", "sans cellule sure", "jamais utile"):
        rs = by_label.get(lab, [])
        if not rs:
            continue
        out.append("**%s** (%d)" % (lab, len(rs)))
        for r in sorted(rs, key=lambda r: r["id"]):
            strong = sorted((c for c in cells(r) if c[3] > 0), key=lambda c: -c[2])
            where = ", ".join("%s %+.1f" % (k, m) for k, _c, m, _lb in strong[:4]) or "-"
            out.append("- %s : %s%s" % (r["id"], where, "  (bot : a tester a la main)" if reliability(r) != "fiable" else ""))
        out.append("")
    out += ["## 4. Ce que ce rapport ne voit pas", "",
            "- le ressenti et le plaisir de jeu ;",
            "- le vrai meta des joueurs (le draft simule prend presque toujours la 1re option) ;",
            "- les combos (paires : scripts/pair_jokers.py, sur demande) ;",
            "- les jokers « mal joues par le bot » en jeu reel : leur force est un plafond, a verifier a la main.", ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cur", default="joker_moments.json")
    parser.add_argument("--prev", default="")
    parser.add_argument("--out", default="joker_report.md")
    args = parser.parse_args()
    with open(args.cur, encoding="utf-8") as f:
        cur = json.load(f)
    prev = None
    if args.prev and os.path.exists(args.prev):
        with open(args.prev, encoding="utf-8") as f:
            prev = json.load(f)
        if not prev.get("rows") or "by_ctx" not in prev["rows"][0]:
            print("version precedente sans by_ctx : comparaison sur force / occasions seulement")
    text = build(cur, prev)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
