"""Table de routage (type_contrat, role_du_siege[, kind]) -> nom du reseau
specialiste (voir DESIGN.md/TRAINING.md). Squelette : pas encore branche a
Godot (le vrai pont vit dans game/whist_ai_bridge_v2.gd, cote whist-4-it,
deja a jour -- voir DESIGN.md section "Phase 2", la session whist-4-it a
corrige le meme ecart misere de son cote le 2026-09-20), remplace a terme
export_weights.py/serve.py de l'ancien projet.
"""

from engine.misere import DECLARER_ROLE_BY_KIND

# role_du_siege : "declarer" | "partner" | "defender" (relatif au contrat)
ROUTING_TABLE = {
    ("emballage", "declarer"): "emballage",
    ("emballage", "partner"): "emballage",
    ("emballage", "defender"): "defenseur",
    ("solo", "declarer"): "solitaire",
    ("solo", "defender"): "defenseur",
    ("misere", "defender"): "defenseur",
    ("trou", "declarer"): "trou",
    ("trou", "partner"): "bouche_trou",
    ("trou", "defender"): "defenseur",
}


def specialist_for(contract_type, seat_role, misere_kind=None):
    """Pendant l'enchere, tous les sieges utilisent 'encherisseur'
    (pas dans cette table -- population
    evolutive, voir agents/evolution.py, choix du siege gere en amont par
    l'appelant). `misere_kind` est OBLIGATOIRE quand contract_type ==
    "misere" et seat_role == "declarer" -- depuis la Phase 1 (2026-09-19),
    la misere n'a plus un seul role partage, voir
    engine/misere.py:DECLARER_ROLE_BY_KIND (meme table que le moteur de
    jeu lui-meme, jamais dupliquee)."""
    if contract_type == "misere" and seat_role == "declarer":
        if misere_kind is None:
            raise ValueError("misere_kind requis pour ('misere', 'declarer') depuis le scindage de la misere")
        return DECLARER_ROLE_BY_KIND[misere_kind]
    key = (contract_type, seat_role)
    if key not in ROUTING_TABLE:
        raise NotImplementedError("pas de route pour %r (voir DESIGN.md)" % (key,))
    return ROUTING_TABLE[key]
