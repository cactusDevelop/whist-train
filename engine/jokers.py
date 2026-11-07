"""Port Python des jokers du roguelike (whist-4-it/roguelike/) pour l'IA
jokers : la decision de CAST d'un bot elite/boss a son tour de pli. Le jeu
de la carte reste celui de l'ExpertBot.

Source de verite : whist-4-it/roguelike/{joker_catalog, joker_def,
joker_effects, joker_rules, match_state, rule_modifiers, rogue_match,
rogue_bosses, rogue_mises}.gd. JOKER_SPEC_HASH ci-dessous = l'empreinte
(tools/joker_spec.mjs) des regles portees ici ; a chaque modification des
jokers cote jeu, porter le diff, mettre a jour le hash, reentrainer
(scripts/train_jokers.py) et reexporter.

v2 (2026-10-13, demande utilisateur « il faut s'entrainer contre des joueurs
qui jouent eventuellement des jokers ») : TOUT le catalogue est porte (49
jokers + versions ameliorees « id+ »), avec le Soupcon et l'Oeil, pour que
le joueur simule du siege 0 (engine/joker_match.py) lance des jokers tires
du catalogue. Les bots ne lancent toujours que les jokers `bot_casts`
(CASTABLE inchange -> ACTION_DIM inchange).

Ecarts connus (sans effet sur les decisions des bots) : Brimborion Dore
inerte (pas de donne a points doubles dans la simulation), pas de
recyclage cote joueur (donc Cendrier / Consigne : defausse seulement), pas
de sanction choisie (Synergies 04b : celle de l'Oeil tombe), pas d'offres
orientees par famille (run_nodes.gd, hors empreinte).
Cartes de pli : une carte de joker est un vrai entier carte (couleur/rang)
pour le moteur et l'ExpertBot, comme dans le jeu ; JokerRules.joker_cards
retient quel siege l'a posee dans le pli en cours.
"""

import random

from engine.cards import NUM_PLAYERS, legal_plays, suit_of, rank_of, trick_winner as classic_trick_winner

JOKER_SPEC_HASH = "f1654a462b62ee8d"  # f165 : garde defausse vide de La Peche2 (whist-4-it) ; acb9 : La Peche au choix (la_peche2), un joker gratuit relance dans la donne coute 1 (whist-4-it 9f25355) ; 473e : grille revue sur la mesure 22d8 (whist-4-it bbb9a97) ; 22d8 : 2 ⚡ par donne sous dominion (3 hors dominion) ; 8ebd : grille G2 raretes/couts (mesure des moments utiles 515d) ; fc90 : defausse due d'un sort de Feu de Paille2 payee apres lui (bots compris) ; 515d : pas de remelange en cours de donne (dominion) ; ee3a : bot_spares joueur
# a76f (whist-4-it, 06/11) : 12 jokers recycles « a la Dominion » (DOMINION_REPLACED), choix en attente
# (MatchState.choices / bot_picks), sort gratuit (cast_free), Complice2 rejoue le sort -> regles changees.
# Empreintes dont les REGLES sont identiques a JOKER_SPEC_HASH (changement de
# texte ou de cablage seulement) : un reseau entraine sous l'une d'elles est
# reexporte tel quel avec JOKER_SPEC_HASH. Une regle qui change -> vider la
# liste (sauf le hash courant) et reentrainer.
RULE_EQUIVALENT_HASHES = {JOKER_SPEC_HASH}  # a76f : regles changees (avant : 8664, 8261)
# 8664 (whist-4-it 6b05a37, 03/11) : JokerAI recoit config + RogueMatch.bosses() ; le Moine (2e regle en Mise
# Doree contre la Baronne) retire pli/aura du masque bot et du combo-bot meme sans priority (avant : seulement
# via turn_error sous priority). Masque des bots seulement, aucune regle de jeu -> 8261 equivalent, reexport seul.
# 8261 (whist-4-it abd18ee, 02/11, decision user sur la mesure 7c98) : Main de Fer coute 3 ⚡ (2 avant) ;
# jamais lance par les bots (CASTABLE inchange) mais le joueur simule le paie -> regle changee.
# 7c98 (whist-4-it 31a19be) : config "pose" (runs neuves, Duel, Belle) : un passif du joueur agit une fois pose
# (0 energie, jamais paye), plus des la pioche ; bots inchanges -> regle changee.
# 61ef (whist-4-it f25a277, Synergies 08-09) : decks de combos des boss (runs a Chaleur), JokerComboBot hors reseau,
# defausses des bots, 6 jokers jamais lances par les bots (CASTABLE/ACTION_DIM inchanges), Faux Depart 2 interdit
# avant la premiere carte du bouche-trou -> regle changee.
# 3285 (whist-4-it, Synergies 05) : Feu de Paille +2 points par ⚡ (3 avant) -> regle changee, joker jamais lance par les bots.
# 6483 (whist-4-it a62a3e1 + c9dd31a + 764a412, Synergies 04a-04c) : faveurs du run et paliers de Chaleur (siege 0,
# config, defaut neutre), Double Valet 2 ⚡, 13 jokers de combos jamais lances par les bots (CASTABLE, OBS_DIM
# inchanges), effets PENDING, ⚡ reportes / dus, joker garde -> regle changee, le champion doit repasser la porte.
# 6cb7 : bot_seen seulement (whist-4-it 203f958, regles identiques a 4d7d)
# 7567 (whist-4-it, 30/10) : Main de Fer devient un sort a 2 ⚡, change l'atout d'un emballage ou d'un solo adverse
# avant le premier pli (jamais lance par les bots -> CASTABLE inchange) -> reentrainement.
# e1f4 (whist-4-it d0e3eec) : Courant d'Air garde sur le nombre de plis classique (Horloge remplacee ensuite ->
# main vide au 13e pli, crash 8f41) -> regle changee, aucune iteration 8f41 acceptee de toute facon.
# 8f41 (whist-4-it 8e28fa9, Rarete 04) : raretes/couts (Silence cave 3, Dette/Double Valet arriere_salle 3,
# Cameleon arriere_salle, Traitre salle 1, Atout Errant/Bluff/Fantome/Fausse Piste 2, Defausse Discrete 3,
# Lune/Zinneke/Main de Fer/Horloge 1, Brouillard 2, La Tournee aura salle 2) ; Valet Borgne abs(), Horloge
# des le 6e pli, Treizieme Carte perd dans la misere de son porteur, Manche Truquee pli pour 2, La Tournee
# toute la donne, Cadeau Empoisonne a chaque defausse, Meche Courte 3 max + 1 joker au 3e -> reentrainement.
# 1f49 (whist-4-it c702e8a) : _bot_fuel aussi dans la branche JokerAI, deja simule ainsi depuis le port 65a7 -> regles = 65a7.
# 65a7 (whist-4-it d74be81 + 48b2dc3 + ff4b72a + f31712b, Chantier Jokers joues) : 2 cases d'effet de donne
# par siege (3 pour elites/boss/habitues, bot_case), passif arme seulement s'il reste une case ; defausse
# d'un joker de la main = +1 energie (pas si due, « sans allumettes » ou passif arme), bots : _bot_fuel ;
# spammers piochent 5 jokers ; 4 jokers de misere (Defausse Discrete, Meche Courte, La Tournee, Cadeau
# Empoisonne ; jamais lances par les bots -> CASTABLE inchange) -> reentrainement.
# f01c2461 (whist-4-it ff4faaa) : apres une Trahison du Duel, play_error -> "betrayal" pour tous jusqu'a la fin
# de la donne, et Trahison ratee partagee entre les 3 autres (whist_table._apply_betrayal). La Trahison est une
# action humaine du Duel jamais simulee ici (l'ExpertBot ne trahit pas) -> regles vues a l'entrainement = d06014bf.
# d06014bf (whist-4-it, rapport 2026-10-24) : "slip" -- aux actes I/II les bots face aux joueurs jouent parfois
# une autre carte legale ; aucun joker, cout ni observation ne change -> regles = 07ac.
# Exportable sans equivalence de regles : champion du gate cloud MESURE sous les regles courantes
# (run 40100438746, 07ac : iter 93 champion_vs_heuristic +26..+41, aucune iteration acceptee devant lui).
MEASURED_CHAMPION_HASHES = {"1f49fc6fb6a25cae", "473e418c736d5419"}  # acb9 : champion 473e (iter 56) reexporte sans reentrainement
# (demande whist-4-it 9f25355, reentrainement acb9 a decider) ; e1f4 : champion 1f49 (iter 93) mesure sous 8f41 (run 40783453624,
# 19 portes, champion_vs_heuristic +12.9..+23.7, se ~3.4) ; 8f41 -> e1f4 ne change que la garde de Courant d'Air
# 65a7 : regles changees, le champion 6c54 doit repasser la porte
# 07ac (whist-4-it 9661351) : Dette de Jeu journalise {"event": "debt"} (aucune regle ne change) -> regles = c8f6.
# c8f6 (whist-4-it 0f73fa4) : Horloge masquee (pointless) avant le 8e pli, comme le port d4c4 -> regles = d4c4.
# d4c4 (whist-4-it, rapport admin 2026-10-23 "trop cheat") : Horloge Arretee seulement a partir du 8e pli -> reentrainement.
# b88c (whist-4-it 396c261) : consigne du joueur humain retiree (jamais posee a l'entrainement) -> regles = 6c54.
# 6c54 (whist-4-it, Chantier Rarete 02) : raretes/couts du tableau 01, deck de depart (cameleon, embuscade,
# main_lourde), DEFENCE = mesure jokers_defence, Main Lourde pioche 2 + defausse 1 (to_discard), Cameleon Valet,
# Courant d'Air la plus forte, Gueule de Bois la plus forte, Fausse Piste tout type sauf passif, Collet = ta
# prochaine carte, Zinneke +3, Treizieme 4 plis, Livre sans jauge pleine, Brouillard / Main de Fer cible
# couleur, Manche Truquee gagne d'office, Pot-de-Vin = 1er pli de sa misere a 0, Valet Borgne max(0, ...),
# Horloge : plis restants au camp en tete, nouveau pli_refile -> reentrainement.
# dc78 (whist-4-it 15cae20) : Soliste = maudit cavalier_seul (solo simple en toute couleur, free_solo), sa
# proposition de couleur retiree ; maudits ContratImpose (sans regle) des boss ; OPPONENT_ONLY -> reentrainement.
# e49f (whist-4-it fdf0754) : l'elite Solo sans solo legal propose sa couleur de solo -> reentrainement.
# d8f1 (whist-4-it c4bce36) : marge d'enchere par famille (emballage 1, solo 2, misere 2) -> reentrainement.
# 3d47 (whist-4-it a9a23cb, Chantier Elites 01) : contrat attitre par elite (deck fixe a doublons, biais
# d'enchere d'un pli la ou ExpertBot passe), elites de l'acte III adaptees en defense, decks des boss refaits
# -> reentrainement.
# 79ac (whist-4-it ebaf225) : elites/boss « spammers » (tout le deck pioche a chaque donne, Pacte du Tenancier
# en plus, +1..+4 ⚡ selon la Mise) ; mises portees (Rouge/Noire/Doree) -> reentrainement.
# 592e (whist-4-it 8a6f635) : joker joue comme une carte perd l egalite face a la vraie carte -> reentrainement.
# 540b (whist-4-it cdc86f6, Chantier Duel 03b) : Soupcon par siege ; Tournee Generale / Livre de Comptes
# agissent sur la jauge de LEUR lanceur (jokers non castables par les bots -> seul le joueur les lance, effet
# inchange en partie roguelike) ; is_opponent via _bot_contract (Trahison) et Oeil "points" du 2e joueur :
# Duel seulement -> regles identiques a 68f8.
# 68f8 (whist-4-it, 18/10) : niveaux de difficulte supprimes, rogue_match.gd appelle server_add_ai_bot sans
# WhistExpertBot.DIFFICULTY (ExpertBot sur les sieges 1-3 comme avant) -> regles identiques a 043f, les
# checkpoints 043f du cloud (whist-train) restent valables.
# 043f (whist-4-it 7b320b0, mesure de rentabilite du 2026-10-18) : Kermesse camp du proprietaire seulement,
# Main de Fer apres l'annonce du proprietaire (emballer sa proposition reste permis), Treizieme Carte 2 derniers
# plis (sans effet dans sa misere), Rachat salle (sans Soupcon), Pacte dette 5, Faux Pas +3 au proprietaire,
# Voeu / Croupier / Chope : pioches au lieu de l'energie, Silence du Cercle cout 1 toute la donne, couts Gueule
# de Bois 1, Remonte-Pli 1, Collet 0, Echange de Manche et Ascete retires -> reentrainement. Fenetre de cast du
# joueur avant l'entame d'un bot : a l'ecran seulement (trick_window false sans affichage), non modelisee.
# 3e5d (whist-4-it 2cc6f15) : ExpertBot sur les sieges 1-3 a tous les actes (EXPERT_SEATS_BY_ACT supprime) ;
# le simulateur jouait deja ExpertBot partout -> ecart de l'acte I resorbe, regles portees identiques a 55e8.
# 55e8 (whist-4-it c6f620e) : renommage roguelite -> roguelike (chemins + classes) ; be8c (3fa4851) : Moine max_hands 5 -> 6
# (le simulateur joue toujours HANDS_PER_MATCH = 5, max_hands non modelise, obs inchangee) + EchangeDeManche
# _send_hand_to_seat (affichage) -> regles portees identiques a cab7, iter 29 reexportee sans reentrainement.
# cab7 (whist-4-it, couts 17/10 depuis joker_profit v2, +-1 par nuit ; remplace 895d) : cout ⚡ de traitre, dette_de_jeu,
# borgne, valet_borgne, silence_du_cercle, faux_depart 1->2, carte_fantome 0->1, manche_truquee 2->3 -> reentrainement.
# Deck de depart (cameleon, coup_de_vent, embuscade) plafonne a 1 ⚡ : le tuto les joue tous les 3 dans une donne a 3 ⚡.
# fd03 (whist-4-it af9d87d + 25a14fa) : boss au siege 2 (en face), complice du Fossoyeur = voisin
# de gauche (PLAYER + 1) -> regles changees, reentrainement. Brim Dore -> Brimborion Dore (nom seul).
# 1bbb (whist-4-it cf7c0ff) : JokerCatalog._translated() (validation des textes, validate() seulement), regles identiques a 43b8.
# 8392 (whist-4-it 51ff230) : renommage trump -> atout dans joker_effects.gd, regles identiques a 43b8.
# 43b8 (whist-4-it 2c63f60 + 6bf4a7c, revue du 2026-10-14) : Rachat, Pot-de-Vin, Inversion
# (aura, lancee par les bots), Cartes sur Table (main exposee), Silence du Cercle, Fossoyeur,
# Brim Dore + 5 nouveaux jokers -> CASTABLE change, reentrainement.
OBS_VERSION = 3  # 2 = enemy_traps rempli ; 3 = cartes exactes, Soupcon/Oeil, jokers par siege (engine/joker_obs.py)

# id -> (type, rarete, cout, bot_casts) -- joker_catalog.gd / joker_effects.gd
CATALOG = {
    "cameleon": ("pli", "salle", 1, False),
    "zero": ("pli", "salle", 1, False),
    "traitre": ("pli", "salle", 2, False),
    "coup_de_vent": ("sort", "salle", 1, True),
    "main_lourde": ("sort", "comptoir", 0, True),
    "embuscade": ("piege", "salle", 1, True),
    "kermesse": ("aura", "salle", 2, True),
    "coup_d_oeil": ("enchere", "comptoir", 1, False),
    "regard_furtif": ("sort", "arriere_salle", 1, False),
    "voeu_de_pauvrete": ("passif", "salle", 0, False),
    "miroir": ("pli", "salle", 2, False),
    "borgne": ("pli", "arriere_salle", 2, False),
    "carte_fantome": ("pli", "salle", 2, False),
    "double_valet": ("pli", "salle", 1, False),
    "atout_errant": ("pli", "salle", 2, False),
    "courant_d_air": ("sort", "comptoir", 0, False),
    "tournee_generale": ("sort", "comptoir", 1, False),
    "gueule_de_bois": ("sort", "salle", 2, False),
    "remonte_pli": ("sort", "comptoir", 1, False),  # 4d7d : L'Hameçon, id inchange
    "la_peche": ("sort", "comptoir", 0, False),
    "faux_depart": ("sort", "salle", 1, False),
    "inversion": ("aura", "cave", 3, True),
    "fausse_piste": ("piege", "comptoir", 2, False),
    "collet": ("piege", "arriere_salle", 1, False),
    "contre_espionnage": ("piege", "comptoir", 1, False),
    "croupier_endormi": ("piege", "comptoir", 1, False),
    "dette_de_jeu": ("piege", "salle", 1, False),
    "fumee_de_pipe": ("aura", "comptoir", 1, False),
    "silence_du_cercle": ("aura", "cave", 3, False),
    "lune_noire": ("aura", "arriere_salle", 2, False),
    "coeur_de_zinneke": ("aura", "comptoir", 1, False),
    "brouillard_de_la_senne": ("aura", "cave", 2, False),
    "cartes_sur_table": ("enchere", "comptoir", 0, False),
    "bluff": ("enchere", "salle", 1, False),
    "rachat": ("enchere", "salle", 2, False),
    "main_de_fer": ("sort", "cave", 3, False),
    "pot_de_vin": ("enchere", "arriere_salle", 1, False),
    "fossoyeur_de_plis": ("passif", "salle", 0, False),
    "brimborion_dore": ("passif", "arriere_salle", 0, False),
    "manche_truquee": ("sort", "salle", 1, False),
    "coup_du_chapeau": ("aura", "arriere_salle", 1, True),
    "pourboire": ("sort", "comptoir", 1, False),
    "chope_vide": ("passif", "comptoir", 0, False),
    "faux_pas": ("sort", "comptoir", 1, True),
    "pli_refile": ("sort", "salle", 1, False),
    "valet_borgne": ("sort", "cave", 3, False),
    "horloge_arretee": ("sort", "salle", 2, False),
    "treizieme_carte": ("aura", "salle", 1, False),
    "livre_de_comptes": ("sort", "cave", 1, False),
    # 65a7 (whist-4-it f31712b) : jokers de misere, jamais lances par les bots
    "defausse_discrete": ("sort", "salle", 3, False),
    "meche_courte": ("aura", "comptoir", 1, False),
    "la_tournee": ("aura", "salle", 2, False),
    "cadeau_empoisonne": ("piege", "salle", 2, False),
    # 6483 (Synergies 04c) : jokers de combos, jamais lances par les bots
    "beau_valet": ("aura", "comptoir", 1, False),
    "bal_des_valets": ("aura", "comptoir", 1, False),
    "valet_de_chambre": ("sort", "comptoir", 0, False),
    "boite_d_allumettes": ("passif", "salle", 0, False),
    "poche_interieure": ("passif", "salle", 0, False),
    "feu_de_paille": ("sort", "salle", 0, False),
    "ardoise": ("sort", "comptoir", 0, False),
    "addition": ("sort", "comptoir", 1, False),
    "complice": ("sort", "comptoir", 0, False),
    "verre_de_trop": ("aura", "salle", 1, False),
    "refrain": ("sort", "comptoir", 1, False),
    "cendrier": ("passif", "salle", 0, False),
    "consigne": ("passif", "comptoir", 0, False),
    # 177e (Synergies 07) : reworks sous de nouveaux ids, jamais lances par les bots
    "pli_refile2": ("sort", "salle", 2, False),
    "coeur_de_zinneke2": ("aura", "comptoir", 0, False),
    "chope_vide2": ("passif", "comptoir", 0, False),
    "croupier_endormi2": ("piege", "comptoir", 1, False),
    "faux_depart2": ("sort", "arriere_salle", 2, False),
    "la_tournee2": ("aura", "salle", 2, False),
    "cadeau_empoisonne2": ("piege", "salle", 2, False),
    # 61ef (Synergies 09) : jamais lances par les bots
    "vieille_habitude": ("passif", "comptoir", 0, False),
    "pot_commun": ("sort", "comptoir", 0, False),
    "trouble_fete": ("aura", "salle", 1, False),
    "mauvais_perdant": ("passif", "comptoir", 0, False),
    "cierge": ("aura", "cave", 2, False),
    "patron": ("passif", "cave", 0, False),
    "cartes_marquees": ("passif", "maudit", 0, False),
    "joueur_fantome": ("passif", "maudit", 0, False),
    "pacte_du_tenancier": ("passif", "maudit", 0, False),
    "heritage_empoisonne": ("passif", "maudit", 0, False),
    # maudits des adversaires (dc78) : raison du contrat impose d'un boss, solo libre de la Soliste
    "regle_du_moine": ("passif", "maudit", 0, False),
    "gant_de_la_baronne": ("passif", "maudit", 0, False),
    "pelle_du_fossoyeur": ("passif", "maudit", 0, False),
    "sceau_du_grand_maitre": ("passif", "maudit", 0, False),
    "cavalier_seul": ("passif", "maudit", 0, False),
    # a76f (whist-4-it, CHANTIER-REWORK-JOKERS-PLAN.md « Code (2026-11-06) ») : les inutiles
    # recycles « a la Dominion », tous des sorts ; bot_casts vrai (DeckEngine), mais HORS du
    # CASTABLE du reseau (DOMINION_IDS) : aucun deck de bot ne les tient, ACTION_DIM inchange.
    "ardoise2": ("sort", "comptoir", 0, True),
    "tournee_generale2": ("sort", "comptoir", 1, True),
    "valet_de_chambre2": ("sort", "comptoir", 0, True),
    "addition2": ("sort", "comptoir", 1, True),
    "vieille_habitude2": ("sort", "comptoir", 1, True),
    "chope_vide3": ("sort", "comptoir", 1, True),
    "complice2": ("sort", "comptoir", 1, True),
    "consigne2": ("sort", "comptoir", 0, True),
    "cendrier2": ("sort", "salle", 0, True),
    "mauvais_perdant2": ("sort", "comptoir", 1, True),
    "feu_de_paille2": ("sort", "salle", 0, True),
    "poche_interieure2": ("sort", "salle", 0, True),
    "la_peche2": ("sort", "salle", 1, True),
}
# JokerCatalog.OPPONENT_ONLY : jamais proposes au joueur (chaise_vide, deck simule)
OPPONENT_ONLY = ("regle_du_moine", "gant_de_la_baronne", "pelle_du_fossoyeur", "sceau_du_grand_maitre", "cavalier_seul")
# JokerCatalog.REPLACED (177e) : ancien -> nouveau ; l'ancien reste au catalogue, hors des decks et des mesures
REPLACED = {"carte_fantome": "zero", "pli_refile": "pli_refile2", "coeur_de_zinneke": "coeur_de_zinneke2",
            "chope_vide": "chope_vide3", "croupier_endormi": "croupier_endormi2", "faux_depart": "faux_depart2",
            "la_tournee": "la_tournee2", "cadeau_empoisonne": "cadeau_empoisonne2"}
# JokerCatalog.DOMINION_REPLACED (a76f) : restent pour les runs d'avant et les decks des boss,
# hors de all_ids() -- ici fusionnes dans REPLACED (hors des decks du joueur et des mesures).
DOMINION_REPLACED = {"ardoise": "ardoise2", "tournee_generale": "tournee_generale2", "valet_de_chambre": "valet_de_chambre2",
                     "addition": "addition2", "vieille_habitude": "vieille_habitude2", "chope_vide2": "chope_vide3",
                     "complice": "complice2", "consigne": "consigne2", "cendrier": "cendrier2",
                     "mauvais_perdant": "mauvais_perdant2", "feu_de_paille": "feu_de_paille2",
                     "poche_interieure": "poche_interieure2", "la_peche": "la_peche2"}
DOMINION_IDS = frozenset(DOMINION_REPLACED.values())
REPLACED.update(DOMINION_REPLACED)
ON_TURN_IDS = ("coup_de_vent", "faux_pas")  # JokerDef.ON_TURN_IDS : a son tour de pli, apres avoir vu une coupe
STARTING_DECK = ["cameleon", "embuscade", "main_lourde"]  # JokerCatalog.STARTING_DECK
VALET_FAMILY = ("cameleon", "double_valet", "beau_valet", "bal_des_valets", "valet_de_chambre", "valet_de_chambre2")  # JokerCatalog.FAMILIES["valets"]
UPGRADE_SUFFIX = "+"  # JokerDef : cout -1 (jamais un cout 0 ni un maudit)
MAX_COPIES = 2
SUSPICION_BY_RARITY = {"arriere_salle": 2, "cave": 4}  # JokerDef.SUSPICION_BY_RARITY
FROM_DECK_IDS = ("brimborion_dore", "patron")  # JokerDef.FROM_DECK_IDS : jamais pioches, agissent depuis le deck
BOT_TYPES = ("sort", "piege", "aura")  # RogueMatch.BOT_TYPES
# Actions de l'IA : 0 = aucun cast, 1..K = ces jokers (ids tries).
CASTABLE = sorted(i for i, (t, _r, _c, bc) in CATALOG.items() if bc and t in BOT_TYPES and i not in DOMINION_IDS)
ACTION_DIM = 1 + len(CASTABLE)

DRAW_PER_HAND = 3  # MatchState
SPAMMER_DRAWS = 5  # 65a7 (ff4b72a) : elites/boss/habitues piochent 5 jokers (tout leur deck avant)
DISCARD_GAIN = 1  # 65a7 (48b2dc3) : un joker defausse de la main = +1 energie
LASTING_CASES = 2  # 65a7 (d74be81) : effets de donne par siege (JokerRules.LASTING_CASES)
SPAMMER_CASES = 3
ENERGY_PER_HAND = 2  # 22d8 (07/11, « Ok on passe a 2 ») : a 3 les ⚡ ne manquaient jamais (1 ⚡ = 0.1 pt)
LEGACY_ENERGY_PER_HAND = 3  # runs d'avant (dominion faux)
SUSPICION_MAX = 10
HEAT_ENERGY_AT = 3  # Synergies 04b : paliers de Chaleur du siege 0 (MatchState)
HEAT_DRAW_AT = 6
CARRY_MAX = 6  # Boite d'Allumettes : le plus qu'une donne ouvre grace aux ⚡ reportes
ELITE_SEATS = (1, 3)
ELITE_EYE = "points"
EYE_POINTS = 20  # RogueMatch
NO_ENERGY = 99
# Elites et boss « spammers » (RogueMatch.SPAMMER_EXTRA, RogueMises.SPAMMER_ENERGY, 79ac)
SPAMMER_EXTRA = ["pacte_du_tenancier"]
SPAMMER_ENERGY = [1, 2, 3, 4]
MISE_COUNT = 4
RED_ACT = 3  # RogueMises
RED_MALUS = 1
BLACK_EARLIER = 2


def boss_deck(boss, heat):
    """RogueBosses.deck."""
    return list(BOSS_COMBO_DECKS.get(boss, BOSS_DECKS[boss]) if heat else BOSS_DECKS[boss])


def spammer_energy(mise):
    return SPAMMER_ENERGY[min(max(mise, 0), MISE_COUNT - 1)]


def mise_energy_malus(mise, act):
    """Mise Rouge : -1 ⚡ par donne au joueur en acte III."""
    return RED_MALUS if mise >= 1 and act == RED_ACT else 0


def eye_threshold(mise):
    """Mise Noire : l'Oeil prend des 8 de Soupcon."""
    return SUSPICION_MAX - (BLACK_EARLIER if mise >= 2 else 0)


def second_rule(mise, boss):
    """Mise Doree : la regle d'un autre boss ("" sinon)."""
    if mise < 3 or boss == "":
        return ""
    return "moine" if boss == "baronne" else "baronne"
RARITIES_BY_ACT = {1: ("comptoir", "salle"), 2: ("comptoir", "salle", "arriere_salle"),
                   3: ("comptoir", "salle", "arriere_salle", "cave")}
BOSS_SEAT = 2  # en face du joueur (rapport 2026-10-15)
PLAYER = 0
BOSS_DECKS = {  # RogueBosses.DEFS (3d47 : decks pour leur contrat ; dc78 : + contract_joker en fin)
    "moine": ["inversion", "inversion", "coup_du_chapeau", "faux_pas", "regle_du_moine"],
    "baronne": ["kermesse", "kermesse", "coup_de_vent", "coup_de_vent", "inversion", "gant_de_la_baronne"],
    "fossoyeur": ["coup_de_vent", "coup_de_vent", "inversion", "kermesse", "kermesse", "pelle_du_fossoyeur"],
    "grand_maitre": ["kermesse", "kermesse", "kermesse", "coup_de_vent", "coup_de_vent", "inversion", "main_lourde",
                     "sceau_du_grand_maitre"],
}
# RogueBosses.DEFS combo_deck (61ef) : run a Chaleur ; le Moine garde son deck
BOSS_COMBO_DECKS = {
    "baronne": ["kermesse", "kermesse", "coup_de_vent", "coup_de_vent", "complice", "gant_de_la_baronne"],
    "fossoyeur": ["coup_de_vent", "coup_de_vent", "kermesse", "kermesse", "main_lourde", "main_lourde", "consigne",
                  "cendrier", "la_peche", "pelle_du_fossoyeur"],
    "grand_maitre": ["kermesse", "kermesse", "kermesse", "coup_de_vent", "coup_de_vent", "main_lourde",
                     "silence_du_cercle", "sceau_du_grand_maitre"],
}
BOSS_SANCTIONS = {"moine": "points", "baronne": "confisque", "fossoyeur": "allumettes", "grand_maitre": "confisque"}
FACE_RANKS = (9, 10, 11)  # J, Q, K (engine/cards.py : rang 12 = As)
ACE = 12
HEARTS = 0  # WhistCard.SUITS : ♥ ♦ ♣ ♠ = 0..3
RED_SUITS = (0, 1)
BLACK_TWIN = {0: 3, 1: 2}  # Lune Noire : ♥ compte comme ♠, ♦ comme ♣


def base_id(jid):
    return jid[:-1] if jid.endswith(UPGRADE_SUFFIX) else jid


def known(jid):
    return base_id(jid) in CATALOG and (jid == base_id(jid) or can_upgrade(base_id(jid)))


def jtype(jid):
    return CATALOG[base_id(jid)][0]


def jrarity(jid):
    return CATALOG[base_id(jid)][1]


def jcost(jid):
    cost = CATALOG[base_id(jid)][2]
    return cost - 1 if jid.endswith(UPGRADE_SUFFIX) else cost


def timing(jid):
    """JokerDef.timing : "bidding", "card", "trick_start" (fenetre), "turn", "none"."""
    t = jtype(jid)
    if t == "enchere":
        return "bidding"
    if t == "pli":
        return "card"
    if t == "passif":
        return "none"
    return "turn" if base_id(jid) in ON_TURN_IDS else "trick_start"


def jsuspicion(jid):
    return SUSPICION_BY_RARITY.get(jrarity(jid), 0)


def acts_from_deck(jid):
    """JokerDef.acts_from_deck : maudits + FROM_DECK_IDS (MatchState.curses)."""
    return jrarity(jid) == "maudit" or base_id(jid) in FROM_DECK_IDS


def can_upgrade(jid):
    return not jid.endswith(UPGRADE_SUFFIX) and CATALOG[jid][2] > 0 and CATALOG[jid][1] != "maudit"


# --- elites (elite_decks.gd, 3d47) --------------------------------------------
SPECIALTY_DECKS = {  # EliteDecks.SPECIALTIES[fam].deck
    "emballage": ["coup_de_vent", "coup_de_vent", "inversion", "kermesse"],
    "solo": ["kermesse", "kermesse", "coup_de_vent", "main_lourde", "cavalier_seul"],
    "misere": ["inversion", "inversion", "coup_du_chapeau", "faux_pas"],
}
FAMILIES = ["emballage", "solo", "misere"]
DEFENCE = {  # contre le type du contrat, pour une elite adaptee en defense (6c54 : mesure jokers_defence dc78)
    "emballage": ["inversion", "inversion", "coup_du_chapeau", "coup_de_vent"],
    "solo": ["inversion", "inversion", "faux_pas", "coup_du_chapeau"],
    "misere": ["faux_pas", "faux_pas", "coup_du_chapeau", "coup_du_chapeau"],
    "trou": ["inversion", "inversion", "coup_du_chapeau", "kermesse"],
}
ADAPT_FROM_ACT = 3
BID_SLACK = {"emballage": 1, "solo": 1, "misere": 1}  # plis de marge par famille (d8f1)


def elite_specialties(rng):
    """EliteDecks.for_elites : sieges 1 puis 3, deux familles differentes."""
    pool = list(FAMILIES)
    return {seat: pool.pop(rng.randint(0, len(pool) - 1)) for seat in ELITE_SEATS}


def habitue_families(rng, seats=(1, 2, 3)):
    """DuelRules.habitue_decks (07ac, rapport 2026-10-23) : chaque habitue du
    Duel tire une famille d'elite de la graine du duel (doublons permis)."""
    return {seat: FAMILIES[rng.randint(0, len(FAMILIES) - 1)] for seat in seats}


def habitue_deck(family):
    """Le deck d'elite de `family` sans ses maledictions (Cavalier Seul)."""
    return [j for j in SPECIALTY_DECKS[family] if CATALOG[base_id(j)][1] != "maudit"]


def attackers(contract):
    """EliteDecks.attackers : les sieges qui jouent le contrat."""
    if "declarers" in contract:
        return contract["declarers"]
    return [contract.get("declarer", -1), contract.get("partner", -1)]


def _declares(seat, contract):
    return (seat in (contract.get("declarers") or []) or seat == contract.get("declarer", -1)
            or seat == contract.get("partner", -1))


def _weakest(cards):
    """Godot : la carte de plus grand rank_index (la plus faible), premiere
    rencontree en cas d'egalite."""
    low = cards[0]
    for c in cards:
        if rank_of(c) < rank_of(low):
            low = c
    return low


def _strongest(cards):
    """Godot : la carte de plus petit rank_index (la plus forte), premiere
    rencontree en cas d'egalite."""
    high = cards[0]
    for c in cards:
        if rank_of(c) > rank_of(high):
            high = c
    return high


def longest_suit(hand):
    counts = [sum(1 for c in hand if suit_of(c) == s) for s in range(4)]
    return counts.index(max(counts))


# --- effets (joker_effects.gd) ---------------------------------------------

TRICK, HAND, INSTANT = "trick", "hand", "instant"
PENDING = "pending"  # 6483 : sans case, jusqu'a ce qu'il soit paye (le prix du prochain joker)


class Effect:
    """JokerEffects.Base : TOUS les hooks du jeu, neutres par defaut. Un
    nouveau joker = une sous-classe qui surcharge ce qu'il change + une
    entree dans CATALOG et EFFECTS (registre generique, aucune reecriture).
    `play` = (siege, carte) ; `rules` = JokerRules."""
    scope = INSTANT

    def __init__(self, jid, owner):
        self.id = jid
        self.owner = owner
        self.spent = False

    def make_card(self, led_suit, rules):  # cartes de pli : la carte posee dans le pli
        return None

    def needs_target(self):
        return False

    def target_kind(self):
        """Ce que needs_target() choisit : "seat", ou "suit" (indice de couleur)."""
        return "seat"

    def accepts_target(self, seat, rules):
        """65a7 : un siege que needs_target() peut viser maintenant (le HUD n'offre que ceux-la)."""
        return True

    def on_play(self, rules, target=-1):
        return {}

    def trump(self, trump, rules):
        return trump

    def suit_as(self, suit):
        """La couleur que `suit` vaut tant que cet effet vit (Lune Noire)."""
        return suit

    def competes(self, play, rules):
        return True

    def raises(self, play):
        """6483 : True = le pli voit ce coup au-dessus de tous les rangs (Bal des Valets)."""
        return False

    def discount(self, jid):
        """6483 : ce que cet effet retire du prix de `jid` tant qu'il vit."""
        return 0

    def on_joker_paid(self, seat, jid, rules):
        """`seat` a paye `jid`, deja compte dans played_this_hand."""
        pass

    def discard_energy(self):
        """⚡ que ce joker rapporte lui-meme defausse (remplace DISCARD_GAIN)."""
        return 0

    def discard_bonus(self, seat, jid):
        """⚡ de plus pour `jid` defausse par `seat`."""
        return 0

    def inverts(self):
        return False

    def trick_value(self, value, winner_play, rules):
        return value

    def voids_trick(self, rules):
        return False

    def on_trick_resolved(self, winner, plays, rules):
        pass

    def bot_wants(self, rules):
        return False

    def bot_casts(self):
        return False

    def pointless(self, rules):
        """Rien a changer dans le contrat de la donne -> injouable (whist-4-it
        bf79074, signalement 2026-10-12 : Coup de Vent en Grande Misere)."""
        return False

    def pointless_reason(self):
        """a76f : la cle d'erreur quand pointless() tient (JokerUi.ERROR_TEXTS)."""
        return "pointless"

    def energy_bonus(self):
        return 0

    def adjust_deltas(self, deltas, contract, rules):
        return deltas

    def blocks(self, jid, seat, rules):
        """True = `seat` ne peut pas jouer `jid` maintenant."""
        return False

    def plays_as_trump(self, play, rules):
        """True = cette carte prend le pli comme l'atout le plus fort (Manche Truquee)."""
        return False

    def takes_trick(self, play, rules):
        """65a7 : True = ce coup prend le pli, quoi qu'il y ait d'autre (Cadeau Empoisonne)."""
        return False

    def legal_cards(self, seat, hand, led, legal, rules):
        """65a7 : les cartes que `seat` peut jouer de `hand`, `legal` = celles
        que les regles permettent jusqu'ici (jokers de misere)."""
        return legal

    def on_hand_end(self, rules):
        """Donne comptee, avant la defausse de ses jokers (Chope Vide)."""
        pass

    def cancels(self, seat, jid, rules):
        return False

    def bot_card(self, seat, card, legal, rules):
        return card

    def forces_pass(self, seat, action, bidding):
        """bot_bid : True = l'annonce `action` ({"action": ...}) du bot `seat`
        devient un passe."""
        return False

    def prepare_bidding(self, bidding):
        """RuleModifiers.prepare_bidding : enchere neuve, avant toute annonce (dc78)."""
        pass

    def next_leader(self, winner):
        return -1

    def tricks_in_hand(self, classic):
        return classic

    def hides_suspicion(self, seat):
        return False

    def absorbs_sanction(self):
        return False

    def on_contract(self, contract, rules):
        return False

    def cost_override(self, jid):
        return -1

    def draw_bonus(self):
        return 0

    def returns_owed(self):
        """Jokers que le proprietaire remet sous sa pioche apres avoir pioche (Le Patron)."""
        return 0

    def on_hand_scored(self, deltas, rules):
        """`deltas` finaux de la donne, tous les jokers compris."""
        pass

    def freezes_suspicion(self):
        return False


class TrickJoker(Effect):
    scope = TRICK

    def _own_card(self, play, rules):
        return play[0] == self.owner and rules.joker_cards.get(self.owner) == self.id


class Cameleon(TrickJoker):
    def make_card(self, led_suit, rules):
        return led_suit * 13 + 9  # Valet (6c54, une Dame avant)


class Zero(TrickJoker):
    def make_card(self, led_suit, rules):
        return led_suit * 13 + 0  # 2

    def competes(self, play, rules):
        return not self._own_card(play, rules)


class Traitor(Zero):
    def inverts(self):
        return True


class CoupDeVent(Effect):
    scope = TRICK

    def trump(self, trump, rules):
        return None

    def pointless(self, rules):
        return rules.trick_trump() is None

    def bot_casts(self):
        return True

    def bot_wants(self, rules):
        trump = rules.contract_trump()
        led = rules.led_suit()
        return trump is not None and led is not None and led != trump and any(
            rules.is_opponent(self.owner, s) and suit_of(c) == trump for s, c in rules.game.current_trick)


class MainLourde(Effect):
    """6c54 : pioche DRAWS, puis le proprietaire defausse un joker, une fois
    celui-ci paye (RogueMatch._spend -> MatchState.must_discard)."""
    DRAWS = 2

    def bot_casts(self):
        return True

    def bot_wants(self, rules):
        return True

    def on_play(self, rules, target=-1):
        rules.state.draw_jokers(self.owner, self.DRAWS)
        return {"discard": 1}


class Embuscade(Effect):
    scope = HAND

    def pointless(self, rules):
        return rules.trick_trump() is None

    def bot_casts(self):
        return True

    def bot_wants(self, rules):
        return True

    def _is_enemy_ruff(self, play, rules):
        seat, card = play
        return (rules.is_opponent(self.owner, seat) and rules.current_trump is not None
                and rules.current_led != rules.current_trump and suit_of(card) == rules.current_trump)

    def competes(self, play, rules):
        return not self._is_enemy_ruff(play, rules)

    def on_trick_resolved(self, winner, plays, rules):
        if any(self._is_enemy_ruff(p, rules) for p in plays):
            self.spent = True


class Kermesse(Effect):
    scope = HAND

    def bot_casts(self):
        return True

    def bot_wants(self, rules):
        return True

    def trick_value(self, value, winner_play, rules):
        """043f : plis du camp du proprietaire seulement."""
        mine = not rules.is_opponent(self.owner, winner_play[0])
        return value + 1 if mine and rank_of(winner_play[1]) in FACE_RANKS else value


class CoupDOeil(Effect):
    """Information seulement (l'ExpertBot du siege 0 ne s'en sert pas)."""
    def needs_target(self):
        return True

    def on_play(self, rules, target=-1):
        if target < 0 or target == self.owner:
            return {"error": "target"}
        return {"seen": True}


class RegardFurtif(CoupDOeil):
    pass


class VoeuDePauvrete(Effect):
    """043f : un joker pioche tous les LOST plis perdus (plus d'energie)."""
    scope = HAND
    LOST = 3

    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.lost = 0

    def on_trick_resolved(self, winner, plays, rules):
        if winner == self.owner:
            return
        self.lost += 1
        if self.lost % self.LOST == 0:
            rules.state.draw_jokers(self.owner, 1)


class Curse(Effect):
    scope = HAND


class PacteDuTenancier(Curse):
    DEBT = 5  # 043f (10 avant)

    def energy_bonus(self):
        return 2

    def adjust_deltas(self, deltas, contract, rules):
        if _declares(self.owner, contract) and deltas[self.owner] < 0:
            deltas[self.owner] -= self.DEBT
        return deltas


class HeritageEmpoisonne(Curse):
    pass


class ContratImpose(Curse):
    """Maudit d'un boss (dc78) : sans regle, le contrat reste boss_contract."""
    pass


class CavalierSeul(Curse):
    """Maudit de la Soliste (dc78) : solo simple dans n'importe quelle couleur."""
    def prepare_bidding(self, bidding):
        bidding.free_solo[self.owner] = True


class Miroir(TrickJoker):
    def make_card(self, led_suit, rules):
        plays = rules.game.current_trick if rules.game is not None else []
        return plays[-1][1] if plays else led_suit * 13 + 0


class Borgne(TrickJoker):
    def make_card(self, led_suit, rules):
        return led_suit * 13 + ACE

    def competes(self, play, rules):
        if not self._own_card(play, rules) or rules.current_trump is None or rules.current_led == rules.current_trump:
            return True
        return not any(suit_of(c) == rules.current_trump and s != self.owner for s, c in rules.current_plays)


class CarteFantome(Zero):
    def competes(self, play, rules):
        return True

    def voids_trick(self, rules):
        return True


class DoubleValet(TrickJoker):
    def make_card(self, led_suit, rules):
        return led_suit * 13 + 9  # Valet

    def trick_value(self, value, winner_play, rules):
        return value + 1 if winner_play[0] == self.owner else value


class AtoutErrant(TrickJoker):
    def make_card(self, led_suit, rules):
        trump = rules.trick_trump() if rules.game is not None else None
        return trump * 13 + 0 if trump is not None else led_suit * 13 + 9


class CourantDAir(Effect):
    def on_play(self, rules, target=-1):
        g = rules.game
        hand = g.hands[self.owner]
        if len(hand) <= rules.classic_tricks() - g.trick_count:  # e1f4 : pas hand_tricks (Horloge retiree ensuite)
            return {"error": "hand"}
        strongest = _strongest(hand)  # 6c54 : la plus forte (la plus faible avant)
        hand.remove(strongest)
        rules.vanished.append(strongest)  # JokerRules.set_aside : sous le paquet ramasse (Match.finish)
        rules.state.draw_jokers(self.owner, 1)
        return {"discarded": strongest}


class TourneeGenerale(Effect):
    RELIEF = 2

    def on_play(self, rules, target=-1):
        rules.state.lower_suspicion(self.RELIEF)
        return {}


class GueuleDeBois(Effect):
    scope = TRICK

    def bot_card(self, seat, card, legal, rules):
        if not rules.is_opponent(self.owner, seat) or not legal:
            return card
        return _strongest(legal)  # 6c54 : la plus forte (la plus faible avant)


class RemontePli(Effect):
    """4d7d, L'Hameçon (id garde) : une carte au hasard de la main visee contre
    la plus faible du lanceur."""
    def needs_target(self):
        return True

    def accepts_target(self, seat, rules):
        return seat != self.owner

    def on_play(self, rules, target=-1):
        if target < 0 or target == self.owner:
            return {"error": "target"}
        theirs, mine = rules.game.hands[target], rules.game.hands[self.owner]
        if not theirs or not mine:
            return {"error": "none"}
        hooked = theirs[rules.rng.randrange(len(theirs))]
        weakest = min(mine, key=rank_of)
        theirs.remove(hooked)
        mine.remove(weakest)
        theirs.append(weakest)
        mine.append(hooked)
        return {"hooked": hooked}


class LaPeche(Effect):
    """4d7d : le dernier joker de la defausse du lanceur revient en main."""
    def pointless(self, rules):
        return not rules.state.discard[self.owner]

    def on_play(self, rules, target=-1):
        return {} if rules.state.take_back(self.owner) else {"error": "none"}


class PliRefile(Effect):
    """6c54 : le dernier pli du proprietaire passe a l'adversaire choisi."""
    def needs_target(self):
        return True

    def on_play(self, rules, target=-1):
        if target < 0 or not rules.is_opponent(self.owner, target):
            return {"error": "target"}
        g = rules.game
        if self.owner not in rules.trick_winners or g.tricks_won[self.owner] <= 0:
            return {"error": "none"}
        g.tricks_won[self.owner] -= 1
        g.tricks_won[target] += 1
        return {"to": target}


class FauxDepart(Effect):
    scope = TRICK

    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.target = -1

    def needs_target(self):
        return True

    def on_play(self, rules, target=-1):
        if target < 0:
            return {"error": "target"}
        self.target = target
        return {}

    def next_leader(self, winner):
        return self.target


class Inversion(Effect):
    """Revue 2026-10-14 : une aura, chaque pli de la donne a la plus faible
    carte de la couleur demandee."""
    scope = HAND

    def inverts(self):
        return True

    def bot_casts(self):
        return True

    def bot_wants(self, rules):
        return (rules.contract or {}).get("type") == "misere"


class Trap(Effect):
    scope = HAND


class FauxPas(Effect):
    """dcf1 : la renonce rendue legale, pour ce pli le proprietaire peut jouer
    atout alors qu'il a la couleur demandee."""
    scope = TRICK

    def bot_casts(self):
        return True

    def _trumps(self, hand, rules):
        t = rules.trick_trump()
        return [c for c in hand if rules.card_suit(c) == t]

    def pointless(self, rules):
        return rules.trick_trump() is None

    def legal_cards(self, seat, hand, led, legal, rules):
        if seat != self.owner:
            return legal
        return list(legal) + [c for c in self._trumps(hand, rules) if c not in legal]

    def _takes_it(self, card, rules):
        plays = list(rules.game.current_trick) + [(self.owner, card)]
        return rules.trick_winner(plays, rules.led_suit(), rules.trick_trump()) == self.owner

    def bot_wants(self, rules):
        g, led = rules.game, rules.led_suit()
        hand = g.hands[self.owner]
        if led is None or rules.suit_name(led) == rules.trick_trump():
            return False
        return (not any(self._takes_it(c, rules) for c in legal_plays(hand, led))
                and any(self._takes_it(c, rules) for c in self._trumps(hand, rules)))

    def bot_card(self, seat, card, legal, rules):
        if seat != self.owner:
            return card
        for c in sorted(self._trumps(rules.game.hands[self.owner], rules), key=rank_of):
            if self._takes_it(c, rules):
                return c
        return card


class FaussePiste(Trap):
    """6c54 : tout joker adverse (un passif s'arme a la pioche, il n'est pas lance)."""
    def cancels(self, seat, jid, rules):
        if self.spent or jtype(jid) == "passif" or not rules.is_opponent(self.owner, seat):
            return False
        self.spent = True
        return True


class Collet(Trap):
    """6c54 : la prochaine carte du proprietaire ne peut pas gagner son pli."""
    def competes(self, play, rules):
        return self.spent or play[0] != self.owner

    def on_trick_resolved(self, winner, plays, rules):
        if any(p[0] == self.owner for p in plays):
            self.spent = True


class ContreEspionnage(Trap):
    def absorbs_sanction(self):
        if self.spent:
            return False
        self.spent = True
        return True


class CroupierEndormi(Trap):
    STREAK = 3
    DRAWS = 2  # 043f : jokers pioches (+2 energie avant)

    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.lost = 0

    def on_trick_resolved(self, winner, plays, rules):
        self.lost = self.lost + 1 if winner != self.owner else 0
        if self.lost >= self.STREAK:
            rules.state.draw_jokers(self.owner, self.DRAWS)
            self.spent = True


class DetteDeJeu(Effect):
    scope = TRICK
    DEBT = 5

    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.target = -1

    def needs_target(self):
        return True

    def on_play(self, rules, target=-1):
        if target < 0 or not rules.is_opponent(self.owner, target):
            return {"error": "target"}
        self.target = target
        return {}

    def on_trick_resolved(self, winner, plays, rules):
        if winner == self.target:
            rules.add_score(self.target, -self.DEBT)
            rules.add_score(self.owner, self.DEBT)


class Aura(Effect):
    scope = HAND


class FumeeDePipe(Aura):
    def hides_suspicion(self, seat):
        return seat == self.owner


class SilenceDuCercle(Aura):
    """043f : toute la donne (3 plis avant)."""
    def blocks(self, jid, seat, rules):
        return rules.is_opponent(self.owner, seat)  # revue 2026-10-14 : adversaires seulement, tout type


class LuneNoire(Aura):
    """dcf1 : ♥ compte comme ♠ et ♦ comme ♣ jusqu'a la fin de la donne, atout
    compris ; entre jumelles, la vraie noire gagne (JokerRules.trick_winner)."""
    def suit_as(self, suit):
        return BLACK_TWIN.get(suit, suit)

    def trump(self, trump, rules):
        return self.suit_as(trump)

    def legal_cards(self, seat, hand, led, legal, rules):
        """Ignore `legal` : passe avant les autres (Faux Pas, Defausse Discrete y ajoutent)."""
        if led is None:
            return legal
        follow = [c for c in hand if self.suit_as(suit_of(c)) == self.suit_as(led)]
        return follow if follow else list(hand)


class CoeurDeZinneke(Aura):
    BONUS = 3  # 6c54

    def on_trick_resolved(self, winner, plays, rules):
        for seat, card in plays:
            if seat == winner and winner == self.owner and rules.card_suit(card) == HEARTS:
                rules.add_score(self.owner, self.BONUS)


class BrouillardDeLaSenne(Aura):
    """6c54 : la couleur choisie au lancement est l'atout des plis qu'entame
    le proprietaire jusqu'a la fin de la donne."""
    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.suit = None

    def needs_target(self):
        return True

    def target_kind(self):
        return "suit"

    def on_play(self, rules, target=-1):
        if target < 0 or target >= 4:
            return {"error": "target"}
        self.suit = target
        return {}

    def trump(self, trump, rules):
        g = rules.game
        leader = g.current_trick[0][0] if g.current_trick else g.turn
        return self.suit if leader == self.owner and self.suit is not None else trump


class CartesSurTable(Aura):
    """Revue 2026-10-14 : le contrat porte le siege expose ; les ExpertBots
    jouent en voyant sa main (WhistExpertBot._revealed, PublicView.revealed)."""
    def on_play(self, rules, target=-1):
        rules.state.gain_energy(self.owner, 1)
        self._expose(rules.contract, rules)
        return {"revealed": True}

    def on_contract(self, contract, rules):
        return self._expose(contract, rules)

    def _expose(self, contract, rules):
        if not contract or self.owner in contract.get("exposed", []):
            return False
        contract["exposed"] = list(contract.get("exposed", [])) + [self.owner]
        if rules.game is not None:
            rules.game.exposed = list(contract["exposed"])
        return True


class Bluff(Aura):
    def adjust_deltas(self, deltas, contract, rules):
        if _declares(self.owner, contract) and deltas[self.owner] < 0:
            deltas[self.owner] = int(deltas[self.owner] / 2)  # division entiere GDScript (vers 0)
        return deltas


class Rachat(Aura):
    """Rapport 2026-10-14 : le camp du proprietaire compte un pli de plus a
    la fin de la donne (pris au declarant, sinon a son partenaire, quand le
    proprietaire defend). Jamais en misere."""
    def adjust_deltas(self, deltas, contract, rules):
        ctype = contract.get("type", "")
        if ctype in ("", "misere") or rules.scorer is None:
            return deltas
        tricks = rules.game.tricks_won
        saved = list(tricks)
        base = rules.score_hand()
        side = [contract.get("declarer", -1), contract.get("partner", -1)]
        side = [-1 if x is None else x for x in side]
        if self.owner in side:
            tricks[self.owner] += 1
        else:
            giver = side[0] if side[0] >= 0 and tricks[side[0]] > 0 else side[1]
            if giver < 0 or tricks[giver] <= 0:
                return deltas
            tricks[giver] -= 1
            tricks[self.owner] += 1
        bought = rules.score_hand()
        tricks[:] = saved
        return [d + b - a for d, a, b in zip(deltas, base, bought)]


class MainDeFer(Effect):
    """7567 : le contrat connu, avant la premiere carte, le proprietaire change
    l'atout d'un emballage ou d'un solo adverse pour la couleur choisie."""
    TYPES = ("emballage", "solo")

    def needs_target(self):
        return True

    def target_kind(self):
        return "suit"

    def pointless(self, rules):
        g, c = rules.game, rules.contract
        return (g.trick_count > 0 or bool(g.current_trick) or c.get("type", "") not in self.TYPES
                or c.get("suit") is None or _declares(self.owner, c))

    def on_play(self, rules, target=-1):
        if target < 0 or target >= 4 or target == rules.contract.get("suit"):
            return {"error": "target"}
        rules.contract["suit"] = target
        rules.game.trump_suit = target
        return {}


class PotDeVin(Aura):
    """6c54 : dans la misere du proprietaire, le premier pli qu'il remporte
    ne compte pour personne (le bot vise passait avant)."""
    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.used = False

    def trick_value(self, value, winner_play, rules):
        c = rules.contract or {}
        if (self.used or winner_play[0] != self.owner or c.get("type", "") != "misere"
                or self.owner not in (c.get("declarers") or [])):
            return value
        self.used = True
        return 0


class FossoyeurDePlis(Aura):
    def on_trick_resolved(self, winner, plays, rules):
        if (rules.contract or {}).get("type") in ("emballage", "trou") and winner != self.owner and not rules.is_opponent(self.owner, winner):
            rules.add_score(self.owner, 1)


class BrimborionDore(Curse):
    pass  # agit depuis le deck ; donne a points doubles : absente de la simulation


class ValetBorgne(Aura):
    def on_play(self, rules, target=-1):
        return {"exhaust": True}

    def adjust_deltas(self, deltas, contract, rules):
        deltas[self.owner] = abs(deltas[self.owner])  # 8f41 : ses points perdus deviennent gagnes
        return deltas


class HorlogeArretee(Aura):
    """6c54 + d4c4 (a partir du 8e pli) : la donne s'arrete apres ce pli ; les plis restants vont au camp
    qui en a le plus (aucun en cas d'egalite), a son siege qui en a le plus."""
    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.stop_at = 13
        self.rest = 0
        self.given = False

    FROM_TRICK = 6  # d4c4 : 8e pli ; 8f41 : 6e

    def pointless(self, rules):
        return rules.game.trick_count + 1 < self.FROM_TRICK

    def on_play(self, rules, target=-1):
        if rules.game.trick_count + 1 < self.FROM_TRICK:
            return {"error": "early"}
        self.stop_at = rules.game.trick_count + 1
        self.rest = max(0, rules.hand_tricks() - self.stop_at)
        return {}

    def tricks_in_hand(self, classic):
        return min(classic, self.stop_at)

    def on_trick_resolved(self, winner, plays, rules):
        # Godot : _tricks_done() + 1 < stop_at, pli en cours exclu ; ici trick_count le compte deja
        if self.given or rules.game.trick_count < self.stop_at:
            return
        self.given = True
        won = rules.game.tricks_won
        best, top, tie = [], -1, False
        for side in rules.sides():
            n = sum(won[s] for s in side)
            if n > top:
                best, top, tie = side, n, False
            elif n == top:
                tie = True
        if tie or self.rest == 0:
            return
        taker = best[0]
        for seat in best:
            if won[seat] > won[taker]:
                taker = seat
        won[taker] += self.rest


class TreiziemeCarte(Aura):
    """043f : les LAST derniers plis (1 avant) ; 8f41 : dans la misere de son
    proprietaire, sa carte ne les gagne jamais."""
    LAST = 4  # 6c54 (2 avant)

    def competes(self, play, rules):
        if rules.game.trick_count < rules.hand_tricks() - self.LAST:
            return True
        if in_misere(self.owner, rules):
            return play[0] != self.owner
        return play[0] == self.owner or not any(s == self.owner for s, _c in rules.current_plays)


class LivreDeComptes(Effect):
    def on_play(self, rules, target=-1):
        return {"seen_all": True}  # 6c54 : plus de jauge remplie, le Soupcon de la cave suffit


class CartesMarquees(Curse):
    def cost_override(self, jid):
        return 0 if jtype(jid) == "sort" else -1

    def freezes_suspicion(self):
        return True


class JoueurFantome(Curse):
    def draw_bonus(self):
        return 1

    def voids_trick(self, rules):
        return rules.game.trick_count % 4 == 3


class MancheTruquee(Effect):
    """6c54 : pour ce pli, la carte du proprietaire le prend comme l'atout le
    plus fort (avec un atout et sans inversion)."""
    scope = TRICK

    def pointless(self, rules):
        return rules.trick_trump() is None

    def plays_as_trump(self, play, rules):
        return play[0] == self.owner

    def trick_value(self, value, winner_play, rules):
        return value + 1 if winner_play[0] == self.owner else value  # 8f41 : le pli compte pour 2


class CoupDuChapeau(Aura):
    STREAK = 3
    BONUS = 6

    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.streak = 0

    def bot_casts(self):
        return True

    def bot_wants(self, rules):
        return True

    def on_trick_resolved(self, winner, plays, rules):
        self.streak = self.streak + 1 if winner == self.owner else 0
        if self.streak >= self.STREAK:
            rules.add_score(self.owner, self.BONUS)
            self.streak = 0


class Pourboire(Effect):
    scope = TRICK

    def on_trick_resolved(self, winner, plays, rules):
        if winner == self.owner:
            rules.state.win_sous(self.owner, 1)


class ChopeVide(Aura):
    """043f : un joker de plus pioche a la donne suivante (+2 energie avant)."""
    def on_hand_end(self, rules):
        if rules.state.played_this_hand[self.owner] == 0:
            rules.state.next_hand_draws[self.owner] += 1


class BaronneSees(Effect):
    """RogueBosses.BaronneSees (regle propre de la Baronne) : la premiere
    carte du joueur de chaque donne ne gagne jamais son pli."""
    scope = HAND

    def competes(self, play, rules):
        if play[0] != PLAYER:
            return True
        for seat, card in list(rules.game.play_log) + list(rules.current_plays):
            if seat == PLAYER:
                return card != play[1]
        return True


# --- jokers de misere (65a7, whist-4-it f31712b) ---------------------------

def in_misere(seat, rules):
    """JokerEffects.in_misere : `seat` declare la misere jouee."""
    c = rules.contract or {}
    return c.get("type", "") == "misere" and seat in c.get("declarers", [])


def misere_opponent(seat, other, rules):
    return in_misere(other, rules) and rules.is_opponent(seat, other)


class DefausseDiscrete(Effect):
    """Dans la misere du proprietaire, ce pli : il n'est pas oblige de fournir."""
    scope = TRICK

    def pointless(self, rules):
        return not in_misere(self.owner, rules)

    def legal_cards(self, seat, hand, led, legal, rules):
        return list(hand) if seat == self.owner else legal


class MecheCourte(Aura):
    """Chaque pli que le proprietaire perd dans sa misere : +1 energie, MAX dans la donne ;
    8f41 : le MAXe pioche aussi un joker."""
    MAX = 3

    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.gained = 0

    def pointless(self, rules):
        return not in_misere(self.owner, rules)

    def on_trick_resolved(self, winner, plays, rules):
        if (self.gained >= self.MAX or winner == self.owner or not in_misere(self.owner, rules)
                or not any(p[0] == self.owner for p in plays)):
            return
        self.gained += 1
        rules.state.gain_energy(self.owner, 1)
        if self.gained == self.MAX:
            rules.state.draw_jokers(self.owner, 1)


class LaTournee(Aura):
    """8f41 : toute la donne (ce pli avant), le declarant de misere vise joue sa plus forte carte
    de la couleur demandee."""

    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.target = -1

    def needs_target(self):
        return True

    def accepts_target(self, seat, rules):
        return misere_opponent(self.owner, seat, rules)

    def pointless(self, rules):
        return not any(self.accepts_target(s, rules) for s in range(NUM_PLAYERS))

    def on_play(self, rules, target=-1):
        if target < 0 or not self.accepts_target(target, rules):
            return {"error": "target"}
        self.target = target
        return {}

    def legal_cards(self, seat, hand, led, legal, rules):
        if seat != self.target:
            return legal
        follow = [c for c in legal if suit_of(c) == led]
        if not follow:
            return legal
        return [max(follow, key=rank_of)]


class CadeauEmpoisonne(Trap):
    """8f41 : chaque fois (une fois avant) que le declarant de misere vise ne fournit pas, sa carte
    prend le pli, toute la donne."""

    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.target = -1

    def needs_target(self):
        return True

    def accepts_target(self, seat, rules):
        return misere_opponent(self.owner, seat, rules)

    def pointless(self, rules):
        return not any(self.accepts_target(s, rules) for s in range(NUM_PLAYERS))

    def on_play(self, rules, target=-1):
        if target < 0 or not self.accepts_target(target, rules):
            return {"error": "target"}
        self.target = target
        return {}

    def _discards(self, play, rules):
        return play[0] == self.target and play[0] not in rules.joker_cards and suit_of(play[1]) != rules.current_led

    def takes_trick(self, play, rules):
        return self._discards(play, rules)


# --- jokers de combos (6483, whist-4-it 764a412) -----------------------------

class BeauValet(Aura):
    def on_trick_resolved(self, winner, plays, rules):
        if winner == self.owner and any(s == self.owner and rank_of(c) == FACE_RANKS[0] for s, c in plays):
            rules.state.gain_energy(self.owner, 1)


class BalDesValets(Aura):
    def raises(self, play):
        return play[0] == self.owner and rank_of(play[1]) == FACE_RANKS[0]


class Pending(Effect):
    scope = PENDING


class ValetDeChambre(Pending):
    def discount(self, jid):
        return jcost(jid) if base_id(jid) in VALET_FAMILY else 0

    def on_joker_paid(self, seat, jid, rules):
        self.spent = self.spent or (seat == self.owner and self.discount(jid) > 0)


class Complice(Pending):
    def discount(self, jid):
        return 1

    def on_joker_paid(self, seat, jid, rules):
        self.spent = self.spent or (seat == self.owner and jcost(jid) > 0)


class Refrain(Pending):
    """Le sort revient en main, sans etre rejoue : il peut demander sa cible."""
    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.spell = ""

    def pointless(self, rules):
        return rules.state.last_spell[self.owner] == ""

    def on_play(self, rules, target=-1):
        self.spell = rules.state.take_spell(self.owner)
        return {} if self.spell else {"error": "no_spell"}

    def discount(self, jid):
        return jcost(jid) if jid == self.spell else 0

    def on_joker_paid(self, seat, jid, rules):
        self.spent = self.spent or (seat == self.owner and jid == self.spell)


class BoiteDAllumettes(Aura):
    def on_hand_end(self, rules):
        rules.state.carried[self.owner] = rules.state.energy[self.owner]


class PocheInterieure(Aura):
    """Le plus cher qui reste, jamais un passif : une nouvelle donne ne l'armerait pas."""
    def on_hand_end(self, rules):
        best = ""
        for jid in rules.state.joker_hand[self.owner]:
            if jtype(jid) != "passif" and (best == "" or jcost(jid) > jcost(best)):
                best = jid
        rules.state.kept[self.owner] = best


class FeuDePaille(Effect):
    PER_MATCH = 2  # 3285 (3 en 6483, mesure bloc 41234459930)

    def pointless(self, rules):
        return rules.state.energy[self.owner] <= 0

    def on_play(self, rules, target=-1):
        burnt = rules.state.energy[self.owner]
        rules.state.energy[self.owner] = 0
        rules.add_score(self.owner, burnt * self.PER_MATCH)
        return {}


class Ardoise(Effect):
    LOAN = 2

    def on_play(self, rules, target=-1):
        rules.state.gain_energy(self.owner, self.LOAN)
        rules.state.debts[self.owner] += self.LOAN
        return {}


class Addition(Effect):
    PER_JOKER = 2

    def pointless(self, rules):
        return rules.state.played_this_hand[self.owner] == 0

    def on_play(self, rules, target=-1):
        rules.add_score(self.owner, rules.state.played_this_hand[self.owner] * self.PER_JOKER)
        return {}


class VerreDeTrop(Aura):
    EVERY = 3

    def on_joker_paid(self, seat, jid, rules):
        if seat == self.owner and rules.state.played_this_hand[self.owner] % self.EVERY == 0:
            rules.state.draw_jokers(self.owner, 1)


class Cendrier(Aura):
    def discard_bonus(self, seat, jid):
        return 1 if seat == self.owner and base_id(jid) != base_id(self.id) else 0


class Consigne(Aura):
    def discard_energy(self):
        return 2


# --- reworks (177e, whist-4-it 3867121, Synergies 07) --------------------------

class PliRefile2(Effect):
    """Le dernier pli de la table change de main : d'un adversaire vers le
    proprietaire, ou du proprietaire vers l'adversaire vise."""
    def needs_target(self):
        return True

    def _last(self, rules):
        return rules.trick_winners[-1] if rules.trick_winners else -1

    def accepts_target(self, seat, rules):
        last = self._last(rules)
        return rules.is_opponent(self.owner, seat) and last in (self.owner, seat) and rules.game.tricks_won[last] > 0

    def pointless(self, rules):
        return not any(self.accepts_target(s, rules) for s in range(NUM_PLAYERS))

    def on_play(self, rules, target=-1):
        if target < 0 or not self.accepts_target(target, rules):
            return {"error": "target"}
        src = self._last(rules)
        to = target if src == self.owner else self.owner
        rules.game.tricks_won[src] -= 1
        rules.game.tricks_won[to] += 1
        return {"to": to}


class CoeurDeZinneke2(Aura):
    BONUS = 1

    def on_trick_resolved(self, winner, plays, rules):
        if winner == self.owner and any(s == self.owner and rules.card_suit(c) != rules.current_trump for s, c in plays):
            rules.add_score(self.owner, self.BONUS)


class ChopeVide2(Aura):
    """Plus aucun autre joker en main a la fin de la donne ; ses copies armees ne comptent pas."""
    def on_hand_end(self, rules):
        state = rules.state
        if all(base_id(j) == base_id(self.id) for j in state.joker_hand[self.owner]):
            state.next_hand_draws[self.owner] += 1


class CroupierEndormi2(CroupierEndormi):
    ENERGY = 2
    DRAW = 1

    def on_trick_resolved(self, winner, plays, rules):
        self.lost = self.lost + 1 if winner != self.owner else 0
        if self.lost >= self.STREAK:
            rules.state.gain_energy(self.owner, self.ENERGY)
            rules.state.draw_jokers(self.owner, self.DRAW)
            self.spent = True


class FauxDepart2(Effect):
    """Joue dans la fenetre : le joueur vise entame ce pli, aussitot."""
    def needs_target(self):
        return True

    def accepts_target(self, seat, rules):
        return seat != rules.game.turn

    def pointless(self, rules):
        # 61ef : la premiere carte d'un Trou fixe son contrat, seul le bouche-trou la pose
        g = rules.game
        return bool(g.current_trick) or (getattr(g, "trou_pending", lambda: False)() and g.turn == g.partner)

    def on_play(self, rules, target=-1):
        if target < 0 or self.pointless(rules) or not self.accepts_target(target, rules):
            return {"error": "target"}
        rules.game.turn = target
        return {}


EASY_LEVEL = 9


def easy_contract(owner, contract):
    """JokerEffects.easy_contract : emballage ou solo (Abondance comprise), niveau <= 9, lanceur non declarant."""
    c = contract or {}
    level = c.get("level")
    return (c.get("type", "") in ("emballage", "solo") and (EASY_LEVEL + 1 if level is None else int(level)) <= EASY_LEVEL
            and not _declares(owner, c))


class LaTournee2(Aura):
    """Les declarants d'un contrat facile jouent leur plus forte carte legale, atout d'abord."""
    def pointless(self, rules):
        return not easy_contract(self.owner, rules.contract)

    def on_play(self, rules, target=-1):
        return {"error": "none"} if self.pointless(rules) else {}

    def legal_cards(self, seat, hand, led, legal, rules):
        c = rules.contract
        if not legal or not easy_contract(self.owner, c) or not _declares(seat, c):
            return legal
        trumps = [x for x in legal if suit_of(x) == c.get("suit")]
        return [_strongest(trumps if trumps else legal)]


class CadeauEmpoisonne2(Trap):
    """Contrat facile : score recalcule avec 2 plis de moins ; chute, le proprietaire gagne la moitie de la perte."""
    TRICKS = 2

    def pointless(self, rules):
        return not easy_contract(self.owner, rules.contract)

    def on_play(self, rules, target=-1):
        return {"error": "none"} if self.pointless(rules) else {}

    def adjust_deltas(self, deltas, contract, rules):
        if not easy_contract(self.owner, contract) or rules.scorer is None:
            return deltas
        tricks = rules.game.tricks_won
        saved = list(tricks)
        base = rules.score_hand()
        declarer = contract["declarer"]
        partner = contract.get("partner", -1)
        partner = -1 if partner is None else partner
        for _ in range(self.TRICKS):
            seat = declarer if tricks[declarer] > 0 or partner < 0 else partner
            tricks[seat] = max(0, tricks[seat] - 1)
        scored = rules.score_hand()
        tricks[:] = saved
        out = [d + s - b for d, s, b in zip(deltas, scored, base)]
        if scored[declarer] < 0:
            out[self.owner] += (-scored[declarer]) // 2
        return out


# --- Synergies 09 (61ef, whist-4-it f25a277) -----------------------------------

class VieilleHabitude(Aura):
    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.used = False

    def on_hand_end(self, rules):
        self.used = False

    def on_joker_paid(self, seat, jid, rules):
        if self.used or seat != self.owner or jsuspicion(jid) == 0 or rules.hides_suspicion(seat):
            return
        self.used = True
        rules.state.gain_energy(self.owner, 1)


class PotCommun(Effect):
    """La Chaleur est celle du joueur : le seul a lancer ce joker."""
    def pointless(self, rules):
        return rules.state.suspicion == 0 or rules.freezes_suspicion(self.owner)

    def on_play(self, rules, target=-1):
        if self.pointless(rules):
            return {"error": "none"}
        heat = rules.state.suspicion
        rules.state.lower_suspicion(heat)
        rules.add_score(self.owner, heat)
        return {}


class TroubleFete(Aura):
    MAX = 3

    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.gained = 0

    def pointless(self, rules):
        return not easy_contract(self.owner, rules.contract)

    def on_play(self, rules, target=-1):
        return {"error": "none"} if self.pointless(rules) else {}

    def on_trick_resolved(self, winner, plays, rules):
        if winner == self.owner and self.gained < self.MAX:
            self.gained += 1
            rules.state.gain_energy(self.owner, 1)


class MauvaisPerdant(Aura):
    ENERGY = 2

    def on_hand_scored(self, deltas, rules):
        if deltas[self.owner] < 0:
            rules.state.debts[self.owner] -= self.ENERGY  # au-dessus du plafond de la Boite, contrairement a carried


class Cierge(Aura):
    """Pas de ⚡ apres le dernier pli : il ne pourrait que se payer."""
    MAX = 6
    PENALTY = 2

    def __init__(self, jid, owner):
        super().__init__(jid, owner)
        self.gained = 0

    def on_trick_resolved(self, winner, plays, rules):
        if self.gained < self.MAX and len(rules.trick_winners) < rules.hand_tricks():
            self.gained += 1
            rules.state.gain_energy(self.owner, 1)

    def on_hand_end(self, rules):
        left = rules.state.energy[self.owner]
        if left > 0:
            rules.add_score(self.owner, -left * self.PENALTY)


class Patron(Curse):
    EXTRA = 2

    def draw_bonus(self):
        return self.EXTRA

    def returns_owed(self):
        return self.EXTRA


# --- a76f (whist-4-it, CHANTIER-REWORK-JOKERS-PLAN.md) : les inutiles recycles a la Dominion ---
# on_play agit avant que le joker quitte la main : il ne se compte jamais lui-meme.

class DeckEngine(Effect):
    """+Cartes / +Actions : un sort qui ne nourrit que les suivants."""
    def bot_casts(self):
        return True

    def bot_wants(self, rules):
        return not self.pointless(rules)

    def _can_draw(self, rules):
        return rules.state.can_draw(self.owner)


class Ardoise2(DeckEngine):
    """Laboratoire : la pioche a credit."""
    DRAWS = 2
    DEBT = 1

    def pointless(self, rules):
        return not self._can_draw(rules)

    def pointless_reason(self):
        return "pile"

    def bot_wants(self, rules):
        return not self.pointless(rules) and rules.state.debts[self.owner] == 0

    def on_play(self, rules, target=-1):
        rules.state.draw_jokers(self.owner, self.DRAWS)
        rules.state.debts[self.owner] += self.DEBT
        return {}


class TourneeGenerale2(DeckEngine):
    """Salle du Conseil : la tournee est pour le proprietaire."""
    DRAWS = 3

    def pointless(self, rules):
        return not self._can_draw(rules)

    def pointless_reason(self):
        return "pile"

    def on_play(self, rules, target=-1):
        rules.state.draw_jokers(self.owner, self.DRAWS)
        for seat in range(NUM_PLAYERS):
            if rules.is_opponent(self.owner, seat):
                rules.state.draw_jokers(seat, 1)
        return {}


def _valet_wanted(jid):
    return base_id(jid) in VALET_FAMILY and base_id(jid) != "valet_de_chambre2"


class ValetDeChambre2(DeckEngine):
    """Atelier : va chercher un Valet dans la pioche, jamais un autre Valet de Chambre."""
    def pointless(self, rules):
        return not any(_valet_wanted(j) for j in rules.state.draw_pile[self.owner])

    def pointless_reason(self):
        return "no_valet"

    def on_play(self, rules, target=-1):
        return {} if rules.state.fetch(self.owner, _valet_wanted) != "" else {"error": "no_valet"}


class Addition2(DeckEngine):
    """Marche."""
    def on_play(self, rules, target=-1):
        rules.state.gain_energy(self.owner, 1)
        rules.state.draw_jokers(self.owner, 1)
        rules.state.win_sous(self.owner, 1)
        return {}


class VieilleHabitude2(DeckEngine):
    """Bibliotheque."""
    HAND = 4

    def _missing(self, rules):
        return self.HAND - (len(rules.state.joker_hand[self.owner]) - 1)

    def pointless(self, rules):
        return self._missing(rules) <= 0 or not self._can_draw(rules)

    def pointless_reason(self):
        return "full"

    def on_play(self, rules, target=-1):
        rules.state.draw_jokers(self.owner, self._missing(rules))
        return {}


class ChopeVide3(DeckEngine):
    """Village."""
    ENERGY = 2

    def on_play(self, rules, target=-1):
        rules.state.draw_jokers(self.owner, 1)
        rules.state.gain_energy(self.owner, self.ENERGY)
        return {}


class Complice2(Pending):
    """Salle du Trone, sorts seulement : JokerRules.play rejoue le prochain."""
    def doubles(self, seat, jid):
        """Un sort qui dure la donne (L'Horloge Arretee) demanderait une 2e case."""
        return (not self.spent and seat == self.owner and jtype(jid) == "sort" and base_id(jid) != "complice2"
                and make_effect(jid, seat).scope != HAND)

    def bot_casts(self):
        return True

    def bot_wants(self, rules):
        return any(jtype(j) == "sort" and base_id(j) != "complice2" and make_effect(j, self.owner).bot_wants(rules)
                   for j in rules.state.joker_hand[self.owner])


class Consigne2(DeckEngine):
    """Cave (Cellar) : ce que le proprietaire jette est du, jamais paye en energie."""
    def _others(self, rules):
        hand = list(rules.state.joker_hand[self.owner])
        if self.id in hand:
            hand.remove(self.id)
        return hand

    def pointless(self, rules):
        return not self._others(rules)

    def pointless_reason(self):
        return "empty_hand"

    def bot_wants(self, rules):
        return any(rules.state.bot_spares(self.owner, j) for j in self._others(rules))

    def on_play(self, rules, target=-1):
        hand = self._others(rules)
        rules.state.offer_choice(self.owner, {"kind": "consigne", "joker": self.id, "options": hand,
                                              "min": 0, "max": len(hand)})
        return {}


class Cendrier2(DeckEngine):
    """Presage (Harbinger) : on fouille les cendres pour la prochaine pioche."""
    def pointless(self, rules):
        return not self._can_draw(rules)

    def pointless_reason(self):
        return "pile"

    def on_play(self, rules, target=-1):
        st = rules.state
        st.draw_jokers(self.owner, 1)
        if st.discard[self.owner]:
            st.offer_choice(self.owner, {"kind": "cendrier", "joker": self.id, "options": list(st.discard[self.owner]),
                                         "min": 1, "max": 1})
        return {}


class LaPeche2(DeckEngine):
    """acb9 : La Peche au choix -- un joker de la defausse, au choix, revient en main."""
    def pointless(self, rules):
        return not rules.state.discard[self.owner]

    def pointless_reason(self):
        return "empty_discard"

    def bot_wants(self, rules):
        return any(not rules.state.bot_spares(self.owner, j) for j in rules.state.discard[self.owner])

    def on_play(self, rules, target=-1):
        st = rules.state
        if st.discard[self.owner]:  # lancee par Feu de Paille2 depuis une defausse ou elle etait seule : rien a reprendre
            st.offer_choice(self.owner, {"kind": "la_peche", "joker": self.id, "options": list(st.discard[self.owner]),
                                         "min": 1, "max": 1})
        return {}


class MauvaisPerdant2(DeckEngine):
    """Milice : chaque victime choisit sa defausse (MatchState.must_discard)."""
    def _victims(self, rules):
        return [s for s in range(NUM_PLAYERS) if rules.is_opponent(self.owner, s) and rules.state.joker_hand[s]]

    def pointless(self, rules):
        return not self._victims(rules)

    def pointless_reason(self):
        return "no_joker"

    def on_play(self, rules, target=-1):
        for seat in self._victims(rules):
            rules.state.must_discard(seat, 1)
        return {}


class FeuDePaille2(DeckEngine):
    """Vassal : le dessus de la pioche part ; un sort est lance pour rien."""
    def pointless(self, rules):
        return not self._can_draw(rules)

    def pointless_reason(self):
        return "pile"

    def on_play(self, rules, target=-1):
        st = rules.state
        jid = st.reveal(self.owner)
        if jid == "":
            return {"error": "pile"}
        if jtype(jid) != "sort":
            return {"revealed": jid}
        effect = make_effect(jid, self.owner)
        if effect.pointless(rules) or rules.blocked(jid, self.owner):
            return {"revealed": jid}
        if not effect.needs_target():
            cast = st.cast_free(self.owner, jid, -1, False)
            return {"revealed": jid, "cast": cast, "discard": cast.get("discard", 0)}
        targets = targets_of(effect, self.owner, rules)
        if not targets:
            return {"revealed": jid}
        if self.owner in st.players:
            st.offer_choice(self.owner, {"kind": "target", "joker": jid, "options": targets, "min": 1, "max": 1,
                                         "target_kind": effect.target_kind()})
            return {"revealed": jid}
        cast = st.cast_free(self.owner, jid, targets[0], False)
        return {"revealed": jid, "cast": cast, "discard": cast.get("discard", 0)}


class PocheInterieure2(DeckEngine):
    """Sentinelle : les 3 prochains de la pioche, jetes ou remis dans l'ordre choisi."""
    LOOK = 3

    def pointless(self, rules):
        return not rules.state.draw_pile[self.owner]

    def pointless_reason(self):
        return "pile"

    def on_play(self, rules, target=-1):
        pile = rules.state.draw_pile[self.owner]
        top = pile[-min(self.LOOK, len(pile)):][::-1]  # le dessus d'abord
        rules.state.offer_choice(self.owner, {"kind": "poche", "joker": self.id, "options": top,
                                              "min": 0, "max": len(top), "ordered": True})
        return {}


def targets_of(effect, seat, rules):
    """JokerEffects.targets_of : les cibles possibles maintenant, sieges (adversaires d'abord) ou couleurs."""
    if effect.target_kind() == "suit":
        return list(range(4))
    seats = [s for s in range(NUM_PLAYERS) if s != seat and effect.accepts_target(s, rules)]
    return sorted(seats, key=lambda s: (not rules.is_opponent(seat, s), s))


EFFECTS = {
    "cameleon": Cameleon, "zero": Zero, "traitre": Traitor, "coup_de_vent": CoupDeVent,
    "main_lourde": MainLourde, "embuscade": Embuscade, "kermesse": Kermesse, "coup_d_oeil": CoupDOeil,
    "regard_furtif": RegardFurtif, "voeu_de_pauvrete": VoeuDePauvrete, "miroir": Miroir, "borgne": Borgne,
    "carte_fantome": CarteFantome, "double_valet": DoubleValet, "atout_errant": AtoutErrant,
    "courant_d_air": CourantDAir, "tournee_generale": TourneeGenerale, "gueule_de_bois": GueuleDeBois,
    "remonte_pli": RemontePli, "la_peche": LaPeche, "faux_depart": FauxDepart,
    "inversion": Inversion, "fausse_piste": FaussePiste, "collet": Collet,
    "contre_espionnage": ContreEspionnage, "croupier_endormi": CroupierEndormi, "dette_de_jeu": DetteDeJeu,
    "fumee_de_pipe": FumeeDePipe, "silence_du_cercle": SilenceDuCercle, "lune_noire": LuneNoire,
    "coeur_de_zinneke": CoeurDeZinneke, "brouillard_de_la_senne": BrouillardDeLaSenne,
    "cartes_sur_table": CartesSurTable, "bluff": Bluff, "rachat": Rachat, "main_de_fer": MainDeFer,
    "pot_de_vin": PotDeVin, "fossoyeur_de_plis": FossoyeurDePlis,
    "brimborion_dore": BrimborionDore, "valet_borgne": ValetBorgne, "horloge_arretee": HorlogeArretee,
    "treizieme_carte": TreiziemeCarte, "livre_de_comptes": LivreDeComptes, "cartes_marquees": CartesMarquees,
    "joueur_fantome": JoueurFantome, "pacte_du_tenancier": PacteDuTenancier,
    "heritage_empoisonne": HeritageEmpoisonne, "manche_truquee": MancheTruquee,
    "regle_du_moine": ContratImpose, "gant_de_la_baronne": ContratImpose, "pelle_du_fossoyeur": ContratImpose,
    "sceau_du_grand_maitre": ContratImpose, "cavalier_seul": CavalierSeul,
    "coup_du_chapeau": CoupDuChapeau, "pourboire": Pourboire, "chope_vide": ChopeVide, "faux_pas": FauxPas,
    "pli_refile": PliRefile,
    "defausse_discrete": DefausseDiscrete, "meche_courte": MecheCourte, "la_tournee": LaTournee,
    "cadeau_empoisonne": CadeauEmpoisonne,
    "beau_valet": BeauValet, "bal_des_valets": BalDesValets, "valet_de_chambre": ValetDeChambre,
    "boite_d_allumettes": BoiteDAllumettes, "poche_interieure": PocheInterieure, "feu_de_paille": FeuDePaille,
    "ardoise": Ardoise, "addition": Addition, "complice": Complice, "verre_de_trop": VerreDeTrop,
    "refrain": Refrain, "cendrier": Cendrier, "consigne": Consigne,
    "pli_refile2": PliRefile2, "coeur_de_zinneke2": CoeurDeZinneke2, "chope_vide2": ChopeVide2,
    "croupier_endormi2": CroupierEndormi2, "faux_depart2": FauxDepart2, "la_tournee2": LaTournee2,
    "cadeau_empoisonne2": CadeauEmpoisonne2,
    "vieille_habitude": VieilleHabitude, "pot_commun": PotCommun, "trouble_fete": TroubleFete,
    "mauvais_perdant": MauvaisPerdant, "cierge": Cierge, "patron": Patron,
    "ardoise2": Ardoise2, "tournee_generale2": TourneeGenerale2, "valet_de_chambre2": ValetDeChambre2,
    "addition2": Addition2, "vieille_habitude2": VieilleHabitude2, "chope_vide3": ChopeVide3,
    "complice2": Complice2, "consigne2": Consigne2, "cendrier2": Cendrier2,
    "mauvais_perdant2": MauvaisPerdant2, "feu_de_paille2": FeuDePaille2, "poche_interieure2": PocheInterieure2,
    "la_peche2": LaPeche2,
}
assert set(EFFECTS) == set(CATALOG)


def make_effect(jid, owner):
    return EFFECTS[base_id(jid)](jid, owner)


# --- economie (match_state.gd) ---------------------------------------------

class MatchState:
    def __init__(self, decks, rng, suspicion=0):
        self.rng = rng
        self.draw_pile = [[j for j in d if not acts_from_deck(j)] for d in decks]
        self.curses = [[j for j in d if acts_from_deck(j)] for d in decks]
        self.joker_hand = [[] for _ in range(NUM_PLAYERS)]
        self.discard = [[] for _ in range(NUM_PLAYERS)]
        self.energy = [0] * NUM_PLAYERS
        self.suspicion = suspicion
        self.next_hand_malus = 0
        self.played_this_hand = [0] * NUM_PLAYERS  # Chope Vide
        self.cast_ids = [[] for _ in range(NUM_PLAYERS)]  # 4a38 : ids payes cette donne (JokerRules.cost_of)
        self.next_hand_draws = [0] * NUM_PLAYERS  # Chope Vide
        self.sous_won = 0  # Pourboire (joueur seulement)
        self.confiscated = []
        self.rules = None
        self.spammers = []  # sieges qui piochent tout leur deck (79ac)
        self.spammer_energy = 0
        self.energy_malus = 0  # Mise Rouge (joueur)
        self.eye_threshold = SUSPICION_MAX  # Mise Noire
        self.borrowed = [[] for _ in range(NUM_PLAYERS)]  # jokers de defense d'une elite adaptee (3d47)
        self.to_discard = [0] * NUM_PLAYERS  # Main Lourde (6c54)
        self.to_return = [0] * NUM_PLAYERS  # 61ef : parmi eux, ceux qui repartent sous la pioche (Le Patron)
        self.dry_hand = False  # donne du joueur sous « sans allumettes » : sa defausse ne rapporte rien
        self.run_energy = 0  # Synergies 04a : faveurs du run, siege 0
        self.run_draws = 0
        self.pose = False  # 7c98 : un passif du joueur agit une fois pose, celui d'un bot des la pioche
        self.heat = False  # Synergies 04b : paliers de Chaleur (jamais au Duel)
        self.carried = [0] * NUM_PLAYERS  # 6483 : ⚡ reportes (Boite d'Allumettes)
        self.debts = [0] * NUM_PLAYERS  # 6483 : ⚡ dus (L'Ardoise)
        self.kept = [""] * NUM_PLAYERS  # 6483 : joker garde (Poche Interieure)
        self.last_spell = [""] * NUM_PLAYERS  # 6483 : dernier sort paye, Refrain a part
        # a76f : par joueur, le choix qu'un joker a laisse ouvert ({} = aucun) ; ceux d'un bot sont
        # faits aussitot (bot_picks). Joueur simule = RogueAutopilot : bot_picks juste apres le paiement.
        self.choices = [{} for _ in range(NUM_PLAYERS)]
        self.players = [PLAYER]
        # 515d (choix utilisateur 06/11, boucle Chope Vide3 / Addition2) : sous `dominion`, une pioche
        # vide ne reprend la defausse qu'au tirage d'ouverture de la donne (runs neuves, Duel).
        self.dominion = False
        self.recycled = []  # MatchState.recycled : les joueurs qui ont deja recycle cette donne
        for pile in self.draw_pile:
            self.rng.shuffle(pile)

    def arm_curses(self):
        self.rules.curses = [e for e in self.rules.curses if not isinstance(e, Curse)]
        for seat in range(NUM_PLAYERS):
            for jid in self.curses[seat]:
                self.rules.curses.append(make_effect(jid, seat))

    @staticmethod
    def energy_for(dominion):
        return ENERGY_PER_HAND if dominion else LEGACY_ENERGY_PER_HAND

    def base_energy(self):
        return self.energy_for(self.dominion)

    def energy_per_hand(self, seat):
        n = self.base_energy() + (self.spammer_energy if seat in self.spammers else 0)
        if seat == PLAYER:
            n += self.run_energy + self.heat_energy()
        return n + (sum(e.energy_bonus() for e in self.rules.curses if e.owner == seat) if self.rules else 0)

    def heat_energy(self):
        return 1 if self.heat and self.suspicion >= HEAT_ENERGY_AT else 0

    def heat_draws(self):
        return 1 if self.heat and self.suspicion >= HEAT_DRAW_AT else 0

    def start_hand(self):
        for seat in range(NUM_PLAYERS):
            malus = self.next_hand_malus + self.energy_malus if seat == PLAYER else 0
            base = self.energy_per_hand(seat)
            opening = max(base, min(CARRY_MAX, base + self.carried[seat]))
            self.energy[seat] = max(0, opening - self.debts[seat] - malus)
            n = SPAMMER_DRAWS if seat in self.spammers else DRAW_PER_HAND
            if seat == PLAYER:
                n += self.run_draws + self.heat_draws()
            self.draw_jokers(seat, n + self.next_hand_draws[seat] + (self.rules.draw_bonus(seat) if self.rules else 0), opening=True)
        self.dry_hand = self.next_hand_malus >= NO_ENERGY
        self.next_hand_malus = 0
        self.next_hand_draws = [0] * NUM_PLAYERS
        self.played_this_hand = [0] * NUM_PLAYERS
        self.cast_ids = [[] for _ in range(NUM_PLAYERS)]
        self.to_discard = [0] * NUM_PLAYERS
        self.carried = [0] * NUM_PLAYERS
        self.debts = [0] * NUM_PLAYERS
        self.kept = [""] * NUM_PLAYERS
        self.last_spell = [""] * NUM_PLAYERS
        self.choices = [{} for _ in range(NUM_PLAYERS)]
        self.recycled = []
        self._owe_returns()

    def _owe_returns(self):
        """Le Patron ne laisse jamais une donne sous sa pioche habituelle (deck mince).
        Siege du joueur seulement ; le joueur simule remet ses plus recents
        (RogueAutopilot._settle_returns)."""
        self.to_return = [0] * NUM_PLAYERS
        if self.rules is None:
            return
        self.to_return[PLAYER] = min(self.rules.returns_owed(PLAYER), max(0, len(self.joker_hand[PLAYER]) - DRAW_PER_HAND))
        self.to_discard[PLAYER] = self.to_return[PLAYER]
        while self.to_return[PLAYER] > 0:
            self.discard_joker(PLAYER, self.joker_hand[PLAYER][-1])

    def adapt(self, seat, jids):
        """Une elite en defense met sa main de cote pour `jids` (empruntes :
        joues ou non, ils partent avec la donne, jamais dans son deck)."""
        self.discard[seat].extend(self.joker_hand[seat])
        self.joker_hand[seat] = list(jids)
        self.borrowed[seat] = list(jids)

    def end_hand(self):
        self._settle_choices()
        self.to_discard = [0] * NUM_PLAYERS
        self.to_return = [0] * NUM_PLAYERS
        for seat in range(NUM_PLAYERS):
            for jid in self.borrowed[seat]:
                if jid in self.joker_hand[seat]:
                    self.joker_hand[seat].remove(jid)
            self.borrowed[seat] = []
            staying = [self.kept[seat]] if self.kept[seat] in self.joker_hand[seat] else []
            going = list(self.joker_hand[seat])
            for jid in staying:
                going.remove(jid)
            self.discard[seat].extend(going)
            self.joker_hand[seat] = staying

    def draw_jokers(self, seat, n, opening=False):
        """`opening` : le tirage du debut de donne, le seul qui remelange sous `dominion`."""
        for _ in range(n):
            if not self.draw_pile[seat]:
                if not self.discard[seat] or (self.dominion and not opening):
                    return
                self.draw_pile[seat] = self.discard[seat]
                self.discard[seat] = []
                self.rng.shuffle(self.draw_pile[seat])
            jid = self.draw_pile[seat].pop()
            self.joker_hand[seat].append(jid)
            # 65a7 : un passif agit depuis une case libre ; les deux prises, il dort dans la main
            if jtype(jid) == "passif" and self.rules is not None and self.arms_on_draw(seat) and self.rules.free_case(seat) >= 0:
                self.rules.play(seat, jid)

    def arms_on_draw(self, seat):
        return not (self.pose and seat == PLAYER)

    def pose_error(self, seat, jid):
        if jid not in self.joker_hand[seat]:
            return "not_in_hand"
        if self.arms_on_draw(seat) or jtype(jid) != "passif":
            return "passive"
        if self.choices[seat]:
            return "choice"
        return "discard" if self.to_discard[seat] > 0 else ""

    def pose_passives(self):
        """RogueAutopilot._pose_passives : chaque passif qui trouve une case
        libre, gratuit, jamais compte comme joker paye (combo, Chope Vide)."""
        for jid in list(self.joker_hand[PLAYER]):
            if self.pose_error(PLAYER, jid) == "" and self.rules.free_case(PLAYER) >= 0:
                if not self.rules.play(PLAYER, jid).get("error"):
                    self.joker_hand[PLAYER].remove(jid)
                    self.discard[PLAYER].append(jid)

    def must_discard(self, seat, n):
        """`seat` doit defausser `n` jokers de sa main ; un bot paie aussitot
        (d'abord un joker qu'il ne lancerait jamais, sinon le plus recent). Le
        joueur simule aussi, le plus recent (RogueAutopilot._try_joker)."""
        self.to_discard[seat] = min(self.to_discard[seat] + n, len(self.joker_hand[seat]))
        while self.to_discard[seat] > 0:
            pick = self.joker_hand[seat][-1]
            if seat != PLAYER:
                pick = self._bot_discard(seat)
            else:  # 6483 : un joueur defausse d'abord ce qui paie (La Consigne), l'autopilote non
                pick = next((j for j in self.joker_hand[seat] if make_effect(j, seat).discard_energy() > 0), pick)
            self.discard_joker(seat, pick)

    def _bot_discard(self, seat):
        """MatchState._bot_discard (61ef) : La Consigne paie son depart, puis ce
        que le bot ne lance jamais, enfin son plus recent."""
        hand = self.joker_hand[seat]
        for jid in hand:
            if make_effect(jid, seat).discard_energy() > 0:
                return jid
        for jid in hand:
            if base_id(jid) not in COMBO_CAST and (jtype(jid) not in BOT_TYPES or not make_effect(jid, seat).bot_casts()):
                return jid
        return hand[-1]

    def discard_gain(self, seat, jid):
        """65a7 : l'energie que defausser `jid` rapporterait a `seat` maintenant
        (rien si la defausse est due, sous « sans allumettes », ou passif arme),
        plus _discard_extra (6483). Remis sous la pioche (Le Patron) : rien."""
        if self.to_return[seat] > 0:
            return 0
        extra = self._discard_extra(seat, jid)
        if self.to_discard[seat] > 0 or (seat == PLAYER and self.dry_hand):
            return extra
        if make_effect(jid, seat).discard_energy() > 0:
            return extra
        if self.rules is not None and self.arms_on_draw(seat) and jtype(jid) == "passif" and self.rules.armed(seat, jid):
            return extra
        return DISCARD_GAIN + extra

    def _discard_extra(self, seat, jid):
        """Ce que les jokers ajoutent quand `jid` quitte la main, du ou non (La Consigne, Cendrier)."""
        if seat == PLAYER and self.dry_hand:
            return 0
        return make_effect(jid, seat).discard_energy() + (self.rules.discard_bonus(seat, jid) if self.rules is not None else 0)

    def discard_joker(self, seat, jid):
        """Due d'abord (Main Lourde), sinon payee DISCARD_GAIN ; -> energie gagnee."""
        gain = self.discard_gain(seat, jid)
        self.joker_hand[seat].remove(jid)
        if self.rules is not None and self.arms_on_draw(seat) and jtype(jid) == "passif":
            self.rules.disarm(seat, jid)
        if self.to_return[seat] > 0:
            self.to_return[seat] -= 1
            self.to_discard[seat] -= 1
            self.draw_pile[seat].insert(0, jid)
            return 0
        if jid in self.borrowed[seat]:
            self.borrowed[seat].remove(jid)
        else:
            self.discard[seat].append(jid)
        if self.to_discard[seat] > 0:
            self.to_discard[seat] -= 1
        self.energy[seat] += gain
        return gain

    def win_sous(self, seat, n):
        if seat == PLAYER:
            self.sous_won += n

    def gain_energy(self, seat, n):
        self.energy[seat] += n

    def lower_suspicion(self, n):
        if self.rules is not None and self.rules.freezes_suspicion(PLAYER):
            return
        self.suspicion = max(0, self.suspicion - n)

    def cost_of(self, seat, jid):
        return self.rules.cost_of(seat, jid) if self.rules is not None else jcost(jid)

    def take_back(self, seat):
        """La Peche : le dessus de la defausse revient en main ; "" sans defausse."""
        if not self.discard[seat]:
            return ""
        jid = self.discard[seat].pop()
        self.joker_hand[seat].append(jid)
        return jid

    def take_spell(self, seat):
        """Refrain : le dernier sort de la donne repasse de la defausse a la
        main ; "" s'il n'y est plus (un remelange l'a pris)."""
        jid = self.last_spell[seat]
        if jid == "" or jid not in self.discard[seat]:
            return ""
        at = len(self.discard[seat]) - 1 - self.discard[seat][::-1].index(jid)
        del self.discard[seat][at]
        self.joker_hand[seat].append(jid)
        self.last_spell[seat] = ""
        return jid

    def play_error(self, seat, jid):
        if not known(jid) or jid not in self.joker_hand[seat]:
            return "not_in_hand"
        if self.to_discard[seat] > 0:
            return "discard"
        if self.choices[seat]:
            return "choice"
        if jtype(jid) == "passif":
            return "passive"
        # match_state.gd : "pointless" en phase PLAYING (ici : une donne liee)
        if self.rules is not None and self.rules.game is not None and make_effect(jid, seat).pointless(self.rules):
            return make_effect(jid, seat).pointless_reason()
        if self.cost_of(seat, jid) > self.energy[seat]:
            return "energy"
        return ""

    def spend(self, seat, jid, free=False):
        """`free` : lance par un autre joker (Feu de Paille2), rien a payer."""
        self.energy[seat] -= 0 if free else self.cost_of(seat, jid)
        self.played_this_hand[seat] += 1
        self.cast_ids[seat].append(jid)
        self.joker_hand[seat].remove(jid)
        # 4a38 : un sort rendu par Refrain (emprunte exclu) reste le seul cas ; boucle gratuite fermee
        if jtype(jid) == "sort" and base_id(jid) != "refrain" and jid not in self.borrowed[seat]:
            self.last_spell[seat] = jid
        if jid in self.borrowed[seat]:
            self.borrowed[seat].remove(jid)
        else:
            self.discard[seat].append(jid)
        if seat == PLAYER and not (self.rules is not None and self.rules.hides_suspicion(seat)):
            self.suspicion = min(SUSPICION_MAX, self.suspicion + jsuspicion(jid))
        if self.rules is not None:
            self.rules.joker_paid(seat, jid)

    # --- a76f : les mouvements des jokers Dominion sur les piles (match_state.gd) ---

    def fetch(self, seat, wanted):
        """Valet de Chambre2 : le premier joker depuis le dessus de la pioche que `wanted`
        accepte, en main ; "" sans."""
        pile = self.draw_pile[seat]
        for i in range(len(pile) - 1, -1, -1):
            if wanted(pile[i]):
                jid = pile.pop(i)
                self._take(seat, jid)
                return jid
        return ""

    def _take(self, seat, jid):
        """En main comme une pioche : un passif s'arme comme pioche."""
        self.joker_hand[seat].append(jid)
        if jtype(jid) == "passif" and self.rules is not None and self.arms_on_draw(seat) and self.rules.free_case(seat) >= 0:
            self.rules.play(seat, jid)

    RECYCLE_COST = 1

    def recycle_error(self, jid, seat=PLAYER):
        """MatchState.recycle_error : "" si le joueur `seat` peut recycler `jid` maintenant."""
        if seat not in self.players or jid not in self.joker_hand[seat]:
            return "not_in_hand"
        if self.to_discard[seat] > 0:
            return "discard"
        if self.choices[seat]:
            return "choice"
        if seat in self.recycled:
            return "recycled"
        if self.energy[seat] < self.RECYCLE_COST:
            return "energy"
        if not self.can_draw(seat):
            return "pile"
        return ""

    def recycle(self, jid, seat=PLAYER):
        """MatchState.recycle (+ RogueMatch.recycle_joker : un passif arme est desarme) : 1 energie,
        le joker part, un autre est pioche avant qu'il n'atteigne la defausse."""
        if self.rules is not None and self.arms_on_draw(seat) and jtype(jid) == "passif":
            self.rules.disarm(seat, jid)
        self.energy[seat] += self._discard_extra(seat, jid) - self.RECYCLE_COST
        self.joker_hand[seat].remove(jid)
        self.recycled.append(seat)
        self.draw_jokers(seat, 1)
        self.discard[seat].append(jid)

    def can_draw(self, seat):
        """515d : pioche non vide, ou (hors dominion) une defausse a remelanger."""
        return bool(self.draw_pile[seat]) or (not self.dominion and bool(self.discard[seat]))

    def reveal(self, seat):
        """Feu de Paille2 : le dessus de la pioche (remelangee depuis la defausse s'il le
        faut) sur la defausse ; "" quand les deux sont vides."""
        if not self.draw_pile[seat]:
            if not self.can_draw(seat):
                return ""
            self.draw_pile[seat] = self.discard[seat]
            self.discard[seat] = []
            self.rng.shuffle(self.draw_pile[seat])
        jid = self.draw_pile[seat].pop()
        self.discard[seat].append(jid)
        return jid

    def cast_free(self, seat, jid, target, owe=True):
        """Feu de Paille2 : le sort `jid`, sur la defausse, lance pour rien. -> ce que
        JokerRules.play a rendu ; {"error"} et de retour sur la defausse s'il ne peut pas.
        fc90 : `owe` False -> ses defausses dues sont rendues a l'appelant (payees apres lui)."""
        d = self.discard[seat]
        if jid not in d or self.rules is None:
            return {"error": "not_in_hand"}
        del d[len(d) - 1 - d[::-1].index(jid)]
        self.joker_hand[seat].append(jid)
        played = self.rules.play(seat, jid, target)
        if played.get("error"):
            self.joker_hand[seat].remove(jid)
            self.discard[seat].append(jid)
            return played
        self.spend(seat, jid, free=True)
        if owe and played.get("discard", 0) > 0:
            self.must_discard(seat, played["discard"])
        return played

    def bot_spares(self, seat, jid):
        """Un bot jette `jid` pour rien qui lui manquerait (La Consigne2). ee3a : un joueur (pilote
        auto) lance aussi ses jokers de pli et d'enchere -> seulement ce qui ne peut plus servir,
        en phase de jeu (une enchere passee, un joker pointless maintenant)."""
        if seat in self.players:
            playing = self.rules is not None and self.rules.game is not None
            return playing and (jtype(jid) == "enchere" or make_effect(jid, seat).pointless(self.rules))
        if base_id(jid) in COMBO_CAST:
            return False
        return jtype(jid) not in BOT_TYPES or not make_effect(jid, seat).bot_casts()

    def offer_choice(self, seat, choice):
        """Le choix d'un joueur attend (settle_player_choice) ; celui d'un bot est fait maintenant."""
        if seat in self.players:
            self.choices[seat] = choice
        else:
            self._apply_choice(seat, choice, self.bot_picks(seat, choice))

    def bot_picks(self, seat, choice):
        """Les choix d'un bot (et du minuteur, et de RogueAutopilot), indices dans les options :
        ce qu'il ne lance jamais part, le reste garde son ordre ; le plus cher qu'il lance
        revient sur la pioche ; la premiere cible."""
        options = choice.get("options", [])
        kind = choice.get("kind", "")
        if kind == "consigne":
            return [i for i in range(len(options)) if self.bot_spares(seat, options[i])]
        if kind == "poche":
            return [i for i in range(len(options)) if not self.bot_spares(seat, options[i])]
        if kind in ("cendrier", "la_peche"):
            best = 0
            for i in range(len(options)):
                better = jcost(options[i]) > jcost(options[best])
                if self.bot_spares(seat, options[best]) or (not self.bot_spares(seat, options[i]) and better):
                    best = i
            return [best]
        return [0] if options else []

    def choose(self, seat, picks):
        """"" une fois le choix du joueur `seat` regle par `picks`, sinon pourquoi pas."""
        choice = self.choices[seat]
        if not choice:
            return "no_choice"
        at = []
        for p in picks:
            if not isinstance(p, int) or p < 0 or p >= len(choice["options"]) or p in at:
                return "picks"
            at.append(p)
        if len(at) < int(choice["min"]) or len(at) > int(choice["max"]):
            return "picks"
        self.choices[seat] = {}
        self._apply_choice(seat, choice, at)
        return ""

    def settle_player_choice(self, seat=PLAYER):
        """RogueAutopilot._try_joker (a76f) : juste apres le joker paye, son choix ouvert
        se regle par bot_picks."""
        if self.choices[seat]:
            self.choose(seat, self.bot_picks(seat, self.choices[seat]))

    def _apply_choice(self, seat, choice, at):
        options = choice["options"]
        kind = choice["kind"]
        if kind == "consigne":
            thrown = 0
            for i in at:
                if options[i] in self.joker_hand[seat]:
                    self.to_discard[seat] += 1
                    self.discard_joker(seat, options[i])
                    thrown += 1
            self.draw_jokers(seat, thrown)
        elif kind == "cendrier":
            jid = options[at[0]]
            d = self.discard[seat]
            if jid in d:
                del d[len(d) - 1 - d[::-1].index(jid)]
                self.draw_pile[seat].append(jid)
        elif kind == "la_peche":
            jid = options[at[0]]
            d = self.discard[seat]
            if jid in d:
                del d[len(d) - 1 - d[::-1].index(jid)]
                self.joker_hand[seat].append(jid)
        elif kind == "poche":
            pile = self.draw_pile[seat]
            found = []
            for jid in options:
                if jid in pile:
                    del pile[len(pile) - 1 - pile[::-1].index(jid)]
                    found.append(jid)
            kept_ids = []
            for i in at:
                if options[i] in found:
                    found.remove(options[i])
                    kept_ids.append(options[i])
            self.discard[seat].extend(found)
            for jid in reversed(kept_ids):
                pile.append(jid)
        elif kind == "target":
            self.cast_free(seat, choice["joker"], int(options[at[0]]))

    def _settle_choices(self):
        """Fin de donne : un choix ouvert se regle comme un bot ; la cible d'un sort tombe."""
        for seat in range(NUM_PLAYERS):
            choice = self.choices[seat]
            self.choices[seat] = {}
            if choice and choice["kind"] != "target":
                self._apply_choice(seat, choice, self.bot_picks(seat, choice))


# --- regles (joker_rules.gd) ------------------------------------------------

def beats(ch, best, led, trump, ch_rank=None, best_rank=None):
    """WhistTrick._beats : meme couleur -> rang ; sinon seul l'atout bat.
    ch_rank / best_rank : le rang vu par le pli s'il differe (Bal des Valets)."""
    if suit_of(ch) == suit_of(best):
        return (rank_of(ch) if ch_rank is None else ch_rank) > (rank_of(best) if best_rank is None else best_rank)
    return trump is not None and suit_of(ch) == trump


class JokerRules:
    """JokerRules (joker_rules.gd) : les effets vivants d'une donne + les
    maledictions/regles de boss (`curses`, jamais videes). `game` = moteur
    de pli de la donne (TeamTrumpDeal / LoneDeclarerDeal), None pendant
    l'enchere. Chaque hook replie les effets vivants dans l'ordre de jeu,
    comme le jeu. `score_bonus` = points ajoutes au score du match pendant
    la donne (table.total_score cote jeu : Dette de Jeu, Coeur de Zinneke,
    Fossoyeur de Plis, sanction « points » de l'Oeil)."""

    def __init__(self, state, rng=None):
        self.state = state
        state.rules = self
        self.rng = rng or random.Random(0)
        self.effects = []
        self.curses = []
        self.game = None
        self.contract = None
        self.deal_hands = None
        self.current_led = None
        self.current_trump = None
        self.current_plays = []
        self.trick_winners = []
        self.joker_cards = {}  # siege -> joker de pli pose dans le pli en cours
        self.score_bonus = [0] * NUM_PLAYERS
        self.bonus_tricks = [0] * NUM_PLAYERS  # dcf1 : plis comptes au-dela de 1 par un joker
        self.vanished = []  # cartes defaussees par Courant d'Air
        # scorer(game, contract) -> score de base de la donne (WhistTable._score_hand) ; une
        # fonction de module (jamais une fermeture : clone() = deepcopy garde la meme fonction)
        self.scorer = None
        # 65a7 (d74be81) : par siege, les effets de donne dans ses cases (None = libre) ;
        # un 3e prend la case de bot_case, celui qu'il remplace s'arrete
        self.lasting_cases = [LASTING_CASES] * NUM_PLAYERS
        self.priority = False  # 4a38 : fenetre de jokers avant chaque pli (run heat, Duel, Belle)
        self.bosses = ()  # regles de boss de RogueMatch._boss_blocks
        self.reset_window()
        self.free_cases()

    def reset_window(self):
        self.passed = []
        self.priority_from = -1

    def boss_blocks(self, jid):
        return "moine" in self.bosses and jtype(jid) in ("pli", "aura")  # RogueBosses.blocks_joker

    def can_cast(self, seat):
        """RogueMatch._can_cast : `seat` tient un joker lancable dans la fenetre."""
        return any(timing(j) == "trick_start" and self.state.play_error(seat, j) == "" and not self.blocked(j, seat)
                   and not self.boss_blocks(j) for j in self.state.joker_hand[seat])

    def priority_seat(self):
        """JokerRules.priority_seat : qui peut lancer dans la fenetre, -1 sinon."""
        g = self.game
        if not self.priority or g is None or g.done or g.current_trick or getattr(g, "pending_discards", None):
            return -1
        first = self.priority_from if self.priority_from >= 0 else (g.turn + 1) % NUM_PLAYERS
        for i in range(NUM_PLAYERS):
            seat = (first + i) % NUM_PLAYERS
            if seat not in self.passed and self.can_cast(seat):
                return seat
        return -1

    def cast_seen(self, seat):
        if self.priority:
            self.passed = []
            self.priority_from = (seat + 1) % NUM_PLAYERS

    def pass_window(self, seat):
        self.passed.append(seat)

    def blocks_card(self, seat):
        return self.priority_seat() not in (-1, seat)

    def turn_error(self, jid, seat):
        """RogueMatch.turn_error sous `priority` (phase de pli)."""
        if self.boss_blocks(jid):
            return "boss"
        if timing(jid) == "trick_start":
            return "" if self.priority_seat() == seat else "turn"
        return "" if self.game.turn == seat and not self.blocks_card(seat) else "turn"

    def set_spammers(self, seats):
        for seat in seats:
            self.lasting_cases[seat] = SPAMMER_CASES
        self.free_cases()

    def free_cases(self):
        self.cases = [[None] * n for n in self.lasting_cases]
        self.case_order = [[] for _ in range(NUM_PLAYERS)]

    def free_case(self, seat):
        return self.cases[seat].index(None) if None in self.cases[seat] else -1

    def bot_case(self, seat):
        """Case dont l'effet est fini (piege parti), sinon la plus ancienne."""
        for i, e in enumerate(self.cases[seat]):
            if e is not None and not any(x is e for x in self.effects):
                return i
        return self.case_order[seat][0] if self.case_order[seat] else -1

    def armed(self, seat, jid):
        return any(e.owner == seat and e.id == jid for e in self.effects)

    def clear_case(self, seat, case):
        """65a7 (48b2dc3) : `seat` defausse le joker de sa case, son effet s'arrete ; -> son id ou ""."""
        if case < 0 or case >= len(self.cases[seat]) or self.cases[seat][case] is None:
            return ""
        e = self.cases[seat][case]
        self.effects = [x for x in self.effects if x is not e]
        self.cases[seat][case] = None
        self.case_order[seat].remove(case)
        return e.id

    def new_hand(self, game, contract):
        self.free_cases()
        self.effects = []
        self.trick_winners = []
        self.joker_cards = {}
        self.score_bonus = [0] * NUM_PLAYERS
        self.bonus_tricks = [0] * NUM_PLAYERS
        self.vanished = []
        self.reset_window()
        self.trick_window = False  # joker_rules.gd in_trick_window : voir joker_match.play_out
        self.bind(game, contract)

    def bind(self, game, contract):
        self.game = game
        self.contract = contract

    def score_hand(self):
        return self.counted(lambda: self.scorer(self.game, self.contract))

    def counted(self, score):
        """dcf1 (WhistTable._taken_by) : un pli double de la defense enleve au camp
        du contrat autant de plis que ses plis bonus, plancher 0."""
        c, tricks = self.contract or {}, self.game.tricks_won
        if not any(self.bonus_tricks) or "declarers" in c or "declarer" not in c:
            return score()
        camp = self.sides()[0]
        saved, cost = list(tricks), sum(self.bonus_tricks[s] for s in range(NUM_PLAYERS) if s not in camp)
        for s in camp:
            taken = min(cost, tricks[s])
            tricks[s] -= taken
            cost -= taken
        try:
            return score()
        finally:
            tricks[:] = saved

    def live(self):
        return self.curses + self.effects

    def sides(self):
        """6c54 : les camps de la donne (chaque declarant seul en misere),
        chacun seul sans contrat."""
        c = self.contract or {}
        if "declarers" in c:
            return [[s] for s in c["declarers"]] + [[s for s in range(NUM_PLAYERS) if s not in c["declarers"]]]
        if "declarer" not in c:
            return [[0], [1], [2], [3]]
        side = [s for s in (c["declarer"], c.get("partner", -1)) if s is not None and s >= 0]
        return [side, [s for s in range(NUM_PLAYERS) if s not in side]]

    def is_opponent(self, seat, other):
        """joker_rules.gd (whist-4-it a3d70cf) : les camps viennent du
        CONTRAT, pas de partner_of. Misere : chaque declarant seul, les
        defenseurs allies ; sinon camp = [declarer, partner] contre le reste
        (les defenseurs d'un emballage/trou/solo forment un camp) ; pas de
        contrat -> tout le monde est adversaire."""
        if other == seat:
            return False
        c = self.contract
        if not c:
            return True
        if "declarers" in c:
            return seat in c["declarers"] or other in c["declarers"]
        if "declarer" not in c:
            return True
        side = (c["declarer"], c.get("partner", -1))
        return (seat in side) != (other in side)

    def contract_trump(self):
        return self.game.trump_suit

    def led_suit(self):
        return suit_of(self.game.current_trick[0][1]) if self.game is not None and self.game.current_trick else None

    def trick_trump(self):
        trump = self.game.trump_suit
        for e in self.live():
            trump = e.trump(trump, self)
        return trump

    def hand_tricks(self):
        return self.tricks_in_hand(self.classic_tricks())

    def classic_tricks(self):
        """WhistTable._classic_tricks (e1f4) : plis de la donne sans modificateurs."""
        return getattr(self.game, "num_tricks", 13)

    def add_score(self, seat, n):
        self.score_bonus[seat] += n

    def blocked(self, jid, seat):
        return any(e.blocks(jid, seat, self) for e in self.live())

    def play(self, seat, jid, target=-1):
        effect = make_effect(jid, seat)
        if self.blocked(jid, seat):
            return {"error": "silence"}
        led = self.led_suit()
        card = effect.make_card(led if led is not None else 0, self)
        if card is not None and led is None:
            return {"error": "lead"}
        case = -1
        if effect.scope == HAND:  # 65a7 : une case libre, sinon celle de bot_case (bots et autopilote)
            case = self.free_case(seat)
            if case < 0:
                case = self.bot_case(seat)
                if case < 0:
                    return {"error": "case"}
        if any(e.cancels(seat, jid, self) for e in self.live()):
            return {"cancelled": True}
        # a76f : depense avant que le sort agisse -- un sort qu'il lance a son tour (Feu de Paille2)
        # n'est pas le prochain sort.
        doubler = next((e for e in self.effects if isinstance(e, Complice2) and e.doubles(seat, jid)), None)
        if doubler is not None:
            doubler.spent = True
        result = effect.on_play(self, target)
        if result.get("error"):
            if doubler is not None:
                doubler.spent = False
            return result
        if effect.scope != INSTANT:
            self.effects.append(effect)
        if doubler is not None:
            self.effects = [x for x in self.effects if x is not doubler]
            again = make_effect(jid, seat)
            twice = again.on_play(self, target)
            if not twice.get("error"):
                if again.scope != INSTANT:
                    self.effects.append(again)
                result["discard"] = result.get("discard", 0) + twice.get("discard", 0)
                result["twice"] = True
        if case >= 0:
            old = self.cases[seat][case]
            if old is not None:
                self.effects = [x for x in self.effects if x is not old]
                result["replaced"] = old.id
            self.cases[seat][case] = effect
            if case in self.case_order[seat]:
                self.case_order[seat].remove(case)
            self.case_order[seat].append(case)
            result["case"] = case
        if card is not None:
            result["card"] = card
        return result

    def disarm(self, seat, jid):
        for e in self.effects:
            if e.owner == seat and e.id == jid:
                self.effects.remove(e)
                for i, x in enumerate(self.cases[seat]):
                    if x is e:
                        self.cases[seat][i] = None
                        self.case_order[seat].remove(i)
                return

    def legal_cards(self, hand, led, seat):
        """65a7 : les jokers de misere plient l'obligation de fournir."""
        legal = legal_plays(hand, led)
        live = self.live()
        for e in [e for e in live if self._remaps(e)] + [e for e in live if not self._remaps(e)]:
            legal = e.legal_cards(seat, hand, led, legal, self)
        return legal

    @staticmethod
    def _remaps(e):
        return e.suit_as(HEARTS) != HEARTS

    def suit_name(self, suit):
        """La couleur que `suit` vaut sous les jokers en jeu (Lune Noire)."""
        for e in self.live():
            suit = e.suit_as(suit)
        return suit

    def card_suit(self, card):
        return self.suit_name(suit_of(card))

    def trick_winner(self, real_plays, real_led, trump):
        # dcf1 : le pli vu sous Lune Noire ; une carte recoloree perd l'egalite comme un joker
        plays = [(s, self.card_suit(c) * 13 + rank_of(c)) for s, c in real_plays]
        led = self.suit_name(real_led)
        blackened = {s for (s, c), (_s, r) in zip(plays, real_plays) if suit_of(c) != suit_of(r)}
        # 6483 : RaisedCard, au-dessus de l'As, et ni joker ni carte recoloree pour l'egalite
        raised = {s for s, c in real_plays if any(e.raises((s, c)) for e in self.live())}
        stands_in = lambda seat: seat not in raised and (seat in self.joker_cards or seat in blackened)
        seen_rank = lambda p: ACE + 1 if p[0] in raised else rank_of(p[1])
        self.current_led = led
        self.current_trump = trump
        self.current_plays = list(plays)
        live = self.live()
        for p in plays:  # 65a7 : Cadeau Empoisonne
            if any(e.takes_trick(p, self) for e in live):
                return p[0]
        competing = [p for p in plays if all(e.competes(p, self) for e in live)]
        if not competing:
            return plays[0][0]
        inverted = any(e.inverts() for e in live)
        if trump is not None and not inverted:  # 6c54 : Manche Truquee prend le pli d'office
            for p in competing:
                if any(e.plays_as_trump(p, self) for e in live):
                    return p[0]
        best = competing[0]
        # 592e (whist-4-it 8a6f635) : un joker qui se joue comme une carte perd l'egalite face a la
        # vraie carte (couleur apres plays_as_trump, meme rang) ; entre deux jokers, le premier pose garde le pli.
        for p in competing:
            if inverted:
                # joker_rules.gd : la plus faible de la couleur demandee gagne
                if suit_of(p[1]) == led and (suit_of(best[1]) != led or seen_rank(p) < seen_rank(best)):
                    best = p
            elif beats(p[1], best[1], led, trump, seen_rank(p), seen_rank(best)):
                best = p
            if (stands_in(best[0]) and not stands_in(p[0])
                    and suit_of(p[1]) == suit_of(best[1]) and rank_of(p[1]) == rank_of(best[1])):
                best = p
        return best[0]

    def trick_value(self, winner, plays):
        winner_play = [p for p in plays if p[0] == winner][0]
        value = 1
        for e in self.live():
            value = e.trick_value(value, winner_play, self)
        if any(e.voids_trick(self) for e in self.live()):
            return 0
        return value

    def next_leader(self, winner):
        for e in self.live():
            if e.next_leader(winner) >= 0:
                return e.next_leader(winner)
        return winner

    def tricks_in_hand(self, classic):
        n = classic
        for e in self.live():
            n = min(n, e.tricks_in_hand(classic))
        return n

    def overrides_bot_card(self):
        return any(type(e).bot_card is not Effect.bot_card for e in self.live())

    def bot_card(self, seat, card, legal):
        for e in self.live():
            card = e.bot_card(seat, card, legal, self)
        return card

    def forces_pass(self, seat, action, bidding):
        return any(e.forces_pass(seat, action, bidding) for e in self.live())

    def prepare_bidding(self, bidding):
        for e in self.live():
            e.prepare_bidding(bidding)

    def hides_suspicion(self, seat):
        return any(e.hides_suspicion(seat) for e in self.live())

    def absorbs_sanction(self):
        for e in self.live():
            if e.absorbs_sanction():
                return True
        return False

    def on_contract(self, contract):
        changed = False
        for e in self.live():
            changed = e.on_contract(contract, self) or changed
        return changed

    def cost_of(self, seat, jid):
        cost = jcost(jid)
        for e in self.curses:
            # 4a38 : sous Cartes Marquees, gratuit une fois par id et par donne
            if (e.owner == seat and e.cost_override(jid) >= 0
                    and not (self.priority and jid in self.state.cast_ids[seat])):
                cost = min(cost, e.cost_override(jid))
        for e in self.effects:
            if e.owner == seat:
                cost = max(0, cost - e.discount(jid))
        # acb9 : un joker gratuit relance dans la meme donne coute 1 (boucle Main Lourde + La Peche)
        if (cost == 0 and jcost(jid) == 0 and self.state is not None and jid in self.state.cast_ids[seat]
                and jid not in self.state.borrowed[seat]):
            cost = 1
        return cost

    def joker_paid(self, seat, jid):
        """6483 : `seat` vient de payer `jid`, les remises servies disparaissent aussitot."""
        for e in self.live():
            e.on_joker_paid(seat, jid, self)
        self.effects = [e for e in self.effects if not e.spent or e.scope != PENDING]

    def discard_bonus(self, seat, jid):
        return sum(e.discard_bonus(seat, jid) for e in self.live())

    def draw_bonus(self, seat):
        return sum(e.draw_bonus() for e in self.curses if e.owner == seat)

    def returns_owed(self, seat):
        return sum(e.returns_owed() for e in self.curses if e.owner == seat)

    def freezes_suspicion(self, seat):
        return any(e.owner == seat and e.freezes_suspicion() for e in self.curses)

    def on_trick_resolved(self, winner, plays):
        self.reset_window()
        self.trick_winners.append(winner)
        for e in self.live():
            e.on_trick_resolved(winner, plays, self)
        self.effects = [e for e in self.effects if not e.spent and e.scope != TRICK]
        self.joker_cards = {}

    def adjust_deltas(self, deltas, contract):
        out = list(deltas)
        for e in self.live():
            out = e.adjust_deltas(out, contract, self)
        for e in self.live():
            e.on_hand_scored(out, self)
        for e in self.live():
            e.on_hand_end(self)
        return out


# --- la donne sous jokers --------------------------------------------------

class JokerHand:
    """Une donne jouee sous JokerRules : delegue au moteur de pli et ne
    corrige que la resolution du pli (WhistTable._commit_play_only avec
    modifiers) : gagnant, valeur du pli (tricks_won += valeur), meneur
    suivant, fin de donne sur tricks_in_hand."""

    def __init__(self, game, contract, state, rules):
        self.game = game
        self.contract = contract
        self.state = state
        self.rules = rules
        self.eye = ""
        self.joker_play_idx = set()  # indices de game.play_order poses par des cartes de joker

    def step(self, seat, card):
        g = self.game
        if getattr(g, "pending_discards", None) or len(g.current_trick) < NUM_PLAYERS - 1:
            g.step(seat, card)
            return
        plays = list(g.current_trick) + [(seat, card)]
        led = suit_of(plays[0][1])
        trump = self.rules.trick_trump()
        winner = self.rules.trick_winner(plays, led, trump)
        value = self.rules.trick_value(winner, plays)
        g.step(seat, card)
        classic = g.turn if not g.done else classic_trick_winner(plays, g.trump_suit)
        g.tricks_won[classic] -= 1
        g.tricks_won[winner] += value
        self.rules.bonus_tricks[winner] += max(0, value - 1)
        leader = self.rules.next_leader(winner)
        self.rules.on_trick_resolved(winner, plays)
        if not g.done and g.trick_count >= self.rules.hand_tricks():
            g.done = True  # WhistTable._hand_over : la donne finit sur son nombre de plis
        if not g.done:
            g.turn = leader

    def step_joker_card(self, seat, card, jid):
        """Carte de joker posee dans le pli a la place d'une carte de la main
        (RogueMatch.play_joker -> table._commit_play_only) : la main ne
        diminue pas."""
        self.rules.joker_cards[seat] = jid
        self.joker_play_idx.add(len(self.game.play_order))
        self.game.hands[seat].append(card)  # le moteur la retire aussitot
        self.step(seat, card)

    def holders(self):
        return [s for s in range(NUM_PLAYERS) if s != PLAYER and (
            self.state.joker_hand[s] or self.state.draw_pile[s] or self.state.discard[s])]

    def legal_casts(self, seat):
        """Masque des actions (0 = aucun cast, toujours legal), comme
        RogueMatch._bot_turn : type de bot, play_error == "" et pas bloque."""
        mask = [True] + [False] * len(CASTABLE)
        if seat not in self.holders():
            return mask
        for i, jid in enumerate(CASTABLE):
            if (self.state.play_error(seat, jid) == "" and not self.rules.blocked(jid, seat)
                    and not self.rules.boss_blocks(jid)  # 8664 : JokerAI.legal_actions(bosses), meme sans priority
                    and not (self.rules.priority and self.rules.turn_error(jid, seat))
                    and not evicts_silence(seat, jid, self.rules)):
                mask[i + 1] = True
        return mask

    def combo_turn(self, seat):
        """RogueMatch._bot_turn (61ef) : JokerComboBot avant le reseau ; True = lance."""
        rules, st = self.rules, self.state
        jid = combo_pick(seat, rules, st, lambda j: st.play_error(seat, j) == "" and not rules.boss_blocks(j) and not (
            rules.priority and rules.turn_error(j, seat)))  # 8664 : _boss_blocks meme sans priority
        if jid == "":
            return False
        played = rules.play(seat, jid)
        if played.get("error"):
            return False
        spend(st, seat, jid, played)
        return True

    def cast(self, seat, action):
        if action == 0:
            return
        jid = CASTABLE[action - 1]
        played = self.rules.play(seat, jid)
        if played.get("error"):
            return
        spend(self.state, seat, jid, played)

    def fuel(self, seat):
        """RogueMatch._bot_fuel (65a7) : sans cast, un joker qu'il ne lancera
        pas part pour l'energie (+DISCARD_GAIN) qui manque a un joker qu'il
        veut ; True = defausse faite (le bot rejoue son tour)."""
        st = self.state
        hand = st.joker_hand[seat]
        for jid in hand:
            if ((self.rules.priority and self.rules.turn_error(jid, seat)) or jtype(jid) not in BOT_TYPES
                    or st.play_error(seat, jid) != "energy"
                    or st.cost_of(seat, jid) > st.energy[seat] + DISCARD_GAIN
                    or not make_effect(jid, seat).bot_wants(self.rules)):
                continue
            for other in hand:
                if other != jid and self._spare(seat, other):
                    st.discard_joker(seat, other)
                    return True
        return False

    def _spare(self, seat, jid):
        if self.state.discard_gain(seat, jid) <= 0:
            return False
        if jtype(jid) == "passif":
            return make_effect(jid, seat).discard_energy() > 0  # La Consigne, faite pour partir
        return jtype(jid) not in BOT_TYPES or not make_effect(jid, seat).bot_wants(self.rules)


# --- JokerComboBot (61ef, joker_combo_bot.gd) : combos des boss, hors reseau ---

COMBO_CAST = ("ardoise", "complice", "silence_du_cercle", "verre_de_trop", "refrain", "la_peche")
COMBO_WORTH_BACK = ("kermesse", "main_lourde", "coup_de_vent", "silence_du_cercle")
COMBO_WORTH_REPEAT = ("coup_de_vent", "main_lourde")
COMBO_SILENCE_FROM = 6
COMBO_VERRE_FROM = 5


def combo_pick(seat, rules, state, allowed):
    hand = state.joker_hand[seat]
    for base in COMBO_CAST:
        for jid in hand:
            if base_id(jid) == base and allowed(jid) and combo_wants(base, seat, rules, state):
                return jid
    return ""


def combo_wants(base, seat, rules, state):
    remaining = rules.hand_tricks() - len(rules.trick_winners)
    if base == "ardoise":
        return state.debts[seat] == 0 and _planned_cost(seat, rules, state) > state.energy[seat]
    if base == "complice":
        return not _armed("complice", seat, rules) and _planned_cost(seat, rules, state) > 0
    if base == "silence_du_cercle":
        return (remaining >= COMBO_SILENCE_FROM and not _armed(base, seat, rules)
                and state.energy[seat] - _trick_jokers_cost(seat, rules, state) >= jcost(base))
    if base == "verre_de_trop":
        return remaining >= COMBO_VERRE_FROM and not _armed(base, seat, rules) and len(state.joker_hand[seat]) >= 3
    if base == "refrain":
        return base_id(state.last_spell[seat]) in COMBO_WORTH_REPEAT
    if base == "la_peche":
        return bool(state.discard[seat]) and base_id(state.discard[seat][-1]) in COMBO_WORTH_BACK
    return False


def evicts_silence(seat, jid, rules):
    """Lancer `jid` chasserait le Silence du siege de sa plus vieille case."""
    if make_effect(jid, seat).scope != HAND or rules.free_case(seat) >= 0 or not rules.case_order[seat]:
        return False
    oldest = rules.cases[seat][rules.case_order[seat][0]]
    return oldest is not None and base_id(oldest.id) == "silence_du_cercle"


def _planned_cost(seat, rules, state):
    total = _trick_jokers_cost(seat, rules, state)
    for jid in state.joker_hand[seat]:
        if base_id(jid) == "silence_du_cercle" and combo_wants("silence_du_cercle", seat, rules, state):
            total += state.cost_of(seat, jid)
    return total


def _trick_jokers_cost(seat, rules, state):
    total = 0
    for jid in state.joker_hand[seat]:
        if jtype(jid) in BOT_TYPES and make_effect(jid, seat).bot_wants(rules):
            total += state.cost_of(seat, jid)
    return total


def _armed(base, seat, rules):
    return any(e.owner == seat and not e.spent and base_id(e.id) == base for e in rules.effects)


def spend(state, seat, jid, played):
    """RogueMatch._spend : paye une fois le joker sorti de la main, la
    defausse de Main Lourde ne peut pas le prendre."""
    state.spend(seat, jid)
    if state.rules is not None:
        state.rules.cast_seen(seat)
    if played.get("discard", 0) > 0:
        state.must_discard(seat, played["discard"])


def watch_eye(state, rules, sanction, jid):
    """RogueMatch._watch_eye : un joker suspect qui remplit la jauge est
    pris ; la sanction tombe, la jauge se vide."""
    if sanction == "" or state.suspicion < state.eye_threshold or jsuspicion(jid) == 0:
        return
    if rules.absorbs_sanction():
        sanction = "absorbed"
    if sanction == "points":
        rules.add_score(PLAYER, -EYE_POINTS)
    elif sanction == "confisque":
        if jid in state.discard[PLAYER]:
            state.discard[PLAYER].remove(jid)
        state.confiscated.append(jid)
    elif sanction == "allumettes":
        state.next_hand_malus = NO_ENERGY
    state.suspicion = 0


def heuristic_action(hand, seat):
    """RogueMatch._bot_turn actuel : le premier joker de la main (ordre de
    pioche) que le bot veut (bot_wants) et peut payer."""
    mask = hand.legal_casts(seat)
    for jid in hand.state.joker_hand[seat]:
        if jid not in CASTABLE or not mask[CASTABLE.index(jid) + 1]:
            continue
        if make_effect(jid, seat).bot_wants(hand.rules):
            return CASTABLE.index(jid) + 1
    return 0


def never_action(hand, seat):
    return 0
