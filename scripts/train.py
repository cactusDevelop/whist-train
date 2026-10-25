"""Phase 1 + Phase 2 (voir DESIGN.md/TRAINING.md). Chaque donne tire au
hasard (`--phase2-prob`, 80% par defaut depuis le 2026-09-26) SOIT :

- Phase 1 (contrat FORCE, inchangee depuis le 2026-09-19) : une SEULE
  famille de declarant au hasard (jamais deux roles de declarant dans la
  meme donne -- structurellement impossible, une donne ne verrouille qu'UNE
  famille), tous les autres sieges utilisent `defenseur`. Garde les 7
  specialistes de jeu de carte au niveau independamment de l'enchere.
- Phase 2 (VRAIE enchere -- voir run_phase2_episode) : 4 sieges = 4
  individus DISTINCTS tires au hasard dans la population evolutive
  d'enchere (agents/evolution.py) -> contrat verrouille ou redonne
  generale -> jeu reel avec les memes specialistes de jeu de carte (+
  `trou`/`bouche_trou`, nouveaux en Phase 2, entraines UNIQUEMENT via
  cette branche -- Trou n'existe pas sans vraie enchere) -> score reel.
  L'enchere elle-meme n'est PAS entrainee par PPO : c'est un algorithme
  genetique (mutation + selection elitiste sur fitness reelle, demande
  explicite de l'utilisateur "un vrai entrainement d'IA classique") --
  voir run_bidding_phase plus bas et scripts/duel_expert.py:evaluate_bidders.

Aucune IA de declarant n'affronte jamais une AUTRE IA de declarant.

Recompense DECLARANT = NOMBRE DE PLIS (dense, une formule par objectif) :
  - solitaire       : +plis du declarant.
  - emballage       : +plis du camp declarant (declarant+partenaire, meme
                       valeur pour les deux -- jamais divisee).
  - piccolo         : -|plis - 1| (objectif = EXACTEMENT 1 pli, pas "au
                       plus").
  - petite/grande/etalee misere : -plis du declarant (objectif = 0 pli).

Recompense DEFENSEUR = VRAI BAREME DE POINTS (`engine/scoring.py`, deja
verifie contre Godot), PAS un proxy en plis -- demande explicite de
l'utilisateur (2026-09-19) : "je le fais en points et pas en pli car faire
des plis une fois un piccolo perdu ne change rien". Au-dela du seuil de
reussite/echec d'un contrat, prendre des plis supplementaires ne change
RIEN au score reel (ex: un Piccolo deja perdu -- le declarant a deja pris
2+ plis -- la defense ne gagne plus rien a en prendre davantage ; c'etait
pourtant recompense en plus par un proxy en plis). Valeur DEFENSEUR de
`engine/misere.py:final_scores`/`solitaire.py:final_scores`/
`emballage.py:final_scores` (level place-holder pour solitaire/emballage,
voir plus bas -- pas de niveau reel sans enchere) -- meme valeur pour
chaque defenseur, jamais divisee. Le declarant, lui, garde sa recompense
dense en plis (INCHANGEE, pas demande par l'utilisateur) -- seul
`defenseur` bascule en points. Voir `defense_points_reward`/`compute_scores`.

Merite attribue AU PLI (depuis le 2026-09-24, voir agents/credit.py) : la
recompense totale d'une donne est inchangee, mais chaque decision n'est
creditee que de ce qui se passe A PARTIR de son pli (retour = valeur
finale - valeur au debut du pli, avantages GAE) -- plus de score final
copie sur les 13 cartes. Ceci ne s'applique QU'au jeu de carte (PPO) : les
decisions d'enchere n'ont pas de "retour PPO" du tout, voir plus haut.

Fitness d'enchere (depuis le 2026-09-27 au soir, decision utilisateur :
"fais le duplicate contre ExpertBot") : en fin de generation, chaque
individu encherit sur les MEMES donnes que les autres, contre l'ExpertBot,
en duplicate (scripts/duel_expert.py:evaluate_bidders). Avant : les points
des contrats pris par le siege pendant les donnes Phase 2 (0 en defense) --
trop peu de contrats propres par individu, note surtout due au hasard, et
la selection degradait l'enchere (elite a -7 pts/donne contre le bot, sa
graine a -1). Les donnes Phase 2 servent toujours a entrainer le jeu de la
carte, avec des encheres tirees de la population.

Plus AUCUN faconnage de la fitness d'enchere depuis le 2026-09-27 : le
malus de "Se coucher" (-4) a ete retire, puis le bonus de la lignee
`encherisseur_amical` (+15 par proposition/emballage, meme si le contrat
chutait), supprimee avec la lignee elle-meme -- il masquait des contrats
propres a -41 points en moyenne.

Tourne indefiniment par defaut. Deux facons d'arreter proprement (sauvegarde
les 7 checkpoints puis s'arrete) :
  - Ctrl+C / SIGINT si le process est lance au premier plan (attache a un
    terminal).
  - Creer le fichier STOP_FLAG_PATH (`.stop_training`, teste a chaque donne)
    -- SEULE methode fiable pour un process DETACHE (`Start-Process` sur
    Windows termine le process sans jamais invoquer le handler SIGTERM ;
    voir memoire projet [[project-background-tasks-die-with-session]]).
  - --until AAAA-MM-JJTHH:MM (heure locale) : meme arret propre a cette heure
    (jobs cloud de duree bornee).
--max-batches borne l'execution (tests).

Toutes les --stats-every donnes (10000 par defaut), imprime ET persiste
(JSON Lines, --stats-file) des statistiques par famille de declarant :
plis moyens du declarant/camp vs plis moyens du camp de defense (2
adversaires pour emballage, 3 partout ailleurs), plus un taux de reussite
(cible EXACTE atteinte) pour piccolo/petite_misere/grande_misere/
misere_etalee -- demande explicite de l'utilisateur pour juger comment
chaque contrat se debrouille face a `defenseur`, avec un format simple
pense pour etre relu facilement cote Godot (voir DESIGN.md, "Statistiques
periodiques").

Usage : venv/Scripts/python.exe scripts/train.py [--batch-size N] [--save-every N]
    [--seed N] [--max-batches N] [--stats-every N] [--stats-file PATH]
"""

import argparse
import json
import os
import copy
import datetime
import random
import signal
import sys
import time

import torch

sys.path.insert(0, ".")

from agents.play_guards import train_play_mask
from agents import credit, evolution, expert_bot, oracle_critic, ppo, rollouts, specialists
from agents.abondance_rule import EMBALLAGE_RAISE_MAX_LEVEL, abondance_rule, apply_bid_guards, emballage_raise_too_high
from agents import bid_ppo as bid_ppo_mod
from agents.checkpoint_io import safe_write_json
from agents.specialists import IMPLEMENTED, PHASE1_ROLES, PLAY_ROLES
from engine.bidding import ACTION_PASS, ACTION_RAISE, Bidding, bid_action_to_dict, first_leader
from engine.bidding import legal_action_ids_with_hand as bid_legal_action_ids_with_hand
from engine.bidding import legal_action_mask_with_hand as bid_legal_action_mask_with_hand
from engine.bidding import observation as bid_observation
from engine.cards import NUM_CARDS, deal
from engine.observation import split_tricks
from engine.emballage import new_emballage_deal
from engine.emballage import final_scores as emballage_final_scores
from engine.misere import DECLARER_ROLE_BY_KIND, new_misere_deal
from engine.misere import final_scores as misere_final_scores
from engine.solitaire import new_solitaire_deal
from engine.solitaire import final_scores as solitaire_final_scores
from engine.trou import detect_trou, new_trou_deal
from engine.trou import final_scores as trou_final_scores
from engine.whisted import DeckChain
from scripts import export_to_godot, godot_sync, record_game
from scripts.team_errors import atout_situation, coupe_situation, surcoupe_situation

DEFENSEUR = "defenseur"

# nom de role specialiste -> kind interne attendu par engine/misere.py
# (engine/scoring.py:MISERE_ORDER). engine/misere.py route deja chaque kind
# vers ce MEME nom de role via DECLARER_ROLE_BY_KIND -- verifie une seule
# fois au chargement du module (voir _assert_role_kind_consistency ci-dessous)
# plutot que reimporte ici en double.
MISERE_FAMILIES = {
    "piccolo": "piccolo",
    "petite_misere": "petite",
    "grande_misere": "grande",
    "misere_etalee": "etalee",
}
DECLARER_FAMILIES = ["solitaire", "emballage"] + list(MISERE_FAMILIES)

# Poids du nombre de declarants simultanes pour les familles misere (le
# mecanisme multi-declarant est reel, voir WHIST-RULES-REFERENCE.txt, mais
# 1 seul declarant reste largement le cas majoritaire dans une vraie
# partie) -- juste assez de poids sur 2-4 pour exposer `defenseur` a des
# tailles de camp de defense variables (1 a 3 defenseurs).
MISERE_DECLARER_COUNTS = [1, 2, 3, 4]
MISERE_DECLARER_WEIGHTS = [70, 20, 7, 3]

# Misere Phase 1 (2026-09-27 soir, demande utilisateur : "change
# l'entrainement pour que l'IA puisse s'entrainer en misere correctement").
# Le declarant d'une misere forcee etait tire au HASARD : une main
# quelconque est presque toujours injouable en misere, l'IA apprenait
# surtout a perdre des miseres perdues d'avance (mesure : 16 % de petites
# miseres reussies meme sur des mains correctes). Desormais on redonne
# jusqu'a MISERE_REDEALS fois pour qu'au moins un siege ait une main que
# les criteres de l'Ecole du whist (agents/expert_bot.py:misere_kinds_for)
# jugent annoncable, et les declarants sont les sieges conformes. Ces
# criteres ne servent qu'a CHOISIR LES MAINS : le jeu de la carte reste
# entierement celui du reseau, sans professeur (l'imitation de l'ExpertBot
# exclut toujours les miseres). MISERE_RANDOM_SHARE des donnes gardent un
# declarant au hasard (variete, et defense contre de mauvaises miseres).
MISERE_REDEALS = 400  # etalee : ~1 main sur 900 (grande 1/210, piccolo 1/45, petite 1/32)
MISERE_RANDOM_SHARE = 0.15

# level=6 : place-holder ignore par new_solitaire_deal (jamais transmis a
# LoneDeclarerDeal, seulement utilise par solitaire.py:final_scores() pour
# le bareme de points reel -- hors-perimetre Phase 1, recompense = plis).
SOLITAIRE_LEVEL_PLACEHOLDER = 6
# level=8 : idem pour new_emballage_deal, ignore par la recompense Phase 1.
EMBALLAGE_LEVEL_PLACEHOLDER = 8

# Nombre d'adversaires "defenseur" par famille -- exactement ce que
# l'utilisateur veut lire dans les statistiques periodiques.
FAMILY_OPPONENTS = {"solitaire": 3, "emballage": 2, "piccolo": 3,
                     "petite_misere": 3, "grande_misere": 3, "misere_etalee": 3}
# Cible EXACTE de plis pour un declarant, familles ou "reussite" a un sens
# binaire meme sans niveau modelise (voir engine/misere.py:MISERE_MAX_TRICKS) --
# solitaire/emballage n'ont pas de cible Phase 1 (pas de niveau), donc pas
# de taux de reussite pour elles, seulement des moyennes de plis.
MISERE_TARGET_TRICKS = {"piccolo": 1, "petite_misere": 0, "grande_misere": 0, "misere_etalee": 0}

STOP_FLAG_PATH = ".stop_training"
STATS_EVERY_DEFAULT = 10000

# Rollouts determinises (agents/rollouts.py, 2026-09-26) : pour quelques
# decisions par donne, valeur de CHAQUE carte legale -> cible de politique
# apprise en plus de PPO (agents/ppo.py:DISTILL_COEF). Pas un faconnage :
# meme score que la recompense, juste compare carte par carte.
# Analyse au hasard : 0 par defaut depuis le 2026-09-26 apres-midi -- a 1-2
# decisions/donne, la surcoupe a legerement AUGMENTE (45 -> 49%) : les
# situations de surcoupe (~2 sur 52 decisions) n'etaient presque jamais
# analysees. Remplacee par l'analyse CIBLEE ci-dessous.
ROLLOUT_PER_DEAL_DEFAULT = 0.0  # decisions analysees au hasard par donne, en moyenne
ROLLOUT_SAMPLES_DEFAULT = 8  # mondes (mains cachees tirees) par decision
# Analyse CIBLEE (accord utilisateur 2026-09-26) : CHAQUE occasion de surcoupe
# (scripts/team_errors.py:surcoupe_situation) hors misere est analysee avec
# 16 mondes (4 : ~1 verdict faux sur 8 ; 16 : aucun sur 80 cas testes). Le
# vrai score juge toujours : pas de malus, egalites gratuites. Misere exclue :
# y voler la main au coequipier est souvent neutre, parfois utile.
# DESACTIVE 2026-09-26 22:40 : en ~36k donnes (14h55 -> 22h36) l'IA a
# generalise "ne pas prendre la main du coequipier" en "ne pas prendre le
# pli" -- en derniere position, adversaire maitre, carte gagnante en main :
# pris 50% -> 20% (emballage, defense). Checkpoints abimes gardes dans
# ../whist-ai-backups/checkpoints_v2_20260926_2237_degrade_rollouts_surcoupe,
# reprise depuis la sauvegarde de 14h55 + observation v4 (bloc N).
ROLLOUT_SURCOUPE_SAMPLES_DEFAULT = 0  # 16 = ancienne valeur
# Analyse CIBLEE des renonces non coupees (signalement utilisateur 2026-10-22 :
# le coequipier entame dans la renonce de l'IA, qui defausse au lieu de couper ;
# scripts/team_errors.py:coupe_situation). Meme mecanisme que la surcoupe, mais
# avec la suite jouee par l'ExpertBot (--rollout-policy expert) : le vrai score
# de la donne juge chaque carte, aucune restriction ni malus -- si defausser
# est meilleur (coequipier deja maitre...), c'est la defausse qui est apprise.
ROLLOUT_COUPE_SAMPLES_DEFAULT = 0
# Analyse CIBLEE de l'entame du camp declarant quand un adversaire a encore de
# l'atout (scripts/team_errors.py:atout_situation ; « ne tire pas l'atout »,
# 49 % le 24/10, inchange apres 2 runs dds). Le score DDS juge : si une entame
# hors atout vaut mieux, c'est elle qui est apprise.
ROLLOUT_ATOUT_SAMPLES_DEFAULT = 0
# Marge (unites de retour : plis ; points/10 pour defenseur) sous laquelle une
# carte compte parmi les meilleures -- absorbe le bruit des 4 mondes.
ROLLOUT_MARGIN_DEFAULT = 0.25
# Marge exprimee EN PLIS (2026-10-25) : remplace --rollout-margin si > 0. La
# marge ci-dessus etait multipliee par return_scale (10) et comparee a des
# POINTS (un pli = 2 pts en emballage/solo) ou des PLIS (declarants Phase 1) :
# 2.5 unites = 1.25 a 2.5 plis -> une coupe qui rapporte UN pli n'etait jamais
# enseignee (renonce non coupee 29 -> 37 % malgre le DDS). Ici : pli de chaque
# siege (rollout["trick_unit"]) x cette marge.
ROLLOUT_MARGIN_TRICKS_DEFAULT = 0.0
POINTS_PER_TRICK = 2.0  # pas d'un pli en vrais points (EMBALLAGE_STEP, SOLO_SUCCESS_STEP)
# Evaluateur omniscient (agents/oracle_critic.py) : lots d'apprentissage d'un
# role avant que ses valeurs remplacent la tete de valeur de l'acteur dans GAE.
ORACLE_WARM_LOTS_DEFAULT = 30

# Etat de l'evaluateur omniscient, rempli par main() si --oracle-critic :
# {"critics": {role: (critic, optimizer)}, "lots": {role: n}, "warm": n, "dir": ...}
ORACLE = None
# Politique qui joue la SUITE de la donne dans les rollouts (2026-10-19).
# "nets" (historique) : les reseaux eux-memes -- leurs propres faiblesses
# faussent Q (surcoupe 26/09 : l'IA a cesse de prendre les plis). "expert" :
# l'ExpertBot aux 4 sieges = PIMC au-dessus du bot, un pas d'amelioration de
# politique qui le BAT par construction ; c'est ce professeur plus fort que le
# bot qui manquait au PPO, plafonne au niveau du bot depuis le 29/09 (duplicate :
# defense IA +0.27 pli concede/donne, stable 3 semaines).
# "dds" (2026-10-23, « ExpertBot est horrible aussi ! Ce n'est pas une
# reference ») : la suite est jouee PARFAITEMENT a cartes ouvertes dans chaque
# monde (engine/dds.c) -- aucun bot a regles dans la cible. Contrats a atout
# seulement (emballage, trou, solo) ; ailleurs aucune cible de rollout (PPO seul).
ROLLOUT_POLICY_DEFAULT = "nets"
DECISIONS_PER_DEAL = 52

# Imitation de l'ExpertBot (agents/expert_bot.py ; accord utilisateur
# 2026-09-27 : "Vas-y pour l'imitation du bot"). Le duplicate du 27/09
# (scripts/expert_bot_match.py) montre le bot nettement meilleur que l'IA
# (declarant 80 % vs 50 % face a la meme defense ; defense IA ~ hasard).
# Methode DAgger : sur les etats que l'IA visite ELLE-MEME, le coup du bot
# devient une cible de politique (meme perte que les rollouts, voir
# agents/ppo.py), a cote de PPO. Poids decroissant lineairement de
# --teacher-coef a 0 sur --teacher-decay-deals donnes (compteur PERSISTANT,
# checkpoints_v2/teacher_state.json) : ensuite PPO seul, libre de depasser
# le professeur. Tous les contrats, attaque ET defense, ecart de petite
# misere compris. Jusqu'au 2026-09-27 23:40 les miseres etaient exclues (le
# bot n'y avait qu'une defense minimale) ; depuis 710f9a8 il les joue selon
# l'Ecole du whist et, sur les memes mains annoncables face a la meme defense
# IA, reussit 76-95 % en declarant contre 40-77 % pour l'IA (accord
# utilisateur pour les inclure).
TEACHER_COEF_DEFAULT = 1.0
TEACHER_DECAY_DEALS_DEFAULT = 400_000

# Joueurs "visiteurs" (demande utilisateur 2026-10-02 : "il faut que l'IA
# s'entraine a affronter des joueurs divers et varies qui ne jouent pas
# forcement de la meilleure des manieres"). Avec cette probabilite, UN camp
# entier de la donne (Phase 1 : declarant(s) OU defense ; Phase 2 : sieges
# 0/2 ou 1/3, enchere comprise) est tenu par un style tire de VISITOR_STYLES ;
# leurs decisions ne sont JAMAIS apprises, seul l'autre camp (IA) apprend. Ces
# donnes sont exclues des stats Phase 1/2 (suivi comparable a avant).
VISITOR_PROB_DEFAULT = 0.25
VISITOR_STYLES = {"expert": 0.3, "bruite": 0.3, "debutant": 0.2, "hasard": 0.2}
VISITOR_NOISE_EPS = 0.25  # "bruite" : IA courante, coup au hasard 25 % du temps, sinon tiree a temperature 2
VISITOR_NOISE_TEMPERATURE = 2.0
# Donnes RICHES (demande utilisateur 2026-10-03 : "l'IA doit savoir reconnaitre
# la main qui pourrait gagner ces gros contrats rares et jouer parfaitement
# dessus"). Mesure scripts/big_contracts_bench.py (03/10, 1000 mains, deck
# whiste) : l'IA JOUE l'abondance mieux que l'ExpertBot (abondance-9 imposee :
# 65 % contre 54 %) mais ne l'ANNONCE presque jamais (4 % des mains jouables,
# solo-7/8 a la place). A 9 plis surs c'est juste (son solo rapporte +11.8 contre
# +4.8 pour l'abondance-9) ; a >= 10 plis surs (~1 % des donnes) elle laisse 10 a
# 25 pts par donne (abondance-9/10 a +24/+41). L'emballage haut, lui, ne paie
# jamais plus a plis egaux (bareme) : rien a apprendre hors surenchere.
# Avec RICH_PROB_DEFAULT, une donne Phase 2 est redonnee (paquets ramasses non
# joues, comme les miseres de la Phase 1) jusqu'a ce qu'un siege ait au moins
# RICH_MIN_TRICKS plis surs (agents/expert_bot.py:estimate_tricks) -- 9 inclus,
# pour apprendre aussi quand NE PAS l'annoncer. Sur ces donnes : pas d'ancre (elle
# retient l'ancien choix, qui n'annoncait pas), pas de visiteurs, et le siege
# fort essaie avec RICH_EXPLORE une abondance/un solo chelem legal dans sa
# meilleure couleur, niveau <= plis surs (rich_explore_actions), a chacune
# de ses decisions ou il en a une (le vrai score juge). Hors stats.
# Niveau plafonne aux plis surs (03/10 13:50) : avec "<= plis surs + 1", apres
# 210 000 donnes l'IA n'annoncait pas plus (banc : 3 % contre 4 % avant).
# Abondance imposee, jeu IA (500 mains du banc) : au niveau plis surs + 1,
# 21-25 % de reussite et -23 a -90 pts pour le soliste, quel que soit le
# nombre de plis surs ; a 9 plis surs, le tirage 9/10 valait -8 en moyenne
# (moins que le solo-7) et apprenait "n'annonce pas". Au niveau plis surs - 1,
# 10 plis surs : +28.6 (98 %) ; 11 : +35.5 (90 %).
RICH_PROB_DEFAULT = 0.1
RICH_MIN_TRICKS = 9
RICH_REDEALS = 400
RICH_EXPLORE = 0.25
TEACHER_STATE_FILE = "teacher_state.json"  # dans specialists.CHECKPOINT_DIR


# Imitation RESERVEE aux declarants de misere (accord utilisateur 2026-09-28
# 23:30). Mesure du soir : l'IA n'annoncait plus AUCUNE misere (0 sur 500
# donnes en Phase 3, ExpertBot 46) parce qu'elle les joue mal en declarante :
# expert_match ia/bot 18-36 % contre 68 % pour le bot sur les memes mains face
# a la meme defense, sans progres pendant 20 h apres la fin de l'imitation
# generale (poids nul a 400k donnes, ~4 h seulement de misere imitee). Meme
# methode DAgger, cible = coup du bot, mais seulement pour les sieges des roles
# MISERE_DECLARER_ROLES ; compteur separe (teacher_misere_state.json) et
# decroissance lente, pour laisser au reseau le temps d'apprendre avant que
# PPO seul reprenne la main. Meme constat pour le camp declarant du TROU
# (duplicate 2000 contrats, 28/09 23:40 : ia/bot 55 % contre bot/bot 81 % sur
# 341 trous, 17 % des donnes) -> roles trou et bouche_trou inclus.
MISERE_DECLARER_ROLES = frozenset(MISERE_FAMILIES) | {"trou", "bouche_trou"}
MISERE_TEACHER_COEF_DEFAULT = 1.0
MISERE_TEACHER_DECAY_DEALS_DEFAULT = 1_000_000
MISERE_TEACHER_STATE_FILE = "teacher_misere_state.json"
# Plancher PERMANENT de cette imitation (accord utilisateur 2026-10-01) :
# mesure du soir, memes donnes, defense ExpertBot fixe -- apres la fin de
# l'imitation (1M donnes) les declarants misere ont regresse (petite 59.5 ->
# 46.7 %, etalee 56.0 -> 46.1 %) alors que la defense n'avait pas bouge ; et
# la regle du Trou du 2026-10-01 demande de reapprendre l'entame du
# bouche-trou (ExpertBot : trou_opening_card). Le poids ne descend plus sous
# ce plancher.
MISERE_TEACHER_FLOOR_DEFAULT = 0.2
# Poids de tirage des familles Phase 1 (1 = uniforme) : chaque famille misere
# pese PHASE1_MISERE_WEIGHT_DEFAULT contre 1 pour solitaire/emballage, qui ont
# deja l'essentiel des donnes de la Phase 2 (~90 decisions par role misere et
# par lot contre ~2600 en emballage avant ce changement).
PHASE1_MISERE_WEIGHT_DEFAULT = 4.0


def teacher_state_path(state_file=TEACHER_STATE_FILE):
    return os.path.join(specialists.CHECKPOINT_DIR, state_file)


def load_teacher_deals(state_file=TEACHER_STATE_FILE):
    try:
        with open(teacher_state_path(state_file), encoding="utf-8") as f:
            return int(json.load(f)["deals"])
    except (OSError, ValueError, KeyError):
        return 0


def teacher_coef_at(deals, coef, decay_deals):
    if coef <= 0 or decay_deals <= 0:
        return 0.0
    return coef * max(0.0, 1.0 - deals / float(decay_deals))


def teacher_target(game, seat):
    """Coup de l'ExpertBot en cible one-hot (tous contrats, ecart compris)."""
    card = expert_bot.choose_card(game, seat)
    target = torch.zeros(NUM_CARDS)
    target[card] = 1.0
    return target, card
STATS_FILE_DEFAULT = "training_stats.jsonl"

# --- Phase 2 : vraie enchere (voir docstring de module) -----------------

# 0.5 jusqu'au 2026-09-26, puis 0.8 (accord utilisateur) : les vraies donnes
# (vraie enchere, mains coherentes avec le contrat) apprennent plus que les
# contrats forces ; la Phase 1 reste pour les familles rares (misere, etc.).
PHASE2_PROB_DEFAULT = 0.8  # probabilite qu'une donne utilise la vraie enchere
# Probabilite qu'une donne soit distribuee en DECK WHISTE (paquet ramasse de
# la donne precedente, coupe une fois -- voir engine/whisted.py), sinon
# melange uniforme. 50/50 = decision utilisateur (deja `--whisted-ratio 0.5`
# avant le reset du 2026-09-18, perdu par ce reset, retabli le 2026-09-24) ;
# 100 % depuis le 2026-10-03 (decision utilisateur : "passe a 100% de deck
# whiste") -- le deck whiste double les mains d'abondance (4.5 % contre 2.2 %).
WHISTED_RATIO_DEFAULT = 1.0
# Mesures periodiques (duplicate ExpertBot, RandomBot, surcoupes) : ratio
# d'avant le 2026-10-03, garde fixe pour que leurs series restent comparables.
MEASURE_WHISTED_RATIO = 0.5

BID_GAMES_PER_INDIVIDUAL_DEFAULT = 50  # ne regle plus que la cadence des generations (voir BIDDER_EVAL_DEALS_DEFAULT)
BIDDER_EVAL_DEALS_DEFAULT = 200  # donnes communes du duplicate vs ExpertBot par generation ; 80 jusqu au 2026-09-28 00:15 : trop peu pour que les solos >= 8 (~3 % des donnes, annonces sous niveau-1 plis surs 59-100 % du temps par le meilleur net) pesent dans la selection (decision utilisateur : plus de donnes plutot qu un garde-fou)
POPULATION_FILE_DEFAULT = os.path.join("checkpoints_v2", "encherisseur_population.pt")

# Export Godot automatique (demande explicite de l'utilisateur, 2026-09-20,
# "tu exportes l'IA automatiquement tous les 100 000 à partir de
# maintenant") -- regenere godot_export/ (voir scripts/export_to_godot.py)
# toutes les EXPORT_EVERY_DEFAULT donnes ET a l'arret, roles de jeu + les 2
# slots d'enchere (meilleur individu par lignee). Depuis 2026-09-23 ("fais
# cet export a chaque fois"), l'export est VERIFIE puis publie dans
# whist-4-it/game/ avec les stats Phase 1/2 (voir scripts/godot_sync.py) ;
# un export divergent n'est jamais copie. --no-godot-sync pour les tests.
EXPORT_EVERY_DEFAULT = 100_000
EXPORT_DIR_DEFAULT = "godot_export"


def new_stats_accumulator():
    return {family: {"donnes": 0, "declarer_tricks_sum": 0.0, "defense_tricks_sum": 0.0,
                      "declarer_seats": 0, "success_count": 0} for family in DECLARER_FAMILIES}


def update_stats(stats, family, game, declarer_seats):
    """Accumule, sur la fenetre courante (remise a zero a chaque emission),
    de quoi calculer plis moyens et taux de reussite -- voir emit_stats()."""
    entry = stats[family]
    total_tricks = sum(game.tricks_won)
    declarer_tricks_sum = sum(game.tricks_won[s] for s in declarer_seats)
    entry["donnes"] += 1
    entry["declarer_tricks_sum"] += declarer_tricks_sum
    entry["defense_tricks_sum"] += (total_tricks - declarer_tricks_sum)
    entry["declarer_seats"] += len(declarer_seats)
    target = MISERE_TARGET_TRICKS.get(family)
    if target is not None:
        entry["success_count"] += sum(1 for s in declarer_seats if game.tricks_won[s] == target)


def emit_stats(stats, total_episodes, elapsed, stats_file):
    """Imprime ET persiste (une ligne JSON par emission, format simple a
    relire cote Godot -- voir DESIGN.md) les stats de la fenetre courante."""
    record = {"phase": 1, "total_donnes": total_episodes, "elapsed_s": round(elapsed, 1),
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "families": {}}
    print("\n=== Statistiques par famille (derniers %d donnes, total=%d) ===" % (
        sum(e["donnes"] for e in stats.values()), total_episodes), flush=True)
    for family in DECLARER_FAMILIES:
        entry = stats[family]
        n, seats = entry["donnes"], entry["declarer_seats"]
        opponents = FAMILY_OPPONENTS[family]
        avg_declarer = (entry["declarer_tricks_sum"] / seats) if seats else 0.0
        avg_defense = (entry["defense_tricks_sum"] / n) if n else 0.0
        block = {"donnes": n, "opponents": opponents,
                 "avg_declarer_tricks_per_seat": round(avg_declarer, 3),
                 "avg_defense_tricks": round(avg_defense, 3)}
        line = "  %-14s donnes=%-5d vs %d defenseur(s)  plis_declarant_moy/siege=%5.2f  plis_defense_moy=%5.2f" % (
            family, n, opponents, avg_declarer, avg_defense)
        target = MISERE_TARGET_TRICKS.get(family)
        if target is not None:
            success_rate = (entry["success_count"] / seats) if seats else 0.0
            block["target_tricks"] = target
            block["success_rate"] = round(success_rate, 4)
            block["declarer_attempts"] = seats
            line += "  reussite(=%d pli)=%5.1f%% (%d/%d)" % (
                target, success_rate * 100, entry["success_count"], seats)
        record["families"][family] = block
        print(line, flush=True)
    with open(stats_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("  (ajoute a %s -- format JSON Lines, une ligne par emission)\n" % stats_file, flush=True)


STATS_FILE_PHASE2_DEFAULT = "training_stats_phase2.jsonl"
PHASE2_STATS_FAMILIES = ["solitaire", "emballage", "trou"] + list(MISERE_FAMILIES)


def new_phase2_stats_accumulator():
    """Stats Phase 2 (vraie enchere), SEPAREES de celles de la Phase 1
    (demande explicite de l'utilisateur, 2026-09-22) : par famille de
    contrat REEL, plus la repartition des contrats, redonnes et actions
    d'enchere sur la meme fenetre."""
    return {"families": {family: {"donnes": 0, "declarer_tricks_sum": 0.0, "defense_tricks_sum": 0.0,
                                   "declarer_seats": 0, "success_count": 0, "success_attempts": 0,
                                   "by_level": {}}
                          for family in PHASE2_STATS_FAMILIES},
            "donnes": 0, "general_pass": 0, "contracts": {}, "bid_actions": {}}


def update_phase2_stats(stats2, role, game, declarer_seats, contract):
    """Reussite = contrat REMPLI : plis du camp >= niveau encheri
    (solitaire/emballage/trou), cible exacte pour piccolo/misere."""
    entry = stats2["families"][role]
    total_tricks = sum(game.tricks_won)
    declarer_tricks_sum = sum(game.tricks_won[s] for s in declarer_seats)
    entry["donnes"] += 1
    entry["declarer_tricks_sum"] += declarer_tricks_sum
    entry["defense_tricks_sum"] += (total_tricks - declarer_tricks_sum)
    entry["declarer_seats"] += len(declarer_seats)
    target = MISERE_TARGET_TRICKS.get(role)
    if target is not None:
        entry["success_attempts"] += len(declarer_seats)
        entry["success_count"] += sum(1 for s in declarer_seats if game.tricks_won[s] == target)
    else:
        level = contract.get("level", getattr(game, "level", None))
        success = 1 if level is not None and declarer_tricks_sum >= level else 0
        entry["success_attempts"] += 1
        entry["success_count"] += success
        if level is not None:
            per_level = entry["by_level"].setdefault(str(level), {"success_count": 0, "attempts": 0})
            per_level["attempts"] += 1
            per_level["success_count"] += success


def phase2_family_blocks(stats2):
    """Bloc "families" d'un record Phase 2 (aussi utilise tel quel par la
    Phase 3, scripts/phase3.py -- meme format lu par whist-4-it), imprime
    une ligne par famille au passage."""
    families = {}
    for family in PHASE2_STATS_FAMILIES:
        entry = stats2["families"][family]
        n, seats = entry["donnes"], entry["declarer_seats"]
        avg_declarer = (entry["declarer_tricks_sum"] / seats) if seats else 0.0
        avg_defense = (entry["defense_tricks_sum"] / n) if n else 0.0
        success_rate = (entry["success_count"] / entry["success_attempts"]) if entry["success_attempts"] else 0.0
        families[family] = {
            "donnes": n, "avg_declarer_tricks_per_seat": round(avg_declarer, 3),
            "avg_defense_tricks": round(avg_defense, 3), "success_rate": round(success_rate, 4),
            "success_attempts": entry["success_attempts"]}
        if entry["by_level"]:
            # Detail par niveau encheri (demande whist-4-it) : cle = niveau en texte.
            families[family]["by_level"] = {
                lvl: {"success_rate": round(v["success_count"] / v["attempts"], 4), "attempts": v["attempts"]}
                for lvl, v in sorted(entry["by_level"].items(), key=lambda kv: int(kv[0]))}
        print("  %-14s donnes=%-5d plis_declarant_moy/siege=%5.2f  plis_defense_moy=%5.2f  contrat_rempli=%5.1f%%" % (
            family, n, avg_declarer, avg_defense, success_rate * 100), flush=True)
    return families


def emit_phase2_stats(stats2, total_episodes, elapsed, stats_file, generation, population):
    record = {"phase": 2, "total_donnes": total_episodes, "phase2_donnes": stats2["donnes"],
              "elapsed_s": round(elapsed, 1), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "general_pass": stats2["general_pass"], "bidder_generation": generation,
              "contracts": dict(sorted(stats2["contracts"].items())),
              "bid_actions": dict(sorted(stats2["bid_actions"].items())),
              "bidder_lineages": {}, "families": {}}
    print("\n=== Statistiques PHASE 2 par famille de contrat reel (%d donnes Phase 2, redonnes=%d, total=%d) ===" % (
        stats2["donnes"], stats2["general_pass"], total_episodes), flush=True)
    record["families"] = phase2_family_blocks(stats2)
    for lineage in evolution.LINEAGES:
        # snapshot_fitness : la fitness duplicate vs ExpertBot est calculee en fin de
        # generation puis remise a zero -> en cours de generation, derniere connue
        fits = [f for f in (ind.snapshot_fitness() for ind in population if ind.lineage == lineage)
                if f != float("-inf")]
        record["bidder_lineages"][lineage] = {
            "effectif": sum(1 for ind in population if ind.lineage == lineage),
            "fitness_moy": round(sum(fits) / len(fits), 3) if fits else None,
            "fitness_max": round(max(fits), 3) if fits else None}
    with open(stats_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("  (ajoute a %s)\n" % stats_file, flush=True)


def _assert_role_kind_consistency():
    from engine.misere import DECLARER_ROLE_BY_KIND
    for role, kind in MISERE_FAMILIES.items():
        assert DECLARER_ROLE_BY_KIND[kind] == role, (
            "desynchro entre scripts/train.py:MISERE_FAMILIES et "
            "engine/misere.py:DECLARER_ROLE_BY_KIND pour kind=%r" % kind
        )


_assert_role_kind_consistency()


def misere_seat_order(hands, kind):
    """Sieges du plus au moins plausible pour une misere `kind` : d'abord
    ceux dont la main la porte (criteres Ecole du whist), puis les moins
    de cartes dangereuses."""
    return sorted(range(4), key=lambda s: (not expert_bot.misere_fits(hands[s], kind),
                                           expert_bot.misere_danger_score(hands[s])))


def deal_for_family(family, rng, chain=None):
    """-> (hands, dealer, whisted, misere_seats). Familles misere : redonne
    jusqu'a trouver une main annoncable (voir MISERE_REDEALS) ; les donnes
    ecartees sont ramassees sans etre jouees, comme une passe generale.
    misere_seats = ordre des declarants plausibles, ou None (hasard)."""
    kind = MISERE_FAMILIES.get(family)
    targeted = kind is not None and rng.random() >= MISERE_RANDOM_SHARE
    attempts = MISERE_REDEALS if targeted else 1
    for attempt in range(attempts):
        if chain is None:
            hands, dealer, whisted = deal(rng), None, False
        else:
            hands, dealer, whisted = chain.deal(rng)
        if not targeted:
            return hands, dealer, whisted, None
        if any(expert_bot.misere_fits(h, kind) for h in hands) or attempt == attempts - 1:
            return hands, dealer, whisted, misere_seat_order(hands, kind)
        if chain is not None:
            chain.finish_unplayed(hands, dealer)
    raise AssertionError("inatteignable")


def build_game(family, hands, rng, dealer=None, whisted=False, misere_seats=None):
    """Contrat SYNTHETIQUE (pas d'enchere en Phase 1) : donneur, declarant(s)
    et atout tires au hasard, mais toujours dans des CONDITIONS DE JEU
    REELLES (moteur/observation/legalite identiques au vrai jeu). Regle
    d'entame = toujours a gauche du donneur -- l'exception Abondance/Solo
    Chelem (le declarant entame) ne s'applique pas ici : `solitaire` est
    Phase 1 volontairement agnostique au niveau (voir DESIGN.md), cette
    exception appartient a la phase d'enchere, pas encore modelisee."""
    if dealer is None:
        dealer = rng.randrange(4)
    leader = (dealer + 1) % 4
    if family == "solitaire":
        declarer = rng.randrange(4)
        trump = rng.randrange(4)
        game = new_solitaire_deal(hands, declarer, trump, SOLITAIRE_LEVEL_PLACEHOLDER, leader)
        seats = {declarer}
    elif family == "emballage":
        declarer, partner = rng.sample(range(4), 2)
        trump = rng.randrange(4)
        game = new_emballage_deal(hands, declarer, partner, trump, EMBALLAGE_LEVEL_PLACEHOLDER, leader)
        seats = {declarer, partner}
    else:
        kind = MISERE_FAMILIES[family]
        n = rng.choices(MISERE_DECLARER_COUNTS, weights=MISERE_DECLARER_WEIGHTS)[0]
        if misere_seats is None:
            declarers = rng.sample(range(4), n)
        else:  # sieges conformes aux criteres, au moins le plus plausible
            fitting = [s for s in misere_seats if expert_bot.misere_fits(hands[s], kind)]
            declarers = fitting[:n] or misere_seats[:1]
        game = new_misere_deal(hands, declarers, kind, leader)
        seats = set(declarers)
    # Phase 1 : aucune enchere, aucun niveau annonce (placeholder interne
    # jamais montre au reseau) -- seul le contexte reellement public.
    game.set_public_context(dealer=dealer, bid_summary=None, whisted=whisted, level_known=False)
    return game, seats


def pick_visitor_style(rng):
    return rng.choices(list(VISITOR_STYLES), weights=list(VISITOR_STYLES.values()))[0]


def visitor_card(style, game, seat, nets, rng):
    """Coup d'un joueur visiteur (voir VISITOR_STYLES) -- jamais appris."""
    legal = [c for c, ok in enumerate(game.legal_action_mask(seat)) if ok]
    if style == "expert":
        return expert_bot.choose_card(game, seat)
    if style == "hasard" or (style == "bruite" and rng.random() < VISITOR_NOISE_EPS):
        return rng.choice(legal)
    if style == "debutant":  # joue toujours sa plus forte carte, meme en misere
        return expert_bot._highest(legal)
    obs = torch.tensor(game.observation(seat), dtype=torch.float32)
    mask = torch.tensor(game.legal_action_mask(seat), dtype=torch.bool)
    with torch.no_grad():
        logits, _value = nets[game.role_of(seat)].forward(obs.unsqueeze(0))
    logits = (logits.squeeze(0) / VISITOR_NOISE_TEMPERATURE).masked_fill(~mask, float("-inf"))
    return torch.distributions.Categorical(logits=logits).sample().item()


def dds_supported(game):
    from engine import dds
    return dds.supported(game)


def play_out(game, nets, rollout=None, teacher=None, visitors=None, rng=None):
    """`visitors` (optionnel) : {siege: style} -- ces sieges jouent en visiteur
    (visitor_card) et n'enregistrent rien.

    -> par siege : [(role, obs, mask, action, logp, valeur_brute, pli,
    cible, entree_oracle)] -- `pli` = nombre de plis deja termines au moment de la
    decision (0 pour l'ecart de petite misere), base du merite au pli ;
    `cible` = cible de politique des rollouts (agents/rollouts.py) ou None.

    `rollout` (optionnel) : {"prob", "samples", "surcoupe_samples",
    "margin", "score_fn", "rng", "count", "corrective", "surcoupes",
    "seconds"} -- chaque occasion de surcoupe hors misere est analysee avec
    `surcoupe_samples` mondes ; chaque autre decision de jeu (hors ecart) a
    `prob` chances d'etre analysee avec `samples` mondes.

    `teacher` (optionnel, imitation de l'ExpertBot) : {"count", "agree"} --
    chaque decision de son domaine recoit le coup du bot en cible."""
    per_seat_steps = [[] for _ in range(4)]
    while not game.done:
        seat = game.turn
        if visitors and seat in visitors:
            game.step(seat, visitor_card(visitors[seat], game, seat, nets, rng))
            continue
        role = game.role_of(seat)
        net = nets[role]
        target = None
        samples = 0
        if rollout is not None and not getattr(game, "pending_discards", None):
            if (rollout["surcoupe_samples"] and game.contract_type != "misere"
                    and surcoupe_situation(game, seat) is not None):
                samples = rollout["surcoupe_samples"]
                rollout["surcoupes"] += 1
            elif rollout.get("coupe_samples") and coupe_situation(game, seat) is not None:
                samples = rollout["coupe_samples"]
                rollout["coupes"] += 1
            elif rollout.get("atout_samples") and atout_situation(game, seat):
                samples = rollout["atout_samples"]
                rollout["atouts"] += 1
            elif rollout["prob"] > 0 and rollout["rng"].random() < rollout["prob"]:
                samples = rollout["samples"]
        if samples and rollout.get("policy") == "dds" and not dds_supported(game):
            samples = 0
        if samples:
            t0 = time.time()
            q = rollouts.action_values(game, seat, nets, rollout["score_fn"], rollout["rng"], samples,
                                       rollout.get("policy", "nets"))
            if q is not None:
                if rollout.get("margin_tricks"):
                    target = rollouts.best_set_target(q[0], q[1], rollout["margin_tricks"] * rollout["trick_unit"][seat])
                else:
                    target = rollouts.best_set_target(q[0], q[1], rollout["margin"], credit.return_scale(role))
                rollout["count"] += 1
                rollout["corrective"] += target is not None
            rollout["seconds"] += time.time() - t0
        teacher_card = None
        if teacher is not None and target is None and (teacher.get("roles") is None or role in teacher["roles"]):
            target, teacher_card = teacher_target(game, seat)
        obs = torch.tensor(game.observation(seat), dtype=torch.float32)
        mask, taught = train_play_mask(game, seat)
        mask = torch.tensor(mask, dtype=torch.bool)
        if taught is not None and target is None:
            # garde-fous enseignes (agents/play_guards.py) : coup jouable, cible = les autres cartes
            target = torch.tensor(taught, dtype=torch.float32)
        with torch.no_grad():
            action, logp, value = net.act(obs, mask)
        value = value.item() * credit.return_scale(role)
        oracle_x = None
        if ORACLE is not None:
            oracle_x = torch.cat([obs, oracle_critic.oracle_features(game, seat)])
            if ORACLE["lots"].get(role, 0) >= ORACLE["warm"]:
                value = oracle_critic.value(ORACLE["critics"][role][0], oracle_x, credit.return_scale(role))
        if teacher_card is not None:
            teacher["count"] += 1
            teacher["agree"] += action == teacher_card
        trick_index = len(game.play_log) // 4
        per_seat_steps[seat].append(
            (role, obs, mask, action, logp, value, trick_index, target, oracle_x))
        game.step(seat, action)
    return per_seat_steps


def declarer_seat_reward(family, tricks):
    """Recompense d'UN siege declarant pour son propre nombre de plis --
    voir docstring de module. `piccolo`/`petite_misere`/`grande_misere`/
    `misere_etalee` ont une CIBLE (1 pli, ou 0) donc leur recompense est
    une distance (toujours <= 0, maximale a la cible) ; `solitaire`/
    `emballage` n'ont pas de cible Phase 1, juste maximiser."""
    if family == "piccolo":
        return -abs(float(tricks) - 1.0)
    if family in ("petite_misere", "grande_misere", "misere_etalee"):
        return -float(tricks)
    return float(tricks)  # solitaire, emballage


def defense_points_reward(family, game):
    """Vraie valeur de score (voir engine/scoring.py, deja verifiee contre
    Godot) recue par CHAQUE siege de defense -- meme valeur pour tous,
    jamais divisee, deja garanti par les engine.*.final_scores existants.
    Reutilise le level place-holder de build_game() pour solitaire/
    emballage (aucun niveau reel sans enchere, voir docstring de module)."""
    if family == "solitaire":
        return solitaire_final_scores(game, SOLITAIRE_LEVEL_PLACEHOLDER)
    if family == "emballage":
        return emballage_final_scores(game)
    return misere_final_scores(game, MISERE_FAMILIES[family])


def compute_scores(family, game, declarer_seats):
    """Declarant = recompense dense en plis (declarer_seat_reward,
    inchangee) ; defenseur = vrai bareme de points (defense_points_reward,
    voir docstring de module -- demande explicite de l'utilisateur pour
    que prendre des plis au-dela du seuil de reussite/echec d'un contrat
    n'apporte plus rien a la defense, comme dans le vrai jeu).

    `declarer_seats` peut contenir 1 a 4 sieges : EQUIPE pour
    solitaire/emballage (tous recoivent LA MEME valeur, calculee sur leur
    total de plis COMBINE -- jamais individuel), CHACUN POUR SOI pour la
    famille misere (recompense individuelle par siege, cible identique
    mais atteinte ou non independamment).

    Depuis le 2026-10-02 (demande utilisateur : "verifie que les bons
    comportements sont chaque fois recompenses et les mauvais punis"), la
    famille misere est notee en VRAIS POINTS pour TOUS les sieges, declarants
    compris : l'ancien -plis punissait autant le 3e pli d'une misere deja
    chutee que le 1er (celui qui la fait chuter), et ne recompensait pas la
    reussite elle-meme. Solitaire/emballage gardent les plis : pas de niveau
    reel en Phase 1."""
    if family in MISERE_FAMILIES:
        return [float(x) for x in defense_points_reward(family, game)]
    scores = [0.0] * 4
    if family in ("solitaire", "emballage"):
        team_tricks = sum(game.tricks_won[s] for s in declarer_seats)
        declarer_reward = declarer_seat_reward(family, team_tricks)
        for s in declarer_seats:
            scores[s] = declarer_reward
    else:
        for s in declarer_seats:
            scores[s] = declarer_seat_reward(family, game.tricks_won[s])
    defense_scores = defense_points_reward(family, game)
    for s in range(4):
        if s not in declarer_seats:
            scores[s] = defense_scores[s]
    return scores


def trick_potentials(game, score_fn):
    """Valeur de chaque siege si la donne s'arretait apres 0..N plis --
    `score_fn(game_termine)` est la MEME fonction que la recompense finale,
    appliquee a une copie du jeu dont seuls les plis gagnes changent."""
    done, _current = split_tricks(game.play_log, game.trump_suit)
    potentials = []
    for tricks in credit.tricks_after_each([w for _plays, w in done]):
        view = copy.copy(game)
        view.tricks_won = tricks
        view.done = True
        potentials.append(score_fn(view))
    return potentials


def record(buffers, role_returns, per_seat_steps, game, score_fn):
    """Merite au pli (agents/credit.py). Retourne les scores finaux."""
    potentials = trick_potentials(game, score_fn)
    final = potentials[-1]
    for seat in range(4):
        steps = per_seat_steps[seat]
        if not steps:
            continue
        returns, advantages = credit.seat_credit(
            [(k, v) for _r, _o, _m, _a, _l, v, k, _t, _x in steps], [p[seat] for p in potentials], final[seat])
        for (role, obs, mask, action, logp, _v, _k, target, oracle_x), ret, adv in zip(steps, returns, advantages):
            buf = buffers[role]
            buf["obs"].append(obs)
            buf["mask"].append(mask)
            buf["action"].append(action)
            buf["logp"].append(logp)
            buf["ret"].append(ret)
            buf["adv"].append(adv)
            if "target" in buf:
                buf["target"].append(target)
            if oracle_x is not None:
                buf["oracle_x"].append(oracle_x)
        role_returns[steps[0][0]].append(final[seat])
    return final


def run_episode(nets, rng, buffers, role_returns, family_counts, stats, chain=None, rollout=None, teacher=None,
                misere_weight=1.0, visitor_prob=0.0, visitor_counts=None):
    """`chain` (engine/whisted.py:DeckChain) : si fourni, la donne vient de
    la suite de donnes (whistee ou uniforme selon son ratio, donneur suivi)
    et alimente le paquet de la suivante ; sinon donne uniforme isolee.
    `misere_weight` : poids de chaque famille misere (voir PHASE1_MISERE_WEIGHT_DEFAULT)."""
    if misere_weight == 1.0:
        family = rng.choice(DECLARER_FAMILIES)
    else:
        family = rng.choices(DECLARER_FAMILIES, weights=[
            misere_weight if f in MISERE_FAMILIES else 1.0 for f in DECLARER_FAMILIES])[0]
    hands, dealer, whisted, misere_seats = deal_for_family(family, rng, chain)
    game, declarer_seats = build_game(family, hands, rng, dealer, whisted, misere_seats)
    score_fn = lambda g: compute_scores(family, g, declarer_seats)
    if rollout is not None:
        rollout["score_fn"] = score_fn
        # declarants solo/emballage Phase 1 notes en plis, tout le reste en points
        rollout["trick_unit"] = [1.0 if family in ("solitaire", "emballage") and s in declarer_seats
                                 else POINTS_PER_TRICK for s in range(4)]
    visitors = None
    if rng.random() < visitor_prob:
        style = pick_visitor_style(rng)
        camp = declarer_seats if rng.random() < 0.5 else set(range(4)) - set(declarer_seats)
        visitors = {s: style for s in camp}
        visitor_counts[style] = visitor_counts.get(style, 0) + 1
    steps = play_out(game, nets, rollout, teacher, visitors, rng)
    if chain is not None:
        chain.finish_played(game, dealer)
    record(buffers, role_returns, steps, game, score_fn)
    family_counts[family] = family_counts.get(family, 0) + 1
    if visitors is None:
        update_stats(stats, family, game, declarer_seats)


def save_all(nets):
    for role in PLAY_ROLES:
        specialists.save(role, nets[role], nets[role + "_optimizer"])
    if ORACLE is not None:
        for role, (critic, optimizer) in ORACLE["critics"].items():
            oracle_critic.save(ORACLE["dir"], role, critic, optimizer)
        oracle_critic.save_state(ORACLE["dir"], ORACLE["lots"])


# --- Phase 2 : vraie enchere ---------------------------------------------

def run_bidding_phase(bidder_for_seat, hands, dealer_index):
    """Vraie enchere (engine/bidding.py), Trou auto-detecte AVANT le debut
    normal (comme le vrai jeu, voir engine/trou.py:detect_trou). Politique
    GREEDY (voir agents/network.py:act_greedy) -- pas d'echantillonnage :
    la diversite de la population evolutive vient des poids MUTES, pas
    d'un bruit d'action a l'evaluation, comme en evolution strategy
    classique ; c'est aussi le comportement REEL une fois exporte vers
    Godot. Retourne (bidding, action_log) -- action_log = [(seat,
    action_id), ...] dans l'ordre reel, pour les stats de decisions
    imprimees par lot."""
    bidding = Bidding(dealer_index)
    trou_info = detect_trou(hands)
    if trou_info is not None:
        bidding.seed_trou(trou_info)
    action_log = []
    while not bidding.finished:
        seat = bidding.turn
        hand = hands[seat]
        bidder = bidder_for_seat[seat]
        if getattr(bidder, "hybrid", False):  # lignee hybride : regles puis reseau (agents/hybrid_bidder.py)
            action_id = bidder.decide(bidding, seat, hand)
        else:
            obs = torch.tensor(bid_observation(bidding, seat, hand), dtype=torch.float32)
            mask = torch.tensor(bid_legal_action_mask_with_hand(bidding, seat, hand), dtype=torch.bool)
            action_id = bidder.act_greedy(obs, mask)
        bidding.apply(seat, action_id)
        action_log.append((seat, action_id))
    return bidding, action_log


def declarer_role_and_seats(contract):
    """Contrat REEL verrouille (engine/bidding.py:Bidding.locked_contract)
    -> (role specialiste, sieges declarants) -- jamais synthetise a la
    main, toujours lu depuis les champs reels du contrat. Reutilise
    DECLARER_ROLE_BY_KIND (engine/misere.py) pour la misere -- LA MEME
    table que le moteur de jeu lui-meme, jamais dupliquee."""
    contract_type = contract["type"]
    if contract_type == "solo":
        return "solitaire", {contract["declarer"]}
    if contract_type == "emballage":
        return "emballage", {contract["declarer"], contract["partner"]}
    if contract_type == "trou":
        return "trou", {contract["declarer"], contract["partner"]}
    if contract_type == "misere":
        return DECLARER_ROLE_BY_KIND[contract["kind"]], set(contract["declarers"])
    raise AssertionError("type de contrat inattendu: %r" % contract_type)


def dispatch_play_phase2(contract, hands, dealer_index, bid_summary=None, whisted=False):
    game = _dispatch_play_phase2(contract, hands, dealer_index)
    game.set_public_context(dealer=dealer_index, bid_summary=bid_summary, whisted=whisted, level_known=True)
    return game


def _dispatch_play_phase2(contract, hands, dealer_index):
    """Construit le moteur de jeu reel depuis le contrat verrouille --
    voir declarer_role_and_seats() pour le mapping role/sieges. `trou` est
    nouveau en Phase 2 (jamais present dans la branche Phase 1 forcee)."""
    leader = first_leader(dealer_index, contract)
    contract_type = contract["type"]
    if contract_type == "solo":
        return new_solitaire_deal(hands, contract["declarer"], contract["suit"], contract["level"], leader)
    if contract_type == "emballage":
        return new_emballage_deal(
            hands, contract["declarer"], contract["partner"], contract["suit"], contract["level"], leader)
    if contract_type == "trou":
        return new_trou_deal(
            hands, contract["declarer"], contract["partner"], contract["anchor_card"], leader,
            royal=contract.get("royal", False), contract=contract)
    if contract_type == "misere":
        return new_misere_deal(hands, contract["declarers"], contract["kind"], leader)
    raise AssertionError("type de contrat inattendu: %r" % contract_type)


def defense_points_reward_real(role, game, contract):
    """Comme defense_points_reward() (Phase 1), mais avec le NIVEAU REEL
    verrouille par l'enchere -- jamais le placeholder de build_game() --
    pour les 3 familles dont le score depend du niveau (solitaire,
    emballage, trou). La misere n'a pas de niveau (cible binaire), inchange."""
    if role == "solitaire":
        return solitaire_final_scores(game, contract["level"])
    if role == "emballage":
        return emballage_final_scores(game)  # game.level deja le niveau reel
    if role == "trou":
        return trou_final_scores(game)  # idem, game.level deja le niveau reel
    return misere_final_scores(game, contract["kind"])


def real_points_by_seat(role, game, contract):
    """Vrai score de la donne pour les 4 sieges (engine/*.final_scores,
    verifies contre Godot) -- la fitness de l'enchere. Les final_scores
    renvoient deja la valeur de CHAQUE siege, declarant(s) compris."""
    return [float(x) for x in defense_points_reward_real(role, game, contract)]


def compute_scores_real(role, game, declarer_seats, contract):
    """VRAIS POINTS du contrat reel pour les 4 sieges, declarants compris
    (depuis le 2026-10-02, demande utilisateur : "verifie que les bons
    comportements sont chaque fois recompenses et les mauvais punis").
    Avant, le declarant recevait ses plis bruts : remplir son contrat ne
    valait pas plus qu'un pli de plus (emballage-9 a 8 plis = 8, a 9 = 9,
    alors que le vrai score passe de -7 a +5), une misere deja chutee
    restait punie a chaque pli, un Trou a 12 plis valait plus qu'a 9 (meme
    score). Le declarant d'un contrat a trumps (solo/emballage/trou) garde un
    petit bonus DECLARER_TRICK_TIEBREAK par pli d'equipe, negligeable devant
    le seuil, pour ne pas jouer au hasard une fois le contrat fait ou perdu."""
    scores = real_points_by_seat(role, game, contract)
    if role in ("solitaire", "emballage", "trou"):
        team_tricks = sum(game.tricks_won[s] for s in declarer_seats)
        for s in declarer_seats:
            scores[s] += DECLARER_TRICK_TIEBREAK * team_tricks
    return scores


DECLARER_TRICK_TIEBREAK = 0.25


def describe_contract(contract):
    """Cle courte pour les stats de contrats verrouilles par lot (voir
    bid_outcome_counts dans main()) -- "ce que choisit l'IA encherisseur",
    demande explicite de l'utilisateur."""
    contract_type = contract["type"]
    if contract_type == "solo":
        return "solo-%d" % contract["level"]
    if contract_type == "emballage":
        return "emballage-%d" % contract["level"]
    if contract_type == "trou":
        return "trou"
    if contract_type == "misere":
        return "misere-%s" % contract["kind"]
    raise AssertionError("type de contrat inattendu: %r" % contract_type)


def strongest_lone_seat(hands):
    """-> (plis surs estimes, siege) du meilleur solo de la donne."""
    return max((expert_bot.estimate_tricks(hands[s], t), s) for s in range(4) for t in range(4))


def rich_deal(chain, rng, hands, dealer, whisted):
    """Redonne (paquets ramasses NON joues) jusqu'a ce qu'un siege ait au moins
    RICH_MIN_TRICKS plis surs -- voir RICH_PROB_DEFAULT."""
    for _ in range(RICH_REDEALS):
        if strongest_lone_seat(hands)[0] >= RICH_MIN_TRICKS:
            break
        chain.finish_unplayed(hands, dealer)
        hands, dealer, whisted = chain.deal(rng)
    return hands, dealer, whisted


def run_phase2_episode(nets, population, rng, buffers, role_returns, bid_action_counts, bid_outcome_counts, stats2=None,
                       chain=None, rollout=None, teacher=None, bid_ppo=None, visitor_prob=0.0, visitor_counts=None,
                       rich_prob=0.0, rich_counts=None):
    """`bid_ppo` (depuis 2026-09-28, agents/bid_ppo.py) : dict {net, buffer, rng}
    -> l'enchere est celle de CE reseau (echantillonnee, decisions enregistrees
    pour PPO), contre lui-meme ou un camp ExpertBot ; `population` n'est alors
    pas utilisee pour encherir.

    Une donne complete : vraie enchere (4 individus DISTINCTS de la
    population, sieges melanges) -> jeu reel -> score reel. Alimente a la
    fois les buffers PPO du jeu de carte (comme run_episode(), Phase 1) ET
    la fitness des 4 individus d'enchere impliques (agents/evolution.py,
    jamais un buffer PPO) -- ce sont deux mecanismes d'apprentissage
    completement independants partageant la MEME donne reelle."""
    if chain is None:
        hands = deal(rng)
        dealer = rng.randrange(4)
        whisted = False
    else:
        hands, dealer, whisted = chain.deal(rng)
    bid_steps = None
    visitors = None
    rich_seat = None
    if chain is not None and bid_ppo is not None and rng.random() < rich_prob:
        hands, dealer, whisted = rich_deal(chain, rng, hands, dealer, whisted)
        est, seat = strongest_lone_seat(hands)
        if est >= RICH_MIN_TRICKS:
            rich_seat = seat
            stats2 = None  # donnes riches hors stats (suivi comparable)
    if rich_seat is None and bid_ppo is not None and rng.random() < visitor_prob:
        style = pick_visitor_style(rng)
        visitors = {s: style for s in ((0, 2) if rng.random() < 0.5 else (1, 3))}
        visitor_counts[style] = visitor_counts.get(style, 0) + 1
    if bid_ppo is not None:
        bidding, action_log, bid_steps = run_bidding_phase_ppo(bid_ppo, hands, dealer, visitors, rich_seat)
        if visitors is not None:
            bid_steps = None  # l'enchere n'apprend pas des tables visiteuses (contrats faciles contre un mauvais jeu)
    else:
        chosen = rng.sample(population, 4)  # 4 individus DISTINCTS, jamais le meme sur 2 sieges
        seats = list(range(4))
        rng.shuffle(seats)
        individual_for_seat = {seat: chosen[i] for i, seat in enumerate(seats)}
        bidder_for_seat = {seat: evolution.policy_of(individual_for_seat[seat]) for seat in range(4)}
        bidding, action_log = run_bidding_phase(bidder_for_seat, hands, dealer)

    if visitors is not None:
        stats2 = None  # tables visiteuses hors stats (suivi comparable a avant le 2026-10-02)
    for seat, action_id in action_log:
        action_name = bid_action_to_dict(action_id)["action"]
        if visitors is None and rich_seat is None:
            bid_action_counts[action_name] = bid_action_counts.get(action_name, 0) + 1
        if stats2 is not None:
            stats2["bid_actions"][action_name] = stats2["bid_actions"].get(action_name, 0) + 1
    if stats2 is not None:
        stats2["donnes"] += 1

    final_points = [0.0] * 4
    if bidding.general_pass:
        if rich_seat is not None and rich_counts is not None:
            rich_counts["donnes"] = rich_counts.get("donnes", 0) + 1
        bid_outcome_counts["general_pass"] = bid_outcome_counts.get("general_pass", 0) + 1
        if stats2 is not None:
            stats2["general_pass"] += 1
        if chain is not None:
            chain.finish_unplayed(hands, dealer)
    else:
        contract = bidding.locked_contract
        role, declarer_seats = declarer_role_and_seats(contract)
        game = dispatch_play_phase2(contract, hands, dealer, bidding.bid_summary, whisted)
        score_fn = lambda g: compute_scores_real(role, g, declarer_seats, contract)
        if rollout is not None:
            rollout["score_fn"] = score_fn
            rollout["trick_unit"] = [POINTS_PER_TRICK] * 4
        steps = play_out(game, nets, rollout, teacher, visitors, rng)
        if chain is not None:
            chain.finish_played(game, dealer, contract["type"])
        record(buffers, role_returns, steps, game, score_fn)
        # (plus de fitness d'enchere ici : voir duel_expert.evaluate_bidders)
        final_points = real_points_by_seat(role, game, contract)
        outcome_key = describe_contract(contract)
        if visitors is None and rich_seat is None:
            bid_outcome_counts[outcome_key] = bid_outcome_counts.get(outcome_key, 0) + 1
        if rich_seat is not None and rich_counts is not None:
            rich_counts["donnes"] = rich_counts.get("donnes", 0) + 1
            if contract["type"] == "solo" and contract["declarer"] == rich_seat and contract["level"] >= 9:
                rich_counts["abondance"] = rich_counts.get("abondance", 0) + 1
                rich_counts["abondance_reussie"] = (rich_counts.get("abondance_reussie", 0)
                                                    + (game.tricks_won[rich_seat] >= contract["level"]))
        if stats2 is not None:
            stats2["contracts"][outcome_key] = stats2["contracts"].get(outcome_key, 0) + 1
            update_phase2_stats(stats2, role, game, declarer_seats, contract)

    if bid_steps is not None:
        for seat, steps in bid_steps.items():
            bid_ppo_mod.record_seat(bid_ppo["buffer"], steps, final_points[seat])
    return bidding.general_pass


def run_bidding_phase_ppo(bid_ppo, hands, dealer_index, visitors=None, rich_seat=None):
    """Comme run_bidding_phase, mais les sieges IA encherissent avec le reseau
    PPO ECHANTILLONNE (exploration) et chaque decision est gardee pour PPO ;
    avec probabilite EXPERT_OPP_PROB un camp (0/2 ou 1/3) est l'ExpertBot.
    Sinon (table 100 % IA), avec probabilite bid_ppo["greedy_prob"], les
    4 sieges jouent GREEDY (voir bid_ppo.GREEDY_TABLE_PROB).
    -> (bidding, action_log, {siege IA: [(obs, mask, action, logp, valeur, cible, poids)]}),
    (cible, poids) = bid_ppo.emballer_guard_target (garde d'emballage haut + ancre)."""
    rng = bid_ppo["rng"]
    net = bid_ppo["net"]
    anchor = None if rich_seat is not None else bid_ppo.get("anchor")  # donne riche : pas d'ancre
    expert_seats = set()
    greedy = False
    if visitors:
        # camp visiteur : style "expert" = ExpertBot, sinon notre reseau avec une part
        # d'encheres au hasard (50 % pour "hasard", 20 % sinon) ; jamais appris
        style = next(iter(visitors.values()))
        if style == "expert":
            expert_seats = set(visitors)
        eps = 0.5 if style == "hasard" else 0.2
    elif rng.random() < bid_ppo_mod.EXPERT_OPP_PROB:
        expert_seats = {0, 2} if rng.random() < 0.5 else {1, 3}
        if rich_seat in expert_seats:  # donne riche : le siege fort reste a l'IA
            expert_seats = {0, 1, 2, 3} - expert_seats
    else:
        greedy = rng.random() < bid_ppo.get("greedy_prob", 0.0)
    bidding = Bidding(dealer_index)
    trou_info = detect_trou(hands)
    if trou_info is not None:
        bidding.seed_trou(trou_info)
    action_log = []
    steps = {s: [] for s in range(4) if s not in expert_seats}
    while not bidding.finished:
        seat = bidding.turn
        hand = hands[seat]
        if seat in expert_seats:
            choice = expert_bid_capped(bidding, seat, hand)
            ok = bidding.apply_raise_to(seat, choice[1]) if isinstance(choice, tuple) else bidding.apply(seat, choice)
            if not ok:
                raise AssertionError("ExpertBot : enchere illegale %r" % (choice,))
            action_id = choice if not isinstance(choice, tuple) else ACTION_RAISE
        elif visitors and seat in visitors:
            mask = bid_legal_action_mask_with_hand(bidding, seat, hand)
            if rng.random() < eps:
                action_id = rng.choice([a for a, ok in enumerate(mask) if ok])
            else:
                obs = torch.tensor(bid_observation(bidding, seat, hand), dtype=torch.float32)
                with torch.no_grad():
                    action_id, _logp, _value = net.act(obs, torch.tensor(mask, dtype=torch.bool))
            if not bidding.apply(seat, action_id):
                raise AssertionError("enchere visiteuse illegale %r" % action_id)
        elif abondance_rule(bidding, seat, hand) is not None:
            # regle d'abondance de l'IA exportee (agents/abondance_rule.py) : meme
            # comportement que dans le jeu, decision imposee donc jamais apprise
            action_id = abondance_rule(bidding, seat, hand)
            if not bidding.apply(seat, action_id):
                raise AssertionError("regle d'abondance illegale %r" % action_id)
        else:
            obs = torch.tensor(bid_observation(bidding, seat, hand), dtype=torch.float32)
            # garde-fous (solos, emballeur) de l'IA exportee (agents/abondance_rule.py) : meme masque qu'en jeu
            mask = torch.tensor(apply_bid_guards(bidding, seat, hand, bid_legal_action_mask_with_hand(bidding, seat, hand)),
                                dtype=torch.bool)
            target, weight = bid_ppo_mod.emballer_guard_target(bidding, hand, mask, obs, anchor)  # avant apply
            big = rich_explore_actions(hand, mask) if seat == rich_seat else []
            with torch.no_grad():
                if big and rng.random() < RICH_EXPLORE:  # exploration de l'abondance (donne riche)
                    action_id = rng.choice(big)
                    logits, value = net.forward(obs.unsqueeze(0))
                    dist = torch.distributions.Categorical(logits=logits.squeeze(0).masked_fill(~mask, float("-inf")))
                    logp, value = dist.log_prob(torch.tensor(action_id)), value.squeeze(0)
                elif greedy:
                    action_id, logp, value = net.act_greedy_logp(obs, mask)
                else:
                    action_id, logp, value = net.act(obs, mask)
            if not bidding.apply(seat, action_id):
                raise AssertionError("enchere PPO illegale %r" % action_id)
            steps[seat].append((obs, mask, action_id, logp, value, target, weight))
        action_log.append((seat, action_id))
    return bidding, action_log, steps


def expert_bid_capped(bidding, seat, hand):
    """Decision de l'ExpertBot ADVERSAIRE d'entrainement, relances de son
    emballage plafonnees a EMBALLAGE_RAISE_MAX_LEVEL (sinon passe) : ses
    emballages 11/12 (relances optimistes, 2026-10-10) faisaient l'essentiel des
    emballage-11/12 < 30 % des stats. L'ExpertBot du jeu (whist-4-it) inchange."""
    choice = expert_bot.choose_bid(bidding, seat, hand)
    if isinstance(choice, tuple):
        if choice[1] <= EMBALLAGE_RAISE_MAX_LEVEL:
            return choice
        c = bidding.contracts[bidding.lie_contract_index[seat]]
        if c["type"] == "emballage" and bidding.can_raise_to(seat, EMBALLAGE_RAISE_MAX_LEVEL):
            return (choice[0], EMBALLAGE_RAISE_MAX_LEVEL)
    elif not emballage_raise_too_high(bidding, seat, choice):
        return choice
    return ACTION_PASS if ACTION_PASS in bid_legal_action_ids_with_hand(bidding, seat, hand) else choice


_BIG_LONE_IDS = [a for a in range(evolution.ACTION_DIM)
                 if bid_action_to_dict(a)["action"] in ("abondance", "solo_chelem")]


def rich_explore_actions(hand, mask):
    """Abondances/solo chelem legaux essayes sur une donne riche : dans la
    meilleure couleur de la main, niveau <= plis surs (a plis surs + 1 :
    21-25 % de reussite, voir RICH_EXPLORE ; au hasard, toutes couleurs et
    niveaux confondus, presque tous chutaient : rien a apprendre)."""
    est, suit = max((expert_bot.estimate_tricks(hand, t), t) for t in range(4))
    out = []
    for a in _BIG_LONE_IDS:
        info = bid_action_to_dict(a)
        if mask[a] and info["suit"] == suit and info.get("level", 13) <= est:
            out.append(a)
    return out


def print_generation_summary(generation, population, pass0=None):
    """`pass0` : {id: score de la passe 0} (tout le monde y est note)."""
    by_lineage = {}
    for ind in population:
        by_lineage.setdefault(ind.lineage, []).append(ind)
    print("=== Enchere (Phase 2) -- fin de generation %d, %d individus ===" % (generation, len(population)), flush=True)
    for lineage in evolution.LINEAGES:
        members = by_lineage.get(lineage, [])
        if not members:
            print("  %-20s EXTINCTE (0 survivant -- NE DEVRAIT PLUS ARRIVER depuis que select_elites "
                  "garantit un survivant/lignee, voir DESIGN.md 2026-09-20 -- signaler si vu)" % lineage, flush=True)
            continue
        if pass0 is not None:
            fits = [pass0[ind.id] for ind in members if pass0.get(ind.id) not in (None, float("-inf"))]
        else:
            fits = [ind.mean_fitness() for ind in members if ind.games]
        if fits:
            print("  %-20s effectif=%-3d fitness min=%+8.3f moy=%+8.3f max=%+8.3f" % (
                lineage, len(members), min(fits), sum(fits) / len(fits), max(fits)), flush=True)
        else:
            print("  %-20s effectif=%-3d (aucune partie jouee cette generation)" % (lineage, len(members)), flush=True)


LATEST_AI_BRANCH = "ia-latest"


def push_latest_ai(paths, total_episodes, branch=LATEST_AI_BRANCH, remote="origin"):
    """A l'arret : publie sur GitHub l'IA la plus recente (checkpoints_v2/ =
    reseaux + optimiseurs + population d'enchere) et les stats, en ECRASANT
    la precedente : branche `ia-latest` a UN SEUL commit sans parent, poussee
    en force -> l'ancienne version devient inaccessible et GitHub la purge, le
    depot ne grossit pas (demande utilisateur 2026-09-27). N'utilise qu'un
    index temporaire : ne touche ni master, ni l'index, ni les fichiers.
    Recuperer : git fetch origin ia-latest && git checkout FETCH_HEAD -- checkpoints_v2
    Un echec n'empeche pas l'arret propre."""
    import subprocess
    import tempfile
    paths = [p for p in paths if p and os.path.exists(p)]
    fd, index = tempfile.mkstemp(prefix="ia_latest_index_")
    os.close(fd)
    os.remove(index)  # git cree lui-meme l'index
    env = dict(os.environ, GIT_INDEX_FILE=index)
    msg = "IA la plus recente : checkpoints + stats a l'arret (%d donnes depuis le lancement)" % total_episodes

    def git(*args):
        r = subprocess.run(["git"] + list(args), capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError("git %s : %s" % (args[0], (r.stdout + r.stderr).strip()[-300:]))
        return r.stdout.strip()

    try:
        git("add", "-f", "--", *paths)  # -f : ces fichiers sont ignores sur master
        commit = git("commit-tree", git("write-tree"), "-m", msg)
        git("push", "-f", remote, "%s:refs/heads/%s" % (commit, branch))
        print("  IA la plus recente publiee sur GitHub (branche %s, %s)." % (branch, commit[:7]), flush=True)
    except RuntimeError as exc:
        print("  !! publication GitHub de l'IA en echec : %s" % exc, flush=True)
    finally:
        if os.path.exists(index):
            os.remove(index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=200, help="donnes par mise a jour PPO")
    parser.add_argument("--save-every", type=int, default=5, help="sauvegarder tous les N lots")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None, help="arreter apres N lots (tests)")
    parser.add_argument("--until", default=None,
                        help="arret propre a cette heure locale (AAAA-MM-JJTHH:MM), meme chemin que .stop_training")
    parser.add_argument("--stats-every", type=int, default=STATS_EVERY_DEFAULT,
                         help="imprimer+persister les stats par famille toutes les N donnes")
    parser.add_argument("--stats-file", default=STATS_FILE_DEFAULT,
                         help="fichier JSON Lines ou ajouter chaque emission de stats (Phase 1)")
    parser.add_argument("--stats-file-phase2", default=STATS_FILE_PHASE2_DEFAULT,
                         help="fichier JSON Lines des stats Phase 2 (vraie enchere), separe de la Phase 1")
    parser.add_argument("--eval-every", type=int, default=None,
                         help="jouer les 7 specialistes (greedy) contre RandomBot toutes les N donnes "
                              "(defaut : scripts/eval_vs_random.py:EVAL_EVERY_DEFAULT, 20000 -- ~25 min)")
    parser.add_argument("--eval-games", type=int, default=None,
                         help="contrats (choisis par encherisseur) joues lors de l'evaluation vs RandomBot "
                              "(defaut : scripts/eval_vs_random.py:EVAL_GAMES_DEFAULT)")
    parser.add_argument("--eval-file", default=None,
                         help="fichier JSON Lines ou ajouter chaque evaluation vs RandomBot "
                              "(defaut : scripts/eval_vs_random.py:EVAL_FILE_DEFAULT)")
    parser.add_argument("--no-record-samples", action="store_true",
                         help="ne pas enregistrer/envoyer les parties d'exemple par famille "
                              "(actif par defaut, EN MEME TEMPS que l'evaluation vs RandomBot -- "
                              "meme cadence --eval-every, demande explicite de l'utilisateur)")
    parser.add_argument("--record-samples-dir", default="recordings",
                         help="dossier local ou garder une copie de chaque envoi (1 sous-dossier horodate, "
                              "1 fichier par famille)")
    parser.add_argument("--godot-dir", default=record_game.DEFAULT_GODOT_RECORDINGS_DIR,
                         help="dossier whist-4-it ou envoyer ces enregistrements "
                              "(defaut : %s)" % record_game.DEFAULT_GODOT_RECORDINGS_DIR)
    parser.add_argument("--visitor-prob", type=float, default=VISITOR_PROB_DEFAULT,
                        help="part des donnes ou un camp est tenu par un joueur visiteur (expert/bruite/debutant/hasard)")
    parser.add_argument("--rich-prob", type=float, default=RICH_PROB_DEFAULT,
                        help="part des donnes Phase 2 redonnees jusqu'a une main d'abondance (voir RICH_PROB_DEFAULT)")
    parser.add_argument("--phase2-prob", type=float, default=PHASE2_PROB_DEFAULT,
                         help="probabilite qu'une donne utilise la vraie enchere (Phase 2) "
                              "plutot que le contrat force (Phase 1)")
    parser.add_argument("--whisted-ratio", type=float, default=WHISTED_RATIO_DEFAULT,
                         help="probabilite qu'une donne soit un deck whiste (paquet ramasse de la donne "
                              "precedente, coupe une fois) plutot qu'un melange uniforme")
    parser.add_argument("--bidder-pop-size", type=int, default=evolution.POP_SIZE_DEFAULT,
                         help="taille de la population evolutive d'enchere (2 lignees)")
    parser.add_argument("--bidder-elite-count", type=int, default=evolution.ELITE_COUNT_DEFAULT,
                         help="nombre d'elites conserves a chaque generation")
    parser.add_argument("--bidder-mutation-sigma", type=float, default=evolution.MUTATION_SIGMA_DEFAULT,
                         help="ecart-type du bruit gaussien de mutation")
    parser.add_argument("--bidder-games-per-individual", type=int, default=BID_GAMES_PER_INDIVIDUAL_DEFAULT,
                         help="cadence des generations : une generation toutes les pop*N/4 donnes Phase 2")
    parser.add_argument("--bidder-eval-deals", type=int, default=BIDDER_EVAL_DEALS_DEFAULT,
                         help="donnes communes (x2, duplicate vs ExpertBot) de la fitness de chaque individu")
    parser.add_argument("--population-file", default=POPULATION_FILE_DEFAULT,
                         help="fichier ou sauvegarder/recharger la population d'enchere")
    parser.add_argument("--export-every", type=int, default=EXPORT_EVERY_DEFAULT,
                         help="regenerer l'export Godot (godot_export/) toutes les N donnes (0 pour desactiver)")
    parser.add_argument("--export-dir", default=EXPORT_DIR_DEFAULT,
                         help="dossier local ou ecrire l'export automatique avant verification")
    parser.add_argument("--godot-project", default=None,
                         help="projet whist-4-it ou publier poids + stats (defaut : detection automatique, "
                              "voir scripts/godot_sync.py)")
    parser.add_argument("--phase3-deals", type=int, default=None,
                         help="Phase 3 (parties reelles de l'IA exportee, sans contrainte, voir scripts/phase3.py) : "
                              "donnes par table jouees avant chaque export Godot (defaut : "
                              "scripts/phase3.py:PHASE3_DEALS_DEFAULT ; 0 pour desactiver)")
    parser.add_argument("--stats-file-phase3", default=None,
                         help="fichier JSON Lines des stats Phase 3 (defaut : training_stats_phase3.jsonl)")
    parser.add_argument("--no-git-push", action="store_true",
                        help="ne pas commiter/pousser l'IA finale sur GitHub a l'arret (voir push_latest_ai)")
    parser.add_argument("--bidder-evolution", action="store_true",
                        help="ancienne enchere par population evolutive (avant 2026-09-28) au lieu de PPO "
                             "(agents/bid_ppo.py)")
    parser.add_argument("--eval-stats-file", default=godot_sync.EVAL_STATS_FILE_DEFAULT,
                        help="historique des evaluations publie vers whist-4-it (une ligne par export) ; "
                             "les runs de test doivent pointer ailleurs")
    parser.add_argument("--bid-greedy-prob", type=float, default=None,
                        help="part des tables d'enchere 100 %% IA jouees en greedy (defaut : "
                             "agents/bid_ppo.py:GREEDY_TABLE_PROB)")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="dossier des checkpoints (defaut checkpoints_v2) -- pour tester sur une COPIE ; "
                             "change aussi --population-file par defaut")
    parser.add_argument("--no-godot-sync", action="store_true",
                         help="ne pas publier dans whist-4-it (export local godot_export/ seulement)")
    parser.add_argument("--rollout-per-deal", type=float, default=ROLLOUT_PER_DEAL_DEFAULT,
                         help="decisions analysees par rollouts (valeur de chaque carte legale) par donne, "
                              "en moyenne (0 pour desactiver, voir agents/rollouts.py)")
    parser.add_argument("--rollout-samples", type=int, default=ROLLOUT_SAMPLES_DEFAULT,
                         help="mondes (mains cachees tirees au hasard) par decision analysee")
    parser.add_argument("--rollout-surcoupe-samples", type=int, default=ROLLOUT_SURCOUPE_SAMPLES_DEFAULT,
                         help="mondes par occasion de surcoupe analysee (hors misere) ; 0 pour desactiver")
    parser.add_argument("--rollout-coupe-samples", type=int, default=ROLLOUT_COUPE_SAMPLES_DEFAULT,
                        help="mondes par renonce non coupee possible (coequipier entame dans notre renonce, "
                             "un atout prendrait) ; 0 pour desactiver")
    parser.add_argument("--rollout-atout-samples", type=int, default=ROLLOUT_ATOUT_SAMPLES_DEFAULT,
                        help="mondes par entame du camp declarant quand un adversaire a encore de l'atout ; "
                             "0 pour desactiver")
    parser.add_argument("--rollout-policy", choices=("nets", "expert", "dds"), default=ROLLOUT_POLICY_DEFAULT,
                        help="qui joue la suite de la donne dans les rollouts (voir ROLLOUT_POLICY_DEFAULT)")
    parser.add_argument("--rollout-workers", type=int, default=0,
                        help="processus qui jouent les rollouts --rollout-policy expert en parallele "
                             "(0 ou 1 = en serie dans le processus principal)")
    parser.add_argument("--rollout-margin", type=float, default=ROLLOUT_MARGIN_DEFAULT,
                         help="ecart de valeur (plis ; points/10 pour defenseur) sous lequel une carte compte "
                              "parmi les meilleures")
    parser.add_argument("--rollout-margin-tricks", type=float, default=ROLLOUT_MARGIN_TRICKS_DEFAULT,
                        help="marge EN PLIS (> 0 remplace --rollout-margin), voir ROLLOUT_MARGIN_TRICKS_DEFAULT")
    parser.add_argument("--oracle-critic", action="store_true",
                        help="evaluateur omniscient (4 mains visibles) pour les avantages PPO, voir agents/oracle_critic.py")
    parser.add_argument("--oracle-warm-lots", type=int, default=ORACLE_WARM_LOTS_DEFAULT,
                        help="lots d'apprentissage de l'evaluateur omniscient avant de l'utiliser dans GAE")
    parser.add_argument("--distill-coef", type=float, default=ppo.DISTILL_COEF,
                         help="poids de la perte vers la cible des rollouts, a cote de PPO")
    parser.add_argument("--teacher-coef", type=float, default=TEACHER_COEF_DEFAULT,
                        help="poids initial de l'imitation de l'ExpertBot (0 pour desactiver), voir TEACHER_*")
    parser.add_argument("--misere-teacher-coef", type=float, default=MISERE_TEACHER_COEF_DEFAULT,
                        help="poids initial de l'imitation de l'ExpertBot reservee aux declarants de misere "
                             "(0 pour desactiver), voir MISERE_TEACHER_*")
    parser.add_argument("--misere-teacher-decay-deals", type=int, default=MISERE_TEACHER_DECAY_DEALS_DEFAULT)
    parser.add_argument("--misere-teacher-floor", type=float, default=MISERE_TEACHER_FLOOR_DEFAULT,
                        help="poids minimal permanent de l'imitation reservee (0 = decroissance jusqu'a 0)")
    parser.add_argument("--phase1-misere-weight", type=float, default=PHASE1_MISERE_WEIGHT_DEFAULT,
                        help="poids de tirage de chaque famille misere en Phase 1 (1 = uniforme)")
    parser.add_argument("--teacher-decay-deals", type=int, default=TEACHER_DECAY_DEALS_DEFAULT,
                        help="donnes (compteur persistant) sur lesquelles ce poids descend lineairement a 0")
    parser.add_argument("--duel-deals", type=int, default=None,
                        help="donnes (x2, duplicate) du duel IA vs ExpertBot en vrais points a chaque "
                             "evaluation (scripts/duel_expert.py ; 0 pour desactiver)")
    parser.add_argument("--champion-deals", type=int, default=None,
                        help="donnes (x2, duplicate) de la porte IA vs championne du jeu de la carte en vrais "
                             "points a chaque evaluation (scripts/champion_play.py ; 0 pour desactiver)")
    parser.add_argument("--expert-match-contracts", type=int, default=None,
                        help="contrats du duplicate vs ExpertBot a chaque evaluation (0 pour desactiver)")
    parser.add_argument("--team-errors-deals", type=int, default=None,
                         help="donnes par jeu (reel/force) pour la mesure des surcoupes a chaque evaluation "
                              "(defaut : scripts/team_errors.py:TEAM_ERRORS_DEALS_DEFAULT ; 0 pour desactiver)")
    parser.add_argument("--team-errors-file", default=None,
                         help="fichier JSON Lines de la mesure des surcoupes (defaut : team_errors.jsonl)")
    args = parser.parse_args()
    if args.bid_greedy_prob is None:
        args.bid_greedy_prob = bid_ppo_mod.GREEDY_TABLE_PROB

    if (args.teacher_coef > 0 and args.rollout_policy == "nets"
            and (args.rollout_per_deal > 0 or args.rollout_surcoupe_samples > 0)):
        raise SystemExit("imitation de l'ExpertBot et rollouts partagent la meme perte de cible : "
                         "choisir l'un ou l'autre (--teacher-coef 0 pour les rollouts)")
    if args.bidder_pop_size < 4:
        raise SystemExit("--bidder-pop-size doit etre >= 4 (4 sieges = 4 individus distincts par donne)")

    # Import differe (pas au niveau module) : scripts/eval_vs_random.py importe
    # lui-meme DECLARER_FAMILIES/DEFENSEUR/MISERE_FAMILIES/MISERE_TARGET_TRICKS
    # depuis scripts.train -- un import au niveau module creerait un cycle
    # (scripts.train pas encore initialise au moment ou eval_vs_random essaierait
    # de le relire). A ce point de main(), le module est deja pleinement charge.
    from scripts import champion_play, duel_expert, eval_vs_random, expert_bot_match, phase3, team_errors
    champion_deals = champion_play.CHAMPION_DEALS_DEFAULT if args.champion_deals is None else args.champion_deals
    duel_deals = duel_expert.DUEL_DEALS_DEFAULT if args.duel_deals is None else args.duel_deals
    expert_match_contracts = (expert_bot_match.EXPERT_MATCH_CONTRACTS_DEFAULT
                              if args.expert_match_contracts is None else args.expert_match_contracts)
    team_errors_deals = (team_errors.TEAM_ERRORS_DEALS_DEFAULT if args.team_errors_deals is None
                         else args.team_errors_deals)
    team_errors_file = args.team_errors_file or team_errors.TEAM_ERRORS_FILE_DEFAULT
    phase3_deals = phase3.PHASE3_DEALS_DEFAULT if args.phase3_deals is None else args.phase3_deals
    stats_file_phase3 = args.stats_file_phase3 or phase3.STATS_FILE_PHASE3_DEFAULT
    eval_every = args.eval_every or eval_vs_random.EVAL_EVERY_DEFAULT
    eval_games = args.eval_games or eval_vs_random.EVAL_GAMES_DEFAULT
    eval_file = args.eval_file or eval_vs_random.EVAL_FILE_DEFAULT

    if os.path.exists(STOP_FLAG_PATH):
        os.remove(STOP_FLAG_PATH)
        print("Fichier %r residuel d'un arret precedent supprime au demarrage." % STOP_FLAG_PATH, flush=True)

    if args.checkpoint_dir:
        if args.population_file == POPULATION_FILE_DEFAULT:
            args.population_file = os.path.join(args.checkpoint_dir, "encherisseur_population.pt")
        specialists.CHECKPOINT_DIR = args.checkpoint_dir
    rng = random.Random(args.seed)
    nets = {}
    for role in PLAY_ROLES:
        net, optimizer = specialists.load_or_create(role)
        nets[role] = net
        nets[role + "_optimizer"] = optimizer
    global ORACLE
    if args.oracle_critic:
        ORACLE = {"critics": {role: oracle_critic.load_or_create(specialists.CHECKPOINT_DIR, role) for role in PLAY_ROLES},
                  "lots": oracle_critic.load_state(specialists.CHECKPOINT_DIR), "warm": args.oracle_warm_lots,
                  "dir": specialists.CHECKPOINT_DIR}
        print("Evaluateur omniscient (4 mains visibles, entrainement seul) : utilise dans GAE apres %d lots "
              "par role ; deja appris : %s." % (args.oracle_warm_lots, ORACLE["lots"] or "rien"), flush=True)

    if os.path.exists(args.population_file):
        population, generation = evolution.load_population(args.population_file)
        repaired = evolution.repair_missing_lineages(population)
        if repaired:
            print("!! lignee(s) manquante(s) reinjectee(s) au rechargement (fichier anterieur a la "
                  "correction du 2026-09-20 -- voir DESIGN.md) : %s" % ", ".join(repaired), flush=True)
        print("Population d'enchere rechargee (%s) : generation %d, %d individus." % (
            args.population_file, generation, len(population)), flush=True)
    else:
        seeds = {lineage: evolution.new_net() for lineage in evolution.LINEAGES}
        population = evolution.seed_population(seeds, args.bidder_pop_size, args.bidder_mutation_sigma)
        generation = 0
        print("Nouvelle population d'enchere creee (generation 0, %d individus, "
              "2 lignees neuves : %s)." % (len(population), ", ".join(evolution.LINEAGES)), flush=True)
    games_per_generation = max(4, (args.bidder_pop_size * args.bidder_games_per_individual) // 4)
    phase2_games_this_generation = 0

    bid_ppo = None
    if not args.bidder_evolution:
        bid_net, bid_opt, bid_id, bid_seed = bid_ppo_mod.load_or_seed(population, specialists.CHECKPOINT_DIR)
        backup = bid_ppo_mod.backup_population_once(args.population_file, generation)
        if backup:
            print("Population evolutive sauvegardee avant le passage a PPO : %s" % backup, flush=True)
        population = bid_ppo_mod.ppo_population(population, bid_net, bid_id)
        bid_ppo = {"net": bid_net, "optimizer": bid_opt, "id": bid_id, "seeded_from": bid_seed,
                   "rng": random.Random(rng.random()), "buffer": bid_ppo_mod.new_buffer(),
                   "greedy_prob": args.bid_greedy_prob,
                   "anchor": bid_ppo_mod.load_anchor(specialists.CHECKPOINT_DIR)}
        print("Enchere entrainee par PPO (agents/bid_ppo.py) : reseau #%d (parti de l'individu #%s), "
              "%.0f%% des donnes contre un camp ExpertBot, retour = vrais points du siege ; "
              "%.0f%% des tables 100 %% IA en greedy." % (
                  bid_id, bid_seed, 100 * bid_ppo_mod.EXPERT_OPP_PROB, 100 * args.bid_greedy_prob), flush=True)

    def _save_population():
        evolution.save_population(args.population_file, population, generation)
        if bid_ppo is not None:
            bid_ppo_mod.save(specialists.CHECKPOINT_DIR, bid_ppo["net"], bid_ppo["optimizer"], bid_ppo["id"],
                             bid_ppo["seeded_from"])

    def _export_godot():
        """Phase 3 (mesure de l'IA exportee), puis export (+ verification et
        publication dans whist-4-it sauf --no-godot-sync). Relit
        checkpoints_v2/ : sauvegarder avant."""
        if phase3_deals:
            phase3.run_phase3(nets, population, rng, phase3_deals, total_episodes,
                              time.time() - start_time, stats_file_phase3, generation)
        godot_sync.append_eval_stats(args.eval_stats_file, duel_expert.DUEL_FILE_DEFAULT,
                                     expert_bot_match.EXPERT_MATCH_FILE_DEFAULT, eval_file)
        if args.no_godot_sync:
            for role in PLAY_ROLES:
                export_to_godot.export_role(role, args.export_dir)
            export_to_godot.export_bidder_roles(args.population_file, args.export_dir)
            return
        godot_sync.publish_to_godot(args.export_dir, args.stats_file, args.stats_file_phase2,
                                    args.godot_project, args.population_file, stats_file_phase3,
                                    eval_stats_file=args.eval_stats_file)

    stop = {"flag": False}

    def _handle_stop(signum, frame):
        print("\nSignal recu, sauvegarde et arret propre...", flush=True)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_stop)

    deadline = datetime.datetime.fromisoformat(args.until) if args.until else None

    def _stop_requested():
        if stop["flag"]:
            return True
        if deadline is not None and datetime.datetime.now() >= deadline:
            print("\nheure d'arret atteinte (%s), sauvegarde et arret propre..." % args.until, flush=True)
            stop["flag"] = True
            return True
        if os.path.exists(STOP_FLAG_PATH):
            print("\n%r detecte, sauvegarde et arret propre..." % STOP_FLAG_PATH, flush=True)
            stop["flag"] = True
            return True
        return False

    batch_num = 0
    total_episodes = 0
    stats_marker = 0
    eval_marker = 0
    export_marker = 0
    stats = new_stats_accumulator()
    stats2 = new_phase2_stats_accumulator()
    chain = DeckChain(args.whisted_ratio, rng)
    start_time = time.time()
    print("Entrainement Phase 1 + Phase 2 demarre (phase2-prob=%.2f, duree indeterminee). "
          "Ctrl+C, ou creer %r, pour arreter proprement." % (args.phase2_prob, STOP_FLAG_PATH), flush=True)
    print("Donnes : %.0f%% deck whiste (paquet ramasse de la donne precedente + 1 coupe), le reste melange uniforme."
          % (100 * args.whisted_ratio), flush=True)
    if bid_ppo is None:
        print("Enchere (Phase 2) : population=%d, elite=%d, sigma=%.3f, generation actuelle=%d "
              "(%d donnes Phase 2/generation)." % (
            args.bidder_pop_size, args.bidder_elite_count, args.bidder_mutation_sigma,
            generation, games_per_generation), flush=True)
    print("Evaluation vs RandomBot toutes les %d donnes (%d parties/config, -> %s)." % (
        eval_every, eval_games, eval_file), flush=True)
    if args.export_every:
        print("Export Godot automatique toutes les %d donnes + a l'arret -> %s/%s." % (
            args.export_every, args.export_dir,
            "" if args.no_godot_sync else ", verifie puis publie dans whist-4-it/game/ (poids + stats)"), flush=True)
    if team_errors_deals:
        print("  + a chaque evaluation : mesure des surcoupes (%d donnes reelles + %d forcees) -> %s." % (
            team_errors_deals, team_errors_deals, team_errors_file), flush=True)
    rollout = None
    if (args.rollout_per_deal > 0 or args.rollout_surcoupe_samples > 0 or args.rollout_coupe_samples > 0
            or args.rollout_atout_samples > 0):
        rollout = {"coupe_samples": args.rollout_coupe_samples, "coupes": 0,
                   "atout_samples": args.rollout_atout_samples, "atouts": 0, "prob": args.rollout_per_deal / DECISIONS_PER_DEAL, "samples": args.rollout_samples,
                   "surcoupe_samples": args.rollout_surcoupe_samples,
                   "margin": args.rollout_margin, "margin_tricks": args.rollout_margin_tricks, "rng": random.Random(rng.random()),
                   "score_fn": None, "policy": args.rollout_policy, "count": 0, "corrective": 0, "surcoupes": 0, "seconds": 0.0}
        print("Rollouts : chaque occasion de surcoupe hors misere (%d mondes) + ~%.1f decision(s)/donne au "
              "hasard (%d mondes), marge=%.2f, poids %.2f a cote de PPO." % (
                  args.rollout_surcoupe_samples, args.rollout_per_deal, args.rollout_samples,
                  args.rollout_margin, args.distill_coef), flush=True)
        if args.rollout_policy in ("expert", "dds") and args.rollout_workers > 1:
            import multiprocessing
            rollouts.set_pool(multiprocessing.Pool(args.rollout_workers), args.rollout_workers)
            print("  rollouts (%s) repartis sur %d processus." % (args.rollout_policy, args.rollout_workers), flush=True)
    teacher = None
    teacher_deals = load_teacher_deals()
    if teacher_coef_at(teacher_deals, args.teacher_coef, args.teacher_decay_deals) > 0:
        teacher = {"count": 0, "agree": 0}
        print("Imitation de l'ExpertBot : poids %.3f -> 0 sur %d donnes (deja %d faites, %s), "
              "tous contrats (miseres et ecart compris)." % (
                  teacher_coef_at(teacher_deals, args.teacher_coef, args.teacher_decay_deals),
                  args.teacher_decay_deals, teacher_deals, teacher_state_path()), flush=True)
    elif args.teacher_coef > 0:
        print("Imitation de l'ExpertBot terminee (%d donnes >= %d) : PPO seul." % (
            teacher_deals, args.teacher_decay_deals), flush=True)
    # imitation reservee aux declarants de misere : seulement quand l'imitation
    # generale est finie (sinon elle les couvre deja)
    teacher_state_file = TEACHER_STATE_FILE
    teacher_coef, teacher_decay = args.teacher_coef, args.teacher_decay_deals
    teacher_floor = 0.0
    misere_floor = args.misere_teacher_floor if args.misere_teacher_coef > 0 else 0.0
    if teacher is None and max(misere_floor, teacher_coef_at(load_teacher_deals(MISERE_TEACHER_STATE_FILE),
                                                             args.misere_teacher_coef,
                                                             args.misere_teacher_decay_deals)) > 0:
        teacher_state_file = MISERE_TEACHER_STATE_FILE
        teacher_coef, teacher_decay = args.misere_teacher_coef, args.misere_teacher_decay_deals
        teacher_floor = misere_floor
        teacher_deals = load_teacher_deals(teacher_state_file)
        teacher = {"count": 0, "agree": 0, "roles": MISERE_DECLARER_ROLES}
        print("Imitation de l'ExpertBot RESERVEE aux declarants de misere (%s) : poids %.3f -> %.2f (plancher) "
              "sur %d donnes (deja %d faites, %s)." % (
                  ", ".join(sorted(MISERE_DECLARER_ROLES)),
                  max(teacher_floor, teacher_coef_at(teacher_deals, teacher_coef, teacher_decay)), teacher_floor,
                  teacher_decay, teacher_deals, teacher_state_path(teacher_state_file)), flush=True)
    print("Phase 1 : poids de tirage %.1f par famille misere (1 pour solitaire/emballage)." % (
        args.phase1_misere_weight), flush=True)
    if champion_deals:
        champion_play.ensure_champion(nets)
        print("  + a chaque evaluation : porte IA vs championne du jeu de la carte (%d donnes x2) -> %s." % (
            champion_deals, champion_play.CHAMPION_FILE_DEFAULT), flush=True)
    if duel_deals:
        print("  + a chaque evaluation : duel IA vs ExpertBot en vrais points (%d donnes x2) -> %s." % (
            duel_deals, duel_expert.DUEL_FILE_DEFAULT), flush=True)
    if expert_match_contracts:
        print("  + a chaque evaluation : duplicate vs ExpertBot (%d contrats) -> %s." % (
            expert_match_contracts, expert_bot_match.EXPERT_MATCH_FILE_DEFAULT), flush=True)
    if not args.no_record_samples:
        print("  + a chaque evaluation : 1 partie/famille (%d donnes chacune, vs defenseur) -> %s + %s." % (
            record_game.SAMPLE_HANDS_PER_FAMILY, args.record_samples_dir, args.godot_dir), flush=True)

    try:
        while not _stop_requested():
            batch_num += 1
            buffers = {role: {"obs": [], "mask": [], "action": [], "logp": [], "ret": [], "adv": [], "target": [], "oracle_x": []}
                       for role in PLAY_ROLES}
            if rollout is not None:
                rollout["count"], rollout["corrective"], rollout["surcoupes"], rollout["seconds"] = 0, 0, 0, 0.0
                rollout["coupes"], rollout["atouts"] = 0, 0
            if teacher is not None:
                teacher["count"], teacher["agree"] = 0, 0
            batch_start_episodes = total_episodes
            role_returns = {role: [] for role in PLAY_ROLES}
            family_counts = {}
            bid_action_counts = {}
            bid_outcome_counts = {}
            visitor_counts = {}
            rich_counts = {}

            for _ in range(args.batch_size):
                if rng.random() < args.phase2_prob:
                    run_phase2_episode(nets, population, rng, buffers, role_returns, bid_action_counts, bid_outcome_counts, stats2,
                                       chain, rollout, teacher, bid_ppo, args.visitor_prob, visitor_counts,
                                       args.rich_prob, rich_counts)
                    phase2_games_this_generation += 1
                    if bid_ppo is None and phase2_games_this_generation >= games_per_generation:
                        # fitness = duplicate contre l'ExpertBot sur des donnes communes,
                        # tirees d'un rng propre a la generation et a la passe (rng
                        # d'entrainement intact) ; selection en deux passes avec memoire
                        # (agents/evolution.py, "Selection sous bruit", 2026-09-28)
                        played = []
                        pass0 = {}

                        def _evaluate(individuals, passe, _g=generation):
                            seed = (1_000_003 if passe == 0 else 2_000_029) * _g + 17
                            played.append(duel_expert.evaluate_bidders(
                                individuals, nets, random.Random(seed), args.bidder_eval_deals))
                            if passe == 0:
                                pass0.update({i.id: i.mean_fitness() for i in individuals})

                        elites = evolution.select_under_noise(population, _evaluate, args.bidder_elite_count)
                        print("  fitness d'enchere : duplicate vs ExpertBot, %d donnes communes x2, 2 passes "
                              "(%s donnes jouees, le reste en cache)" % (args.bidder_eval_deals, "+".join(map(str, played))), flush=True)
                        print_generation_summary(generation, population, pass0)
                        print("  elite retenue (moyenne memoire / donnes) : %s" % ", ".join(
                            "#%d(%s,%+.2f/%d)" % (e.id, e.lineage, e.life_mean(), e.life_n) for e in elites), flush=True)
                        population = evolution.spawn_generation(elites, args.bidder_pop_size, args.bidder_mutation_sigma)
                        generation += 1
                        phase2_games_this_generation = 0
                        _save_population()
                else:
                    run_episode(nets, rng, buffers, role_returns, family_counts, stats, chain, rollout, teacher,
                                args.phase1_misere_weight, args.visitor_prob, visitor_counts)
                total_episodes += 1
                if _stop_requested():
                    break

            distill_coef = args.distill_coef
            if teacher is not None:
                distill_coef = max(teacher_floor, teacher_coef_at(teacher_deals, teacher_coef, teacher_decay))
                teacher_deals += total_episodes - batch_start_episodes
                safe_write_json({"deals": teacher_deals}, teacher_state_path(teacher_state_file))
            for role in PLAY_ROLES:
                buf = buffers[role]
                if not buf["obs"]:
                    continue
                # imitation reservee (misere/trou) : son poids ne vaut que pour ses roles -- jusqu'au
                # 24/10 il s'appliquait a TOUS, cibles dds des defenseurs/emballeurs a 0.2 au lieu de 1
                role_coef = distill_coef
                if teacher is not None and teacher.get("roles") is not None and role not in teacher["roles"]:
                    role_coef = args.distill_coef
                ppo_stats = ppo.update(
                    nets[role], nets[role + "_optimizer"],
                    buf["obs"], buf["mask"], buf["action"], buf["logp"], buf["ret"], buf["adv"],
                    return_scale=credit.return_scale(role), targets=buf["target"], distill_coef=role_coef,
                )
                avg_return = sum(role_returns[role]) / len(role_returns[role])
                elapsed = time.time() - start_time
                print("[lot %d | %d donnes | %.0fs] %-14s decisions=%-5d ret_moy=%+7.3f loss=%.4f cibles=%-4d "
                      "distill=%.4f" % (
                          batch_num, total_episodes, elapsed, role, len(buf["obs"]), avg_return,
                          ppo_stats["policy_loss"], ppo_stats["targets"], ppo_stats["distill_loss"]),
                      flush=True)
                if ORACLE is not None and len(buf["oracle_x"]) == len(buf["ret"]):
                    critic, critic_opt = ORACLE["critics"][role]
                    oracle_loss = oracle_critic.update(critic, critic_opt, buf["oracle_x"], buf["ret"],
                                                       credit.return_scale(role))
                    ORACLE["lots"][role] = ORACLE["lots"].get(role, 0) + 1
                    # perte de valeur acteur (aveugle) vs omniscient : l'ecart = la chance retiree des avantages
                    print("    %-14s valeur aveugle=%.4f omnisciente=%.4f (%s)" % (
                        role, ppo_stats["value_loss"], oracle_loss,
                        "dans GAE" if ORACLE["lots"][role] >= ORACLE["warm"] else
                        "chauffe %d/%d" % (ORACLE["lots"][role], ORACLE["warm"])), flush=True)

            if bid_ppo is not None and bid_ppo["buffer"]["obs"]:
                buf = bid_ppo["buffer"]
                bid_stats = ppo.update(bid_ppo["net"], bid_ppo["optimizer"], buf["obs"], buf["mask"], buf["action"],
                                       buf["logp"], buf["ret"], buf["adv"], return_scale=bid_ppo_mod.RETURN_SCALE,
                                       targets=buf["target"], distill_coef=bid_ppo_mod.EMBALLER_GUARD_COEF,
                                       target_weights=buf["weight"])
                print("[lot %d | %d donnes | %.0fs] %-14s decisions=%-5d ret_moy=%+7.3f loss=%.4f entropie=%.3f "
                      "garde_emballage=%d ancre=%d distill=%.4f" % (
                          batch_num, total_episodes, time.time() - start_time, "encherisseur", len(buf["obs"]),
                          sum(buf["ret"]) / len(buf["ret"]), bid_stats["policy_loss"], bid_stats["entropy"],
                          sum(w == bid_ppo_mod.EMBALLER_GUARD_WEIGHT for w in buf["weight"]),
                          sum(w == 1.0 for w in buf["weight"]), bid_stats["distill_loss"]), flush=True)
                bid_ppo["buffer"] = bid_ppo_mod.new_buffer()
            print("  familles (phase1): %s" % family_counts, flush=True)
            if visitor_counts:
                print("  donnes contre visiteurs (hors stats) : %s" % visitor_counts, flush=True)
            if rich_counts:
                print("  donnes riches (hors stats) : %s" % rich_counts, flush=True)
            if rollout is not None:
                print("  rollouts : %d decisions analysees dont %d occasions de surcoupe, %d renonces a couper, "
                      "%d entames atout (%d avec une carte nettement moins bonne), %.0fs" % (
                          rollout["count"], rollout["surcoupes"], rollout["coupes"], rollout["atouts"],
                          rollout["corrective"],
                          rollout["seconds"]),
                      flush=True)
            if teacher is not None:
                print("  imitation ExpertBot : poids %.3f, %d decisions ciblees, accord IA/bot %.1f%% "
                      "(%d donnes d'imitation au total)" % (
                          distill_coef, teacher["count"], 100.0 * teacher["agree"] / max(1, teacher["count"]),
                          teacher_deals), flush=True)
                if max(teacher_floor, teacher_coef_at(teacher_deals, teacher_coef, teacher_decay)) <= 0:
                    teacher = None
                    print("  imitation ExpertBot terminee : PPO seul a partir du prochain lot.", flush=True)
            if bid_outcome_counts or bid_action_counts:
                print("  enchere (phase2) contrats=%s" % bid_outcome_counts, flush=True)
                print("  enchere (phase2) actions=%s" % bid_action_counts, flush=True)

            if batch_num % args.save_every == 0 or stop["flag"]:
                save_all(nets)
                _save_population()
                print("  checkpoints sauvegardes (checkpoints_v2/, dont population enchere).", flush=True)

            if total_episodes // args.stats_every > stats_marker:
                stats_marker = total_episodes // args.stats_every
                emit_stats(stats, total_episodes, time.time() - start_time, args.stats_file)
                stats = new_stats_accumulator()
                if stats2["donnes"]:
                    emit_phase2_stats(stats2, total_episodes, time.time() - start_time,
                                      args.stats_file_phase2, generation, population)
                stats2 = new_phase2_stats_accumulator()

            if total_episodes // eval_every > eval_marker:
                eval_marker = total_episodes // eval_every
                # Graine fixe, rng d'entrainement intact : evaluations comparables entre elles.
                eval_vs_random.run_and_report(
                    nets, random.Random(eval_vs_random.EVAL_SEED), total_episodes, time.time() - start_time,
                    eval_games, eval_file, population)
                if team_errors_deals:
                    team_errors.measure(nets, population, team_errors_deals, total_episodes,
                                        time.time() - start_time, team_errors_file)
                if expert_match_contracts:
                    expert_bot_match.measure_and_record(nets, total_episodes, time.time() - start_time,
                                                        expert_match_contracts)
                if champion_deals:
                    champion_play.measure_and_gate(nets, population, total_episodes, time.time() - start_time,
                                                   champion_deals)
                if duel_deals:
                    duel_expert.measure_and_record(nets, population, total_episodes, time.time() - start_time,
                                                   duel_deals)

                # EN MEME TEMPS que l'evaluation vs RandomBot (meme cadence
                # eval_every, demande explicite de l'utilisateur) : 1 partie
                # de plusieurs donnes par famille (voir scripts/record_game.py:
                # SAMPLE_HANDS_PER_FAMILY), envoyee a whist-4-it sous un nom
                # fixe par famille -- remplace l'ancienne partie de cette
                # famille sans toucher aux autres fichiers du dossier.
                if not args.no_record_samples:
                    sample_nets = {role: record_game.GreedyPolicy(nets[role]) for role in PHASE1_ROLES}
                    games = record_game.build_family_sample_games(
                        sample_nets, rng, generated_by="whist-ai-training")
                    local_dir = os.path.join(args.record_samples_dir, "training_samples_%d" % total_episodes)
                    os.makedirs(local_dir, exist_ok=True)
                    for family, data in games.items():
                        with open(os.path.join(local_dir, "%s.json" % family), "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                    results = record_game.send_family_samples_to_godot(games, godot_dir=args.godot_dir)
                    n_sent = sum(1 for p in results.values() if p)
                    print("  %d partie(s)/famille (%d donnes chacune) enregistrees (%s), "
                          "%d/%d envoyees a whist-4-it (%s)." % (
                              len(games), record_game.SAMPLE_HANDS_PER_FAMILY, local_dir,
                              n_sent, len(games), args.godot_dir), flush=True)

            if args.export_every and total_episodes // args.export_every > export_marker:
                export_marker = total_episodes // args.export_every
                save_all(nets)  # l'export relit depuis checkpoints_v2/ -- jamais un export en retard
                _save_population()
                print("  export Godot automatique (%d donnes) :" % total_episodes, flush=True)
                _export_godot()

            if args.max_batches is not None and batch_num >= args.max_batches:
                break
    finally:
        save_all(nets)
        _save_population()
        print("Export Godot final :", flush=True)
        try:
            _export_godot()
        except Exception as exc:  # l'arret doit rester propre meme si l'export echoue
            print("  !! export Godot final en echec : %r" % exc, flush=True)
        if not args.no_git_push and not args.no_godot_sync and args.max_batches is None:
            push_latest_ai([specialists.CHECKPOINT_DIR, args.stats_file, args.stats_file_phase2,
                            stats_file_phase3, eval_file, team_errors_file,
                            expert_bot_match.EXPERT_MATCH_FILE_DEFAULT, duel_expert.DUEL_FILE_DEFAULT,
                            args.eval_stats_file],
                           total_episodes)
        if os.path.exists(STOP_FLAG_PATH):
            os.remove(STOP_FLAG_PATH)
        print("Arret. Checkpoints finaux sauvegardes. Total: %d donnes en %.0fs." % (
            total_episodes, time.time() - start_time), flush=True)


if __name__ == "__main__":
    main()
