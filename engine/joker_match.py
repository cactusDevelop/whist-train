"""Matchs roguelike simules pour l'IA jokers : un match = plusieurs donnes
avec la meme economie de jokers (pioche/defausse qui tournent), comme
RogueMatch. Elites aux sieges 1 et 3 (contrat attitre, deck EliteDecks fixe,
biais d'enchere ; acte III : jokers de defense contre le contrat adverse)
ou boss au siege 2, en face du joueur (deck, contrat impose et donne arrangee de RogueBosses ;
regle propre de la Baronne portee, blocage des jokers de pli/aura par le
Moine aussi). Jeu de la carte ET enchere : ExpertBot partout, comme le jeu
depuis whist-4-it 2cc6f15 (3e5d).

Joueur (siege 0) AVEC jokers depuis la v2 (2026-10-13, demande
utilisateur) : deck tire du catalogue de l'acte (STARTING_DECK + gains,
ameliorations, parfois un maudit), Soupcon et Oeil du combat, et un style
tire par match (aucun joker / autopilote du jeu qui lance tout ce qui
passe / selectif au hasard) ; cibles : un adversaire au hasard.
"""

import copy
import random

from agents import expert_bot
from engine import jokers as J
from engine.bidding import Bidding, bid_action_to_dict, legal_action_ids_with_hand
from engine.cards import NUM_PLAYERS, legal_plays, suit_of, rank_of
from engine.jokers import BOSS_DECKS, BOSS_SEAT, ELITE_SEATS, PLAYER, JokerHand, JokerRules, MatchState
from engine.trou import detect_trou
from engine.whisted import DeckChain, deal_uniform

HANDS_PER_MATCH = 5
TIEBREAK_OWN = 0.1


def hand_reward(points, seat):
    """Objectif d'un bot a jokers : finir devant le joueur (condition
    « premier », la plus courante) -- ecart de points au joueur, plus un
    petit terme de points propres (departage quand il est le partenaire
    du joueur)."""
    return (points[seat] - points[PLAYER]) + TIEBREAK_OWN * points[seat]


class DuelDeals:
    """Donnes du Duel (DuplicateMatch : whisted_deck_enabled = false) : la
    k-ieme donne et le premier donneur ne dependent que de la graine du duel,
    les memes sur chaque table quels que soient les jokers. Meme interface
    que DeckChain (Match.finish y range pack / donneur, ignores ici)."""

    def __init__(self, duel_seed):
        self.duel_seed = duel_seed
        self.k = 0
        self.pack = None
        self.dealer = random.Random(duel_seed * 4 + 1).randrange(NUM_PLAYERS)  # DuelRules.first_dealer
        self.next_is_brimborion = False

    def deal(self, _rng):
        hands = deal_uniform(random.Random(self.duel_seed * 1009 + self.k))
        self.k += 1
        return hands, self.dealer, False

    def finish_unplayed(self, hands, dealer):  # passe generale : meme donneur
        self.dealer = dealer


class Match:
    def __init__(self, config, seed, deals=None):
        """`deals` : liste de (mains, donneur, whiste) a rejouer dans
        l'ordre (duplicate exact entre politiques, voir record_deals) ;
        None = chaine whistee normale."""
        self.config = config
        self.deals = None if deals is None else list(deals)
        self.recorded = []
        self.rng = random.Random(seed)
        decks = [[] for _ in range(NUM_PLAYERS)]
        self.specialties = {}  # {siege: famille} (RogueRun.match_config, 3d47)
        self.adapts = []
        if config["kind"] == "duel":  # DuplicateMatch : habitues a deck d'elite (07ac)
            for s, fam in config["habitues"].items():
                decks[int(s)] = J.habitue_deck(fam)
        elif config["kind"] == "elite":
            self.specialties = J.elite_specialties(self.rng)
            for s in ELITE_SEATS:
                decks[s] = list(J.SPECIALTY_DECKS[self.specialties[s]])
            if config["act"] >= J.ADAPT_FROM_ACT:
                self.adapts = list(ELITE_SEATS)
        else:
            decks[BOSS_SEAT] = list(BOSS_DECKS[config["boss"]])
        spammers = [s for s in range(NUM_PLAYERS) if s != PLAYER and decks[s]]
        for s in spammers:  # RogueMatch : elites/boss = spammers (79ac)
            decks[s] = decks[s] + J.SPAMMER_EXTRA
        decks[PLAYER] = list(config.get("deck", []))
        mise = config.get("mise", 0)
        self.state = MatchState(decks, random.Random(self.rng.random()), config.get("suspicion", 0))
        self.state.spammers = spammers
        self.state.spammer_energy = J.spammer_energy(mise)
        self.state.energy_malus = J.mise_energy_malus(mise, config["act"])
        self.state.eye_threshold = J.eye_threshold(mise)
        self.rules = JokerRules(self.state, random.Random(self.rng.random()))
        self.rules.set_spammers(spammers)  # 65a7 : 3 cases d'effet de donne
        boss = config.get("boss", "")
        # regle du boss + celle d'un autre en Mise Doree (RogueMises.second_rule)
        self.bosses = tuple(b for b in (boss, J.second_rule(mise, boss)) if b)
        self.state.arm_curses()
        if "baronne" in self.bosses:
            self.rules.curses.append(J.BaronneSees("baronne", BOSS_SEAT))
        self.player = Player(config, random.Random(self.rng.random()))
        if config["kind"] == "duel":
            self.chain = DuelDeals(config["duel_seed"])
        else:
            self.chain = DeckChain(1.0, self.rng)  # entrainement 100 % whiste

    def next_hand(self):
        """Nouvelle donne (RogueMatch._on_new_hand : end_hand, new_hand,
        start_hand) jusqu'a un contrat joue. Une passe generale redonne et
        repioche, comme une nouvelle donne du jeu. -> JokerHand."""
        while True:
            self.state.end_hand()
            self.rules.new_hand(None, None)
            self.state.start_hand()
            if self.deals is not None:
                hands, dealer, whisted = self.deals.pop(0)
                hands = [list(h) for h in hands]
            else:
                hands, dealer, whisted = self.chain.deal(self.rng)
                self.recorded.append(([list(h) for h in hands], dealer, whisted))
            self.rules.deal_hands = hands
            if self.config.get("force_misere"):  # mesure ciblee (scripts/misere_jokers.py), jamais a l'entrainement
                contract = forced_misere(self.config["force_misere"], hands, self.rng)
                bid_summary = None
                if self.player is not None:  # 8f41 : ses jokers d'enchere (Pot-de-Vin) avant la misere imposee
                    self.player.bidding_turn(self.state, self.rules, self.config.get("eye", ""), self.bosses)
            elif self.config["kind"] == "boss":
                contract = boss_contract(self.config["boss"], hands, self.rng)
                bid_summary = None
            else:
                bidding = run_bot_bidding(hands, dealer, self)
                if bidding.general_pass:
                    if self.deals is None:
                        self.chain.finish_unplayed(hands, dealer)
                    continue
                contract = bidding.locked_contract
                bid_summary = bidding.bid_summary
            from scripts.train import dispatch_play_phase2
            contract = dict(contract)
            for seat in self.adapts:  # RogueMatch._on_contract_locked, avant rules.on_contract
                if seat not in J.attackers(contract) and contract.get("type", "") in J.DEFENCE:
                    self.state.adapt(seat, J.DEFENCE[contract["type"]])
            self.rules.on_contract(contract)  # Rachat
            game = dispatch_play_phase2(contract, hands, dealer, bid_summary, whisted)
            self.rules.bind(game, contract)
            self.rules.scorer = base_score
            game.exposed = list(contract.get("exposed", []))  # Cartes sur Table
            hand = JokerHand(game, contract, self.state, self.rules)
            hand.dealer = dealer
            hand.player = self.player
            hand.eye = self.config.get("eye", "")
            hand.boss = self.bosses
            return hand

    def finish(self, hand):
        """DeckChain.finish_played sous jokers (JokerRules.gathered_pack) :
        les cartes jouees SAUF les cartes de joker, puis ce que les sieges
        tiennent encore (Horloge Arretee, carte de joker jouee a la place
        d'une carte), puis set_aside (Courant d'Air), puis les ecarts."""
        if self.deals is not None:
            return
        g = hand.game
        pack = [c for i, c in enumerate(g.play_order) if i not in hand.joker_play_idx]
        for h in g.hands:
            pack.extend(h)
        pack.extend(self.rules.vanished)  # set_aside (whist-4-it cc566e0)
        discards = getattr(g, "discards", {})
        for offset in (1, 2, 3, 4):
            seat = (hand.dealer + offset) % NUM_PLAYERS
            if seat in discards:
                pack.append(discards[seat])
        assert sorted(pack) == list(range(52)), "paquet ramasse incomplet (jokers)"
        self.chain.pack = pack
        self.chain.dealer = (hand.dealer + 1) % NUM_PLAYERS
        if hand.contract["type"] == "trou":
            self.chain.next_is_brimborion = True


def run_bot_bidding(hands, dealer, match=None):
    """Enchere ExpertBot ; sous jokers (`match`) : le joueur peut lancer ses
    jokers d'enchere a son tour, et JokerRules.bot_bid fait passer les bots
    vises : comme net/whist_table.gd, le bot choisit puis la regle remplace
    son annonce par un passe (s'il est legal)."""
    bidding = Bidding(dealer)
    trou = detect_trou(hands)
    if trou is not None:
        bidding.seed_trou(trou)
    elif match is not None:  # WhistTable._start_bidding -> RuleModifiers.prepare_bidding (Cavalier Seul, dc78)
        match.rules.prepare_bidding(bidding)
    while not bidding.finished:
        seat = bidding.turn
        if match is not None and seat == PLAYER:
            match.player.bidding_turn(match.state, match.rules, match.config.get("eye", ""), match.bosses)
        if match is not None and seat != PLAYER:
            # 4d7d (rogue_run.gd) : enchere adoucie a l'acte I et dans tout match sans boss
            expert_bot.set_bid_params(expert_bot.ACT1_BID_PARAMS if match.config.get("act") == 1 or not match.bosses else {})
            try:
                choice = expert_bot.choose_bid(bidding, seat, hands[seat])
                info = {"action": "raise"} if isinstance(choice, tuple) else bid_action_to_dict(choice)
                if info["action"] == "pass" and seat in match.specialties:  # JokerRules.bot_bid, avant les effets
                    biased = specialty_bid(match.specialties[seat], hands[seat], bidding, seat)
                    if biased is not None:
                        choice, info = biased, bid_action_to_dict(biased)
            finally:
                expert_bot.set_bid_params({})
            if match.rules.forces_pass(seat, info, bidding):
                passes = [a for a in legal_action_ids_with_hand(bidding, seat, hands[seat])
                          if bid_action_to_dict(a)["action"] == "pass"]
                if passes and bidding.apply(seat, passes[0]):
                    continue
            ok = bidding.apply_raise_to(seat, choice[1]) if isinstance(choice, tuple) else bidding.apply(seat, choice)
            if not ok:
                raise AssertionError("enchere illegale %r siege %d" % (choice, seat))
            continue
        expert_bot.apply_bid(bidding, seat, hands[seat])
    return bidding


def _best_scored(entries, needed_of, hand, propose):
    """WhistExpertBot._best_scored : max (plis estimes, id) parmi les
    couleurs qui atteignent needed_of(couleur)."""
    best = None
    for a, info in entries:
        suit = info["suit"]
        if propose and not expert_bot.propose_suit_ok(hand, suit):
            continue
        est = expert_bot.estimate_tricks(hand, suit)
        if est >= needed_of(suit) and (best is None or (est, a) > best):
            best = (est, a)
    return None if best is None else best[1]


def specialty_bid(family, hand, bidding, seat):
    """EliteDecks.specialty_bid : la ou ExpertBot passe, l'annonce de la
    famille avec sa marge BID_SLACK, ou None."""
    by_kind = {}
    for a in legal_action_ids_with_hand(bidding, seat, hand):
        info = bid_action_to_dict(a)
        by_kind.setdefault(info["action"], []).append((a, info))
    slack = J.BID_SLACK[family]
    choice = None
    if family == "emballage":
        choice = _best_scored(by_kind.get("emballer", []),
                              lambda s: expert_bot.emballer_min(bidding, s) - slack, hand, False)
        if choice is None:
            choice = _best_scored(by_kind.get("propose", []),
                                  lambda s: expert_bot.PROPOSE_MIN - slack, hand, True)
    elif family == "solo":
        best_est = 0
        for s in range(4):
            est = expert_bot.estimate_tricks(hand, s) + slack
            c = expert_bot._cheapest_solo(by_kind, est, s, hand, ("solo",))
            if c is not None and (choice is None or est > best_est):
                choice, best_est = c, est
    elif family == "misere":
        offered = {info["kind"]: a for a, info in by_kind.get("misere", [])}
        spread = expert_bot.misere_slack(hand)
        for kind in ("etalee", "grande"):
            if choice is None and kind in offered and spread <= expert_bot.MISERE_SLACK_MAX[kind] + slack:
                choice = offered[kind]
    return choice


FORCED_MISERE_KINDS = ("petite", "piccolo", "grande")


def forced_misere(who, hands, rng):
    """Mesure des jokers de misere : misere imposee, declarant = le joueur
    ("player") ou un bot ("bot"), qui recoit la main la plus « misere »
    (expert_bot.misere_slack) parmi les sieges eligibles (mains echangees)."""
    seats = [PLAYER] if who == "player" else [1, 2, 3]
    best = min(range(NUM_PLAYERS), key=lambda s: (expert_bot.misere_slack(hands[s]), s))
    seat = rng.choice(seats)
    hands[seat], hands[best] = hands[best], hands[seat]
    return {"type": "misere", "kind": rng.choice(FORCED_MISERE_KINDS), "declarers": [seat]}


# --- boss (rogue_bosses.gd) --------------------------------------------------

def _longest_suit(hand):
    counts = [sum(1 for c in hand if suit_of(c) == s) for s in range(4)]
    return counts.index(max(counts))


def _pull(hands, suit, rank):
    me = hands[BOSS_SEAT]
    card = suit * 13 + rank
    if card in me:
        return
    for seat in range(NUM_PLAYERS):
        if seat != BOSS_SEAT and card in hands[seat]:
            options = [c for c in me if suit_of(c) != suit and rank_of(c) != 12]
            if not options:
                return
            give = min(options, key=rank_of)
            me.remove(give)
            hands[seat].remove(card)
            me.append(card)
            hands[seat].append(give)
            return


def boss_contract(boss, hands, rng):
    """RogueBosses.rig + imposed_contract (mains modifiees sur place)."""
    me = hands[BOSS_SEAT]
    if boss == "moine":
        for c in [c for c in me if rank_of(c) == 12]:
            options = [(s, o) for s in range(NUM_PLAYERS) if s != BOSS_SEAT for o in hands[s] if rank_of(o) <= 5]
            if options:
                s, o = rng.choice(options)
                me.remove(c)
                hands[s].remove(o)
                me.append(o)
                hands[s].append(c)
        contract = {"type": "misere", "kind": "grande", "declarers": [BOSS_SEAT]}
    elif boss == "baronne":
        suit = rng.randrange(4)
        for rank in (12, 11, 10):
            _pull(hands, suit, rank)
        contract = {"type": "solo", "suit": suit, "level": 8, "declarer": BOSS_SEAT}
    elif boss == "fossoyeur":
        suit = _longest_suit(me)
        _pull(hands, suit, 12)
        contract = {"type": "emballage", "suit": suit, "level": 10, "declarer": BOSS_SEAT,
                    "partner": (PLAYER + 1) % 4}  # RogueBosses.partner_seat : voisin de gauche
    else:
        suit = _longest_suit(me)
        for s in range(4):
            _pull(hands, s, 12)
        for rank in (11, 10):
            _pull(hands, suit, rank)
        contract = {"type": "solo", "suit": suit, "level": 11, "declarer": BOSS_SEAT}
    for h in hands:
        h.sort()
    return contract


# --- jeu ------------------------------------------------------------------------

def base_score(game, contract):
    """WhistTable._score_hand : le score de la donne sur les plis actuels."""
    from scripts.train import declarer_role_and_seats, real_points_by_seat
    role, _seats = declarer_role_and_seats(contract)
    return real_points_by_seat(role, game, contract)


def points_of(hand):
    """Ecarts de la donne (adjust_deltas des jokers : Bluff, Pacte, Valet
    Borgne...) + points ajoutes au score pendant la donne (score_bonus)."""
    from scripts.train import declarer_role_and_seats, real_points_by_seat
    role, _seats = declarer_role_and_seats(hand.contract)
    base = hand.rules.counted(lambda: real_points_by_seat(role, hand.game, hand.contract))
    deltas = hand.rules.adjust_deltas(base, hand.contract)
    return [d + b for d, b in zip(deltas, hand.rules.score_bonus)]


def play_out(hand, policy, first_action=None, on_decision=None, before_player=None):
    """Joue la donne jusqu'au bout. `policy(hand, seat) -> action` pour
    chaque tour de pli d'un siege a jokers (avant sa carte) ;
    `first_action` force la decision du premier de ces tours ;
    `on_decision(hand, seat, action)` est appele AVANT le cast. Le joueur
    simule (hand.player) lance ses jokers a son tour de pli ;
    `before_player(hand)` est appele juste avant (test de rentabilite).
    7567 : au premier pli, il a aussi la fenetre avant l'entame d'un bot
    (joker_rules.gd in_trick_window), simulee pour Main de Fer seul."""
    g = hand.game
    holders = set(hand.holders())
    player = getattr(hand, "player", None)
    window = player is not None and g.trick_count == 0 and not g.current_trick and g.turn != PLAYER
    while not g.done:
        seat = g.turn
        discarding = getattr(g, "pending_discards", None)
        if window and not discarding:
            window = False
            hand.rules.trick_window = True
            if before_player is not None:
                before_player(hand)
            player.trick_turn(hand)
            hand.rules.trick_window = False
        if seat in holders and not discarding:
            if first_action is not None:
                action, first_action = first_action, None
            else:
                action = policy(hand, seat)
            if on_decision is not None:
                on_decision(hand, seat, action)
            hand.cast(seat, action)
            if action == 0 and hand.fuel(seat):  # RogueMatch._bot_fuel (65a7) puis _bot_turn a nouveau
                action = policy(hand, seat)
                if on_decision is not None:
                    on_decision(hand, seat, action)
                hand.cast(seat, action)
        if seat == PLAYER and player is not None and not discarding and before_player is not None:
            before_player(hand)
        if seat == PLAYER and player is not None and not discarding and player.trick_turn(hand):
            continue  # carte de joker posee dans le pli a la place d'une carte
        card = expert_bot.choose_card(g, seat, rules_seen(hand, seat))
        if not discarding:
            card = _rules_card(hand, seat, card)
        hand.step(seat, card)
    return points_of(hand)


def rules_seen(hand, seat):
    """Lot 1 du 31/10 (ExpertBot sensible aux regles, bots et joueur) : l'atout
    effectif du pli, les cartes que les jokers rendent jouables, l'inversion.
    None quand les jokers ne changent rien : le jeu reste exactement classique.
    Lune Noire (couleurs jumelles) : atout et legalite seulement, lot 2."""
    g, rules = hand.game, hand.rules
    if not rules.live() or getattr(g, "pending_discards", None):
        return None
    led = suit_of(g.current_trick[0][1]) if g.current_trick else None
    cards = sorted(g.hands[seat])
    trump = rules.trick_trump()
    legal = rules.legal_cards(cards, led, seat)
    inverted = any(e.inverts() for e in rules.live())
    if trump == g.trump_suit and not inverted and sorted(legal) == sorted(legal_plays(cards, led)):
        return None
    return {"trump": trump, "legal": legal, "inverted": inverted}


def _rules_card(hand, seat, card):
    """WhistTable._pump_bots : bot_card des jokers, puis une carte que les
    jokers rendent illegale (La Tournee) -> la premiere legale. Joueur dans
    sa misere sous Defausse Discrete : sa plus haute carte qui ne prend pas
    le pli (un humain s'en sert ; l'autopilote du jeu non)."""
    g, rules = hand.game, hand.rules
    led = suit_of(g.current_trick[0][1]) if g.current_trick else None
    legal = rules.legal_cards(sorted(g.hands[seat]), led, seat)
    if seat != PLAYER and rules.overrides_bot_card():
        card = rules.bot_card(seat, card, expert_bot.PublicView(g, seat).legal)
    if seat == PLAYER:  # dcf1 : le joueur qui a lance Faux Pas coupe avec le plus petit atout qui prend
        for e in rules.effects:
            if e.owner == PLAYER and J.base_id(e.id) == "faux_pas":
                card = e.bot_card(seat, card, legal, rules)
    if seat == PLAYER and led is not None and J.in_misere(PLAYER, rules) and len(legal) > len(legal_plays(g.hands[seat], led)):
        trump = rules.trick_trump()
        safe = [c for c in legal if rules.trick_winner(list(g.current_trick) + [(seat, c)], led, trump) != seat]
        if safe:
            return max(safe, key=lambda c: (rank_of(c), c))
    return card if card in legal else legal[0]


def clone(hand):
    return copy.deepcopy(hand)


SPARE_DEALS = 30  # donnes fixes de la suite du match (passes generales comprises)


def spare_deals(match, seed):
    """Donnes fixes pour la suite du match, communes a toutes les actions d'une
    decision (seul l'etat des jokers differe entre rollouts)."""
    chain, rng = copy.deepcopy(match.chain), random.Random(seed)
    out = []
    for _ in range(SPARE_DEALS):
        hands, dealer, whisted = chain.deal(rng)
        out.append(([list(h) for h in hands], dealer, whisted))
        chain.finish_unplayed(hands, dealer)
    return out


def match_rollout(match, hand, seat, action, policy, hands_left, deals, seats=None):
    """-> (recompense de la donne pour `seat`, recompense donne + donnes
    restantes du match pour `seat` ou sommee sur `seats`), `action` forcee a
    la premiere decision, puis `policy`."""
    seats = [seat] if seats is None else seats
    m2, h2 = copy.deepcopy((match, hand))
    m2.deals = list(deals)
    pts = play_out(h2, policy, first_action=action)
    qh = hand_reward(pts, seat)
    qm = sum(hand_reward(pts, s) for s in seats)
    for _ in range(hands_left):
        m2.finish(h2)
        h2 = m2.next_hand()
        pts = play_out(h2, policy)
        qm += sum(hand_reward(pts, s) for s in seats)
    return qh, qm


BOSS_ACT = {"moine": 1, "baronne": 2, "fossoyeur": 2, "grand_maitre": 3}
# deck du joueur : jokers gagnes en plus de STARTING_DECK, par acte (bornes incluses)
EXTRA_JOKERS = {1: (0, 4), 2: (2, 7), 3: (4, 10)}
UPGRADE_PROB = 0.25
CURSE_PROB = 0.15
# style du joueur simule ; "expert" (2026-10-16) = regles de agents/joker_expert.py
STYLES = (("none", 0.1), ("greedy", 0.2), ("selective", 0.2), ("expert", 0.5))


def player_deck(act, rng):
    """Deck d'un run a l'acte `act` : STARTING_DECK + jokers des raretes de
    l'acte (MAX_COPIES par joker), certains ameliores, parfois un maudit."""
    deck = list(J.STARTING_DECK)
    pool = sorted(i for i, (_t, r, _c, _b) in J.CATALOG.items() if r in J.RARITIES_BY_ACT[act])
    lo, hi = EXTRA_JOKERS[act]
    for _ in range(rng.randint(lo, hi)):
        jid = rng.choice(pool)
        if sum(1 for d in deck if J.base_id(d) == jid) < J.MAX_COPIES:
            deck.append(jid)
    deck = [d + J.UPGRADE_SUFFIX if J.can_upgrade(d) and rng.random() < UPGRADE_PROB else d for d in deck]
    if rng.random() < CURSE_PROB:
        deck.append(rng.choice(sorted(i for i, v in J.CATALOG.items() if v[1] == "maudit" and i not in J.OPPONENT_ONLY)))
    return deck


def random_config(rng):
    if rng.random() < 0.6:
        cfg = {"kind": "elite", "act": rng.choice((1, 2, 3)), "eye": J.ELITE_EYE}
    else:
        boss = rng.choice(sorted(BOSS_DECKS))
        cfg = {"kind": "boss", "boss": boss, "act": BOSS_ACT[boss], "eye": J.BOSS_SANCTIONS[boss]}
    x, style = rng.random(), STYLES[-1][0]
    for name, p in STYLES:
        if x < p:
            style = name
            break
        x -= p
    cfg.update({"deck": player_deck(cfg["act"], rng), "suspicion": rng.randint(0, 7),
                "style": style, "cast_prob": round(rng.uniform(0.15, 0.6), 3),
                "mise": rng.randrange(J.MISE_COUNT)})  # Mises 0..3 (79ac)
    return cfg


class Player:
    """Joueur simule du siege 0 (le jeu de la carte reste l'ExpertBot).
    `greedy` = RogueAutopilot._try_joker (le premier joker de la main qui
    passe a ce tour, tant qu'il en reste) ; `selective` = chaque joker qui
    passe est lance avec la probabilite cast_prob ; `none` = jamais.
    Cibles : un adversaire du joueur au hasard (sinon un bot au hasard)."""

    MAX_CASTS = 6

    def __init__(self, config, rng):
        self.style = config.get("style", "none")
        self.cast_prob = config.get("cast_prob", 0.3)
        self.rng = rng
        # Test de rentabilite (scripts/joker_profit.py) : joker impose au prochain
        # tour d'enchere / de pli ("" = aucun), et candidats du 1er tour d'enchere.
        self.force_bid = None
        self.force_trick = None
        self.forced_ok = None
        self.bid_probe = None

    def _target(self, rules, jid=None):
        if jid is not None and J.make_effect(jid, PLAYER).target_kind() == "suit":  # 6c54 : sa plus longue couleur
            cards = rules.game.hands[PLAYER] if rules.game is not None else (rules.deal_hands or [[]])[PLAYER]
            if J.base_id(jid) == "main_de_fer":
                from agents import joker_expert as X
                return X.main_de_fer_suit(cards, rules.contract_trump())
            return J.longest_suit(cards)
        if jid is not None and rules.game is not None:  # 65a7 : le HUD n'offre que les sieges acceptes
            eff = J.make_effect(jid, PLAYER)
            ok = [s for s in (1, 2, 3) if eff.accepts_target(s, rules)]
            if ok and len(ok) < 3:
                return self.rng.choice(ok)
        opp = [s for s in (1, 2, 3) if rules.is_opponent(PLAYER, s)]
        return self.rng.choice(opp or [1, 2, 3])

    def _candidates(self, state, rules, boss, bidding):
        out = []
        for jid in list(state.joker_hand[PLAYER]):
            t = J.jtype(jid)
            if (t == "enchere") != bidding or state.play_error(PLAYER, jid) != "":
                continue
            if rules.trick_window and J.base_id(jid) != "main_de_fer":
                continue
            if "moine" in boss and t in ("pli", "aura"):  # RogueBosses.blocks_joker (boss = regles des boss)
                continue
            if t == "pli" and rules.led_suit() is None:
                continue
            out.append(jid)
        return out

    def _pick(self, state, rules, boss, bidding, tried, hand=None, cards=None):
        """-> (jid, cible) ou None."""
        if self.style == "none":
            return None
        if self.style == "expert":
            from agents import joker_expert as X
            cands = [j for j in self._candidates(state, rules, boss, bidding) if j not in tried]
            if not cands:
                return None
            got = X.player_bid_choice(cards, state, cands) if bidding else X.player_trick_choice(hand, cands)
            return None if got is None else (got[0], got[1])
        for jid in self._candidates(state, rules, boss, bidding):
            if jid in tried:
                continue
            if self.style == "greedy" or self.rng.random() < self.cast_prob:
                return jid, None
        return None

    def _cast(self, state, rules, eye, jid, target=None):
        """RogueMatch.play_joker : -> resultat de rules.play, None si refuse."""
        if target is None or target < 0:
            target = self._target(rules, jid)
        played = rules.play(PLAYER, jid, target)
        if played.get("error"):
            return None
        J.spend(state, PLAYER, jid, played)
        if played.get("exhaust") and jid in state.discard[PLAYER]:
            state.discard[PLAYER].remove(jid)
        J.watch_eye(state, rules, eye, jid)
        return played

    def _forced(self, state, rules, eye, jid, target=None):
        # _cast -> None si refuse ; un joker accepte peut renvoyer {} (Coup de Vent, Bluff...)
        played = self._cast(state, rules, eye, jid, target) if jid else None
        self.forced_ok = played is not None
        return played

    def bidding_turn(self, state, rules, eye, boss):
        if self.bid_probe is not None and not self.bid_probe:
            self.bid_probe.append(self._candidates(state, rules, boss, True))
        if self.force_bid is not None:
            jid, self.force_bid = self.force_bid, None
            tgt = None
            if jid and self.style == "expert":
                from agents import joker_expert as X
                tgt = X.bid_target(jid)
            self._forced(state, rules, eye, jid, tgt)
            return
        tried = set()
        cards = rules.deal_hands[PLAYER] if rules.deal_hands else []
        for _ in range(self.MAX_CASTS):
            got = self._pick(state, rules, boss, True, tried, cards=cards)
            if got is None:
                return
            jid, tgt = got
            tried.add(jid)
            self._cast(state, rules, eye, jid, tgt)

    def trick_turn(self, hand):
        """Avant la carte du joueur ; True si une carte de joker a ete posee
        a sa place."""
        if self.force_trick is not None:
            jid, self.force_trick = self.force_trick, None
            tgt = None
            if jid and self.style == "expert":
                from agents import joker_expert as X
                tgt = X.target(X.view(hand, PLAYER), jid)
            played = self._forced(hand.state, hand.rules, hand.eye, jid, tgt)
            if played is not None and played.get("card") is not None:
                hand.step_joker_card(PLAYER, played["card"], jid)
                return True
            return False
        tried = set()
        for _ in range(self.MAX_CASTS):
            got = self._pick(hand.state, hand.rules, hand.boss, False, tried, hand=hand)
            if got is None:
                return False
            jid, tgt = got
            tried.add(jid)
            played = self._cast(hand.state, hand.rules, hand.eye, jid, tgt)
            if played is not None and played.get("card") is not None:
                hand.step_joker_card(PLAYER, played["card"], jid)
                return True
        return False
