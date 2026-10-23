/* Solveur double-mort (cartes ouvertes) du jeu de la carte, pour les contrats
 * a atout (emballage, trou, solo) : nombre de plis restants du camp A quand
 * les 4 joueurs voient toutes les mains et jouent au mieux (A maximise, B
 * minimise). Les camps sont quelconques (partenaires voisins ou non, 1 contre
 * 3) -- les solveurs de bridge supposent N-S / E-O, d'ou ce solveur.
 *
 * Carte = couleur * 13 + rang (rang 12 = As), main = masque 64 bits.
 * Alpha-beta "fail-soft" + table de transposition aux debuts de pli (bornes
 * basse/haute par position) + cartes equivalentes (sequences dans une main,
 * les cartes deja jouees ne separant plus rien) + ordre des coups simple.
 * La valeur exacte est trouvee par sondes a fenetre nulle (dichotomie).
 * Charge par engine/dds.py (ctypes). */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t u64;

#define TT_BITS 20
#define TT_SIZE (1u << TT_BITS)

typedef struct {
    u64 h[4];
    uint32_t gen;
    int8_t leader, lb, ub;
} Entry;

static Entry *tt = NULL;
static uint32_t gen = 0;
static int trump;
static int side[4];
static u64 H[4];
static int tcard[4], tseat[4], nt, leader_s;
static long long nodes;

static inline int suit_of(int c) { return c / 13; }
static inline int popc(u64 x) { return __builtin_popcountll(x); }
static inline u64 suit_mask(int s) { return 0x1FFFULL << (13 * s); }

/* a bat-il b (b = carte maitresse du pli jusqu'ici, couleur demandee led) ? */
static inline int beats(int a, int b) {
    int sa = suit_of(a), sb = suit_of(b);
    if (sa == sb) return a > b;
    return sa == trump;
}

static int trick_winner_idx(int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (beats(tcard[i], tcard[best])) best = i;
    return best;
}

static inline uint32_t tt_index(int leader) {
    u64 x = H[0] * 0x9E3779B97F4A7C15ULL ^ H[1] * 0xC2B2AE3D27D4EB4FULL ^ H[2] * 0x165667B19E3779F9ULL
            ^ H[3] * 0x27D4EB2F165667C5ULL ^ (u64)leader * 0x94D049BB133111EBULL;
    x ^= x >> 29;
    return (uint32_t)(x & (TT_SIZE - 1));
}

/* coups a essayer (un par classe d'equivalence), dans l'ordre */
static int gen_moves(int turn, int *moves) {
    u64 legal = H[turn];
    if (nt > 0) {
        u64 follow = H[turn] & suit_mask(suit_of(tcard[0]));
        if (follow) legal = follow;
    }
    u64 live = H[0] | H[1] | H[2] | H[3];
    for (int i = 0; i < nt; i++) live |= 1ULL << tcard[i];
    int n = 0;
    int keys[52];
    int win_idx = nt > 0 ? trick_winner_idx(nt) : -1;
    int mate_winning = nt > 0 && side[tseat[win_idx]] == side[turn];
    for (int s = 0; s < 4; s++) {
        if (!(legal & suit_mask(s))) continue;
        int prev_mine = 0;
        for (int r = 12; r >= 0; r--) {
            int c = s * 13 + r;
            u64 b = 1ULL << c;
            if (!(live & b)) continue;
            if (legal & b) {
                if (!prev_mine) {
                    int key;
                    if (nt == 0) {
                        key = 100 + r;  /* entame : les fortes d'abord */
                    } else {
                        int wins = beats(c, tcard[win_idx]);
                        if (mate_winning) key = wins ? -r : 50 - r;      /* partenaire maitre : petite carte */
                        else key = wins ? 100 - r - (s == trump ? 13 : 0) : 30 - r; /* prendre au moins cher */
                    }
                    moves[n] = c;
                    keys[n] = key;
                    n++;
                }
                prev_mine = 1;
            } else {
                prev_mine = 0;
            }
        }
    }
    for (int i = 1; i < n; i++) {  /* tri par cle decroissante */
        int m = moves[i], k = keys[i], j = i - 1;
        while (j >= 0 && keys[j] < k) { moves[j + 1] = moves[j]; keys[j + 1] = keys[j]; j--; }
        moves[j + 1] = m;
        keys[j + 1] = k;
    }
    return n;
}

static int search(int alpha, int beta) {
    nodes++;
    int turn = (leader_s + nt) & 3;
    Entry *e = NULL;
    if (nt == 0) {
        int R = popc(H[leader_s]);
        if (R == 0) return 0;
        if (R == 1) {  /* dernier pli : chaque main a une carte */
            for (int i = 0; i < 4; i++) {
                int s = (leader_s + i) & 3;
                tcard[i] = __builtin_ctzll(H[s]);
                tseat[i] = s;
            }
            return side[tseat[trick_winner_idx(4)]];
        }
        e = &tt[tt_index(leader_s)];
        if (e->gen == gen && e->leader == leader_s && e->h[0] == H[0] && e->h[1] == H[1] && e->h[2] == H[2]
            && e->h[3] == H[3]) {
            if (e->lb >= beta) return e->lb;
            if (e->ub <= alpha) return e->ub;
            if (e->lb > alpha) alpha = e->lb;
            if (e->ub < beta) beta = e->ub;
        }
    }
    int moves[13];
    int n = gen_moves(turn, moves);
    int maximizing = side[turn];
    int best = maximizing ? -1 : 99;
    int a = alpha, b = beta;
    for (int i = 0; i < n; i++) {
        int m = moves[i];
        H[turn] &= ~(1ULL << m);
        tcard[nt] = m;
        tseat[nt] = turn;
        nt++;
        int v;
        if (nt == 4) {
            int w = tseat[trick_winner_idx(4)];
            int add = side[w];
            int sc[4], ss[4], sl = leader_s;
            memcpy(sc, tcard, sizeof sc);
            memcpy(ss, tseat, sizeof ss);
            nt = 0;
            leader_s = w;
            v = add + search(a - add, b - add);
            leader_s = sl;
            memcpy(tcard, sc, sizeof sc);
            memcpy(tseat, ss, sizeof ss);
            nt = 4;
        } else {
            v = search(a, b);
        }
        nt--;
        H[turn] |= 1ULL << m;
        if (maximizing) {
            if (v > best) best = v;
            if (best > a) a = best;
        } else {
            if (v < best) best = v;
            if (best < b) b = best;
        }
        if (a >= b) break;
    }
    if (e != NULL) {
        if (!(e->gen == gen && e->leader == leader_s && e->h[0] == H[0] && e->h[1] == H[1] && e->h[2] == H[2]
              && e->h[3] == H[3])) {
            memcpy(e->h, H, sizeof H);
            e->gen = gen;
            e->leader = (int8_t)leader_s;
            e->lb = 0;
            e->ub = (int8_t)popc(H[leader_s]);
        }
        if (best <= alpha) { if (best < e->ub) e->ub = (int8_t)best; }
        else if (best >= beta) { if (best > e->lb) e->lb = (int8_t)best; }
        else { e->lb = (int8_t)best; e->ub = (int8_t)best; }
    }
    return best;
}

/* Plis restants (pli en cours compris) du camp A depuis la position :
 * hands[4] (masques), trump (-1 = sans atout), side_a (bit s = siege s dans
 * A), leader du pli en cours, n_trick cartes deja posees (trick_cards /
 * trick_seats, dans l'ordre). new_gen = 0 garde la table (meme atout et
 * memes camps que l'appel precedent). */
long long dds_nodes(void) { return nodes; }

int dds_solve(const u64 *hands, int trump_, int side_a, int leader, int n_trick, const int *trick_cards,
              const int *trick_seats, int new_gen) {
    if (tt == NULL) {
        tt = (Entry *)calloc(TT_SIZE, sizeof(Entry));
        if (tt == NULL) return -1;
    }
    if (new_gen || gen == 0) gen++;
    trump = trump_;
    for (int s = 0; s < 4; s++) {
        side[s] = (side_a >> s) & 1;
        H[s] = hands[s];
    }
    nt = n_trick;
    leader_s = leader;
    for (int i = 0; i < n_trick; i++) {
        tcard[i] = trick_cards[i];
        tseat[i] = trick_seats[i];
    }
    int turn = (leader + n_trick) & 3;
    int hi = popc(H[turn]);
    int lo = 0;
    if (hi == 0) return 0;
    while (lo < hi) {  /* dichotomie par sondes a fenetre nulle */
        int mid = (lo + hi + 1) / 2;
        nt = n_trick;
        leader_s = leader;
        for (int i = 0; i < n_trick; i++) {
            tcard[i] = trick_cards[i];
            tseat[i] = trick_seats[i];
        }
        int v = search(mid - 1, mid);
        if (v >= mid) {  /* fail-soft : valeur >= v */
            lo = v < hi ? v : hi;
        } else {         /* valeur <= v */
            hi = v > lo ? v : lo;
        }
    }
    return lo;
}
