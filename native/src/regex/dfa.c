/* Port of regex 1.8.4 src/dfa.rs (with src/sparse.rs's SparseSet), MIT OR Apache-2.0. */
/* The lazy DFA: states are built from the byte program during a search and cached, with
 * dfa.rs's state keys (flags byte + delta-varint instruction pointers), transition table,
 * start-state cache, size accounting against dfa_size_limit, cache flushes and the
 * flush-rate Quit, the Unicode \b Quit transitions for non-ASCII bytes, prefix scanning in
 * start states, and the unrolled inner loops. Only single-regex, leftmost-first or reverse
 * searches are reached (quit_after_match is false), so the regex-set branches are not
 * ported. Rust's StateMap HashMap<State, StatePtr> is an open-addressing table here. */
#include "prog.h"
#include <stdlib.h>
#include <string.h>

typedef uint32_t state_ptr;

#define STATE_UNKNOWN ((state_ptr)1 << 31)
#define STATE_DEAD (STATE_UNKNOWN + 1)
#define STATE_QUIT (STATE_DEAD + 1)
#define STATE_START ((state_ptr)1 << 30)
#define STATE_MATCH ((state_ptr)1 << 29)
#define STATE_MAX (STATE_MATCH - 1)

/* StateFlags */
#define FLAG_MATCH 0x01u
#define FLAG_WORD 0x02u
#define FLAG_EMPTY 0x04u

/* Byte::eof */
#define BYTE_EOF 256

/* Rust's mem::size_of::<State>() (an Arc<[u8]>) and of StatePtr / InstPtr. */
#define RUST_STATE_SIZE ((size_t)16)
#define RUST_PTR_SIZE ((size_t)4)

/* EmptyFlags */
typedef struct empty_flags {
    bool start, end, start_line, end_line, word_boundary, not_word_boundary;
} empty_flags;

/* SparseSet */
typedef struct sparse_set {
    uint32_t *dense;
    uint32_t *sparse;
    size_t len;
} sparse_set;

/* A state's data in the arena. */
typedef struct state_ref {
    size_t off;
    size_t len;
} state_ref;

/* Cache { inner: CacheInner, qcur, qnext } */
struct chn_dfa_cache {
    size_t num_byte_classes;
    /* StateMap */
    state_ref *states;
    size_t states_len;
    size_t states_cap;
    uint8_t *arena;
    size_t arena_len;
    size_t arena_cap;
    uint32_t *map; /* open addressing: state index + 1, 0 is empty */
    size_t map_cap;
    /* Transitions */
    state_ptr *trans;
    size_t trans_len;
    size_t trans_cap;
    state_ptr start_states[256];
    uint32_t *stack;
    size_t stack_cap;
    uint64_t flush_count;
    size_t size;
    uint8_t *scratch; /* insts_scratch_space */
    size_t scratch_cap;
    sparse_set qcur;
    sparse_set qnext;
};

/* Fsm */
typedef struct fsm {
    const chn_prog *prog;
    const chn_literal_searcher *prefixes;
    state_ptr start;
    size_t at;
    state_ptr last_match_si;
    size_t last_cache_flush;
    chn_dfa_cache *cache;
    bool oom;
} fsm;

void chn_dfa_cache_free(chn_dfa_cache *cache) {
    if (!cache) return;
    free(cache->states);
    free(cache->arena);
    free(cache->map);
    free(cache->trans);
    free(cache->stack);
    free(cache->scratch);
    free(cache->qcur.dense);
    free(cache->qcur.sparse);
    free(cache->qnext.dense);
    free(cache->qnext.sparse);
    free(cache);
}

bool chn_dfa_can_exec(const chn_prog *prog) {
    if (prog->dfa_size_limit == 0 || prog->len > (size_t)INT32_MAX) return false;
    for (size_t i = 0; i < prog->len; ++i) {
        if (prog->insts[i].kind == CHN_INST_CHAR || prog->insts[i].kind == CHN_INST_RANGES) return false;
    }
    return true;
}

/* CacheInner::reset_size (the stack is empty between epsilon walks). */
static void reset_size(chn_dfa_cache *c) {
    c->size = 256 * RUST_PTR_SIZE;
}

/* Cache::new */
static chn_dfa_cache *cache_new(const chn_prog *prog) {
    chn_dfa_cache *c = (chn_dfa_cache *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->num_byte_classes = (size_t)prog->byte_classes[255] + 1 + 1;
    for (size_t i = 0; i < 256; ++i) c->start_states[i] = STATE_UNKNOWN;
    const size_t n = prog->len ? prog->len : 1;
    c->qcur.dense = (uint32_t *)malloc(n * sizeof(uint32_t));
    c->qcur.sparse = (uint32_t *)calloc(n, sizeof(uint32_t));
    c->qnext.dense = (uint32_t *)malloc(n * sizeof(uint32_t));
    c->qnext.sparse = (uint32_t *)calloc(n, sizeof(uint32_t));
    c->stack_cap = n + 1;
    c->stack = (uint32_t *)malloc(c->stack_cap * sizeof(uint32_t));
    c->map_cap = 64;
    c->map = (uint32_t *)calloc(c->map_cap, sizeof(uint32_t));
    if (!c->qcur.dense || !c->qcur.sparse || !c->qnext.dense || !c->qnext.sparse || !c->stack || !c->map) {
        chn_dfa_cache_free(c);
        return NULL;
    }
    reset_size(c);
    return c;
}

static inline bool set_contains(const sparse_set *s, uint32_t v) {
    const uint32_t i = s->sparse[v];
    return i < s->len && s->dense[i] == v;
}

static inline void set_insert(sparse_set *s, uint32_t v) {
    s->dense[s->len] = v;
    s->sparse[v] = (uint32_t)s->len;
    ++s->len;
}

/* ---- the state map ---- */

static uint64_t hash_bytes(const uint8_t *p, size_t n) {
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h ^ (h >> 29);
}

static const uint8_t *state_data(const chn_dfa_cache *c, size_t index, size_t *len) {
    *len = c->states[index].len;
    return c->arena + c->states[index].off;
}

/* StateMap::get_ptr */
static bool map_get(const chn_dfa_cache *c, const uint8_t *key, size_t len, state_ptr *si) {
    const size_t mask = c->map_cap - 1;
    for (size_t i = (size_t)hash_bytes(key, len) & mask;; i = (i + 1) & mask) {
        const uint32_t slot = c->map[i];
        if (!slot) return false;
        size_t n;
        const uint8_t *d = state_data(c, slot - 1, &n);
        if (n == len && memcmp(d, key, len) == 0) {
            *si = (state_ptr)((slot - 1) * c->num_byte_classes);
            return true;
        }
    }
}

static void map_place(chn_dfa_cache *c, size_t index) {
    const size_t mask = c->map_cap - 1;
    size_t n;
    const uint8_t *d = state_data(c, index, &n);
    size_t i = (size_t)hash_bytes(d, n) & mask;
    while (c->map[i]) i = (i + 1) & mask;
    c->map[i] = (uint32_t)(index + 1);
}

/* StateMap::insert (the state's data goes to the arena). */
static bool map_insert(chn_dfa_cache *c, const uint8_t *key, size_t len) {
    if (c->arena_cap - c->arena_len < len) {
        size_t cap = c->arena_cap ? c->arena_cap : 4096;
        while (cap - c->arena_len < len) cap *= 2;
        uint8_t *arena = (uint8_t *)realloc(c->arena, cap);
        if (!arena) return false;
        c->arena = arena;
        c->arena_cap = cap;
    }
    if (c->states_len == c->states_cap) {
        const size_t cap = c->states_cap ? c->states_cap * 2 : 64;
        state_ref *states = (state_ref *)realloc(c->states, cap * sizeof(*states));
        if (!states) return false;
        c->states = states;
        c->states_cap = cap;
    }
    if ((c->states_len + 1) * 2 > c->map_cap) {
        const size_t cap = c->map_cap * 2;
        uint32_t *map = (uint32_t *)calloc(cap, sizeof(uint32_t));
        if (!map) return false;
        free(c->map);
        c->map = map;
        c->map_cap = cap;
        for (size_t i = 0; i < c->states_len; ++i) map_place(c, i);
    }
    memcpy(c->arena + c->arena_len, key, len);
    c->states[c->states_len].off = c->arena_len;
    c->states[c->states_len].len = len;
    c->arena_len += len;
    map_place(c, c->states_len);
    ++c->states_len;
    return true;
}

/* StateMap::clear */
static void map_clear(chn_dfa_cache *c) {
    c->states_len = 0;
    c->arena_len = 0;
    memset(c->map, 0, c->map_cap * sizeof(uint32_t));
}

/* ---- Fsm helpers ---- */

static inline size_t byte_class(const fsm *f, int b) {
    return b == BYTE_EOF ? f->cache->num_byte_classes - 1 : f->prog->byte_classes[b];
}

static inline bool is_ascii_word(int b) {
    return b != BYTE_EOF && chn_hir_is_word_byte((uint8_t)b);
}

static inline state_ptr next_si(const fsm *f, state_ptr si, const uint8_t *text, size_t i) {
    return f->cache->trans[si + f->prog->byte_classes[text[i]]];
}

/* The flags byte of the state at si. */
static inline uint8_t state_flags_of(const fsm *f, state_ptr si) {
    return f->cache->arena[f->cache->states[si / f->cache->num_byte_classes].off];
}

static bool has_prefix(const fsm *f) {
    return !f->prog->is_reverse && !chn_literal_searcher_is_empty(f->prefixes) && !f->prog->is_anchored_start;
}

/* Fsm::start_ptr */
static state_ptr start_ptr(const fsm *f, state_ptr si) {
    return has_prefix(f) ? si | STATE_START : si;
}

/* Fsm::follow_epsilons */
static void follow_epsilons(fsm *f, uint32_t ip0, sparse_set *q, empty_flags flags) {
    const chn_prog *prog = f->prog;
    uint32_t *stack = f->cache->stack;
    size_t len = 0;
    stack[len++] = ip0;
    while (len) {
        uint32_t ip = stack[--len];
        for (;;) {
            if (set_contains(q, ip)) break;
            set_insert(q, ip);
            const chn_inst *inst = &prog->insts[ip];
            if (inst->kind == CHN_INST_MATCH || inst->kind == CHN_INST_BYTES) break;
            if (inst->kind == CHN_INST_EMPTY_LOOK) {
                bool ok;
                switch ((chn_empty_look)inst->look) {
                case CHN_LOOK_START_LINE: ok = flags.start_line; break;
                case CHN_LOOK_END_LINE: ok = flags.end_line; break;
                case CHN_LOOK_START_TEXT: ok = flags.start; break;
                case CHN_LOOK_END_TEXT: ok = flags.end; break;
                case CHN_LOOK_WORD_BOUNDARY_ASCII:
                case CHN_LOOK_WORD_BOUNDARY: ok = flags.word_boundary; break;
                default: ok = flags.not_word_boundary; break;
                }
                if (!ok) break;
                ip = (uint32_t)inst->next;
            } else if (inst->kind == CHN_INST_SAVE) {
                ip = (uint32_t)inst->next;
            } else {
                /* Split: every instruction is inserted at most once, so the stack holds
                 * at most prog.len() + 1 entries. */
                stack[len++] = (uint32_t)inst->next2;
                ip = (uint32_t)inst->next;
            }
        }
    }
}

/* push_inst_ptr: delta-encoded with write_vari32 / write_varu32. */
static size_t push_inst_ptr(uint8_t *out, uint32_t *prev, uint32_t ip) {
    const int32_t delta = (int32_t)ip - (int32_t)*prev;
    uint32_t un = (uint32_t)delta << 1;
    if (delta < 0) un = ~un;
    size_t n = 0;
    while (un >= 0x80) {
        out[n++] = (uint8_t)(un | 0x80);
        un >>= 7;
    }
    out[n++] = (uint8_t)un;
    *prev = ip;
    return n;
}

/* read_vari32 over a state's data from *pos. */
static uint32_t read_inst_ptr(const uint8_t *data, size_t *pos, uint32_t base) {
    uint32_t un = 0, shift = 0;
    for (;;) {
        const uint8_t b = data[(*pos)++];
        if (b < 0x80) {
            un |= (uint32_t)b << shift;
            break;
        }
        un |= ((uint32_t)b & 0x7F) << shift;
        shift += 7;
    }
    int32_t n = (int32_t)(un >> 1);
    if (un & 1) n = ~n;
    return (uint32_t)((int32_t)base + n);
}

/* Fsm::cached_state_key: the key in the scratch space; false is None. */
static bool cached_state_key(fsm *f, const sparse_set *q, uint8_t *state_flags, size_t *key_len) {
    chn_dfa_cache *c = f->cache;
    /* Each pointer takes at most 5 bytes. */
    const size_t need = 1 + q->len * 5;
    if (c->scratch_cap < need) {
        uint8_t *s = (uint8_t *)realloc(c->scratch, need);
        if (!s) {
            f->oom = true;
            return false;
        }
        c->scratch = s;
        c->scratch_cap = need;
    }
    uint8_t *insts = c->scratch;
    size_t n = 1;
    uint32_t prev = 0;
    for (size_t i = 0; i < q->len; ++i) {
        const uint32_t ip = q->dense[i];
        const chn_inst *inst = &f->prog->insts[ip];
        if (inst->kind == CHN_INST_SAVE || inst->kind == CHN_INST_SPLIT) continue;
        if (inst->kind == CHN_INST_EMPTY_LOOK) *state_flags |= FLAG_EMPTY;
        n += push_inst_ptr(insts + n, &prev, ip);
        if (inst->kind == CHN_INST_MATCH && !f->prog->is_reverse) break;
    }
    if (n == 1 && !(*state_flags & FLAG_MATCH)) return false;
    insts[0] = *state_flags;
    *key_len = n;
    return true;
}

/* Fsm::add_state: false is None (state limit or out of memory). */
static bool add_state(fsm *f, const uint8_t *key, size_t len, state_ptr *out) {
    chn_dfa_cache *c = f->cache;
    const size_t si = c->trans_len;
    if (si > STATE_MAX) return false;
    const size_t nbc = c->num_byte_classes;
    if (c->trans_cap - c->trans_len < nbc) {
        size_t cap = c->trans_cap ? c->trans_cap : nbc * 64;
        while (cap - c->trans_len < nbc) cap *= 2;
        state_ptr *t = (state_ptr *)realloc(c->trans, cap * sizeof(state_ptr));
        if (!t) {
            f->oom = true;
            return false;
        }
        c->trans = t;
        c->trans_cap = cap;
    }
    for (size_t k = 0; k < nbc; ++k) c->trans[si + k] = STATE_UNKNOWN;
    c->trans_len += nbc;
    if (f->prog->has_unicode_word_boundary) {
        for (unsigned b = 128; b < 256; ++b) c->trans[si + byte_class(f, (int)b)] = STATE_QUIT;
    }
    c->size += nbc * RUST_PTR_SIZE + len + 2 * RUST_STATE_SIZE + RUST_PTR_SIZE;
    if (!map_insert(c, key, len)) {
        f->oom = true;
        return false;
    }
    *out = (state_ptr)si;
    return true;
}

/* Fsm::restore_state */
static bool restore_state(fsm *f, const uint8_t *data, size_t len, state_ptr *out) {
    if (map_get(f->cache, data, len, out)) return true;
    return add_state(f, data, len, out);
}

/* A copy of a state's data (State::clone across a cache wipe). */
static uint8_t *copy_state(fsm *f, state_ptr si, size_t *len) {
    const uint8_t *d = state_data(f->cache, si / f->cache->num_byte_classes, len);
    uint8_t *copy = (uint8_t *)malloc(*len);
    if (!copy) {
        f->oom = true;
        return NULL;
    }
    memcpy(copy, d, *len);
    return copy;
}

/* Fsm::clear_cache */
static bool clear_cache(fsm *f) {
    chn_dfa_cache *c = f->cache;
    const size_t nstates = c->states_len;
    if (c->flush_count >= 3 && f->at >= f->last_cache_flush && (f->at - f->last_cache_flush) <= 10 * nstates) {
        return false;
    }
    f->last_cache_flush = f->at;
    c->flush_count += 1;
    size_t start_len, last_len = 0;
    uint8_t *start = copy_state(f, f->start & ~STATE_START, &start_len);
    if (!start) return false;
    uint8_t *last = NULL;
    if (f->last_match_si <= STATE_MAX) {
        last = copy_state(f, f->last_match_si, &last_len);
        if (!last) {
            free(start);
            return false;
        }
    }
    reset_size(c);
    c->trans_len = 0;
    map_clear(c);
    for (size_t i = 0; i < 256; ++i) c->start_states[i] = STATE_UNKNOWN;
    state_ptr sp;
    bool ok = restore_state(f, start, start_len, &sp);
    if (ok) f->start = start_ptr(f, sp);
    if (ok && last) {
        ok = restore_state(f, last, last_len, &sp);
        if (ok) f->last_match_si = sp;
    }
    free(start);
    free(last);
    if (!ok) f->oom = true;
    return ok;
}

/* Fsm::clear_cache_and_save */
static bool clear_cache_and_save(fsm *f, state_ptr *current) {
    if (f->cache->states_len == 0) return true;
    if (!current) return clear_cache(f);
    size_t len;
    uint8_t *cur = copy_state(f, *current, &len);
    if (!cur) return false;
    bool ok = clear_cache(f);
    if (ok) {
        ok = restore_state(f, cur, len, current);
        if (!ok) f->oom = true;
    }
    free(cur);
    return ok;
}

/* Fsm::cached_state: false is None; *out may be STATE_DEAD. */
static bool cached_state(fsm *f, const sparse_set *q, uint8_t state_flags, state_ptr *current, state_ptr *out) {
    size_t len;
    if (!cached_state_key(f, q, &state_flags, &len)) {
        if (f->oom) return false;
        *out = STATE_DEAD;
        return true;
    }
    chn_dfa_cache *c = f->cache;
    if (map_get(c, c->scratch, len, out)) return true;
    if (c->size > f->prog->dfa_size_limit && !clear_cache_and_save(f, current)) return false;
    return add_state(f, c->scratch, len, out);
}

/* Fsm::exec_byte: false is None. */
static bool exec_byte(fsm *f, state_ptr si, int b, state_ptr *out) {
    chn_dfa_cache *c = f->cache;
    sparse_set *qcur = &c->qcur, *qnext = &c->qnext;
    size_t dlen;
    const uint8_t *data = state_data(c, si / c->num_byte_classes, &dlen);
    const uint8_t sflags = data[0];
    qcur->len = 0;
    {
        size_t pos = 1;
        uint32_t base = 0;
        while (pos < dlen) {
            base = read_inst_ptr(data, &pos, base);
            set_insert(qcur, base);
        }
    }
    const bool is_word_last = (sflags & FLAG_WORD) != 0;
    const bool is_word = is_ascii_word(b);
    if (sflags & FLAG_EMPTY) {
        empty_flags flags = {false, false, false, false, false, false};
        if (b == BYTE_EOF) {
            flags.end = true;
            flags.end_line = true;
        } else if (b == '\n') {
            flags.end_line = true;
        }
        if (is_word_last == is_word) flags.not_word_boundary = true;
        else flags.word_boundary = true;
        qnext->len = 0;
        for (size_t i = 0; i < qcur->len; ++i) follow_epsilons(f, qcur->dense[i], qnext, flags);
        const sparse_set tmp = *qcur;
        *qcur = *qnext;
        *qnext = tmp;
    }
    empty_flags eflags = {false, false, false, false, false, false};
    uint8_t state_flags = 0;
    eflags.start_line = b == '\n';
    if (is_ascii_word(b)) state_flags |= FLAG_WORD;
    qnext->len = 0;
    for (size_t i = 0; i < qcur->len; ++i) {
        const uint32_t ip = qcur->dense[i];
        const chn_inst *inst = &f->prog->insts[ip];
        if (inst->kind == CHN_INST_MATCH) {
            state_flags |= FLAG_MATCH;
            if (!f->prog->is_reverse) break;
        } else if (inst->kind == CHN_INST_BYTES) {
            if (b != BYTE_EOF && inst->start <= (uint8_t)b && (uint8_t)b <= inst->end) {
                follow_epsilons(f, (uint32_t)inst->next, qnext, eflags);
            }
        }
    }
    state_ptr next;
    if (!cached_state(f, qnext, state_flags, &si, &next)) return false;
    if ((f->start & ~STATE_START) == next) next = start_ptr(f, next);
    if (next <= STATE_MAX && (state_flags_of(f, next) & FLAG_MATCH)) next |= STATE_MATCH;
    c->trans[si + byte_class(f, b)] = next;
    *out = next;
    return true;
}

/* Fsm::next_state: false is None. */
static bool next_state(fsm *f, state_ptr si, int b, state_ptr *out) {
    if (si == STATE_DEAD) {
        *out = STATE_DEAD;
        return true;
    }
    const state_ptr nsi = f->cache->trans[si + byte_class(f, b)];
    if (nsi == STATE_UNKNOWN) return exec_byte(f, si, b, out);
    if (nsi == STATE_QUIT) return false;
    *out = nsi;
    return true;
}

/* Fsm::start_state: false is None; *out may be STATE_DEAD. */
static bool start_state(fsm *f, empty_flags ef, uint8_t state_flags, state_ptr *out) {
    const size_t flagi = (size_t)ef.start | ((size_t)ef.end << 1) | ((size_t)ef.start_line << 2)
        | ((size_t)ef.end_line << 3) | ((size_t)ef.word_boundary << 4) | ((size_t)ef.not_word_boundary << 5)
        | ((size_t)((state_flags & FLAG_WORD) != 0) << 6);
    chn_dfa_cache *c = f->cache;
    if (c->start_states[flagi] != STATE_UNKNOWN) {
        *out = c->start_states[flagi];
        return true;
    }
    c->qcur.len = 0;
    follow_epsilons(f, (uint32_t)f->prog->start, &c->qcur, ef);
    state_ptr sp;
    if (!cached_state(f, &c->qcur, state_flags, NULL, &sp)) return false;
    sp = start_ptr(f, sp);
    c->start_states[flagi] = sp;
    *out = sp;
    return true;
}

/* Fsm::prefix_at */
static bool prefix_at(const fsm *f, const uint8_t *text, size_t len, size_t at, size_t *out) {
    size_t s, e;
    if (!chn_literal_searcher_find(f->prefixes, text + at, len - at, &s, &e)) return false;
    *out = at + s;
    return true;
}

static chn_dfa_result result_of(chn_dfa_status status, size_t value) {
    const chn_dfa_result r = {status, value};
    return r;
}

static chn_dfa_result quit(const fsm *f) {
    return result_of(f->oom ? CHN_DFA_NOMEM : CHN_DFA_QUIT, 0);
}

/* Fsm::exec_at */
static chn_dfa_result exec_at(fsm *f, const uint8_t *text, size_t len) {
    chn_dfa_result result = result_of(CHN_DFA_NO_MATCH, f->at);
    state_ptr prev_si = f->start, next = f->start;
    size_t at = f->at;
    while (at < len) {
        while (next <= STATE_MAX && at < len) {
            prev_si = next_si(f, next, text, at);
            at += 1;
            if (prev_si > STATE_MAX || at + 2 >= len) {
                const state_ptr t = prev_si;
                prev_si = next;
                next = t;
                break;
            }
            next = next_si(f, prev_si, text, at);
            at += 1;
            if (next > STATE_MAX) break;
            prev_si = next_si(f, next, text, at);
            at += 1;
            if (prev_si > STATE_MAX) {
                const state_ptr t = prev_si;
                prev_si = next;
                next = t;
                break;
            }
            next = next_si(f, prev_si, text, at);
            at += 1;
        }
        if (next & STATE_MATCH) {
            next &= ~STATE_MATCH;
            result = result_of(CHN_DFA_MATCH, at - 1);
            f->last_match_si = next;
            prev_si = next;
            const size_t cur = at;
            while ((next & ~STATE_MATCH) == prev_si && at + 2 < len) {
                next = next_si(f, next & ~STATE_MATCH, text, at);
                at += 1;
            }
            if (at > cur) result = result_of(CHN_DFA_MATCH, at - 2);
        } else if (next & STATE_START) {
            next &= ~STATE_START;
            prev_si = next;
            size_t i;
            if (!prefix_at(f, text, len, at, &i)) return result_of(CHN_DFA_NO_MATCH, len);
            at = i;
        } else if (next >= STATE_UNKNOWN) {
            if (next == STATE_QUIT) return quit(f);
            if (at == 0) abort(); /* text[at - 1]: unreachable, a start state is never dead here */
            const int byte = text[at - 1];
            prev_si &= STATE_MAX;
            f->at = at;
            if (!next_state(f, prev_si, byte, &next)) return quit(f);
            if (next == STATE_DEAD) {
                if (result.status == CHN_DFA_NO_MATCH) result.value = at;
                return result;
            }
            if (next & STATE_MATCH) {
                next &= ~STATE_MATCH;
                result = result_of(CHN_DFA_MATCH, at - 1);
                f->last_match_si = next;
            }
            prev_si = next;
        } else {
            prev_si = next;
        }
    }
    prev_si &= STATE_MAX;
    if (!next_state(f, prev_si, BYTE_EOF, &prev_si)) return quit(f);
    if (prev_si == STATE_DEAD) {
        if (result.status == CHN_DFA_NO_MATCH) result.value = len;
        return result;
    }
    prev_si &= ~STATE_START;
    if (prev_si & STATE_MATCH) {
        prev_si &= ~STATE_MATCH;
        f->last_match_si = prev_si;
        result = result_of(CHN_DFA_MATCH, len);
    }
    return result;
}

/* Fsm::exec_at_reverse */
static chn_dfa_result exec_at_reverse(fsm *f, const uint8_t *text) {
    chn_dfa_result result = result_of(CHN_DFA_NO_MATCH, f->at);
    state_ptr prev_si = f->start, next = f->start;
    size_t at = f->at;
    while (at > 0) {
        while (next <= STATE_MAX && at > 0) {
            at -= 1;
            prev_si = next_si(f, next, text, at);
            if (prev_si > STATE_MAX || at <= 4) {
                const state_ptr t = prev_si;
                prev_si = next;
                next = t;
                break;
            }
            at -= 1;
            next = next_si(f, prev_si, text, at);
            if (next > STATE_MAX) break;
            at -= 1;
            prev_si = next_si(f, next, text, at);
            if (prev_si > STATE_MAX) {
                const state_ptr t = prev_si;
                prev_si = next;
                next = t;
                break;
            }
            at -= 1;
            next = next_si(f, prev_si, text, at);
        }
        if (next & STATE_MATCH) {
            next &= ~STATE_MATCH;
            result = result_of(CHN_DFA_MATCH, at + 1);
            f->last_match_si = next;
            prev_si = next;
            const size_t cur = at;
            while ((next & ~STATE_MATCH) == prev_si && at >= 2) {
                at -= 1;
                next = next_si(f, next & ~STATE_MATCH, text, at);
            }
            if (at < cur) result = result_of(CHN_DFA_MATCH, at + 2);
        } else if (next >= STATE_UNKNOWN) {
            if (next == STATE_QUIT) return quit(f);
            const int byte = text[at];
            prev_si &= STATE_MAX;
            f->at = at;
            if (!next_state(f, prev_si, byte, &next)) return quit(f);
            if (next == STATE_DEAD) {
                if (result.status == CHN_DFA_NO_MATCH) result.value = at;
                return result;
            }
            if (next & STATE_MATCH) {
                next &= ~STATE_MATCH;
                result = result_of(CHN_DFA_MATCH, at + 1);
                f->last_match_si = next;
            }
            prev_si = next;
        } else {
            prev_si = next;
        }
    }
    if (!next_state(f, prev_si, BYTE_EOF, &prev_si)) return quit(f);
    if (prev_si == STATE_DEAD) {
        if (result.status == CHN_DFA_NO_MATCH) result.value = 0;
        return result;
    }
    if (prev_si & STATE_MATCH) {
        prev_si &= ~STATE_MATCH;
        f->last_match_si = prev_si;
        result = result_of(CHN_DFA_MATCH, 0);
    }
    return result;
}

static bool ensure_cache(const chn_prog *prog, chn_dfa_cache **cache) {
    if (!*cache) *cache = cache_new(prog);
    return *cache != NULL;
}

chn_dfa_result chn_dfa_forward(const chn_prog *prog, const chn_literal_searcher *prefixes,
    chn_dfa_cache **cache, const uint8_t *text, size_t len, size_t at) {
    if (!ensure_cache(prog, cache)) return result_of(CHN_DFA_NOMEM, 0);
    fsm f = {prog, prefixes, 0, at, STATE_UNKNOWN, at, *cache, false};
    /* Fsm::start_flags */
    empty_flags ef = {false, false, false, false, false, false};
    uint8_t state_flags = 0;
    ef.start = at == 0;
    ef.end = len == 0;
    ef.start_line = at == 0 || text[at - 1] == '\n';
    ef.end_line = len == 0;
    const bool is_word_last = at > 0 && is_ascii_word(text[at - 1]);
    const bool is_word = at < len && is_ascii_word(text[at]);
    if (is_word_last) state_flags |= FLAG_WORD;
    if (is_word == is_word_last) ef.not_word_boundary = true;
    else ef.word_boundary = true;
    state_ptr start;
    if (!start_state(&f, ef, state_flags, &start)) return quit(&f);
    if (start == STATE_DEAD) return result_of(CHN_DFA_NO_MATCH, at);
    f.start = start;
    return exec_at(&f, text, len);
}

chn_dfa_result chn_dfa_reverse(const chn_prog *prog, chn_dfa_cache **cache, const uint8_t *text,
    size_t len, size_t at) {
    if (!ensure_cache(prog, cache)) return result_of(CHN_DFA_NOMEM, 0);
    /* The reverse program's prefixes are LiteralSearcher::empty() (and has_prefix is false
     * for a reverse program anyway): a zeroed searcher is Matcher::Empty. */
    static const chn_literal_searcher no_prefixes = {0};
    fsm f = {prog, &no_prefixes, 0, at, STATE_UNKNOWN, at, *cache, false};
    /* Fsm::start_flags_reverse */
    empty_flags ef = {false, false, false, false, false, false};
    uint8_t state_flags = 0;
    ef.start = at == len;
    ef.end = len == 0;
    ef.start_line = at == len || text[at] == '\n';
    ef.end_line = len == 0;
    const bool is_word_last = at < len && is_ascii_word(text[at]);
    const bool is_word = at > 0 && is_ascii_word(text[at - 1]);
    if (is_word_last) state_flags |= FLAG_WORD;
    if (is_word == is_word_last) ef.not_word_boundary = true;
    else ef.word_boundary = true;
    state_ptr start;
    if (!start_state(&f, ef, state_flags, &start)) return quit(&f);
    if (start == STATE_DEAD) return result_of(CHN_DFA_NO_MATCH, at);
    f.start = start;
    return exec_at_reverse(&f, text);
}
