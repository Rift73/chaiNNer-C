/* Port of regex-syntax 0.7.2 src/hir/literal.rs (Extractor, Seq, Literal, PreferenceTrie) and regex 1.8.4 src/exec.rs (literal_analysis), MIT OR Apache-2.0. */
/* Only what regex 1.8.4 reaches is ported: Extractor with its default limits (prefix and
 * suffix), and the Seq/Literal methods exec.rs and literal/imp.rs call. Rust aborts on
 * allocation failure; here an out-of-memory flag travels through the extraction and the
 * caller gets false. */
#include "hir_literal.h"
#include <stdlib.h>
#include <string.h>

/* Extractor { kind, limit_class: 10, limit_repeat: 10, limit_literal_len: 100,
 * limit_total: 250 } plus the out-of-memory flag. */
typedef struct chn_extractor {
    bool suffix; /* ExtractKind::Suffix */
    bool oom;
} chn_extractor;

enum {
    LIMIT_CLASS = 10,
    LIMIT_REPEAT = 10,
    LIMIT_LITERAL_LEN = 100,
    LIMIT_TOTAL = 250
};

/* ---- Literal ---- */

static void lit_drop(chn_lit *lit) {
    free(lit->bytes);
    lit->bytes = NULL;
    lit->len = lit->cap = 0;
}

static bool lit_reserve(chn_lit *lit, size_t extra) {
    if (lit->cap - lit->len >= extra) return true;
    size_t cap = lit->cap ? lit->cap : 8;
    while (cap - lit->len < extra) cap *= 2;
    uint8_t *bytes = (uint8_t *)realloc(lit->bytes, cap);
    if (!bytes) return false;
    lit->bytes = bytes;
    lit->cap = cap;
    return true;
}

/* Literal::exact(bytes) (a copy) */
static chn_lit lit_exact(chn_extractor *x, const uint8_t *bytes, size_t len) {
    chn_lit lit = {NULL, 0, 0, true};
    if (len) {
        if (!lit_reserve(&lit, len)) {
            x->oom = true;
            return lit;
        }
        memcpy(lit.bytes, bytes, len);
        lit.len = len;
    }
    return lit;
}

static chn_lit lit_clone(chn_extractor *x, const chn_lit *src) {
    chn_lit lit = lit_exact(x, src->bytes, src->len);
    lit.exact = src->exact;
    return lit;
}

static bool lit_eq(const chn_lit *a, const chn_lit *b) {
    return a->len == b->len && a->exact == b->exact && (a->len == 0 || memcmp(a->bytes, b->bytes, a->len) == 0);
}

static bool lit_bytes_eq(const chn_lit *a, const chn_lit *b) {
    return a->len == b->len && (a->len == 0 || memcmp(a->bytes, b->bytes, a->len) == 0);
}

/* Literal::extend */
static void lit_extend(chn_extractor *x, chn_lit *lit, const chn_lit *other) {
    if (!lit->exact) return;
    if (!other->len) return;
    if (!lit_reserve(lit, other->len)) {
        x->oom = true;
        return;
    }
    memcpy(lit->bytes + lit->len, other->bytes, other->len);
    lit->len += other->len;
}

/* Literal::keep_first_bytes */
static void lit_keep_first_bytes(chn_lit *lit, size_t len) {
    if (len >= lit->len) return;
    lit->exact = false;
    lit->len = len;
}

/* Literal::keep_last_bytes */
static void lit_keep_last_bytes(chn_lit *lit, size_t len) {
    if (len >= lit->len) return;
    lit->exact = false;
    memmove(lit->bytes, lit->bytes + (lit->len - len), len);
    lit->len = len;
}

/* ---- Seq ---- */

void chn_seq_init_empty(chn_seq *seq) {
    seq->finite = true;
    seq->lits = NULL;
    seq->len = seq->cap = 0;
}

void chn_seq_drop(chn_seq *seq) {
    for (size_t i = 0; i < seq->len; ++i) lit_drop(&seq->lits[i]);
    free(seq->lits);
    chn_seq_init_empty(seq);
}

static chn_seq seq_empty(void) {
    chn_seq seq;
    chn_seq_init_empty(&seq);
    return seq;
}

/* Seq::infinite */
static chn_seq seq_infinite(void) {
    chn_seq seq = seq_empty();
    seq.finite = false;
    return seq;
}

/* Seq::make_infinite */
static void seq_make_infinite(chn_seq *seq) {
    chn_seq_drop(seq);
    seq->finite = false;
}

static bool seq_reserve(chn_seq *seq, size_t extra) {
    if (seq->cap - seq->len >= extra) return true;
    size_t cap = seq->cap ? seq->cap : 4;
    while (cap - seq->len < extra) cap *= 2;
    chn_lit *lits = (chn_lit *)realloc(seq->lits, cap * sizeof(*lits));
    if (!lits) return false;
    seq->lits = lits;
    seq->cap = cap;
    return true;
}

/* Vec::push of a literal the sequence takes over. */
static void seq_push_raw(chn_extractor *x, chn_seq *seq, chn_lit lit) {
    if (!seq_reserve(seq, 1)) {
        x->oom = true;
        lit_drop(&lit);
        return;
    }
    seq->lits[seq->len++] = lit;
}

/* Seq::singleton */
static chn_seq seq_singleton(chn_extractor *x, chn_lit lit) {
    chn_seq seq = seq_empty();
    seq_push_raw(x, &seq, lit);
    return seq;
}

/* Seq::push */
static void seq_push(chn_extractor *x, chn_seq *seq, chn_lit lit) {
    if (!seq->finite) {
        lit_drop(&lit);
        return;
    }
    if (seq->len && lit_eq(&seq->lits[seq->len - 1], &lit)) {
        lit_drop(&lit);
        return;
    }
    seq_push_raw(x, seq, lit);
}

void chn_seq_make_inexact(chn_seq *seq) {
    for (size_t i = 0; i < seq->len; ++i) seq->lits[i].exact = false;
}

/* Seq::dedup (Vec::dedup_by: a duplicate of the previous kept literal is dropped, and both
 * become inexact when their exactness differs). */
static void seq_dedup(chn_seq *seq) {
    if (!seq->finite || seq->len < 2) return;
    size_t w = 1;
    for (size_t r = 1; r < seq->len; ++r) {
        chn_lit *prev = &seq->lits[w - 1];
        chn_lit *cur = &seq->lits[r];
        if (lit_bytes_eq(cur, prev)) {
            if (cur->exact != prev->exact) {
                cur->exact = false;
                prev->exact = false;
            }
            lit_drop(cur);
            continue;
        }
        seq->lits[w++] = *cur;
    }
    seq->len = w;
}

/* Seq::keep_first_bytes / keep_last_bytes */
static void seq_keep_first_bytes(chn_seq *seq, size_t len) {
    for (size_t i = 0; i < seq->len; ++i) lit_keep_first_bytes(&seq->lits[i], len);
}

static void seq_keep_last_bytes(chn_seq *seq, size_t len) {
    for (size_t i = 0; i < seq->len; ++i) lit_keep_last_bytes(&seq->lits[i], len);
}

bool chn_seq_is_exact(const chn_seq *seq) {
    if (!seq->finite) return false;
    for (size_t i = 0; i < seq->len; ++i) {
        if (!seq->lits[i].exact) return false;
    }
    return true;
}

/* Seq::is_inexact */
static bool seq_is_inexact(const chn_seq *seq) {
    if (!seq->finite) return true;
    for (size_t i = 0; i < seq->len; ++i) {
        if (seq->lits[i].exact) return false;
    }
    return true;
}

bool chn_seq_min_literal_len(const chn_seq *seq, size_t *out) {
    if (!seq->finite || seq->len == 0) return false;
    size_t min = seq->lits[0].len;
    for (size_t i = 1; i < seq->len; ++i) {
        if (seq->lits[i].len < min) min = seq->lits[i].len;
    }
    *out = min;
    return true;
}

/* Seq::max_union_len / max_cross_len: false is None. */
static bool seq_max_union_len(const chn_seq *a, const chn_seq *b, size_t *out) {
    if (!a->finite || !b->finite) return false;
    const size_t sum = a->len + b->len;
    *out = sum < a->len ? SIZE_MAX : sum;
    return true;
}

static bool seq_max_cross_len(const chn_seq *a, const chn_seq *b, size_t *out) {
    if (!a->finite || !b->finite) return false;
    if (a->len != 0 && b->len > SIZE_MAX / a->len) *out = SIZE_MAX;
    else *out = a->len * b->len;
    return true;
}

bool chn_seq_longest_common_prefix(const chn_seq *seq, const uint8_t **ptr, size_t *len) {
    if (!seq->finite || seq->len == 0) return false;
    const chn_lit *base = &seq->lits[0];
    size_t n = base->len;
    for (size_t i = 1; i < seq->len; ++i) {
        const chn_lit *m = &seq->lits[i];
        size_t k = 0;
        while (k < m->len && k < n && m->bytes[k] == base->bytes[k]) ++k;
        n = k;
        if (n == 0) break;
    }
    *ptr = base->bytes;
    *len = n;
    return true;
}

bool chn_seq_longest_common_suffix(const chn_seq *seq, const uint8_t **ptr, size_t *len) {
    if (!seq->finite || seq->len == 0) return false;
    const chn_lit *base = &seq->lits[0];
    size_t n = base->len;
    for (size_t i = 1; i < seq->len; ++i) {
        const chn_lit *m = &seq->lits[i];
        size_t k = 0;
        while (k < m->len && k < n && m->bytes[m->len - 1 - k] == base->bytes[base->len - 1 - k]) ++k;
        n = k;
        if (n == 0) break;
    }
    *ptr = base->bytes ? base->bytes + (base->len - n) : NULL;
    *len = n;
    return true;
}

/* Seq::cross_preamble: true when both sides are finite and the cross must run. */
static bool seq_cross_preamble(chn_seq *self, chn_seq *other) {
    if (!other->finite) {
        size_t min;
        if (chn_seq_min_literal_len(self, &min) && min == 0) {
            seq_make_infinite(self);
        } else {
            chn_seq_make_inexact(self);
        }
        return false;
    }
    if (!self->finite) {
        chn_seq_drop(other);
        return false;
    }
    return true;
}

/* Seq::cross_forward */
static void seq_cross_forward(chn_extractor *x, chn_seq *self, chn_seq *other) {
    if (!seq_cross_preamble(self, other)) return;
    chn_seq old = *self;
    chn_seq_init_empty(self);
    for (size_t i = 0; i < old.len; ++i) {
        chn_lit *selflit = &old.lits[i];
        if (!selflit->exact) {
            seq_push_raw(x, self, *selflit);
            selflit->bytes = NULL;
            continue;
        }
        for (size_t j = 0; j < other->len; ++j) {
            const chn_lit *otherlit = &other->lits[j];
            chn_lit newlit = {NULL, 0, 0, true};
            lit_extend(x, &newlit, selflit);
            lit_extend(x, &newlit, otherlit);
            if (!otherlit->exact) newlit.exact = false;
            seq_push_raw(x, self, newlit);
        }
    }
    chn_seq_drop(&old);
    chn_seq_drop(other);
    seq_dedup(self);
}

/* Seq::cross_reverse */
static void seq_cross_reverse(chn_extractor *x, chn_seq *self, chn_seq *other) {
    if (!seq_cross_preamble(self, other)) return;
    chn_seq selflits = *self;
    chn_seq_init_empty(self);
    for (size_t i = 0; i < other->len; ++i) {
        const chn_lit *otherlit = &other->lits[i];
        for (size_t j = 0; j < selflits.len; ++j) {
            const chn_lit *selflit = &selflits.lits[j];
            if (!selflit->exact) {
                if (i == 0) seq_push_raw(x, self, lit_clone(x, selflit));
                continue;
            }
            chn_lit newlit = {NULL, 0, 0, true};
            lit_extend(x, &newlit, otherlit);
            lit_extend(x, &newlit, selflit);
            if (!otherlit->exact) newlit.exact = false;
            seq_push_raw(x, self, newlit);
        }
    }
    chn_seq_drop(&selflits);
    chn_seq_drop(other);
    seq_dedup(self);
}

/* Seq::union */
static void seq_union(chn_extractor *x, chn_seq *self, chn_seq *other) {
    if (!other->finite) {
        seq_make_infinite(self);
        return;
    }
    if (!self->finite) {
        chn_seq_drop(other);
        return;
    }
    if (!seq_reserve(self, other->len)) {
        x->oom = true;
        chn_seq_drop(other);
        return;
    }
    for (size_t i = 0; i < other->len; ++i) self->lits[self->len++] = other->lits[i];
    other->len = 0;
    chn_seq_drop(other);
    seq_dedup(self);
}

bool chn_seq_union(chn_seq *seq, chn_seq *other) {
    chn_extractor x = {false, false};
    seq_union(&x, seq, other);
    return !x.oom;
}

static chn_seq seq_clone(chn_extractor *x, const chn_seq *src) {
    chn_seq seq = seq_empty();
    seq.finite = src->finite;
    if (src->len && !seq_reserve(&seq, src->len)) {
        x->oom = true;
        return seq;
    }
    for (size_t i = 0; i < src->len; ++i) seq.lits[seq.len++] = lit_clone(x, &src->lits[i]);
    return seq;
}

/* ---- PreferenceTrie ---- */

typedef struct pt_trans {
    uint8_t byte;
    size_t next;
} pt_trans;

typedef struct pt_state {
    pt_trans *trans;
    size_t len;
    size_t cap;
    size_t literal_index; /* SIZE_MAX: None */
} pt_state;

typedef struct pt_trie {
    pt_state *states;
    size_t len;
    size_t cap;
    size_t next_literal_index;
} pt_trie;

static bool pt_create_state(pt_trie *t, size_t *id) {
    if (t->len == t->cap) {
        const size_t cap = t->cap ? t->cap * 2 : 16;
        pt_state *states = (pt_state *)realloc(t->states, cap * sizeof(*states));
        if (!states) return false;
        t->states = states;
        t->cap = cap;
    }
    pt_state *s = &t->states[t->len];
    s->trans = NULL;
    s->len = s->cap = 0;
    s->literal_index = SIZE_MAX;
    *id = t->len++;
    return true;
}

/* PreferenceTrie::insert: 1 Ok, 0 Err(*idx), -1 out of memory. */
static int pt_insert(pt_trie *t, const uint8_t *bytes, size_t len, size_t *idx) {
    size_t prev = 0;
    if (t->len == 0 && !pt_create_state(t, &prev)) return -1;
    if (t->states[prev].literal_index != SIZE_MAX) {
        *idx = t->states[prev].literal_index;
        return 0;
    }
    for (size_t k = 0; k < len; ++k) {
        const uint8_t b = bytes[k];
        pt_state *s = &t->states[prev];
        size_t lo = 0, hi = s->len;
        while (lo < hi) {
            const size_t mid = lo + (hi - lo) / 2;
            if (s->trans[mid].byte < b) lo = mid + 1;
            else hi = mid;
        }
        if (lo < s->len && s->trans[lo].byte == b) {
            prev = s->trans[lo].next;
            if (t->states[prev].literal_index != SIZE_MAX) {
                *idx = t->states[prev].literal_index;
                return 0;
            }
        } else {
            size_t next;
            if (!pt_create_state(t, &next)) return -1;
            s = &t->states[prev];
            if (s->len == s->cap) {
                const size_t cap = s->cap ? s->cap * 2 : 2;
                pt_trans *trans = (pt_trans *)realloc(s->trans, cap * sizeof(*trans));
                if (!trans) return -1;
                s->trans = trans;
                s->cap = cap;
            }
            memmove(s->trans + lo + 1, s->trans + lo, (s->len - lo) * sizeof(*s->trans));
            s->trans[lo].byte = b;
            s->trans[lo].next = next;
            ++s->len;
            prev = next;
        }
    }
    *idx = t->next_literal_index++;
    t->states[prev].literal_index = *idx;
    return 1;
}

/* PreferenceTrie::minimize(literals, keep_exact = false) */
static void pt_minimize(chn_extractor *x, chn_seq *seq) {
    pt_trie t = {NULL, 0, 0, 0};
    size_t *make_inexact = NULL, n_inexact = 0;
    if (seq->len && !(make_inexact = (size_t *)malloc(seq->len * sizeof(size_t)))) {
        x->oom = true;
        return;
    }
    size_t w = 0;
    for (size_t r = 0; r < seq->len; ++r) {
        size_t idx = 0;
        const int res = pt_insert(&t, seq->lits[r].bytes, seq->lits[r].len, &idx);
        if (res < 0) {
            x->oom = true;
            seq->lits[w++] = seq->lits[r];
            continue;
        }
        if (res == 1) {
            seq->lits[w++] = seq->lits[r];
        } else {
            make_inexact[n_inexact++] = idx;
            lit_drop(&seq->lits[r]);
        }
    }
    seq->len = w;
    for (size_t i = 0; i < n_inexact; ++i) {
        if (make_inexact[i] < seq->len) seq->lits[make_inexact[i]].exact = false;
    }
    free(make_inexact);
    for (size_t i = 0; i < t.len; ++i) free(t.states[i].trans);
    free(t.states);
}

/* ---- Extractor ---- */

static chn_seq extract(chn_extractor *x, const chn_hir *hir);

/* Extractor::enforce_literal_len */
static void enforce_literal_len(const chn_extractor *x, chn_seq *seq) {
    if (x->suffix) seq_keep_last_bytes(seq, LIMIT_LITERAL_LEN);
    else seq_keep_first_bytes(seq, LIMIT_LITERAL_LEN);
}

/* Extractor::cross */
static chn_seq ex_cross(chn_extractor *x, chn_seq seq1, chn_seq *seq2) {
    size_t n;
    if (seq_max_cross_len(&seq1, seq2, &n) && n > LIMIT_TOTAL) seq_make_infinite(seq2);
    if (x->suffix) seq_cross_reverse(x, &seq1, seq2);
    else seq_cross_forward(x, &seq1, seq2);
    enforce_literal_len(x, &seq1);
    return seq1;
}

/* Extractor::union */
static chn_seq ex_union(chn_extractor *x, chn_seq seq1, chn_seq *seq2) {
    size_t n;
    if (seq_max_union_len(&seq1, seq2, &n) && n > LIMIT_TOTAL) {
        if (x->suffix) {
            seq_keep_last_bytes(&seq1, 4);
            seq_keep_last_bytes(seq2, 4);
        } else {
            seq_keep_first_bytes(&seq1, 4);
            seq_keep_first_bytes(seq2, 4);
        }
        seq_dedup(&seq1);
        seq_dedup(seq2);
        if (seq_max_union_len(&seq1, seq2, &n) && n > LIMIT_TOTAL) seq_make_infinite(seq2);
    }
    seq_union(x, &seq1, seq2);
    return seq1;
}

/* Extractor::extract_concat (forward for prefixes, reversed for suffixes) */
static chn_seq extract_concat(chn_extractor *x, const chn_hir_vec *hirs) {
    chn_seq seq = seq_singleton(x, lit_exact(x, NULL, 0));
    for (size_t k = 0; k < hirs->len; ++k) {
        const chn_hir *hir = &hirs->ptr[x->suffix ? hirs->len - 1 - k : k];
        if (seq_is_inexact(&seq)) break;
        chn_seq sub = extract(x, hir);
        seq = ex_cross(x, seq, &sub);
        chn_seq_drop(&sub);
    }
    return seq;
}

/* Extractor::extract_alternation */
static chn_seq extract_alternation(chn_extractor *x, const chn_hir_vec *hirs) {
    chn_seq seq = seq_empty();
    for (size_t k = 0; k < hirs->len; ++k) {
        if (!seq.finite) break;
        chn_seq sub = extract(x, &hirs->ptr[k]);
        seq = ex_union(x, seq, &sub);
        chn_seq_drop(&sub);
    }
    return seq;
}

/* Extractor::extract_repetition */
static chn_seq extract_repetition(chn_extractor *x, const chn_hir_repetition *rep) {
    chn_seq subseq = extract(x, rep->sub);
    if (rep->min == 0) {
        if (!(rep->max.some && rep->max.value == 1)) chn_seq_make_inexact(&subseq);
        chn_seq empty = seq_singleton(x, lit_exact(x, NULL, 0));
        if (!rep->greedy) {
            const chn_seq tmp = subseq;
            subseq = empty;
            empty = tmp;
        }
        chn_seq out = ex_union(x, subseq, &empty);
        chn_seq_drop(&empty);
        return out;
    }
    if (rep->max.some && rep->min <= rep->max.value) {
        const uint32_t limit = LIMIT_REPEAT;
        chn_seq seq = seq_singleton(x, lit_exact(x, NULL, 0));
        const uint32_t count = rep->min < limit ? rep->min : limit;
        for (uint32_t i = 0; i < count; ++i) {
            if (seq_is_inexact(&seq)) break;
            chn_seq copy = seq_clone(x, &subseq);
            seq = ex_cross(x, seq, &copy);
            chn_seq_drop(&copy);
        }
        if (rep->min == rep->max.value) {
            if (rep->min > limit) chn_seq_make_inexact(&seq);
        } else {
            chn_seq_make_inexact(&seq);
        }
        chn_seq_drop(&subseq);
        return seq;
    }
    chn_seq_make_inexact(&subseq);
    return subseq;
}

/* char::encode_utf8 */
static size_t encode_utf8(uint32_t cp, uint8_t *out) {
    if (cp < 0x80) {
        out[0] = (uint8_t)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (uint8_t)(0xC0 | (cp >> 6));
        out[1] = (uint8_t)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (uint8_t)(0xE0 | (cp >> 12));
        out[1] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (uint8_t)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (uint8_t)(0xF0 | (cp >> 18));
    out[1] = (uint8_t)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (uint8_t)(0x80 | (cp & 0x3F));
    return 4;
}

/* Extractor::extract_class_unicode (with class_over_limit_unicode) */
static chn_seq extract_class_unicode(chn_extractor *x, const chn_hir_class_unicode *cls) {
    size_t count = 0;
    bool over = false;
    for (size_t i = 0; i < cls->len; ++i) {
        if (count > LIMIT_CLASS) {
            over = true;
            break;
        }
        count += (size_t)(cls->ranges[i].end - cls->ranges[i].start) + 1;
    }
    if (over || count > LIMIT_CLASS) return seq_infinite();
    chn_seq seq = seq_empty();
    for (size_t i = 0; i < cls->len; ++i) {
        for (uint32_t ch = cls->ranges[i].start;; ++ch) {
            if (ch < 0xD800 || ch > 0xDFFF) {
                uint8_t buf[4];
                const size_t n = encode_utf8(ch, buf);
                seq_push(x, &seq, lit_exact(x, buf, n));
            }
            if (ch == cls->ranges[i].end) break;
        }
    }
    enforce_literal_len(x, &seq);
    return seq;
}

/* Extractor::extract_class_bytes (with class_over_limit_bytes) */
static chn_seq extract_class_bytes(chn_extractor *x, const chn_hir_class_bytes *cls) {
    size_t count = 0;
    bool over = false;
    for (size_t i = 0; i < cls->len; ++i) {
        if (count > LIMIT_CLASS) {
            over = true;
            break;
        }
        count += (size_t)(cls->ranges[i].end - cls->ranges[i].start) + 1;
    }
    if (over || count > LIMIT_CLASS) return seq_infinite();
    chn_seq seq = seq_empty();
    for (size_t i = 0; i < cls->len; ++i) {
        for (unsigned b = cls->ranges[i].start; b <= cls->ranges[i].end; ++b) {
            const uint8_t byte = (uint8_t)b;
            seq_push(x, &seq, lit_exact(x, &byte, 1));
        }
    }
    enforce_literal_len(x, &seq);
    return seq;
}

/* Extractor::extract */
static chn_seq extract(chn_extractor *x, const chn_hir *hir) {
    switch (hir->kind) {
    case CHN_HIR_EMPTY:
    case CHN_HIR_LOOK:
        return seq_singleton(x, lit_exact(x, NULL, 0));
    case CHN_HIR_LITERAL: {
        chn_seq seq = seq_singleton(x, lit_exact(x, hir->literal.bytes, hir->literal.len));
        enforce_literal_len(x, &seq);
        return seq;
    }
    case CHN_HIR_CLASS:
        if (hir->cls.kind == CHN_HIR_CLASS_UNICODE) return extract_class_unicode(x, &hir->cls.unicode);
        return extract_class_bytes(x, &hir->cls.bytes);
    case CHN_HIR_REPETITION:
        return extract_repetition(x, &hir->repetition);
    case CHN_HIR_CAPTURE:
        return extract(x, hir->capture.sub);
    case CHN_HIR_CONCAT:
        return extract_concat(x, &hir->concat);
    case CHN_HIR_ALTERNATION:
        return extract_alternation(x, &hir->alternation);
    }
    abort();
}

/* exec.rs literal_analysis's shrinking loop for one kind. */
static void shrink(chn_extractor *x, chn_seq *seq) {
    static const size_t attempts[3][2] = {{5, 50}, {4, 30}, {3, 20}};
    for (size_t i = 0; i < 3; ++i) {
        if (!seq->finite) break;
        if (seq->len <= attempts[i][1]) break;
        if (x->suffix) seq_keep_last_bytes(seq, attempts[i][0]);
        else seq_keep_first_bytes(seq, attempts[i][0]);
        pt_minimize(x, seq);
    }
}

bool chn_literal_analysis(const chn_hir *expr, chn_seq *prefixes, chn_seq *suffixes) {
    chn_extractor x = {false, false};
    *prefixes = extract(&x, expr);
    shrink(&x, prefixes);
    x.suffix = true;
    *suffixes = extract(&x, expr);
    shrink(&x, suffixes);
    if (x.oom) {
        chn_seq_drop(prefixes);
        chn_seq_drop(suffixes);
        return false;
    }
    return true;
}
