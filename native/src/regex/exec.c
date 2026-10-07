/* Port of regex 1.8.4 src/exec.rs (with src/re_unicode.rs, src/re_builder.rs and src/error.rs) and chainner_ext 0.3.10 crates/regex-py/src/lib.rs (Regex::new), MIT OR Apache-2.0. */
/* ExecBuilder::parse/build (literal analysis, the three programs, the literal searchers,
 * the Aho-Corasick alternation), ExecReadOnly::choose_match_type, and ExecNoSync's
 * find_at / captures_read_at dispatch with every DFA Quit fallback. A panic of 0.3.10
 * (a slice or an index past the text) is reproduced at the same point of the same path,
 * with Rust's message. The pool is a few atomic slots of caches: a search takes one (or
 * creates one) and returns it, so a compiled regex can be searched from several threads. */
#include "prog.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif

/* ---- the pool ---- */

static void *atomic_exchange_ptr(void *volatile *p, void *value) {
#ifdef _MSC_VER
    return _InterlockedExchangePointer(p, value);
#else
    return __atomic_exchange_n(p, value, __ATOMIC_ACQ_REL);
#endif
}

static bool atomic_cas_ptr(void *volatile *p, void *expected, void *desired) {
#ifdef _MSC_VER
    return _InterlockedCompareExchangePointer(p, desired, expected) == expected;
#else
    return __atomic_compare_exchange_n(p, &expected, desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
#endif
}

static void cache_free(chn_cache *cache) {
    chn_pikevm_cache_drop(&cache->pikevm);
    chn_backtrack_cache_drop(&cache->backtrack);
    chn_dfa_cache_free(cache->dfa);
    chn_dfa_cache_free(cache->dfa_reverse);
    free(cache);
}

/* Pool::get */
static chn_cache *pool_get(const chn_regex *re) {
    chn_regex *mut = (chn_regex *)re; /* the pool is the regex's interior mutability */
    for (size_t i = 0; i < CHN_POOL_SLOTS; ++i) {
        if (!mut->pool[i]) continue;
        chn_cache *cache = (chn_cache *)atomic_exchange_ptr(&mut->pool[i], NULL);
        if (cache) return cache;
    }
    return (chn_cache *)calloc(1, sizeof(chn_cache));
}

/* PoolGuard's drop */
static void pool_put(const chn_regex *re, chn_cache *cache) {
    chn_regex *mut = (chn_regex *)re;
    for (size_t i = 0; i < CHN_POOL_SLOTS; ++i) {
        if (!mut->pool[i] && atomic_cas_ptr(&mut->pool[i], NULL, cache)) return;
    }
    cache_free(cache);
}

/* ---- panics ---- */

static chn_regex_status panic_with(chn_regex_string *panic, const char *fmt, size_t a, size_t b) {
    char buf[160];
    const int n = snprintf(buf, sizeof(buf), fmt, (unsigned long long)a, (unsigned long long)b);
    panic->ptr = NULL;
    panic->len = 0;
    if (n < 0) return CHN_REGEX_ERROR_NOMEM;
    if (!(panic->ptr = (char *)malloc((size_t)n + 1))) return CHN_REGEX_ERROR_NOMEM;
    memcpy(panic->ptr, buf, (size_t)n + 1);
    panic->len = (size_t)n;
    return CHN_REGEX_PANIC;
}

/* `&text[start..]` past the end. */
static chn_regex_status panic_slice_start(chn_regex_string *panic, size_t start, size_t len) {
    return panic_with(panic, "range start index %llu out of range for slice of length %llu", start, len);
}

/* `text[index]` past the end. */
static chn_regex_status panic_index(chn_regex_string *panic, size_t len, size_t index) {
    return panic_with(panic, "index out of bounds: the len is %llu but the index is %llu", len, index);
}

/* ---- ExecNoSync ---- */

typedef struct searcher {
    const chn_regex *re;
    chn_cache *cache;
    const uint8_t *text;
    size_t len;
    chn_regex_string *panic;
} searcher;

/* Searches report through these. */
enum {
    R_NONE = 0,  /* None / NoMatch */
    R_SOME = 1,  /* Some / Match */
    R_QUIT = 2,  /* dfa::Result::Quit */
    R_PANIC = 3, /* *panic is set */
    R_NOMEM = 4
};

/* ExecNoSync::exec_nfa (matches = [false]; quit_after_match and quit_after_match_with_pos
 * are false on every path regex-py reaches). */
static int exec_nfa(searcher *s, chn_match_type ty, size_t *slots, size_t nslots, size_t start, size_t end) {
    const chn_prog *nfa = &s->re->nfa;
    if (ty == CHN_MATCH_NFA_AUTO) {
        ty = chn_backtrack_should_exec(nfa->len, s->len) ? CHN_MATCH_NFA_BACKTRACK : CHN_MATCH_NFA_PIKEVM;
    }
    const chn_input input = {s->text, s->len, chn_prog_uses_bytes(nfa), nfa->only_utf8};
    bool matched = false;
    int r;
    if (ty == CHN_MATCH_NFA_PIKEVM) {
        r = chn_pikevm_exec(nfa, &s->cache->pikevm, &matched, slots, nslots, false, &input, start, end);
    } else {
        r = chn_backtrack_exec(nfa, &s->cache->backtrack, &matched, slots, nslots, &input, start, end);
    }
    return r < 0 ? R_NOMEM : (r ? R_SOME : R_NONE);
}

/* ExecNoSync::find_nfa */
static int find_nfa(searcher *s, chn_match_type ty, size_t start, size_t *ms, size_t *me) {
    size_t slots[2] = {CHN_REGEX_NO_SLOT, CHN_REGEX_NO_SLOT};
    const int r = exec_nfa(s, ty, slots, 2, start, s->len);
    if (r != R_SOME) return r;
    if (slots[0] == CHN_REGEX_NO_SLOT || slots[1] == CHN_REGEX_NO_SLOT) return R_NONE;
    *ms = slots[0];
    *me = slots[1];
    return R_SOME;
}

/* ExecNoSync::captures_nfa_type */
static int captures_nfa_type(searcher *s, chn_match_type ty, size_t *slots, size_t nslots, size_t start,
    size_t end) {
    const int r = exec_nfa(s, ty, slots, nslots, start, end);
    if (r != R_SOME) return r;
    if (slots[0] == CHN_REGEX_NO_SLOT || slots[1] == CHN_REGEX_NO_SLOT) return R_NONE;
    return R_SOME;
}

/* ExecNoSync::find_literals */
static int find_literals(searcher *s, size_t start, size_t *ms, size_t *me) {
    const chn_regex *re = s->re;
    bool found;
    size_t a = 0, b = 0;
    if (re->match_type == CHN_MATCH_LITERAL_ANCHORED_START && !(start == 0 || !re->nfa.is_anchored_start)) {
        return R_NONE;
    }
    if (start > s->len) {
        panic_slice_start(s->panic, start, s->len);
        return s->panic->ptr ? R_PANIC : R_NOMEM;
    }
    const uint8_t *hay = s->text + start;
    const size_t hay_len = s->len - start;
    switch (re->match_type) {
    case CHN_MATCH_LITERAL_UNANCHORED:
        found = chn_literal_searcher_find(&re->nfa.prefixes, hay, hay_len, &a, &b);
        break;
    case CHN_MATCH_LITERAL_ANCHORED_START:
        found = chn_literal_searcher_find_start(&re->nfa.prefixes, hay, hay_len, &a, &b);
        break;
    case CHN_MATCH_LITERAL_ANCHORED_END:
        found = chn_literal_searcher_find_end(&re->suffixes, hay, hay_len, &a, &b);
        break;
    default:
        found = chn_multi_find(&re->ac, re->fast, hay, hay_len, &a, &b);
        break;
    }
    if (!found) return R_NONE;
    *ms = start + a;
    *me = start + b;
    return R_SOME;
}

static int dfa_result(chn_dfa_result r, size_t *value) {
    switch (r.status) {
    case CHN_DFA_MATCH:
        *value = r.value;
        return R_SOME;
    case CHN_DFA_NO_MATCH:
        *value = r.value;
        return R_NONE;
    case CHN_DFA_QUIT:
        return R_QUIT;
    default:
        return R_NOMEM;
    }
}

/* Fsm::forward with start_flags's `text[at - 1]` read reproduced. */
static int dfa_forward(searcher *s, size_t at, size_t *value) {
    if (at > s->len) {
        panic_index(s->panic, s->len, at - 1);
        return s->panic->ptr ? R_PANIC : R_NOMEM;
    }
    return dfa_result(chn_dfa_forward(&s->re->dfa, &s->re->nfa.prefixes, &s->cache->dfa, s->text, s->len, at),
        value);
}

/* Fsm::reverse(&text[start..slice_end], at) (in bounds). */
static int dfa_reverse(searcher *s, size_t start, size_t slice_end, size_t at, size_t *value) {
    return dfa_result(chn_dfa_reverse(&s->re->dfa_reverse, &s->cache->dfa_reverse, s->text + start,
        slice_end - start, at), value);
}

/* ExecNoSync::find_dfa_forward */
static int find_dfa_forward(searcher *s, size_t start, size_t *ms, size_t *me) {
    size_t end;
    const int r = dfa_forward(s, start, &end);
    if (r != R_SOME) return r;
    if (end == start) {
        *ms = *me = start;
        return R_SOME;
    }
    size_t rs;
    const int rr = dfa_reverse(s, start, s->len, end - start, &rs);
    if (rr != R_SOME) return rr;
    *ms = start + rs;
    *me = end;
    return R_SOME;
}

/* ExecNoSync::find_dfa_anchored_reverse */
static int find_dfa_anchored_reverse(searcher *s, size_t start, size_t *ms, size_t *me) {
    if (start > s->len) {
        panic_slice_start(s->panic, start, s->len);
        return s->panic->ptr ? R_PANIC : R_NOMEM;
    }
    size_t rs;
    const int r = dfa_reverse(s, start, s->len, s->len - start, &rs);
    if (r != R_SOME) return r;
    *ms = start + rs;
    *me = s->len;
    return R_SOME;
}

/* ExecNoSync::exec_dfa_reverse_suffix: R_NONE / R_SOME / R_QUIT as Some(..), or -1 for
 * the outer None (fall back to a forward scan). */
static int exec_dfa_reverse_suffix(searcher *s, size_t original_start, size_t *ms, size_t *me) {
    const chn_memmem *lcs = &s->re->suffixes.lcs;
    size_t start = original_start, end = start, last_literal = start;
    while (end <= s->len) {
        size_t i;
        if (!chn_memmem_find(lcs, s->re->fast, s->text + last_literal, s->len - last_literal, &i)) return R_NONE;
        last_literal += i;
        end = last_literal + lcs->len;
        size_t at;
        const int r = dfa_reverse(s, start, end, end - start, &at);
        if (r == R_SOME || r == R_NONE) {
            if (at == 0) return -1;
            if (r == R_SOME) {
                *ms = start + at;
                *me = end;
                return R_SOME;
            }
            start += at;
            last_literal += 1;
            continue;
        }
        return r;
    }
    return R_NONE;
}

/* ExecNoSync::find_dfa_reverse_suffix */
static int find_dfa_reverse_suffix(searcher *s, size_t start, size_t *ms, size_t *me) {
    size_t match_start, ignored;
    const int r = exec_dfa_reverse_suffix(s, start, &match_start, &ignored);
    if (r < 0) return find_dfa_forward(s, start, ms, me);
    if (r != R_SOME) return r;
    size_t e;
    const int f = dfa_forward(s, match_start, &e);
    if (f == R_NONE) {
        const char *msg = "BUG: reverse match implies forward match";
        s->panic->len = strlen(msg);
        s->panic->ptr = (char *)malloc(s->panic->len + 1);
        if (!s->panic->ptr) {
            s->panic->len = 0;
            return R_NOMEM;
        }
        memcpy(s->panic->ptr, msg, s->panic->len + 1);
        return R_PANIC;
    }
    if (f != R_SOME) return f;
    *ms = match_start;
    *me = e;
    return R_SOME;
}

/* ExecNoSync::is_anchor_end_match */
static bool is_anchor_end_match(const searcher *s) {
    const chn_regex *re = s->re;
    if (s->len > ((size_t)1 << 20) && re->nfa.is_anchored_end) {
        const chn_memmem *lcs = &re->suffixes.lcs;
        if (lcs->len >= 1 && !chn_memmem_is_suffix(lcs, s->text, s->len)) return false;
    }
    return true;
}

/* ExecNoSync::find_at */
static int find_at(searcher *s, size_t start, size_t *ms, size_t *me) {
    if (!is_anchor_end_match(s)) return R_NONE;
    int r;
    switch (s->re->match_type) {
    case CHN_MATCH_LITERAL_UNANCHORED:
    case CHN_MATCH_LITERAL_ANCHORED_START:
    case CHN_MATCH_LITERAL_ANCHORED_END:
    case CHN_MATCH_LITERAL_AHO_CORASICK:
        return find_literals(s, start, ms, me);
    case CHN_MATCH_DFA:
        r = find_dfa_forward(s, start, ms, me);
        break;
    case CHN_MATCH_DFA_ANCHORED_REVERSE:
        r = find_dfa_anchored_reverse(s, start, ms, me);
        break;
    case CHN_MATCH_DFA_SUFFIX:
        r = find_dfa_reverse_suffix(s, start, ms, me);
        break;
    case CHN_MATCH_NFA_AUTO:
    case CHN_MATCH_NFA_BACKTRACK:
    case CHN_MATCH_NFA_PIKEVM:
        return find_nfa(s, s->re->match_type, start, ms, me);
    default:
        return R_NONE;
    }
    if (r == R_QUIT) return find_nfa(s, CHN_MATCH_NFA_AUTO, start, ms, me);
    return r;
}

/* ExecNoSync::captures_read_at (the slots are already all None). */
static int captures_read_at(searcher *s, size_t *slots, size_t nslots, size_t start) {
    size_t ms = 0, me = 0;
    if (nslots == 0) return find_at(s, start, &ms, &me);
    if (nslots == 2) {
        const int r = find_at(s, start, &ms, &me);
        if (r == R_SOME) {
            slots[0] = ms;
            slots[1] = me;
        }
        return r;
    }
    if (!is_anchor_end_match(s)) return R_NONE;
    const chn_regex *re = s->re;
    int r;
    switch (re->match_type) {
    case CHN_MATCH_LITERAL_UNANCHORED:
    case CHN_MATCH_LITERAL_ANCHORED_START:
    case CHN_MATCH_LITERAL_ANCHORED_END:
    case CHN_MATCH_LITERAL_AHO_CORASICK:
        r = find_literals(s, start, &ms, &me);
        if (r != R_SOME) return r;
        return captures_nfa_type(s, CHN_MATCH_NFA_AUTO, slots, nslots, ms, me);
    case CHN_MATCH_DFA:
        if (re->nfa.is_anchored_start) return captures_nfa_type(s, CHN_MATCH_NFA_AUTO, slots, nslots, start, s->len);
        r = find_dfa_forward(s, start, &ms, &me);
        break;
    case CHN_MATCH_DFA_ANCHORED_REVERSE:
        r = find_dfa_anchored_reverse(s, start, &ms, &me);
        break;
    case CHN_MATCH_DFA_SUFFIX:
        r = find_dfa_reverse_suffix(s, start, &ms, &me);
        break;
    case CHN_MATCH_NFA_AUTO:
    case CHN_MATCH_NFA_BACKTRACK:
    case CHN_MATCH_NFA_PIKEVM:
        return captures_nfa_type(s, re->match_type, slots, nslots, start, s->len);
    default:
        return R_NONE;
    }
    if (r == R_SOME) return captures_nfa_type(s, CHN_MATCH_NFA_AUTO, slots, nslots, ms, me);
    if (r == R_QUIT) return captures_nfa_type(s, CHN_MATCH_NFA_AUTO, slots, nslots, start, s->len);
    return r;
}

static chn_regex_status to_status(int r) {
    switch (r) {
    case R_SOME: return CHN_REGEX_OK;
    case R_NONE: return CHN_REGEX_NO_MATCH;
    case R_PANIC: return CHN_REGEX_PANIC;
    default: return CHN_REGEX_ERROR_NOMEM;
    }
}

chn_regex_status chn_exec_find_at(const chn_regex *re, const uint8_t *text, size_t len, size_t start,
    size_t *s, size_t *e, chn_regex_string *panic) {
    panic->ptr = NULL;
    panic->len = 0;
    *s = *e = CHN_REGEX_NO_SLOT;
    chn_cache *cache = pool_get(re);
    if (!cache) return CHN_REGEX_ERROR_NOMEM;
    searcher sr = {re, cache, text, len, panic};
    size_t ms = 0, me = 0;
    const int r = find_at(&sr, start, &ms, &me);
    pool_put(re, cache);
    if (r == R_SOME) {
        *s = ms;
        *e = me;
    }
    return to_status(r);
}

chn_regex_status chn_exec_captures_read_at(const chn_regex *re, size_t *slots, size_t nslots,
    const uint8_t *text, size_t len, size_t start, chn_regex_string *panic) {
    panic->ptr = NULL;
    panic->len = 0;
    for (size_t i = 0; i < nslots; ++i) slots[i] = CHN_REGEX_NO_SLOT;
    chn_cache *cache = pool_get(re);
    if (!cache) return CHN_REGEX_ERROR_NOMEM;
    searcher sr = {re, cache, text, len, panic};
    const int r = captures_read_at(&sr, slots, nslots, start);
    pool_put(re, cache);
    if (r != R_SOME) {
        for (size_t i = 0; i < nslots; ++i) slots[i] = CHN_REGEX_NO_SLOT;
    }
    return to_status(r);
}

/* ---- ExecBuilder::build ---- */

/* exec.rs alternation_literals: the literals of a pure alternation of literals. */
static bool alternation_literals(const chn_hir *expr, chn_seq *out, bool *nomem) {
    chn_seq_init_empty(out);
    *nomem = false;
    if (!expr->props.alternation_literal || expr->kind != CHN_HIR_ALTERNATION) return false;
    const chn_hir_vec *alts = &expr->alternation;
    out->lits = (chn_lit *)calloc(alts->len ? alts->len : 1, sizeof(chn_lit));
    if (!out->lits) {
        *nomem = true;
        return false;
    }
    out->cap = alts->len;
    for (size_t i = 0; i < alts->len; ++i) {
        const chn_hir *alt = &alts->ptr[i];
        const chn_hir *parts = alt;
        size_t nparts = 1;
        if (alt->kind == CHN_HIR_CONCAT) {
            parts = alt->concat.ptr;
            nparts = alt->concat.len;
        } else if (alt->kind != CHN_HIR_LITERAL) {
            abort(); /* unreachable!("expected literal or concat") */
        }
        size_t total = 0;
        for (size_t k = 0; k < nparts; ++k) {
            if (parts[k].kind != CHN_HIR_LITERAL) abort(); /* unreachable!("expected literal") */
            total += parts[k].literal.len;
        }
        chn_lit *lit = &out->lits[out->len++];
        lit->exact = true;
        lit->bytes = (uint8_t *)malloc(total ? total : 1);
        if (!lit->bytes) {
            chn_seq_drop(out);
            *nomem = true;
            return false;
        }
        lit->cap = total;
        for (size_t k = 0; k < nparts; ++k) {
            memcpy(lit->bytes + lit->len, parts[k].literal.bytes, parts[k].literal.len);
            lit->len += parts[k].literal.len;
        }
    }
    return true;
}

/* ExecReadOnly::should_suffix_scan */
static bool should_suffix_scan(const chn_regex *re) {
    if (chn_literal_searcher_is_empty(&re->suffixes)) return false;
    const size_t lcs_len = re->suffixes.lcs.char_len;
    return lcs_len >= 3 && lcs_len > re->nfa.prefixes.lcp.char_len;
}

/* ExecReadOnly::choose_match_type (the DFA harness mode skips the literal types). */
static chn_match_type choose_match_type(const chn_regex *re, chn_regex_engine engine) {
    if (engine == CHN_REGEX_ENGINE_NFA) return CHN_MATCH_NFA_PIKEVM;
    if (engine == CHN_REGEX_ENGINE_BACKTRACK) return CHN_MATCH_NFA_BACKTRACK;
    if (re->nfa.len == 0) return CHN_MATCH_NOTHING;
    if (engine != CHN_REGEX_ENGINE_DFA) {
        /* choose_literal_match_type */
        if (re->has_ac) return CHN_MATCH_LITERAL_AHO_CORASICK;
        if (chn_literal_searcher_complete(&re->nfa.prefixes)) {
            return re->nfa.is_anchored_start ? CHN_MATCH_LITERAL_ANCHORED_START : CHN_MATCH_LITERAL_UNANCHORED;
        }
        if (chn_literal_searcher_complete(&re->suffixes) && re->nfa.is_anchored_end) {
            return CHN_MATCH_LITERAL_ANCHORED_END;
        }
    }
    /* choose_dfa_match_type */
    if (chn_dfa_can_exec(&re->dfa)) {
        if (!re->nfa.is_anchored_start && re->nfa.is_anchored_end) return CHN_MATCH_DFA_ANCHORED_REVERSE;
        if (should_suffix_scan(re)) return CHN_MATCH_DFA_SUFFIX;
        return CHN_MATCH_DFA;
    }
    return CHN_MATCH_NFA_AUTO;
}

/* format!("Invalid regex: {}", e) */
static chn_regex_status invalid(chn_regex_string *error, const char *text, size_t len) {
    static const char prefix[] = "Invalid regex: ";
    const size_t n = sizeof(prefix) - 1;
    error->ptr = (char *)malloc(n + len + 1);
    if (!error->ptr) {
        error->len = 0;
        return CHN_REGEX_ERROR_NOMEM;
    }
    memcpy(error->ptr, prefix, n);
    if (len) memcpy(error->ptr + n, text, len);
    error->ptr[n + len] = '\0';
    error->len = n + len;
    return CHN_REGEX_ERROR_INVALID;
}

static chn_regex_status compile_error(chn_regex_string *error, chn_compile_status st, const char *message) {
    if (st == CHN_COMPILE_SYNTAX) return invalid(error, message, strlen(message));
    if (st == CHN_COMPILE_TOO_BIG) {
        char buf[96];
        const int n = snprintf(buf, sizeof(buf), "Compiled regex exceeds size limit of %llu bytes.",
            (unsigned long long)CHN_REGEX_SIZE_LIMIT);
        return invalid(error, buf, n > 0 ? (size_t)n : 0);
    }
    return CHN_REGEX_ERROR_NOMEM;
}

void chn_regex_free(chn_regex *re) {
    if (!re) return;
    for (size_t i = 0; i < CHN_POOL_SLOTS; ++i) {
        if (re->pool[i]) cache_free((chn_cache *)re->pool[i]);
    }
    chn_prog_drop(&re->nfa);
    chn_prog_drop(&re->dfa);
    chn_prog_drop(&re->dfa_reverse);
    chn_literal_searcher_drop(&re->suffixes);
    if (re->has_ac) chn_multi_drop(&re->ac);
    free(re->pattern);
    free(re);
}

chn_regex_status chn_regex_compile(const char *pattern, size_t pattern_len,
    const chn_regex_options *options, chn_regex **out, chn_regex_string *error) {
    const chn_regex_options production = {CHN_REGEX_ENGINE_DEFAULT, true};
    if (!options) options = &production;
    *out = NULL;
    error->ptr = NULL;
    error->len = 0;

    /* ExecBuilder::parse */
    chn_hir expr;
    chn_hir_string perr = {NULL, 0};
    const chn_hir_status ps = chn_hir_parse(pattern, pattern_len, CHN_REGEX_NEST_LIMIT, &expr, &perr);
    if (ps == CHN_HIR_ERROR_SYNTAX) {
        const chn_regex_status st = invalid(error, perr.ptr, perr.len);
        free(perr.ptr);
        chn_hir_drop(&expr);
        return st;
    }
    free(perr.ptr);
    if (ps == CHN_HIR_ERROR_UNIMPLEMENTED) return CHN_REGEX_UNIMPLEMENTED;
    if (ps != CHN_HIR_OK) return CHN_REGEX_ERROR_NOMEM;

    const chn_hir_properties *props = &expr.props;
    const bool bytes = !props->utf8 || chn_hir_look_set_contains(props->look_set, CHN_HIR_LOOK_WORD_ASCII_NEGATE);
    bool use_prefixes = true, use_suffixes = true;
    if (!chn_hir_look_set_contains(props->look_set_prefix, CHN_HIR_LOOK_START)
        && chn_hir_look_set_contains(props->look_set, CHN_HIR_LOOK_START)) {
        use_prefixes = false;
    } else if (chn_hir_look_set_contains_word(props->look_set_prefix_any)) {
        use_prefixes = false;
    } else if (chn_hir_look_set_contains(props->look_set_prefix_any, CHN_HIR_LOOK_START_LF)) {
        use_prefixes = false;
    }
    if (!chn_hir_look_set_contains(props->look_set_suffix, CHN_HIR_LOOK_END)
        && chn_hir_look_set_contains(props->look_set, CHN_HIR_LOOK_END)) {
        use_suffixes = false;
    } else if (chn_hir_look_set_contains_word(props->look_set_suffix_any)) {
        use_suffixes = false;
    } else if (chn_hir_look_set_contains(props->look_set_suffix_any, CHN_HIR_LOOK_END_LF)) {
        use_suffixes = false;
    }
    chn_seq pres, suffs, prefixes, suffixes;
    chn_seq_init_empty(&prefixes);
    chn_seq_init_empty(&suffixes);
    if (!use_prefixes && !use_suffixes) {
        chn_seq_init_empty(&pres);
        chn_seq_init_empty(&suffs);
        pres.finite = suffs.finite = false;
    } else if (!chn_literal_analysis(&expr, &pres, &suffs)) {
        chn_hir_drop(&expr);
        return CHN_REGEX_ERROR_NOMEM;
    }
    if (!chn_hir_look_set_is_empty(props->look_set)) {
        chn_seq_make_inexact(&pres);
        chn_seq_make_inexact(&suffs);
    }
    bool ok = true;
    if (use_prefixes) ok = chn_seq_union(&prefixes, &pres) && ok;
    if (use_suffixes) ok = chn_seq_union(&suffixes, &suffs) && ok;
    chn_seq_drop(&pres);
    chn_seq_drop(&suffs);

    chn_regex *re = (chn_regex *)calloc(1, sizeof(chn_regex));
    if (!ok || !re) {
        free(re);
        chn_seq_drop(&prefixes);
        chn_seq_drop(&suffixes);
        chn_hir_drop(&expr);
        return CHN_REGEX_ERROR_NOMEM;
    }
    chn_prog_init(&re->nfa);
    chn_prog_init(&re->dfa);
    chn_prog_init(&re->dfa_reverse);
    chn_literal_searcher_empty(&re->suffixes);
    re->fast = options->prefilter;

    /* ExecBuilder::build: the three programs. */
    const char *message = NULL;
    chn_compile_status cs = chn_compile(&expr, CHN_REGEX_SIZE_LIMIT, bytes, false, false, &re->nfa, &message);
    if (cs == CHN_COMPILE_OK) cs = chn_compile(&expr, CHN_REGEX_SIZE_LIMIT, false, true, false, &re->dfa, &message);
    if (cs == CHN_COMPILE_OK) {
        cs = chn_compile(&expr, CHN_REGEX_SIZE_LIMIT, false, true, true, &re->dfa_reverse, &message);
    }
    chn_regex_status st = CHN_REGEX_OK;
    if (cs != CHN_COMPILE_OK) st = compile_error(error, cs, message);

    /* build_aho_corasick */
    if (st == CHN_REGEX_OK) {
        chn_seq lits;
        bool nomem;
        if (alternation_literals(&expr, &lits, &nomem) && lits.len > 32) {
            if (chn_multi_new(&re->ac, &lits) == CHN_LITS_OK) re->has_ac = true;
            else st = CHN_REGEX_ERROR_NOMEM;
        }
        if (nomem) st = CHN_REGEX_ERROR_NOMEM;
        chn_seq_drop(&lits);
    }
    /* nfa.prefixes = LiteralSearcher::prefixes(..); suffixes = LiteralSearcher::suffixes(..)
     * (dfa.prefixes is the same searcher, read through nfa.prefixes). */
    if (st == CHN_REGEX_OK && chn_literal_searcher_new(&re->nfa.prefixes, &prefixes, false, re->fast) != CHN_LITS_OK) {
        st = CHN_REGEX_ERROR_NOMEM;
    }
    if (st == CHN_REGEX_OK && chn_literal_searcher_new(&re->suffixes, &suffixes, true, re->fast) != CHN_LITS_OK) {
        st = CHN_REGEX_ERROR_NOMEM;
    }
    chn_seq_drop(&prefixes);
    chn_seq_drop(&suffixes);
    chn_hir_drop(&expr);
    if (st == CHN_REGEX_OK) {
        re->pattern = (char *)malloc(pattern_len + 1);
        if (!re->pattern) st = CHN_REGEX_ERROR_NOMEM;
        else {
            if (pattern_len) memcpy(re->pattern, pattern, pattern_len);
            re->pattern[pattern_len] = '\0';
            re->pattern_len = pattern_len;
        }
    }
    if (st != CHN_REGEX_OK) {
        chn_regex_free(re);
        return st;
    }
    re->match_type = choose_match_type(re, options->engine);
    *out = re;
    return CHN_REGEX_OK;
}

void chn_regex_string_drop(chn_regex_string *s) {
    free(s->ptr);
    s->ptr = NULL;
    s->len = 0;
}

const char *chn_regex_pattern(const chn_regex *re, size_t *len) {
    *len = re->pattern_len;
    return re->pattern;
}

size_t chn_regex_groups(const chn_regex *re) {
    return re->nfa.captures_len - 1;
}

size_t chn_regex_captures_len(const chn_regex *re) {
    return re->nfa.captures_len;
}

const char *chn_regex_capture_name(const chn_regex *re, size_t index, size_t *len) {
    const chn_hir_string *name = &re->nfa.captures[index];
    *len = name->len;
    return name->ptr;
}

chn_regex_status chn_regex_search_at(const chn_regex *re, const char *text, size_t text_len,
    size_t byte_pos, size_t *slots, size_t slots_len, chn_regex_string *panic) {
    return chn_exec_captures_read_at(re, slots, slots_len, (const uint8_t *)text, text_len, byte_pos, panic);
}
