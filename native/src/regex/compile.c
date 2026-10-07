/* Port of regex 1.8.4 src/compile.rs, MIT OR Apache-2.0. */
/* Compiler, Hole/Patch backpatching, MaybeInst, CompileClass (UTF-8 automata through
 * regex-syntax's Utf8Sequences), SuffixCache and ByteClassSet, for a single regex
 * (num_exprs == 1) with only_utf8 on. check_size charges Rust's 32 bytes per instruction
 * plus extra_inst_bytes, exactly where compile.rs calls it. Rust's Hole tree is a linked
 * list of instruction pointers here; its order is kept. */
#include "prog.h"
#include <stdlib.h>
#include <string.h>

/* MaybeInst's state; the instruction's own fields hold InstHole's payload. */
enum {
    MAYBE_COMPILED = 0,   /* MaybeInst::Compiled */
    MAYBE_UNCOMPILED = 1, /* MaybeInst::Uncompiled(InstHole) */
    MAYBE_SPLIT = 2,      /* MaybeInst::Split */
    MAYBE_SPLIT1 = 3,     /* MaybeInst::Split1(goto1): next holds goto1 */
    MAYBE_SPLIT2 = 4      /* MaybeInst::Split2(goto2): next2 holds goto2 */
};

#define NO_GOTO SIZE_MAX /* Option<InstPtr>::None in fill_split */

/* Hole: a list of hole nodes (head and tail are node indices, 0 is the empty list). */
typedef struct hole {
    size_t head;
    size_t tail;
} hole;

/* Patch { hole, entry } */
typedef struct patch {
    hole h;
    size_t entry;
} patch;

/* SuffixCacheKey / SuffixCacheEntry */
typedef struct sc_entry {
    size_t from_inst;
    uint8_t start;
    uint8_t end;
    size_t pc;
} sc_entry;

#define SUFFIX_CACHE_SIZE 1000

typedef struct compiler {
    chn_prog *p;            /* compiled (its insts are the MaybeInsts) */
    uint8_t *maybe;         /* MaybeInst state per instruction */
    size_t maybe_cap;
    size_t size_limit;
    size_t extra_inst_bytes;
    size_t *hole_pc;        /* hole nodes */
    size_t *hole_next;
    size_t hole_len;
    size_t hole_cap;
    size_t *sc_sparse;      /* SuffixCache::sparse */
    sc_entry *sc_dense;     /* SuffixCache::dense */
    size_t sc_len;
    size_t sc_cap;
    chn_hir_utf8_sequences seqs; /* utf8_seqs */
    bool seqs_ready;
    bool byte_set[256];     /* ByteClassSet */
    chn_compile_status err;
    const char *message;
} compiler;

/* ---- errors ---- */

static bool fail(compiler *c, chn_compile_status err, const char *message) {
    c->err = err;
    c->message = message;
    return false;
}

static bool nomem(compiler *c) {
    return fail(c, CHN_COMPILE_NOMEM, NULL);
}

/* Compiler::check_size */
static bool check_size(compiler *c) {
    const size_t size = c->extra_inst_bytes + c->p->len * CHN_RUST_INST_SIZE;
    if (size > c->size_limit) return fail(c, CHN_COMPILE_TOO_BIG, NULL);
    return true;
}

/* ---- ByteClassSet ---- */

static void set_range(compiler *c, uint8_t start, uint8_t end) {
    if (start > 0) c->byte_set[start - 1] = true;
    c->byte_set[end] = true;
}

static void set_word_boundary(compiler *c) {
    unsigned b1 = 0;
    while (b1 <= 255) {
        unsigned b2 = b1 + 1;
        while (b2 <= 255 && chn_hir_is_word_byte((uint8_t)b1) == chn_hir_is_word_byte((uint8_t)b2)) ++b2;
        set_range(c, (uint8_t)b1, (uint8_t)(b2 - 1));
        b1 = b2;
    }
}

static void byte_classes(const compiler *c, uint8_t *out) {
    uint8_t cls = 0;
    for (size_t i = 0;; ++i) {
        out[i] = cls;
        if (i >= 255) break;
        if (c->byte_set[i]) ++cls;
    }
}

/* ---- instructions and holes ---- */

static bool push_inst(compiler *c, const chn_inst *inst, uint8_t state) {
    chn_prog *p = c->p;
    if (p->len == p->cap) {
        const size_t cap = p->cap ? p->cap * 2 : 16;
        chn_inst *insts = (chn_inst *)realloc(p->insts, cap * sizeof(*insts));
        if (!insts) return nomem(c);
        p->insts = insts;
        p->cap = cap;
    }
    if (p->len == c->maybe_cap) {
        const size_t cap = c->maybe_cap ? c->maybe_cap * 2 : 16;
        uint8_t *maybe = (uint8_t *)realloc(c->maybe, cap);
        if (!maybe) return nomem(c);
        c->maybe = maybe;
        c->maybe_cap = cap;
    }
    p->insts[p->len] = *inst;
    c->maybe[p->len] = state;
    ++p->len;
    return true;
}

static bool hole_one(compiler *c, size_t pc, hole *out) {
    if (c->hole_len >= c->hole_cap) {
        const size_t cap = c->hole_cap ? c->hole_cap * 2 : 64;
        size_t *pcs = (size_t *)realloc(c->hole_pc, cap * sizeof(size_t));
        if (!pcs) return nomem(c);
        c->hole_pc = pcs;
        size_t *next = (size_t *)realloc(c->hole_next, cap * sizeof(size_t));
        if (!next) return nomem(c);
        c->hole_next = next;
        c->hole_cap = cap;
    }
    const size_t node = c->hole_len++;
    c->hole_pc[node] = pc;
    c->hole_next[node] = 0;
    out->head = out->tail = node;
    return true;
}

static hole hole_none(void) {
    hole h = {0, 0};
    return h;
}

/* Hole::Many: b appended to a. */
static void hole_append(compiler *c, hole *a, hole b) {
    if (!b.head) return;
    if (!a->head) {
        *a = b;
        return;
    }
    c->hole_next[a->tail] = b.head;
    a->tail = b.tail;
}

/* Compiler::push_hole */
static bool push_hole(compiler *c, const chn_inst *inst, hole *out) {
    const size_t pc = c->p->len;
    if (!push_inst(c, inst, MAYBE_UNCOMPILED)) return false;
    return hole_one(c, pc, out);
}

/* Compiler::push_split_hole */
static bool push_split_hole(compiler *c, hole *out) {
    chn_inst inst;
    memset(&inst, 0, sizeof(inst));
    inst.kind = CHN_INST_SPLIT;
    const size_t pc = c->p->len;
    if (!push_inst(c, &inst, MAYBE_SPLIT)) return false;
    return hole_one(c, pc, out);
}

/* Compiler::pop_split_hole: Ok(None) */
static bool pop_split_hole(compiler *c, bool *some) {
    --c->p->len;
    *some = false;
    return true;
}

/* MaybeInst::fill */
static void maybe_fill(compiler *c, size_t pc, size_t goto_) {
    chn_inst *inst = &c->p->insts[pc];
    switch (c->maybe[pc]) {
    case MAYBE_SPLIT:
        inst->next = goto_;
        c->maybe[pc] = MAYBE_SPLIT1;
        return;
    case MAYBE_UNCOMPILED:
        inst->next = goto_;
        c->maybe[pc] = MAYBE_COMPILED;
        return;
    case MAYBE_SPLIT1:
        inst->next2 = goto_;
        c->maybe[pc] = MAYBE_COMPILED;
        return;
    case MAYBE_SPLIT2:
        inst->next = goto_;
        c->maybe[pc] = MAYBE_COMPILED;
        return;
    default:
        abort(); /* unreachable!: not all instructions were compiled */
    }
}

/* Compiler::fill */
static void fill(compiler *c, hole h, size_t goto_) {
    for (size_t node = h.head; node; node = c->hole_next[node]) maybe_fill(c, c->hole_pc[node], goto_);
}

/* Compiler::fill_to_next */
static void fill_to_next(compiler *c, hole h) {
    fill(c, h, c->p->len);
}

/* Compiler::fill_split */
static hole fill_split(compiler *c, hole h, size_t goto1, size_t goto2) {
    hole out = hole_none();
    size_t node = h.head;
    while (node) {
        const size_t next = c->hole_next[node];
        const size_t pc = c->hole_pc[node];
        chn_inst *inst = &c->p->insts[pc];
        if (c->maybe[pc] != MAYBE_SPLIT) abort(); /* unreachable!: must be a Split */
        if (goto1 != NO_GOTO && goto2 != NO_GOTO) {
            inst->next = goto1;
            inst->next2 = goto2;
            c->maybe[pc] = MAYBE_COMPILED;
        } else if (goto1 != NO_GOTO) {
            inst->next = goto1;
            c->maybe[pc] = MAYBE_SPLIT1;
            c->hole_next[node] = 0;
            hole one = {node, node};
            hole_append(c, &out, one);
        } else if (goto2 != NO_GOTO) {
            inst->next2 = goto2;
            c->maybe[pc] = MAYBE_SPLIT2;
            c->hole_next[node] = 0;
            hole one = {node, node};
            hole_append(c, &out, one);
        } else {
            abort(); /* unreachable!: at least one of the split holes must be filled */
        }
        node = next;
    }
    return out;
}

/* Compiler::next_inst */
static patch next_inst(const compiler *c) {
    patch p = {{0, 0}, c->p->len};
    return p;
}

/* ---- compilation ---- */

static bool c_expr(compiler *c, const chn_hir *expr, patch *out, bool *some);

/* Compiler::c_empty */
static bool c_empty(compiler *c, bool *some) {
    c->extra_inst_bytes += CHN_RUST_INST_SIZE;
    *some = false;
    return true;
}

/* Compiler::c_capture */
static bool c_capture(compiler *c, size_t first_slot, const chn_hir *expr, patch *out, bool *some) {
    if (c->p->is_dfa) return c_expr(c, expr, out, some);
    const size_t entry = c->p->len;
    chn_inst save;
    memset(&save, 0, sizeof(save));
    save.kind = CHN_INST_SAVE;
    save.slot = first_slot;
    hole h;
    if (!push_hole(c, &save, &h)) return false;
    patch sub;
    bool sub_some;
    if (!c_expr(c, expr, &sub, &sub_some)) return false;
    if (!sub_some) sub = next_inst(c);
    fill(c, h, sub.entry);
    fill_to_next(c, sub.h);
    save.slot = first_slot + 1;
    if (!push_hole(c, &save, &h)) return false;
    out->h = h;
    out->entry = entry;
    *some = true;
    return true;
}

/* ---- CompileClass ---- */

/* SuffixCache::hash (FNV-1a) */
static size_t sc_hash(size_t from_inst, uint8_t start, uint8_t end) {
    const uint64_t prime = 1099511628211ull;
    uint64_t h = 14695981039346656037ull;
    h = (h ^ (uint64_t)from_inst) * prime;
    h = (h ^ (uint64_t)start) * prime;
    h = (h ^ (uint64_t)end) * prime;
    return (size_t)(h % SUFFIX_CACHE_SIZE);
}

/* SuffixCache::get: true with *cached when the key is cached; otherwise the key is
 * recorded at pc. */
static bool sc_get(compiler *c, size_t from_inst, uint8_t start, uint8_t end, size_t pc, size_t *cached) {
    const size_t h = sc_hash(from_inst, start, end);
    const size_t pos = c->sc_sparse[h];
    if (pos < c->sc_len) {
        const sc_entry *e = &c->sc_dense[pos];
        if (e->from_inst == from_inst && e->start == start && e->end == end) {
            *cached = e->pc;
            return true;
        }
    }
    if (c->sc_len == c->sc_cap) {
        const size_t cap = c->sc_cap * 2;
        sc_entry *dense = (sc_entry *)realloc(c->sc_dense, cap * sizeof(*dense));
        if (!dense) {
            nomem(c);
            return false;
        }
        c->sc_dense = dense;
        c->sc_cap = cap;
    }
    c->sc_sparse[h] = c->sc_len;
    sc_entry *e = &c->sc_dense[c->sc_len++];
    e->from_inst = from_inst;
    e->start = start;
    e->end = end;
    e->pc = pc;
    return false;
}

/* CompileClass::c_utf8_seq / c_utf8_seq_ */
static bool c_utf8_seq(compiler *c, const chn_hir_utf8_sequence *seq, patch *out) {
    size_t from_inst = SIZE_MAX;
    hole last_hole = hole_none();
    for (size_t k = 0; k < seq->len; ++k) {
        const chn_hir_utf8_range *r = &seq->ranges[c->p->is_reverse ? k : seq->len - 1 - k];
        size_t cached;
        if (sc_get(c, from_inst, r->start, r->end, c->p->len, &cached)) {
            from_inst = cached;
            continue;
        }
        if (c->err != CHN_COMPILE_OK) return false;
        set_range(c, r->start, r->end);
        chn_inst inst;
        memset(&inst, 0, sizeof(inst));
        inst.kind = CHN_INST_BYTES;
        inst.start = r->start;
        inst.end = r->end;
        if (from_inst == SIZE_MAX) {
            if (!push_hole(c, &inst, &last_hole)) return false;
        } else {
            inst.next = from_inst;
            if (!push_inst(c, &inst, MAYBE_COMPILED)) return false;
        }
        from_inst = c->p->len - 1;
    }
    out->h = last_hole;
    out->entry = from_inst;
    return true;
}

/* CompileClass::compile */
static bool compile_class(compiler *c, const chn_char_range *ranges, size_t n, patch *out) {
    hole holes = hole_none();
    bool have_entry = false;
    size_t initial_entry = 0;
    hole last_split = hole_none();
    if (!c->seqs_ready) {
        if (chn_hir_utf8_sequences_init(&c->seqs, 0, 0) != CHN_HIR_OK) return nomem(c);
        c->seqs_ready = true;
    }
    c->sc_len = 0; /* suffix_cache.clear() */
    for (size_t i = 0; i < n; ++i) {
        const bool is_last_range = i + 1 == n;
        if (chn_hir_utf8_sequences_reset(&c->seqs, ranges[i].start, ranges[i].end) != CHN_HIR_OK) return nomem(c);
        chn_hir_utf8_sequence cur, nxt;
        int have_cur = chn_hir_utf8_sequences_next(&c->seqs, &cur);
        while (have_cur > 0) {
            const int have_next = chn_hir_utf8_sequences_next(&c->seqs, &nxt);
            if (have_next < 0) return nomem(c);
            patch p;
            if (is_last_range && have_next == 0) {
                if (!c_utf8_seq(c, &cur, &p)) return false;
                hole_append(c, &holes, p.h);
                fill(c, last_split, p.entry);
                last_split = hole_none();
                if (!have_entry) {
                    have_entry = true;
                    initial_entry = p.entry;
                }
            } else {
                if (!have_entry) {
                    have_entry = true;
                    initial_entry = c->p->len;
                }
                fill_to_next(c, last_split);
                if (!push_split_hole(c, &last_split)) return false;
                if (!c_utf8_seq(c, &cur, &p)) return false;
                hole_append(c, &holes, p.h);
                last_split = fill_split(c, last_split, p.entry, NO_GOTO);
            }
            cur = nxt;
            have_cur = have_next;
        }
        if (have_cur < 0) return nomem(c);
    }
    if (!have_entry) abort(); /* initial_entry.unwrap() on a non-empty class */
    out->h = holes;
    out->entry = initial_entry;
    return true;
}

/* Compiler::c_class */
static bool c_class(compiler *c, const chn_char_range *ranges, size_t n, patch *out, bool *some) {
    if (n == 0) return fail(c, CHN_COMPILE_SYNTAX, "empty character classes are not allowed");
    if (chn_prog_uses_bytes(c->p)) {
        *some = true;
        return compile_class(c, ranges, n, out);
    }
    chn_inst inst;
    memset(&inst, 0, sizeof(inst));
    if (n == 1 && ranges[0].start == ranges[0].end) {
        inst.kind = CHN_INST_CHAR;
        inst.c = ranges[0].start;
    } else {
        c->extra_inst_bytes += n * (sizeof(uint32_t) * 2);
        chn_prog *p = c->p;
        if (p->ranges_cap - p->ranges_len < n) {
            size_t cap = p->ranges_cap ? p->ranges_cap : 16;
            while (cap - p->ranges_len < n) cap *= 2;
            chn_char_range *rs = (chn_char_range *)realloc(p->ranges, cap * sizeof(*rs));
            if (!rs) return nomem(c);
            p->ranges = rs;
            p->ranges_cap = cap;
        }
        memcpy(p->ranges + p->ranges_len, ranges, n * sizeof(*ranges));
        inst.kind = CHN_INST_RANGES;
        inst.ranges = p->ranges_len;
        inst.nranges = n;
        p->ranges_len += n;
    }
    hole h;
    if (!push_hole(c, &inst, &h)) return false;
    out->h = h;
    out->entry = c->p->len - 1;
    *some = true;
    return true;
}

/* Compiler::c_class_bytes */
static bool c_class_bytes(compiler *c, const chn_hir_class_bytes_range *ranges, size_t n, patch *out,
    bool *some) {
    if (n == 0) return fail(c, CHN_COMPILE_SYNTAX, "empty character classes are not allowed");
    const size_t first_split_entry = c->p->len;
    hole holes = hole_none();
    hole prev_hole = hole_none();
    chn_inst inst;
    memset(&inst, 0, sizeof(inst));
    inst.kind = CHN_INST_BYTES;
    for (size_t i = 0; i + 1 < n; ++i) {
        fill_to_next(c, prev_hole);
        hole split;
        if (!push_split_hole(c, &split)) return false;
        const size_t next = c->p->len;
        set_range(c, ranges[i].start, ranges[i].end);
        inst.start = ranges[i].start;
        inst.end = ranges[i].end;
        hole h;
        if (!push_hole(c, &inst, &h)) return false;
        hole_append(c, &holes, h);
        prev_hole = fill_split(c, split, next, NO_GOTO);
    }
    const size_t next = c->p->len;
    set_range(c, ranges[n - 1].start, ranges[n - 1].end);
    inst.start = ranges[n - 1].start;
    inst.end = ranges[n - 1].end;
    hole h;
    if (!push_hole(c, &inst, &h)) return false;
    hole_append(c, &holes, h);
    fill(c, prev_hole, next);
    out->h = holes;
    out->entry = first_split_entry;
    *some = true;
    return true;
}

/* Compiler::c_char */
static bool c_char(compiler *c, uint32_t ch, patch *out, bool *some) {
    if (chn_prog_uses_bytes(c->p)) {
        if (ch < 0x80) {
            chn_inst inst;
            memset(&inst, 0, sizeof(inst));
            inst.kind = CHN_INST_BYTES;
            inst.start = inst.end = (uint8_t)ch;
            hole h;
            if (!push_hole(c, &inst, &h)) return false;
            set_range(c, (uint8_t)ch, (uint8_t)ch);
            out->h = h;
            out->entry = c->p->len - 1;
            *some = true;
            return true;
        }
        const chn_char_range r = {ch, ch};
        return c_class(c, &r, 1, out, some);
    }
    chn_inst inst;
    memset(&inst, 0, sizeof(inst));
    inst.kind = CHN_INST_CHAR;
    inst.c = ch;
    hole h;
    if (!push_hole(c, &inst, &h)) return false;
    out->h = h;
    out->entry = c->p->len - 1;
    *some = true;
    return true;
}

/* Compiler::c_byte */
static bool c_byte(compiler *c, uint8_t b, patch *out, bool *some) {
    const chn_hir_class_bytes_range r = {b, b};
    return c_class_bytes(c, &r, 1, out, some);
}

/* Compiler::c_empty_look */
static bool c_empty_look(compiler *c, chn_empty_look look, patch *out, bool *some) {
    chn_inst inst;
    memset(&inst, 0, sizeof(inst));
    inst.kind = CHN_INST_EMPTY_LOOK;
    inst.look = (uint8_t)look;
    hole h;
    if (!push_hole(c, &inst, &h)) return false;
    out->h = h;
    out->entry = c->p->len - 1;
    *some = true;
    return true;
}

/* core::str::from_utf8(bytes).is_ok() */
static bool is_utf8(const uint8_t *bytes, size_t len) {
    size_t i = 0;
    while (i < len) {
        size_t n = 0;
        if (chn_decode_utf8(bytes + i, len - i, &n) == CHN_CHAR_NONE) return false;
        i += n;
    }
    return true;
}

/* Compiler::c_literal */
static bool c_literal(compiler *c, const uint8_t *bytes, size_t len, patch *out, bool *some) {
    const bool chars = is_utf8(bytes, len);
    if (!chars && !chn_prog_uses_bytes(c->p)) abort(); /* assert!(self.compiled.uses_bytes()) */
    size_t i = 0;
    bool have = false;
    patch acc = {{0, 0}, 0};
    while (i < len) {
        patch p;
        bool p_some;
        if (chars) {
            size_t n = 0;
            const uint32_t ch = chn_decode_utf8(bytes + i, len - i, &n);
            i += n;
            if (!c_char(c, ch, &p, &p_some)) return false;
        } else {
            if (!c_byte(c, bytes[i++], &p, &p_some)) return false;
        }
        if (!p_some) continue;
        if (!have) {
            acc = p;
            have = true;
        } else {
            fill(c, acc.h, p.entry);
            acc.h = p.h;
        }
    }
    if (!have) return c_empty(c, some);
    *out = acc;
    *some = true;
    return true;
}

/* Compiler::c's Literal arm in a reverse program: bytes.to_vec(), reverse(), c_literal. */
static bool c_literal_reversed(compiler *c, const chn_hir_literal *lit, patch *out, bool *some) {
    const size_t n = lit->len;
    /* c_literal of no bytes is c_empty(). A Hir literal is never empty (Hir::literal makes
     * an empty one Hir::empty()), but deciding it here keeps the copy below always full. */
    if (n == 0) return c_empty(c, some);
    uint8_t *rev = (uint8_t *)malloc(n);
    if (!rev) return nomem(c);
    for (size_t i = 0; i < n; ++i) rev[i] = lit->bytes[n - 1 - i];
    const bool ok = c_literal(c, rev, n, out, some);
    free(rev);
    return ok;
}

/* The iterators c_concat is called with. */
typedef enum concat_mode {
    CONCAT_FORWARD,   /* es.iter() */
    CONCAT_REVERSE,   /* es.iter().rev() */
    CONCAT_REPEAT     /* iter::repeat(expr).take(count) */
} concat_mode;

/* Compiler::c_concat */
static bool c_concat(compiler *c, const chn_hir *base, size_t count, concat_mode mode, patch *out, bool *some) {
    bool have = false;
    patch acc = {{0, 0}, 0};
    for (size_t k = 0; k < count; ++k) {
        const chn_hir *e = mode == CONCAT_REPEAT ? base : &base[mode == CONCAT_REVERSE ? count - 1 - k : k];
        patch p;
        bool p_some;
        if (!c_expr(c, e, &p, &p_some)) return false;
        if (!p_some) continue;
        if (!have) {
            acc = p;
            have = true;
        } else {
            fill(c, acc.h, p.entry);
            acc.h = p.h;
        }
    }
    if (!have) return c_empty(c, some);
    *out = acc;
    *some = true;
    return true;
}

/* Compiler::c_alternate */
static bool c_alternate(compiler *c, const chn_hir *es, size_t n, patch *out, bool *some) {
    const size_t first_split_entry = c->p->len;
    hole holes = hole_none();
    hole prev_hole = hole_none();
    bool prev_second = false;
    for (size_t i = 0; i + 1 < n; ++i) {
        if (prev_second) {
            const size_t next = c->p->len;
            fill_split(c, prev_hole, NO_GOTO, next);
        } else {
            fill_to_next(c, prev_hole);
        }
        hole split;
        if (!push_split_hole(c, &split)) return false;
        patch p;
        bool p_some;
        if (!c_expr(c, &es[i], &p, &p_some)) return false;
        if (p_some) {
            hole_append(c, &holes, p.h);
            prev_hole = fill_split(c, split, p.entry, NO_GOTO);
            prev_second = false;
        } else {
            /* Hole::dup_one */
            hole split2;
            if (!hole_one(c, c->hole_pc[split.head], &split2)) return false;
            hole_append(c, &holes, split);
            prev_hole = split2;
            prev_second = true;
        }
    }
    patch p;
    bool p_some;
    if (!c_expr(c, &es[n - 1], &p, &p_some)) return false;
    if (p_some) {
        hole_append(c, &holes, p.h);
        if (prev_second) fill_split(c, prev_hole, NO_GOTO, p.entry);
        else fill(c, prev_hole, p.entry);
    } else {
        hole_append(c, &holes, prev_hole);
    }
    out->h = holes;
    out->entry = first_split_entry;
    *some = true;
    return true;
}

/* Compiler::c_repeat_zero_or_one */
static bool c_repeat_zero_or_one(compiler *c, const chn_hir *expr, bool greedy, patch *out, bool *some) {
    const size_t split_entry = c->p->len;
    hole split;
    if (!push_split_hole(c, &split)) return false;
    patch rep;
    bool rep_some;
    if (!c_expr(c, expr, &rep, &rep_some)) return false;
    if (!rep_some) return pop_split_hole(c, some);
    const hole split_hole = greedy ? fill_split(c, split, rep.entry, NO_GOTO) : fill_split(c, split, NO_GOTO, rep.entry);
    hole holes = rep.h;
    hole_append(c, &holes, split_hole);
    out->h = holes;
    out->entry = split_entry;
    *some = true;
    return true;
}

/* Compiler::c_repeat_zero_or_more */
static bool c_repeat_zero_or_more(compiler *c, const chn_hir *expr, bool greedy, patch *out, bool *some) {
    const size_t split_entry = c->p->len;
    hole split;
    if (!push_split_hole(c, &split)) return false;
    patch rep;
    bool rep_some;
    if (!c_expr(c, expr, &rep, &rep_some)) return false;
    if (!rep_some) return pop_split_hole(c, some);
    fill(c, rep.h, split_entry);
    out->h = greedy ? fill_split(c, split, rep.entry, NO_GOTO) : fill_split(c, split, NO_GOTO, rep.entry);
    out->entry = split_entry;
    *some = true;
    return true;
}

/* Compiler::c_repeat_one_or_more */
static bool c_repeat_one_or_more(compiler *c, const chn_hir *expr, bool greedy, patch *out, bool *some) {
    patch rep;
    bool rep_some;
    if (!c_expr(c, expr, &rep, &rep_some)) return false;
    if (!rep_some) {
        *some = false;
        return true;
    }
    fill_to_next(c, rep.h);
    hole split;
    if (!push_split_hole(c, &split)) return false;
    out->h = greedy ? fill_split(c, split, rep.entry, NO_GOTO) : fill_split(c, split, NO_GOTO, rep.entry);
    out->entry = rep.entry;
    *some = true;
    return true;
}

/* Compiler::c_repeat_range_min_or_more */
static bool c_repeat_range_min_or_more(compiler *c, const chn_hir *expr, bool greedy, uint32_t min,
    patch *out, bool *some) {
    patch concat;
    bool concat_some;
    if (!c_concat(c, expr, min, CONCAT_REPEAT, &concat, &concat_some)) return false;
    if (!concat_some) concat = next_inst(c);
    patch rep;
    bool rep_some;
    if (!c_repeat_zero_or_more(c, expr, greedy, &rep, &rep_some)) return false;
    if (!rep_some) {
        *some = false;
        return true;
    }
    fill(c, concat.h, rep.entry);
    out->h = rep.h;
    out->entry = concat.entry;
    *some = true;
    return true;
}

/* Compiler::c_repeat_range */
static bool c_repeat_range(compiler *c, const chn_hir *expr, bool greedy, uint32_t min, uint32_t max,
    patch *out, bool *some) {
    patch concat;
    bool concat_some;
    if (!c_concat(c, expr, min, CONCAT_REPEAT, &concat, &concat_some)) return false;
    if (min == max) {
        *out = concat;
        *some = concat_some;
        return true;
    }
    if (!concat_some) concat = next_inst(c);
    const size_t initial_entry = concat.entry;
    hole holes = hole_none();
    hole prev_hole = concat.h;
    for (uint32_t i = min; i < max; ++i) {
        fill_to_next(c, prev_hole);
        hole split;
        if (!push_split_hole(c, &split)) return false;
        patch p;
        bool p_some;
        if (!c_expr(c, expr, &p, &p_some)) return false;
        if (!p_some) return pop_split_hole(c, some);
        prev_hole = p.h;
        hole_append(c, &holes, greedy ? fill_split(c, split, p.entry, NO_GOTO) : fill_split(c, split, NO_GOTO, p.entry));
    }
    hole_append(c, &holes, prev_hole);
    out->h = holes;
    out->entry = initial_entry;
    *some = true;
    return true;
}

/* Compiler::c_repeat */
static bool c_repeat(compiler *c, const chn_hir_repetition *rep, patch *out, bool *some) {
    if (rep->min == 0 && rep->max.some && rep->max.value == 1) return c_repeat_zero_or_one(c, rep->sub, rep->greedy, out, some);
    if (rep->min == 0 && !rep->max.some) return c_repeat_zero_or_more(c, rep->sub, rep->greedy, out, some);
    if (rep->min == 1 && !rep->max.some) return c_repeat_one_or_more(c, rep->sub, rep->greedy, out, some);
    if (!rep->max.some) return c_repeat_range_min_or_more(c, rep->sub, rep->greedy, rep->min, out, some);
    return c_repeat_range(c, rep->sub, rep->greedy, rep->min, rep->max.value, out, some);
}

/* The capture group's name (Compiler::c's Capture arm). */
static bool push_capture_name(compiler *c, const chn_hir_string *name) {
    chn_prog *p = c->p;
    if (p->captures_len == p->captures_cap) {
        const size_t cap = p->captures_cap ? p->captures_cap * 2 : 4;
        chn_hir_string *caps = (chn_hir_string *)realloc(p->captures, cap * sizeof(*caps));
        if (!caps) return nomem(c);
        p->captures = caps;
        p->captures_cap = cap;
    }
    chn_hir_string copy = {NULL, 0};
    if (name->ptr) {
        if (!(copy.ptr = (char *)malloc(name->len + 1))) return nomem(c);
        memcpy(copy.ptr, name->ptr, name->len);
        copy.ptr[name->len] = '\0';
        copy.len = name->len;
    }
    p->captures[p->captures_len++] = copy;
    return true;
}

/* Compiler::c */
static bool c_expr(compiler *c, const chn_hir *expr, patch *out, bool *some) {
    if (!check_size(c)) return false;
    switch (expr->kind) {
    case CHN_HIR_EMPTY:
        return c_empty(c, some);
    case CHN_HIR_LITERAL:
        if (c->p->is_reverse) return c_literal_reversed(c, &expr->literal, out, some);
        return c_literal(c, expr->literal.bytes, expr->literal.len, out, some);
    case CHN_HIR_CLASS:
        if (expr->cls.kind == CHN_HIR_CLASS_UNICODE) {
            /* ClassUnicodeRange and chn_char_range are both { u32 start, u32 end }. */
            const chn_hir_class_unicode *u = &expr->cls.unicode;
            chn_char_range small[16];
            chn_char_range *rs = u->len <= 16 ? small : (chn_char_range *)malloc(u->len * sizeof(*rs));
            if (!rs) return nomem(c);
            for (size_t i = 0; i < u->len; ++i) {
                rs[i].start = u->ranges[i].start;
                rs[i].end = u->ranges[i].end;
            }
            const bool ok = c_class(c, rs, u->len, out, some);
            if (rs != small) free(rs);
            return ok;
        }
        if (chn_prog_uses_bytes(c->p)) return c_class_bytes(c, expr->cls.bytes.ranges, expr->cls.bytes.len, out, some);
        {
            const chn_hir_class_bytes *b = &expr->cls.bytes;
            chn_char_range small[16];
            chn_char_range *rs = b->len <= 16 ? small : (chn_char_range *)malloc(b->len * sizeof(*rs));
            if (!rs) return nomem(c);
            for (size_t i = 0; i < b->len; ++i) {
                if (b->ranges[i].end > 0x7F) abort(); /* assert!(cls.is_ascii()) */
                rs[i].start = b->ranges[i].start;
                rs[i].end = b->ranges[i].end;
            }
            const bool ok = c_class(c, rs, b->len, out, some);
            if (rs != small) free(rs);
            return ok;
        }
    case CHN_HIR_LOOK:
        switch (expr->look) {
        case CHN_HIR_LOOK_START:
            return c_empty_look(c, c->p->is_reverse ? CHN_LOOK_END_TEXT : CHN_LOOK_START_TEXT, out, some);
        case CHN_HIR_LOOK_END:
            return c_empty_look(c, c->p->is_reverse ? CHN_LOOK_START_TEXT : CHN_LOOK_END_TEXT, out, some);
        case CHN_HIR_LOOK_START_LF:
            set_range(c, '\n', '\n');
            return c_empty_look(c, c->p->is_reverse ? CHN_LOOK_END_LINE : CHN_LOOK_START_LINE, out, some);
        case CHN_HIR_LOOK_END_LF:
            set_range(c, '\n', '\n');
            return c_empty_look(c, c->p->is_reverse ? CHN_LOOK_START_LINE : CHN_LOOK_END_LINE, out, some);
        case CHN_HIR_LOOK_START_CRLF:
        case CHN_HIR_LOOK_END_CRLF:
            return fail(c, CHN_COMPILE_SYNTAX, "CRLF-aware line anchors are not supported yet");
        case CHN_HIR_LOOK_WORD_ASCII:
            set_word_boundary(c);
            return c_empty_look(c, CHN_LOOK_WORD_BOUNDARY_ASCII, out, some);
        case CHN_HIR_LOOK_WORD_ASCII_NEGATE:
            set_word_boundary(c);
            return c_empty_look(c, CHN_LOOK_NOT_WORD_BOUNDARY_ASCII, out, some);
        case CHN_HIR_LOOK_WORD_UNICODE:
            c->p->has_unicode_word_boundary = true;
            set_word_boundary(c);
            set_range(c, 0, 0x7F);
            return c_empty_look(c, CHN_LOOK_WORD_BOUNDARY, out, some);
        case CHN_HIR_LOOK_WORD_UNICODE_NEGATE:
            c->p->has_unicode_word_boundary = true;
            set_word_boundary(c);
            set_range(c, 0, 0x7F);
            return c_empty_look(c, CHN_LOOK_NOT_WORD_BOUNDARY, out, some);
        }
        abort();
    case CHN_HIR_CAPTURE:
        if (expr->capture.index >= c->p->captures_len && !push_capture_name(c, &expr->capture.name)) return false;
        return c_capture(c, 2 * (size_t)expr->capture.index, expr->capture.sub, out, some);
    case CHN_HIR_CONCAT:
        return c_concat(c, expr->concat.ptr, expr->concat.len, c->p->is_reverse ? CONCAT_REVERSE : CONCAT_FORWARD,
            out, some);
    case CHN_HIR_ALTERNATION:
        return c_alternate(c, expr->alternation.ptr, expr->alternation.len, out, some);
    case CHN_HIR_REPETITION:
        return c_repeat(c, &expr->repetition, out, some);
    }
    abort();
}

/* Compiler::c_dotstar: c(&Hir::repetition({0, None, lazy, Hir::dot(AnyChar)})).unwrap() */
static bool c_dotstar(compiler *c, patch *out) {
    chn_hir_class_unicode_range any = {0, 0x10FFFF};
    chn_hir dot;
    memset(&dot, 0, sizeof(dot));
    dot.kind = CHN_HIR_CLASS;
    dot.cls.kind = CHN_HIR_CLASS_UNICODE;
    dot.cls.unicode.ranges = &any;
    dot.cls.unicode.len = dot.cls.unicode.cap = 1;
    chn_hir rep;
    memset(&rep, 0, sizeof(rep));
    rep.kind = CHN_HIR_REPETITION;
    rep.repetition.min = 0;
    rep.repetition.max.some = false;
    rep.repetition.greedy = false;
    rep.repetition.sub = &dot;
    bool some;
    if (!c_expr(c, &rep, out, &some)) return false;
    if (!some) abort();
    return true;
}

chn_compile_status chn_compile(const chn_hir *expr, size_t size_limit, bool bytes, bool dfa,
    bool reverse, chn_prog *out, const char **message) {
    compiler c;
    memset(&c, 0, sizeof(c));
    chn_prog_init(out);
    out->is_bytes = bytes;
    out->is_dfa = dfa;
    out->is_reverse = reverse;
    c.p = out;
    c.size_limit = size_limit;
    *message = NULL;
    c.sc_sparse = (size_t *)calloc(SUFFIX_CACHE_SIZE, sizeof(size_t));
    c.sc_dense = (sc_entry *)malloc(SUFFIX_CACHE_SIZE * sizeof(sc_entry));
    c.sc_cap = SUFFIX_CACHE_SIZE;
    c.hole_len = 1; /* node 0 is the empty list */
    bool ok = c.sc_sparse && c.sc_dense ? true : nomem(&c);
    /* Compiler::compile_one */
    if (ok && chn_hir_look_set_contains(expr->props.look_set, CHN_HIR_LOOK_WORD_ASCII_NEGATE)) {
        ok = fail(&c, CHN_COMPILE_SYNTAX,
            "ASCII-only \\B is not allowed in Unicode regexes because it may result in invalid UTF-8 matches");
    }
    patch dotstar = {{0, 0}, 0};
    if (ok) {
        out->is_anchored_start = chn_hir_look_set_contains(expr->props.look_set_prefix, CHN_HIR_LOOK_START);
        out->is_anchored_end = chn_hir_look_set_contains(expr->props.look_set_suffix, CHN_HIR_LOOK_END);
        if (chn_prog_needs_dotstar(out)) {
            ok = c_dotstar(&c, &dotstar);
            out->start = dotstar.entry;
        }
    }
    if (ok) {
        const chn_hir_string none = {NULL, 0};
        ok = push_capture_name(&c, &none); /* captures = vec![None] */
    }
    patch p = {{0, 0}, 0};
    bool some = false;
    if (ok) ok = c_capture(&c, 0, expr, &p, &some);
    if (ok) {
        if (!some) p = next_inst(&c);
        if (chn_prog_needs_dotstar(out)) fill(&c, dotstar.h, p.entry);
        else out->start = p.entry;
        fill_to_next(&c, p.h);
        out->match_pc = out->len;
        chn_inst m;
        memset(&m, 0, sizeof(m));
        m.kind = CHN_INST_MATCH;
        m.match_index = 0;
        ok = push_inst(&c, &m, MAYBE_COMPILED);
    }
    if (ok) {
        /* compile_finish */
        for (size_t i = 0; i < out->len; ++i) {
            if (c.maybe[i] != MAYBE_COMPILED) abort(); /* MaybeInst::unwrap */
        }
        byte_classes(&c, out->byte_classes);
    }
    free(c.maybe);
    free(c.hole_pc);
    free(c.hole_next);
    free(c.sc_sparse);
    free(c.sc_dense);
    if (c.seqs_ready) chn_hir_utf8_sequences_drop(&c.seqs);
    if (!ok) {
        chn_prog_drop(out);
        *message = c.message;
        return c.err;
    }
    return CHN_COMPILE_OK;
}
