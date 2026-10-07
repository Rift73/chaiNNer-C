/* Port of regex-syntax 0.7.2 src/hir/translate.rs (with src/ast/visitor.rs's walk order), MIT OR Apache-2.0. */
/* AST to HIR. The walk calls the Visitor methods in ast::visit's order, recursing (the
 * nest limit, 250, bounds the depth); the translator's own HirFrame stack is kept as
 * in Rust, so literal merging and flag scoping behave identically. TranslatorBuilder's
 * defaults: utf8 on, every flag unset. With utf8 on, ast_literal_to_scalar never yields
 * a raw byte, so push_byte has no port. */
#include "ast.h"
#include "interval.h"
#include "unicode.h"
#include <stdlib.h>
#include <string.h>

/* Flags: Option<bool> per flag as -1 (None), 0, 1. */
typedef struct flags {
    int8_t case_insensitive;
    int8_t multi_line;
    int8_t dot_matches_new_line;
    int8_t swap_greed;
    int8_t unicode;
    int8_t crlf;
} flags;

typedef enum frame_kind {
    FRAME_EXPR,
    FRAME_LITERAL,
    FRAME_CLASS_UNICODE,
    FRAME_CLASS_BYTES,
    FRAME_REPETITION,
    FRAME_GROUP,
    FRAME_CONCAT,
    FRAME_ALTERNATION,
    FRAME_ALTERNATION_BRANCH
} frame_kind;

/* HirFrame */
typedef struct frame {
    frame_kind kind;
    union {
        chn_hir expr;                    /* Expr(Hir) */
        struct {
            uint8_t *ptr;
            size_t len;
            size_t cap;
        } literal;                       /* Literal(Vec<u8>) */
        chn_hir_class_unicode unicode;   /* ClassUnicode */
        chn_hir_class_bytes bytes;       /* ClassBytes */
        flags old_flags;                 /* Group { old_flags } */
    };
} frame;

typedef struct translator {
    frame *stack;
    size_t len;
    size_t cap;
    flags flags;
    chn_syntax_error *err;
} translator;

static bool fail(translator *t, chn_syntax_error_kind kind, chn_ast_span span) {
    t->err->kind = kind;
    t->err->span = span;
    t->err->has_aux = false;
    return false;
}

static bool nomem(translator *t) {
    chn_ast_span none;
    memset(&none, 0, sizeof(none));
    return fail(t, CHN_SYNTAX_ERR_NOMEM, none);
}

static bool ok_or_nomem(translator *t, chn_hir_status status) {
    return status == CHN_HIR_OK ? true : nomem(t);
}

/* ---- flags ---- */

static bool flag_on(int8_t f, bool fallback) {
    return f < 0 ? fallback : f == 1;
}

static bool f_case_insensitive(const translator *t) { return flag_on(t->flags.case_insensitive, false); }
static bool f_multi_line(const translator *t) { return flag_on(t->flags.multi_line, false); }
static bool f_dot_matches_new_line(const translator *t) { return flag_on(t->flags.dot_matches_new_line, false); }
static bool f_swap_greed(const translator *t) { return flag_on(t->flags.swap_greed, false); }
static bool f_unicode(const translator *t) { return flag_on(t->flags.unicode, true); }
static bool f_crlf(const translator *t) { return flag_on(t->flags.crlf, false); }

/* TranslatorI::set_flags: Flags::from_ast merged over the active flags; returns the old. */
static flags set_flags(translator *t, const chn_ast_flags *ast_flags) {
    const flags old = t->flags;
    flags next = {-1, -1, -1, -1, -1, -1};
    int8_t enable = 1;
    for (size_t i = 0; i < ast_flags->len; ++i) {
        switch (ast_flags->items[i].kind) {
        case CHN_AST_FLAG_NEGATION: enable = 0; break;
        case CHN_AST_FLAG_CASE_INSENSITIVE: next.case_insensitive = enable; break;
        case CHN_AST_FLAG_MULTI_LINE: next.multi_line = enable; break;
        case CHN_AST_FLAG_DOT_MATCHES_NEW_LINE: next.dot_matches_new_line = enable; break;
        case CHN_AST_FLAG_SWAP_GREED: next.swap_greed = enable; break;
        case CHN_AST_FLAG_UNICODE: next.unicode = enable; break;
        case CHN_AST_FLAG_CRLF: next.crlf = enable; break;
        case CHN_AST_FLAG_IGNORE_WHITESPACE: break;
        }
    }
    if (next.case_insensitive < 0) next.case_insensitive = old.case_insensitive;
    if (next.multi_line < 0) next.multi_line = old.multi_line;
    if (next.dot_matches_new_line < 0) next.dot_matches_new_line = old.dot_matches_new_line;
    if (next.swap_greed < 0) next.swap_greed = old.swap_greed;
    if (next.unicode < 0) next.unicode = old.unicode;
    if (next.crlf < 0) next.crlf = old.crlf;
    t->flags = next;
    return old;
}

/* ---- the frame stack ---- */

static void frame_drop(frame *f) {
    switch (f->kind) {
    case FRAME_EXPR: chn_hir_drop(&f->expr); break;
    case FRAME_LITERAL: free(f->literal.ptr); break;
    case FRAME_CLASS_UNICODE: chn_class_unicode_drop(&f->unicode); break;
    case FRAME_CLASS_BYTES: chn_class_bytes_drop(&f->bytes); break;
    default: break;
    }
}

static bool push(translator *t, const frame *f) {
    if (t->len == t->cap) {
        const size_t cap = t->cap ? t->cap * 2 : 16;
        frame *stack = (frame *)realloc(t->stack, cap * sizeof(frame));
        if (!stack) {
            frame copy = *f;
            frame_drop(&copy);
            return nomem(t);
        }
        t->stack = stack;
        t->cap = cap;
    }
    t->stack[t->len++] = *f;
    return true;
}

static bool push_marker(translator *t, frame_kind kind) {
    frame f;
    memset(&f, 0, sizeof(f));
    f.kind = kind;
    return push(t, &f);
}

/* push(HirFrame::Expr(*hir)): *hir is moved. */
static bool push_expr(translator *t, chn_hir *hir) {
    frame f;
    f.kind = FRAME_EXPR;
    f.expr = *hir;
    chn_hir_make_empty(hir);
    return push(t, &f);
}

/* push(HirFrame::ClassUnicode(empty)) or ClassBytes, by the unicode flag. */
static bool push_empty_class(translator *t) {
    frame f;
    memset(&f, 0, sizeof(f));
    if (f_unicode(t)) {
        f.kind = FRAME_CLASS_UNICODE;
        chn_class_unicode_init(&f.unicode);
    } else {
        f.kind = FRAME_CLASS_BYTES;
        chn_class_bytes_init(&f.bytes);
    }
    return push(t, &f);
}

/* TranslatorI::push_char */
static bool push_char(translator *t, uint32_t c) {
    uint8_t buf[4];
    size_t n;
    if (c < 0x80) {
        buf[0] = (uint8_t)c;
        n = 1;
    } else if (c < 0x800) {
        buf[0] = (uint8_t)(0xC0 | (c >> 6));
        buf[1] = (uint8_t)(0x80 | (c & 0x3F));
        n = 2;
    } else if (c < 0x10000) {
        buf[0] = (uint8_t)(0xE0 | (c >> 12));
        buf[1] = (uint8_t)(0x80 | ((c >> 6) & 0x3F));
        buf[2] = (uint8_t)(0x80 | (c & 0x3F));
        n = 3;
    } else {
        buf[0] = (uint8_t)(0xF0 | (c >> 18));
        buf[1] = (uint8_t)(0x80 | ((c >> 12) & 0x3F));
        buf[2] = (uint8_t)(0x80 | ((c >> 6) & 0x3F));
        buf[3] = (uint8_t)(0x80 | (c & 0x3F));
        n = 4;
    }
    if (t->len && t->stack[t->len - 1].kind == FRAME_LITERAL) {
        frame *top = &t->stack[t->len - 1];
        if (top->literal.cap - top->literal.len < n) {
            const size_t cap = top->literal.cap * 2 + n;
            uint8_t *ptr = (uint8_t *)realloc(top->literal.ptr, cap);
            if (!ptr) return nomem(t);
            top->literal.ptr = ptr;
            top->literal.cap = cap;
        }
        memcpy(top->literal.ptr + top->literal.len, buf, n);
        top->literal.len += n;
        return true;
    }
    frame f;
    f.kind = FRAME_LITERAL;
    f.literal.ptr = (uint8_t *)malloc(16);
    if (!f.literal.ptr) return nomem(t);
    memcpy(f.literal.ptr, buf, n);
    f.literal.len = n;
    f.literal.cap = 16;
    return push(t, &f);
}

/* HirFrame::unwrap_expr (Expr, or Literal through Hir::literal). */
static void frame_into_expr(frame *f, chn_hir *out) {
    if (f->kind == FRAME_EXPR) {
        *out = f->expr;
    } else {
        chn_hir_make_literal(out, f->literal.ptr, f->literal.len);
    }
}

/* Pops a frame (the stack is never empty where this is called). */
static frame pop(translator *t) {
    return t->stack[--t->len];
}

/* ---- class helpers ---- */

/* hir::Class::Unicode(cls) through Hir::class, pushed as an Expr. *cls is consumed. */
static bool push_class_unicode(translator *t, chn_hir_class_unicode *cls) {
    chn_hir_class c;
    c.kind = CHN_HIR_CLASS_UNICODE;
    c.unicode = *cls;
    chn_class_unicode_init(cls);
    chn_hir hir;
    if (!ok_or_nomem(t, chn_hir_make_class(&hir, &c))) return false;
    return push_expr(t, &hir);
}

static bool push_class_bytes(translator *t, chn_hir_class_bytes *cls) {
    chn_hir_class c;
    c.kind = CHN_HIR_CLASS_BYTES;
    c.bytes = *cls;
    chn_class_bytes_init(cls);
    chn_hir hir;
    if (!ok_or_nomem(t, chn_hir_make_class(&hir, &c))) return false;
    return push_expr(t, &hir);
}

/* TranslatorI::unicode_fold_and_negate (case folding is always available). */
static bool unicode_fold_and_negate(translator *t, bool negated, chn_hir_class_unicode *cls) {
    if (f_case_insensitive(t) && !ok_or_nomem(t, chn_class_unicode_case_fold_simple(cls))) return false;
    if (negated && !ok_or_nomem(t, chn_class_unicode_negate(cls))) return false;
    return true;
}

/* TranslatorI::bytes_fold_and_negate */
static bool bytes_fold_and_negate(translator *t, chn_ast_span span, bool negated, chn_hir_class_bytes *cls) {
    if (f_case_insensitive(t) && !ok_or_nomem(t, chn_class_bytes_case_fold_simple(cls))) return false;
    if (negated && !ok_or_nomem(t, chn_class_bytes_negate(cls))) return false;
    if (!chn_class_bytes_is_ascii(cls)) return fail(t, CHN_HIR_ERR_INVALID_UTF8, span);
    return true;
}

/* ascii_class */
static size_t ascii_class(chn_ast_ascii_kind kind, chn_hir_class_bytes_range out[6]) {
    static const uint8_t table[][13] = {
        /* count, then (start, end) pairs */
        {3, '0', '9', 'A', 'Z', 'a', 'z'},                      /* Alnum */
        {2, 'A', 'Z', 'a', 'z'},                                /* Alpha */
        {1, 0x00, 0x7F},                                        /* Ascii */
        {2, '\t', '\t', ' ', ' '},                              /* Blank */
        {2, 0x00, 0x1F, 0x7F, 0x7F},                            /* Cntrl */
        {1, '0', '9'},                                          /* Digit */
        {1, '!', '~'},                                          /* Graph */
        {1, 'a', 'z'},                                          /* Lower */
        {1, ' ', '~'},                                          /* Print */
        {4, '!', '/', ':', '@', '[', '`', '{', '~'},            /* Punct */
        {6, '\t', '\t', '\n', '\n', 0x0B, 0x0B, 0x0C, 0x0C, '\r', '\r', ' ', ' '}, /* Space */
        {1, 'A', 'Z'},                                          /* Upper */
        {4, '0', '9', 'A', 'Z', '_', '_', 'a', 'z'},            /* Word */
        {3, '0', '9', 'A', 'F', 'a', 'f'},                      /* Xdigit */
    };
    const uint8_t *row = table[kind];
    for (size_t i = 0; i < row[0]; ++i) {
        out[i].start = row[1 + 2 * i];
        out[i].end = row[2 + 2 * i];
    }
    return row[0];
}

/* hir_ascii_class_bytes / ClassBytes::new(ascii_class(kind)) */
static bool ascii_class_bytes(translator *t, chn_ast_ascii_kind kind, chn_hir_class_bytes *cls) {
    chn_hir_class_bytes_range ranges[6];
    const size_t n = ascii_class(kind, ranges);
    return ok_or_nomem(t, chn_class_bytes_new(cls, ranges, n));
}

/* TranslatorI::hir_ascii_unicode_class */
static bool hir_ascii_unicode_class(translator *t, const chn_ast_class_ascii *ast, chn_hir_class_unicode *cls) {
    chn_hir_class_bytes_range ranges[6];
    chn_hir_class_unicode_range wide[6];
    const size_t n = ascii_class(ast->kind, ranges);
    for (size_t i = 0; i < n; ++i) {
        wide[i].start = ranges[i].start;
        wide[i].end = ranges[i].end;
    }
    if (!ok_or_nomem(t, chn_class_unicode_new(cls, wide, n))) return false;
    if (unicode_fold_and_negate(t, ast->negated, cls)) return true;
    chn_class_unicode_drop(cls);
    return false;
}

/* TranslatorI::hir_ascii_byte_class */
static bool hir_ascii_byte_class(translator *t, const chn_ast_class_ascii *ast, chn_hir_class_bytes *cls) {
    if (!ascii_class_bytes(t, ast->kind, cls)) return false;
    if (bytes_fold_and_negate(t, ast->span, ast->negated, cls)) return true;
    chn_class_bytes_drop(cls);
    return false;
}

/* TranslatorI::hir_perl_unicode_class */
static bool hir_perl_unicode_class(translator *t, const chn_ast_class_perl *ast, chn_hir_class_unicode *cls) {
    chn_hir_status status;
    switch (ast->kind) {
    case CHN_AST_PERL_DIGIT: status = chn_unicode_perl_digit(cls); break;
    case CHN_AST_PERL_SPACE: status = chn_unicode_perl_space(cls); break;
    default: status = chn_unicode_perl_word(cls); break;
    }
    if (status == CHN_HIR_OK && ast->negated) status = chn_class_unicode_negate(cls);
    if (status == CHN_HIR_OK) return true;
    chn_class_unicode_drop(cls);
    return nomem(t);
}

/* TranslatorI::hir_perl_byte_class */
static bool hir_perl_byte_class(translator *t, const chn_ast_class_perl *ast, chn_hir_class_bytes *cls) {
    const chn_ast_ascii_kind kind = ast->kind == CHN_AST_PERL_DIGIT ? CHN_AST_ASCII_DIGIT
        : (ast->kind == CHN_AST_PERL_SPACE ? CHN_AST_ASCII_SPACE : CHN_AST_ASCII_WORD);
    if (!ascii_class_bytes(t, kind, cls)) return false;
    if (ast->negated && chn_class_bytes_negate(cls) != CHN_HIR_OK) {
        chn_class_bytes_drop(cls);
        return nomem(t);
    }
    if (!chn_class_bytes_is_ascii(cls)) {
        chn_class_bytes_drop(cls);
        return fail(t, CHN_HIR_ERR_INVALID_UTF8, ast->span);
    }
    return true;
}

/* TranslatorI::hir_unicode_class. As in 0.7.2, `negated` is the \P flag alone (the `!=`
 * op is not consulted). */
static bool hir_unicode_class(translator *t, const chn_ast_class_unicode *ast, chn_hir_class_unicode *cls) {
    if (!f_unicode(t)) return fail(t, CHN_HIR_ERR_UNICODE_NOT_ALLOWED, ast->span);
    chn_unicode_error err;
    switch (ast->kind) {
    case CHN_AST_UNICODE_ONE_LETTER:
        err = chn_unicode_class(CHN_UNICODE_QUERY_ONE_LETTER, ast->letter, NULL, 0, NULL, 0, cls);
        break;
    case CHN_AST_UNICODE_NAMED:
        err = chn_unicode_class(CHN_UNICODE_QUERY_BINARY, 0, ast->name.ptr, ast->name.len, NULL, 0, cls);
        break;
    default:
        err = chn_unicode_class(CHN_UNICODE_QUERY_BY_VALUE, 0, ast->name.ptr, ast->name.len, ast->value.ptr,
            ast->value.len, cls);
        break;
    }
    switch (err) {
    case CHN_UNICODE_OK: break;
    case CHN_UNICODE_PROPERTY_NOT_FOUND: return fail(t, CHN_HIR_ERR_UNICODE_PROPERTY_NOT_FOUND, ast->span);
    case CHN_UNICODE_PROPERTY_VALUE_NOT_FOUND:
        return fail(t, CHN_HIR_ERR_UNICODE_PROPERTY_VALUE_NOT_FOUND, ast->span);
    default: return nomem(t);
    }
    if (unicode_fold_and_negate(t, ast->negated, cls)) return true;
    chn_class_unicode_drop(cls);
    return false;
}

/* TranslatorI::ast_literal_to_scalar with utf8 on: the char, or InvalidUtf8 for a
 * non-ASCII \xNN outside Unicode mode. */
static bool ast_literal_to_scalar(translator *t, const chn_ast_literal *lit, uint32_t *c) {
    *c = lit->c;
    if (f_unicode(t)) return true;
    if (lit->kind != CHN_AST_LIT_HEX_FIXED || lit->hex != CHN_AST_HEX_X) return true; /* Literal::byte is None */
    if (lit->c <= 0x7F) return true;
    return fail(t, CHN_HIR_ERR_INVALID_UTF8, lit->span);
}

/* TranslatorI::class_literal_byte */
static bool class_literal_byte(translator *t, const chn_ast_literal *lit, uint8_t *byte) {
    uint32_t c;
    if (!ast_literal_to_scalar(t, lit, &c)) return false;
    if (c > 0x7F) return fail(t, CHN_HIR_ERR_UNICODE_NOT_ALLOWED, lit->span);
    *byte = (uint8_t)c;
    return true;
}

/* TranslatorI::case_fold_char: *folded tells whether *out holds the folded class. */
static bool case_fold_char(translator *t, chn_ast_span span, uint32_t c, bool *folded, chn_hir *out) {
    *folded = false;
    if (!f_case_insensitive(t)) return true;
    if (f_unicode(t)) {
        if (!chn_unicode_case_fold_overlaps(c, c)) return true;
        chn_hir_class_unicode cls;
        const chn_hir_class_unicode_range r = {c, c};
        if (!ok_or_nomem(t, chn_class_unicode_new(&cls, &r, 1))) return false;
        if (chn_class_unicode_case_fold_simple(&cls) != CHN_HIR_OK) {
            chn_class_unicode_drop(&cls);
            return nomem(t);
        }
        chn_hir_class hc;
        hc.kind = CHN_HIR_CLASS_UNICODE;
        hc.unicode = cls;
        if (!ok_or_nomem(t, chn_hir_make_class(out, &hc))) return false;
        *folded = true;
        return true;
    }
    if (c >= 0x80) return fail(t, CHN_HIR_ERR_UNICODE_NOT_ALLOWED, span);
    if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))) return true;
    chn_hir_class_bytes cls;
    const chn_hir_class_bytes_range r = {(uint8_t)c, (uint8_t)c};
    if (!ok_or_nomem(t, chn_class_bytes_new(&cls, &r, 1))) return false;
    if (chn_class_bytes_case_fold_simple(&cls) != CHN_HIR_OK) {
        chn_class_bytes_drop(&cls);
        return nomem(t);
    }
    chn_hir_class hc;
    hc.kind = CHN_HIR_CLASS_BYTES;
    hc.bytes = cls;
    if (!ok_or_nomem(t, chn_hir_make_class(out, &hc))) return false;
    *folded = true;
    return true;
}

/* ---- the Visitor ---- */

static bool visit_pre(translator *t, const chn_ast *ast) {
    switch (ast->kind) {
    case CHN_AST_CLASS_BRACKETED:
        return push_empty_class(t);
    case CHN_AST_REPETITION:
        return push_marker(t, FRAME_REPETITION);
    case CHN_AST_GROUP: {
        frame f;
        memset(&f, 0, sizeof(f));
        f.kind = FRAME_GROUP;
        f.old_flags = ast->group.kind == CHN_AST_GROUP_NON_CAPTURING ? set_flags(t, &ast->group.flags) : t->flags;
        return push(t, &f);
    }
    case CHN_AST_CONCAT:
        return ast->concat.asts.len == 0 || push_marker(t, FRAME_CONCAT);
    case CHN_AST_ALTERNATION:
        return ast->alternation.asts.len == 0
            || (push_marker(t, FRAME_ALTERNATION) && push_marker(t, FRAME_ALTERNATION_BRANCH));
    default:
        return true;
    }
}

/* Hir::concat / Hir::alternation over the exprs popped down to the opening marker. */
static bool pop_seq(translator *t, bool alternation) {
    const frame_kind open = alternation ? FRAME_ALTERNATION : FRAME_CONCAT;
    size_t count = 0, i = t->len;
    while (i > 0 && t->stack[i - 1].kind != open) {
        --i;
        if (t->stack[i].kind == FRAME_EXPR || t->stack[i].kind == FRAME_LITERAL) ++count;
    }
    chn_hir_vec exprs = {NULL, 0, 0};
    if (count) {
        exprs.ptr = (chn_hir *)malloc(count * sizeof(chn_hir));
        if (!exprs.ptr) return nomem(t);
        exprs.cap = count;
    }
    /* The frames above the marker, oldest first: (Branch, expr)* for an alternation;
     * exprs for a concat, where Empty ones are skipped. */
    for (size_t k = i; k < t->len; ++k) {
        frame *f = &t->stack[k];
        if (f->kind != FRAME_EXPR && f->kind != FRAME_LITERAL) continue;
        chn_hir hir;
        frame_into_expr(f, &hir);
        if (!alternation && hir.kind == CHN_HIR_EMPTY) continue;
        exprs.ptr[exprs.len++] = hir;
    }
    t->len = i > 0 ? i - 1 : 0; /* drop the marker too */
    chn_hir result;
    const chn_hir_status status = alternation ? chn_hir_make_alternation(&result, &exprs)
                                              : chn_hir_make_concat(&result, &exprs);
    if (!ok_or_nomem(t, status)) return false;
    return push_expr(t, &result);
}

static bool visit_post(translator *t, const chn_ast *ast) {
    chn_hir hir;
    switch (ast->kind) {
    case CHN_AST_EMPTY:
        chn_hir_make_empty(&hir);
        return push_expr(t, &hir);
    case CHN_AST_FLAGS:
        set_flags(t, &ast->set_flags.flags);
        chn_hir_make_empty(&hir);
        return push_expr(t, &hir);
    case CHN_AST_LITERAL: {
        uint32_t c;
        if (!ast_literal_to_scalar(t, &ast->literal, &c)) return false;
        if (!f_unicode(t) && c >= 0x80) return fail(t, CHN_HIR_ERR_UNICODE_NOT_ALLOWED, ast->literal.span);
        bool folded;
        if (!case_fold_char(t, ast->literal.span, c, &folded, &hir)) return false;
        return folded ? push_expr(t, &hir) : push_char(t, c);
    }
    case CHN_AST_DOT: {
        if (!f_unicode(t)) return fail(t, CHN_HIR_ERR_INVALID_UTF8, ast->dot);
        chn_hir_dot dot;
        if (f_dot_matches_new_line(t)) dot = CHN_HIR_DOT_ANY_CHAR;
        else dot = f_crlf(t) ? CHN_HIR_DOT_ANY_CHAR_EXCEPT_CRLF : CHN_HIR_DOT_ANY_CHAR_EXCEPT_LF;
        if (!ok_or_nomem(t, chn_hir_make_dot(&hir, dot))) return false;
        return push_expr(t, &hir);
    }
    case CHN_AST_ASSERTION: {
        const bool unicode = f_unicode(t), multi_line = f_multi_line(t), crlf = f_crlf(t);
        chn_hir_look look;
        switch (ast->assertion.kind) {
        case CHN_AST_ASSERT_START_LINE:
            look = multi_line ? (crlf ? CHN_HIR_LOOK_START_CRLF : CHN_HIR_LOOK_START_LF) : CHN_HIR_LOOK_START;
            break;
        case CHN_AST_ASSERT_END_LINE:
            look = multi_line ? (crlf ? CHN_HIR_LOOK_END_CRLF : CHN_HIR_LOOK_END_LF) : CHN_HIR_LOOK_END;
            break;
        case CHN_AST_ASSERT_START_TEXT: look = CHN_HIR_LOOK_START; break;
        case CHN_AST_ASSERT_END_TEXT: look = CHN_HIR_LOOK_END; break;
        case CHN_AST_ASSERT_WORD_BOUNDARY:
            look = unicode ? CHN_HIR_LOOK_WORD_UNICODE : CHN_HIR_LOOK_WORD_ASCII;
            break;
        default:
            look = unicode ? CHN_HIR_LOOK_WORD_UNICODE_NEGATE : CHN_HIR_LOOK_WORD_ASCII_NEGATE;
            break;
        }
        chn_hir_make_look(&hir, look);
        return push_expr(t, &hir);
    }
    case CHN_AST_CLASS_PERL:
        if (f_unicode(t)) {
            chn_hir_class_unicode cls;
            return hir_perl_unicode_class(t, &ast->class_perl, &cls) && push_class_unicode(t, &cls);
        } else {
            chn_hir_class_bytes cls;
            return hir_perl_byte_class(t, &ast->class_perl, &cls) && push_class_bytes(t, &cls);
        }
    case CHN_AST_CLASS_UNICODE: {
        chn_hir_class_unicode cls;
        return hir_unicode_class(t, &ast->class_unicode, &cls) && push_class_unicode(t, &cls);
    }
    case CHN_AST_CLASS_BRACKETED: {
        frame f = pop(t);
        if (f.kind == FRAME_CLASS_UNICODE) {
            if (!unicode_fold_and_negate(t, ast->class_bracketed.negated, &f.unicode)) {
                frame_drop(&f);
                return false;
            }
            return push_class_unicode(t, &f.unicode);
        }
        if (!bytes_fold_and_negate(t, ast->class_bracketed.span, ast->class_bracketed.negated, &f.bytes)) {
            frame_drop(&f);
            return false;
        }
        return push_class_bytes(t, &f.bytes);
    }
    case CHN_AST_REPETITION: {
        frame f = pop(t);
        chn_hir *sub = (chn_hir *)malloc(sizeof(chn_hir));
        if (!sub) {
            frame_drop(&f);
            return nomem(t);
        }
        frame_into_expr(&f, sub);
        t->len -= 1; /* the Repetition sentinel */
        const chn_ast_repetition *rep = &ast->repetition;
        chn_hir_repetition r;
        r.min = rep->kind == CHN_AST_REP_ONE_OR_MORE ? 1u
            : (rep->kind == CHN_AST_REP_ZERO_OR_ONE || rep->kind == CHN_AST_REP_ZERO_OR_MORE ? 0u : rep->m);
        r.max.some = rep->kind == CHN_AST_REP_ZERO_OR_ONE || rep->kind == CHN_AST_REP_EXACTLY
            || rep->kind == CHN_AST_REP_BOUNDED;
        r.max.value = rep->kind == CHN_AST_REP_ZERO_OR_ONE ? 1u
            : (rep->kind == CHN_AST_REP_EXACTLY ? rep->m : (rep->kind == CHN_AST_REP_BOUNDED ? rep->n : 0u));
        r.greedy = f_swap_greed(t) ? !rep->greedy : rep->greedy;
        r.sub = sub;
        chn_hir_make_repetition(&hir, &r);
        return push_expr(t, &hir);
    }
    case CHN_AST_GROUP: {
        frame f = pop(t);
        const frame g = pop(t);
        t->flags = g.old_flags;
        const chn_ast_group *group = &ast->group;
        if (group->kind == CHN_AST_GROUP_NON_CAPTURING) {
            frame_into_expr(&f, &hir);
            return push_expr(t, &hir);
        }
        chn_hir_capture cap;
        cap.index = group->index;
        cap.name.ptr = NULL;
        cap.name.len = 0;
        cap.sub = (chn_hir *)malloc(sizeof(chn_hir));
        if (group->kind == CHN_AST_GROUP_CAPTURE_NAME && cap.sub) {
            cap.name.ptr = (char *)malloc(group->name.len + 1);
            if (cap.name.ptr) {
                memcpy(cap.name.ptr, group->name.ptr, group->name.len);
                cap.name.ptr[group->name.len] = '\0';
                cap.name.len = group->name.len;
            }
        }
        if (!cap.sub || (group->kind == CHN_AST_GROUP_CAPTURE_NAME && !cap.name.ptr)) {
            free(cap.sub);
            frame_drop(&f);
            return nomem(t);
        }
        frame_into_expr(&f, cap.sub);
        chn_hir_make_capture(&hir, &cap);
        return push_expr(t, &hir);
    }
    case CHN_AST_CONCAT:
        return pop_seq(t, false);
    case CHN_AST_ALTERNATION:
        return pop_seq(t, true);
    }
    return true;
}

static bool visit_class_item_pre(translator *t, const chn_ast_class_item *item) {
    return item->kind != CHN_AST_ITEM_BRACKETED || push_empty_class(t);
}

static bool visit_class_item_post(translator *t, const chn_ast_class_item *item) {
    frame *top = &t->stack[t->len - 1];
    switch (item->kind) {
    case CHN_AST_ITEM_LITERAL:
    case CHN_AST_ITEM_RANGE: {
        const chn_ast_literal *start = item->kind == CHN_AST_ITEM_LITERAL ? &item->literal : &item->range.start;
        const chn_ast_literal *end = item->kind == CHN_AST_ITEM_LITERAL ? &item->literal : &item->range.end;
        if (f_unicode(t)) return ok_or_nomem(t, chn_class_unicode_push(&top->unicode, start->c, end->c));
        uint8_t s, e;
        if (!class_literal_byte(t, start, &s) || !class_literal_byte(t, end, &e)) return false;
        return ok_or_nomem(t, chn_class_bytes_push(&top->bytes, s, e));
    }
    case CHN_AST_ITEM_ASCII: {
        bool ok;
        if (f_unicode(t)) {
            chn_hir_class_unicode x;
            if (!hir_ascii_unicode_class(t, &item->ascii, &x)) return false;
            ok = ok_or_nomem(t, chn_class_unicode_union(&t->stack[t->len - 1].unicode, &x));
            chn_class_unicode_drop(&x);
        } else {
            chn_hir_class_bytes x;
            if (!hir_ascii_byte_class(t, &item->ascii, &x)) return false;
            ok = ok_or_nomem(t, chn_class_bytes_union(&t->stack[t->len - 1].bytes, &x));
            chn_class_bytes_drop(&x);
        }
        return ok;
    }
    case CHN_AST_ITEM_UNICODE: {
        chn_hir_class_unicode x;
        if (!hir_unicode_class(t, &item->unicode, &x)) return false;
        const bool ok = ok_or_nomem(t, chn_class_unicode_union(&t->stack[t->len - 1].unicode, &x));
        chn_class_unicode_drop(&x);
        return ok;
    }
    case CHN_AST_ITEM_PERL: {
        bool ok;
        if (f_unicode(t)) {
            chn_hir_class_unicode x;
            if (!hir_perl_unicode_class(t, &item->perl, &x)) return false;
            ok = ok_or_nomem(t, chn_class_unicode_union(&t->stack[t->len - 1].unicode, &x));
            chn_class_unicode_drop(&x);
        } else {
            chn_hir_class_bytes x;
            if (!hir_perl_byte_class(t, &item->perl, &x)) return false;
            ok = ok_or_nomem(t, chn_class_bytes_union(&t->stack[t->len - 1].bytes, &x));
            chn_class_bytes_drop(&x);
        }
        return ok;
    }
    case CHN_AST_ITEM_BRACKETED: {
        frame cls1 = pop(t);
        bool ok;
        if (f_unicode(t)) {
            ok = unicode_fold_and_negate(t, item->bracketed->negated, &cls1.unicode)
                && ok_or_nomem(t, chn_class_unicode_union(&t->stack[t->len - 1].unicode, &cls1.unicode));
        } else {
            ok = bytes_fold_and_negate(t, item->bracketed->span, item->bracketed->negated, &cls1.bytes)
                && ok_or_nomem(t, chn_class_bytes_union(&t->stack[t->len - 1].bytes, &cls1.bytes));
        }
        frame_drop(&cls1);
        return ok;
    }
    default: /* Empty, Union */
        return true;
    }
}

/* visit_class_set_binary_op_post */
static bool visit_class_binop_post(translator *t, const chn_ast_class_binop *op) {
    frame rhs = pop(t), lhs = pop(t);
    frame *cls = &t->stack[t->len - 1];
    chn_hir_status status = CHN_HIR_OK;
    if (f_unicode(t)) {
        if (f_case_insensitive(t)) {
            status = chn_class_unicode_case_fold_simple(&rhs.unicode);
            if (status == CHN_HIR_OK) status = chn_class_unicode_case_fold_simple(&lhs.unicode);
        }
        if (status == CHN_HIR_OK) {
            switch (op->kind) {
            case CHN_AST_BINOP_INTERSECTION: status = chn_class_unicode_intersect(&lhs.unicode, &rhs.unicode); break;
            case CHN_AST_BINOP_DIFFERENCE: status = chn_class_unicode_difference(&lhs.unicode, &rhs.unicode); break;
            default: status = chn_class_unicode_symmetric_difference(&lhs.unicode, &rhs.unicode); break;
            }
        }
        if (status == CHN_HIR_OK) status = chn_class_unicode_union(&cls->unicode, &lhs.unicode);
    } else {
        if (f_case_insensitive(t)) {
            status = chn_class_bytes_case_fold_simple(&rhs.bytes);
            if (status == CHN_HIR_OK) status = chn_class_bytes_case_fold_simple(&lhs.bytes);
        }
        if (status == CHN_HIR_OK) {
            switch (op->kind) {
            case CHN_AST_BINOP_INTERSECTION: status = chn_class_bytes_intersect(&lhs.bytes, &rhs.bytes); break;
            case CHN_AST_BINOP_DIFFERENCE: status = chn_class_bytes_difference(&lhs.bytes, &rhs.bytes); break;
            default: status = chn_class_bytes_symmetric_difference(&lhs.bytes, &rhs.bytes); break;
            }
        }
        if (status == CHN_HIR_OK) status = chn_class_bytes_union(&cls->bytes, &lhs.bytes);
    }
    frame_drop(&rhs);
    frame_drop(&lhs);
    return ok_or_nomem(t, status);
}

static bool visit_class_set(translator *t, const chn_ast_class_set *set);

static bool visit_class_item(translator *t, const chn_ast_class_item *item) {
    if (!visit_class_item_pre(t, item)) return false;
    if (item->kind == CHN_AST_ITEM_BRACKETED) {
        if (!visit_class_set(t, &item->bracketed->kind)) return false;
    } else if (item->kind == CHN_AST_ITEM_UNION) {
        for (size_t i = 0; i < item->union_.items.len; ++i) {
            if (!visit_class_item(t, &item->union_.items.ptr[i])) return false;
        }
    }
    return visit_class_item_post(t, item);
}

static bool visit_class_set(translator *t, const chn_ast_class_set *set) {
    if (!set->is_op) return visit_class_item(t, &set->item);
    /* binary_op_pre and binary_op_in each push an empty class (the result and the rhs);
     * the lhs gathers into the result's slot pushed before it. */
    return push_empty_class(t) && visit_class_set(t, &set->op->lhs) && push_empty_class(t)
        && visit_class_set(t, &set->op->rhs) && visit_class_binop_post(t, set->op);
}

static bool visit(translator *t, const chn_ast *ast) {
    if (!visit_pre(t, ast)) return false;
    switch (ast->kind) {
    case CHN_AST_CLASS_BRACKETED:
        if (!visit_class_set(t, &ast->class_bracketed.kind)) return false;
        break;
    case CHN_AST_REPETITION:
        if (!visit(t, ast->repetition.ast)) return false;
        break;
    case CHN_AST_GROUP:
        if (!visit(t, ast->group.ast)) return false;
        break;
    case CHN_AST_CONCAT:
        for (size_t i = 0; i < ast->concat.asts.len; ++i) {
            if (!visit(t, ast->concat.asts.ptr[i])) return false;
        }
        break;
    case CHN_AST_ALTERNATION:
        for (size_t i = 0; i < ast->alternation.asts.len; ++i) {
            if (i > 0 && !push_marker(t, FRAME_ALTERNATION_BRANCH)) return false; /* visit_alternation_in */
            if (!visit(t, ast->alternation.asts.ptr[i])) return false;
        }
        break;
    default:
        break;
    }
    return visit_post(t, ast);
}

bool chn_translate(const chn_ast *ast, chn_hir *out, chn_syntax_error *err) {
    translator t;
    memset(&t, 0, sizeof(t));
    t.flags.case_insensitive = t.flags.multi_line = t.flags.dot_matches_new_line = -1;
    t.flags.swap_greed = t.flags.unicode = t.flags.crlf = -1;
    t.err = err;
    const bool ok = visit(&t, ast);
    if (ok) {
        /* Visitor::finish: exactly one frame, unwrapped as an expression. */
        frame f = pop(&t);
        frame_into_expr(&f, out);
    }
    for (size_t i = 0; i < t.len; ++i) frame_drop(&t.stack[i]);
    free(t.stack);
    if (!ok) chn_hir_make_empty(out);
    return ok;
}
