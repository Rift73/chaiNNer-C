/* Port of regex-syntax 0.7.2 src/ast/parse.rs (with src/ast/visitor.rs's walk order and src/parser.rs), MIT OR Apache-2.0. */
/* The AST parser: one pass with explicit group and class stacks (no recursion), the
 * nest limit check, then translate.c; chn_hir_parse is parser.rs's Parser::parse. Every
 * error carries the span parse.rs gives it. Options are regex 1.8.4's defaults: octal
 * off and ignore_whitespace off (the x flag can still turn it on). */
#include "ast.h"
#include "unicode.h"
#include <stdlib.h>
#include <string.h>

/* GroupState: Group { concat, group, ignore_whitespace } | Alternation(alternation). */
typedef struct group_state {
    bool is_alternation;
    chn_ast_seq concat;  /* Group: the concatenation before the group */
    chn_ast *group;      /* Group: the opened group (its ast set when closed) */
    bool ignore_whitespace;
    chn_ast_seq alternation;
} group_state;

/* ClassState: Open { union, set } | Op { kind, lhs }. */
typedef struct class_state {
    bool is_op;
    chn_ast_class_union union_;      /* Open */
    chn_ast_class_bracketed *set;    /* Open */
    chn_ast_binop_kind kind;         /* Op */
    chn_ast_class_set lhs;           /* Op */
} class_state;

/* A CaptureName in the parser's sorted list. */
typedef struct capture_name {
    const char *ptr;
    size_t len;
    chn_ast_span span;
} capture_name;

typedef struct parser {
    const char *pattern;
    size_t len;
    chn_ast_position pos;
    uint32_t capture_index;
    uint32_t nest_limit;
    bool ignore_whitespace;
    group_state *groups;
    size_t groups_len, groups_cap;
    class_state *classes;
    size_t classes_len, classes_cap;
    capture_name *names;
    size_t names_len, names_cap;
    chn_arena arena;
    chn_syntax_error err;
} parser;

/* Primitive: Literal | Assertion | Dot | Perl | Unicode. */
typedef enum primitive_kind {
    PRIM_LITERAL,
    PRIM_ASSERTION,
    PRIM_DOT,
    PRIM_PERL,
    PRIM_UNICODE
} primitive_kind;

typedef struct primitive {
    primitive_kind kind;
    chn_ast_literal literal;
    chn_ast_assertion assertion;
    chn_ast_span dot;
    chn_ast_class_perl perl;
    chn_ast_class_unicode unicode;
} primitive;

/* ---- errors ---- */

static bool fail(parser *p, chn_syntax_error_kind kind, chn_ast_span span) {
    p->err.kind = kind;
    p->err.span = span;
    p->err.has_aux = false;
    return false;
}

static bool fail_aux(parser *p, chn_syntax_error_kind kind, chn_ast_span span, chn_ast_span aux) {
    fail(p, kind, span);
    p->err.has_aux = true;
    p->err.aux = aux;
    return false;
}

static bool nomem(parser *p) {
    chn_ast_span none;
    memset(&none, 0, sizeof(none));
    return fail(p, CHN_SYNTAX_ERR_NOMEM, none);
}

/* ---- characters ---- */

/* Decodes the char at byte offset i of the (valid UTF-8) pattern. */
static uint32_t char_at(const parser *p, size_t i) {
    const unsigned char *s = (const unsigned char *)p->pattern + i;
    const unsigned char b = s[0];
    if (b < 0x80) return b;
    if (b < 0xE0) return ((uint32_t)(b & 0x1F) << 6) | (s[1] & 0x3Fu);
    if (b < 0xF0) return ((uint32_t)(b & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) | (s[2] & 0x3Fu);
    return ((uint32_t)(b & 0x07) << 18) | ((uint32_t)(s[1] & 0x3F) << 12) | ((uint32_t)(s[2] & 0x3F) << 6)
        | (s[3] & 0x3Fu);
}

static size_t len_utf8(uint32_t c) {
    return c < 0x80 ? 1 : (c < 0x800 ? 2 : (c < 0x10000 ? 3 : 4));
}

/* char::is_whitespace (the White_Space property). */
static bool is_whitespace(uint32_t c) {
    return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680
        || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F
        || c == 0x3000;
}

static bool is_hex(uint32_t c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

static uint32_t hex_value(uint32_t c) {
    return c <= '9' ? c - '0' : (c >= 'a' ? c - 'a' + 10 : c - 'A' + 10);
}

/* is_capture_char */
static bool is_capture_char(uint32_t c, bool first) {
    if (first) return c == '_' || chn_unicode_is_alphabetic(c);
    return c == '_' || c == '.' || c == '[' || c == ']' || chn_unicode_is_alphabetic(c) || chn_unicode_is_numeric(c);
}

/* regex_syntax::is_meta_character */
static bool is_meta_character(uint32_t c) {
    switch (c) {
    case '\\': case '.': case '+': case '*': case '?': case '(': case ')': case '|': case '[':
    case ']': case '{': case '}': case '^': case '$': case '#': case '&': case '-': case '~':
        return true;
    default:
        return false;
    }
}

/* regex_syntax::is_escapeable_character */
static bool is_escapeable_character(uint32_t c) {
    if (is_meta_character(c)) return true;
    if (c >= 0x80) return false;
    if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) return false;
    if (c == '<' || c == '>') return false;
    return true;
}

/* ---- the parser's position ---- */

static bool is_eof(const parser *p) {
    return p->pos.offset == p->len;
}

static uint32_t cur(const parser *p) {
    return char_at(p, p->pos.offset);
}

static chn_ast_span span_new(chn_ast_position start, chn_ast_position end) {
    chn_ast_span s;
    s.start = start;
    s.end = end;
    return s;
}

/* ParserI::span */
static chn_ast_span span_here(const parser *p) {
    return span_new(p->pos, p->pos);
}

/* ParserI::span_char */
static chn_ast_span span_char(const parser *p) {
    const uint32_t c = cur(p);
    chn_ast_position next;
    next.offset = p->pos.offset + len_utf8(c);
    next.line = p->pos.line;
    next.column = p->pos.column + 1;
    if (c == '\n') {
        next.line += 1;
        next.column = 1;
    }
    return span_new(p->pos, next);
}

/* ParserI::bump */
static bool bump(parser *p) {
    if (is_eof(p)) return false;
    const uint32_t c = cur(p);
    if (c == '\n') {
        p->pos.line += 1;
        p->pos.column = 1;
    } else {
        p->pos.column += 1;
    }
    p->pos.offset += len_utf8(c);
    return !is_eof(p);
}

/* ParserI::bump_if */
static bool bump_if(parser *p, const char *prefix) {
    const size_t n = strlen(prefix);
    if (p->len - p->pos.offset < n || memcmp(p->pattern + p->pos.offset, prefix, n) != 0) return false;
    for (size_t i = 0; i < n; ++i) bump(p); /* every prefix here is ASCII */
    return true;
}

/* ParserI::is_lookaround_prefix */
static bool is_lookaround_prefix(parser *p) {
    return bump_if(p, "?=") || bump_if(p, "?!") || bump_if(p, "?<=") || bump_if(p, "?<!");
}

/* ParserI::bump_space (comments are skipped; nothing reads them). */
static void bump_space(parser *p) {
    if (!p->ignore_whitespace) return;
    while (!is_eof(p)) {
        const uint32_t c = cur(p);
        if (is_whitespace(c)) {
            bump(p);
        } else if (c == '#') {
            bump(p);
            while (!is_eof(p)) {
                const uint32_t d = cur(p);
                bump(p);
                if (d == '\n') break;
            }
        } else {
            break;
        }
    }
}

/* ParserI::bump_and_bump_space */
static bool bump_and_bump_space(parser *p) {
    if (!bump(p)) return false;
    bump_space(p);
    return !is_eof(p);
}

/* ParserI::peek: the char after the current one, or -1. */
static int64_t peek(const parser *p) {
    if (is_eof(p)) return -1;
    const size_t next = p->pos.offset + len_utf8(cur(p));
    return next < p->len ? (int64_t)char_at(p, next) : -1;
}

/* ParserI::peek_space, including its quirks: a comment ends at its first non-space char,
 * and when only spaces follow, it answers the char right after the current one. */
static int64_t peek_space(const parser *p) {
    if (!p->ignore_whitespace) return peek(p);
    if (is_eof(p)) return -1;
    size_t start = p->pos.offset + len_utf8(cur(p));
    bool in_comment = false;
    for (size_t i = start; i < p->len;) {
        const uint32_t c = char_at(p, i);
        if (is_whitespace(c)) {
            /* continue */
        } else if (!in_comment && c == '#') {
            in_comment = true;
        } else if (in_comment && c == '\n') {
            in_comment = false;
        } else {
            start = i;
            break;
        }
        i += len_utf8(c);
    }
    return start < p->len ? (int64_t)char_at(p, start) : -1;
}

/* ---- AST storage ---- */

static chn_ast *new_node(parser *p, chn_ast_kind kind) {
    chn_ast *node = (chn_ast *)chn_arena_alloc(&p->arena, sizeof(chn_ast));
    if (!node) return NULL;
    memset(node, 0, sizeof(*node));
    node->kind = kind;
    return node;
}

static bool ast_vec_push(parser *p, chn_ast_vec *v, chn_ast *node) {
    if (v->len == v->cap) {
        const size_t cap = v->cap ? v->cap * 2 : 4;
        chn_ast **ptr = (chn_ast **)chn_arena_alloc(&p->arena, cap * sizeof(chn_ast *));
        if (!ptr) return nomem(p);
        if (v->len) memcpy(ptr, v->ptr, v->len * sizeof(chn_ast *));
        v->ptr = ptr;
        v->cap = cap;
    }
    v->ptr[v->len++] = node;
    return true;
}

static bool item_vec_push(parser *p, chn_ast_item_vec *v, const chn_ast_class_item *item) {
    if (v->len == v->cap) {
        const size_t cap = v->cap ? v->cap * 2 : 4;
        chn_ast_class_item *ptr = (chn_ast_class_item *)chn_arena_alloc(&p->arena, cap * sizeof(chn_ast_class_item));
        if (!ptr) return nomem(p);
        if (v->len) memcpy(ptr, v->ptr, v->len * sizeof(chn_ast_class_item));
        v->ptr = ptr;
        v->cap = cap;
    }
    v->ptr[v->len++] = *item;
    return true;
}

/* Copies chars into the arena (the parser's scratch strings). */
static bool arena_str(parser *p, const char *s, size_t n, chn_ast_str *out) {
    char *copy = (char *)chn_arena_alloc(&p->arena, n + 1);
    if (!copy) return nomem(p);
    if (n) memcpy(copy, s, n);
    copy[n] = '\0';
    out->ptr = copy;
    out->len = n;
    return true;
}

/* Ast::span */
static chn_ast_span ast_span(const chn_ast *ast) {
    switch (ast->kind) {
    case CHN_AST_EMPTY: return ast->empty;
    case CHN_AST_FLAGS: return ast->set_flags.span;
    case CHN_AST_LITERAL: return ast->literal.span;
    case CHN_AST_DOT: return ast->dot;
    case CHN_AST_ASSERTION: return ast->assertion.span;
    case CHN_AST_CLASS_UNICODE: return ast->class_unicode.span;
    case CHN_AST_CLASS_PERL: return ast->class_perl.span;
    case CHN_AST_CLASS_BRACKETED: return ast->class_bracketed.span;
    case CHN_AST_REPETITION: return ast->repetition.span;
    case CHN_AST_GROUP: return ast->group.span;
    case CHN_AST_ALTERNATION: return ast->alternation.span;
    case CHN_AST_CONCAT: return ast->concat.span;
    }
    return ast->empty;
}

/* ClassSetItem::span */
static chn_ast_span item_span(const chn_ast_class_item *item) {
    switch (item->kind) {
    case CHN_AST_ITEM_EMPTY: return item->empty;
    case CHN_AST_ITEM_LITERAL: return item->literal.span;
    case CHN_AST_ITEM_RANGE: return item->range.span;
    case CHN_AST_ITEM_ASCII: return item->ascii.span;
    case CHN_AST_ITEM_UNICODE: return item->unicode.span;
    case CHN_AST_ITEM_PERL: return item->perl.span;
    case CHN_AST_ITEM_BRACKETED: return item->bracketed->span;
    case CHN_AST_ITEM_UNION: return item->union_.span;
    }
    return item->empty;
}

/* ClassSet::span */
static chn_ast_span set_span(const chn_ast_class_set *set) {
    return set->is_op ? set->op->span : item_span(&set->item);
}

/* Concat::into_ast and Alternation::into_ast (kind: CONCAT or ALTERNATION). */
static chn_ast *seq_into_ast(parser *p, chn_ast_seq *seq, chn_ast_kind kind) {
    if (seq->asts.len == 1) return seq->asts.ptr[0];
    chn_ast *node = new_node(p, seq->asts.len == 0 ? CHN_AST_EMPTY : kind);
    if (!node) {
        nomem(p);
        return NULL;
    }
    if (seq->asts.len == 0) node->empty = seq->span;
    else if (kind == CHN_AST_CONCAT) node->concat = *seq;
    else node->alternation = *seq;
    return node;
}

/* ClassSetUnion::push */
static bool union_push(parser *p, chn_ast_class_union *u, const chn_ast_class_item *item) {
    const chn_ast_span s = item_span(item);
    if (u->items.len == 0) u->span.start = s.start;
    u->span.end = s.end;
    return item_vec_push(p, &u->items, item);
}

/* ClassSetUnion::into_item */
static chn_ast_class_item union_into_item(const chn_ast_class_union *u) {
    chn_ast_class_item item;
    memset(&item, 0, sizeof(item));
    if (u->items.len == 0) {
        item.kind = CHN_AST_ITEM_EMPTY;
        item.empty = u->span;
    } else if (u->items.len == 1) {
        item = u->items.ptr[0];
    } else {
        item.kind = CHN_AST_ITEM_UNION;
        item.union_ = *u;
    }
    return item;
}

static chn_ast_class_union union_new(chn_ast_span span) {
    chn_ast_class_union u;
    memset(&u, 0, sizeof(u));
    u.span = span;
    return u;
}

/* ---- stacks ---- */

static bool group_push(parser *p, const group_state *g) {
    if (p->groups_len == p->groups_cap) {
        const size_t cap = p->groups_cap ? p->groups_cap * 2 : 8;
        group_state *ptr = (group_state *)realloc(p->groups, cap * sizeof(group_state));
        if (!ptr) return nomem(p);
        p->groups = ptr;
        p->groups_cap = cap;
    }
    p->groups[p->groups_len++] = *g;
    return true;
}

static bool class_push(parser *p, const class_state *c) {
    if (p->classes_len == p->classes_cap) {
        const size_t cap = p->classes_cap ? p->classes_cap * 2 : 8;
        class_state *ptr = (class_state *)realloc(p->classes, cap * sizeof(class_state));
        if (!ptr) return nomem(p);
        p->classes = ptr;
        p->classes_cap = cap;
    }
    p->classes[p->classes_len++] = *c;
    return true;
}

/* ParserI::next_capture_index */
static bool next_capture_index(parser *p, chn_ast_span span, uint32_t *index) {
    if (p->capture_index == UINT32_MAX) return fail(p, CHN_AST_ERR_CAPTURE_LIMIT_EXCEEDED, span);
    p->capture_index += 1;
    *index = p->capture_index;
    return true;
}

static int name_cmp(const char *a, size_t a_len, const char *b, size_t b_len) {
    const int c = memcmp(a, b, a_len < b_len ? a_len : b_len);
    if (c != 0) return c;
    return a_len < b_len ? -1 : (a_len > b_len ? 1 : 0);
}

/* ParserI::add_capture_name */
static bool add_capture_name(parser *p, const char *name, size_t len, chn_ast_span span) {
    size_t lo = 0, hi = p->names_len;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const int c = name_cmp(p->names[mid].ptr, p->names[mid].len, name, len);
        if (c == 0) return fail_aux(p, CHN_AST_ERR_GROUP_NAME_DUPLICATE, span, p->names[mid].span);
        if (c < 0) lo = mid + 1;
        else hi = mid;
    }
    if (p->names_len == p->names_cap) {
        const size_t cap = p->names_cap ? p->names_cap * 2 : 8;
        capture_name *ptr = (capture_name *)realloc(p->names, cap * sizeof(capture_name));
        if (!ptr) return nomem(p);
        p->names = ptr;
        p->names_cap = cap;
    }
    memmove(p->names + lo + 1, p->names + lo, (p->names_len - lo) * sizeof(capture_name));
    p->names[lo].ptr = name;
    p->names[lo].len = len;
    p->names[lo].span = span;
    p->names_len += 1;
    return true;
}

/* ---- flags ---- */

/* Flags::add_item: the index of a duplicate, or -1 after adding. */
static int flags_add_item(chn_ast_flags *flags, chn_ast_span span, chn_ast_flag kind) {
    for (size_t i = 0; i < flags->len; ++i) {
        if (flags->items[i].kind == kind) return (int)i;
    }
    flags->items[flags->len].span = span;
    flags->items[flags->len].kind = kind;
    flags->len += 1;
    return -1;
}

/* Flags::flag_state: -1 None, 0 Some(false), 1 Some(true). */
static int flags_state(const chn_ast_flags *flags, chn_ast_flag flag) {
    bool negated = false;
    for (size_t i = 0; i < flags->len; ++i) {
        if (flags->items[i].kind == CHN_AST_FLAG_NEGATION) negated = true;
        else if (flags->items[i].kind == flag) return negated ? 0 : 1;
    }
    return -1;
}

/* ParserI::parse_flag */
static bool parse_flag(parser *p, chn_ast_flag *flag) {
    switch (cur(p)) {
    case 'i': *flag = CHN_AST_FLAG_CASE_INSENSITIVE; return true;
    case 'm': *flag = CHN_AST_FLAG_MULTI_LINE; return true;
    case 's': *flag = CHN_AST_FLAG_DOT_MATCHES_NEW_LINE; return true;
    case 'U': *flag = CHN_AST_FLAG_SWAP_GREED; return true;
    case 'u': *flag = CHN_AST_FLAG_UNICODE; return true;
    case 'R': *flag = CHN_AST_FLAG_CRLF; return true;
    case 'x': *flag = CHN_AST_FLAG_IGNORE_WHITESPACE; return true;
    default: return fail(p, CHN_AST_ERR_FLAG_UNRECOGNIZED, span_char(p));
    }
}

/* ParserI::parse_flags */
static bool parse_flags(parser *p, chn_ast_flags *flags) {
    memset(flags, 0, sizeof(*flags));
    flags->span = span_here(p);
    bool last_was_negation = false;
    chn_ast_span negation_span;
    memset(&negation_span, 0, sizeof(negation_span));
    while (cur(p) != ':' && cur(p) != ')') {
        if (cur(p) == '-') {
            last_was_negation = true;
            negation_span = span_char(p);
            const int dup = flags_add_item(flags, span_char(p), CHN_AST_FLAG_NEGATION);
            if (dup >= 0) {
                return fail_aux(p, CHN_AST_ERR_FLAG_REPEATED_NEGATION, span_char(p), flags->items[dup].span);
            }
        } else {
            last_was_negation = false;
            chn_ast_flag flag;
            if (!parse_flag(p, &flag)) return false;
            const int dup = flags_add_item(flags, span_char(p), flag);
            if (dup >= 0) return fail_aux(p, CHN_AST_ERR_FLAG_DUPLICATE, span_char(p), flags->items[dup].span);
        }
        if (!bump(p)) return fail(p, CHN_AST_ERR_FLAG_UNEXPECTED_EOF, span_here(p));
    }
    if (last_was_negation) return fail(p, CHN_AST_ERR_FLAG_DANGLING_NEGATION, negation_span);
    flags->span.end = p->pos;
    return true;
}

/* ---- groups and alternations ---- */

/* ParserI::parse_capture_name */
static bool parse_capture_name(parser *p, uint32_t capture_index, chn_ast_group *group) {
    if (is_eof(p)) return fail(p, CHN_AST_ERR_GROUP_NAME_UNEXPECTED_EOF, span_here(p));
    const chn_ast_position start = p->pos;
    for (;;) {
        if (cur(p) == '>') break;
        if (!is_capture_char(cur(p), p->pos.offset == start.offset)) {
            return fail(p, CHN_AST_ERR_GROUP_NAME_INVALID, span_char(p));
        }
        if (!bump(p)) break;
    }
    const chn_ast_position end = p->pos;
    if (is_eof(p)) return fail(p, CHN_AST_ERR_GROUP_NAME_UNEXPECTED_EOF, span_here(p));
    bump(p);
    if (end.offset == start.offset) return fail(p, CHN_AST_ERR_GROUP_NAME_EMPTY, span_new(start, start));
    group->kind = CHN_AST_GROUP_CAPTURE_NAME;
    group->index = capture_index;
    group->name_span = span_new(start, end);
    group->name.ptr = p->pattern + start.offset;
    group->name.len = end.offset - start.offset;
    return add_capture_name(p, group->name.ptr, group->name.len, group->name_span);
}

/* ParserI::parse_group: *node is the Ast::Flags or Ast::Group parsed. */
static bool parse_group(parser *p, chn_ast **node) {
    const chn_ast_span open_span = span_char(p);
    bump(p);
    bump_space(p);
    if (is_lookaround_prefix(p)) {
        return fail(p, CHN_AST_ERR_UNSUPPORTED_LOOK_AROUND, span_new(open_span.start, p->pos));
    }
    const chn_ast_span inner_span = span_here(p);
    if (bump_if(p, "?P<") || bump_if(p, "?<")) {
        uint32_t capture_index;
        if (!next_capture_index(p, open_span, &capture_index)) return false;
        chn_ast *g = new_node(p, CHN_AST_GROUP);
        if (!g) return nomem(p);
        if (!parse_capture_name(p, capture_index, &g->group)) return false;
        g->group.span = open_span;
        *node = g;
        return true;
    }
    if (bump_if(p, "?")) {
        if (is_eof(p)) return fail(p, CHN_AST_ERR_GROUP_UNCLOSED, open_span);
        chn_ast_flags flags;
        if (!parse_flags(p, &flags)) return false;
        const uint32_t char_end = cur(p);
        bump(p);
        if (char_end == ')') {
            if (flags.len == 0) return fail(p, CHN_AST_ERR_REPETITION_MISSING, inner_span);
            chn_ast *f = new_node(p, CHN_AST_FLAGS);
            if (!f) return nomem(p);
            f->set_flags.span = span_new(open_span.start, p->pos);
            f->set_flags.flags = flags;
            *node = f;
            return true;
        }
        chn_ast *g = new_node(p, CHN_AST_GROUP);
        if (!g) return nomem(p);
        g->group.span = open_span;
        g->group.kind = CHN_AST_GROUP_NON_CAPTURING;
        g->group.flags = flags;
        *node = g;
        return true;
    }
    uint32_t capture_index;
    if (!next_capture_index(p, open_span, &capture_index)) return false;
    chn_ast *g = new_node(p, CHN_AST_GROUP);
    if (!g) return nomem(p);
    g->group.span = open_span;
    g->group.kind = CHN_AST_GROUP_CAPTURE_INDEX;
    g->group.index = capture_index;
    *node = g;
    return true;
}

static chn_ast_seq seq_new(chn_ast_span span) {
    chn_ast_seq s;
    memset(&s, 0, sizeof(s));
    s.span = span;
    return s;
}

/* ParserI::push_group */
static bool push_group(parser *p, chn_ast_seq *concat) {
    chn_ast *node;
    if (!parse_group(p, &node)) return false;
    if (node->kind == CHN_AST_FLAGS) {
        const int ignore = flags_state(&node->set_flags.flags, CHN_AST_FLAG_IGNORE_WHITESPACE);
        if (ignore >= 0) p->ignore_whitespace = ignore == 1;
        return ast_vec_push(p, &concat->asts, node);
    }
    const bool old_ignore_whitespace = p->ignore_whitespace;
    bool new_ignore_whitespace = old_ignore_whitespace;
    if (node->group.kind == CHN_AST_GROUP_NON_CAPTURING) {
        const int ignore = flags_state(&node->group.flags, CHN_AST_FLAG_IGNORE_WHITESPACE);
        if (ignore >= 0) new_ignore_whitespace = ignore == 1;
    }
    group_state g;
    memset(&g, 0, sizeof(g));
    g.concat = *concat;
    g.group = node;
    g.ignore_whitespace = old_ignore_whitespace;
    if (!group_push(p, &g)) return false;
    p->ignore_whitespace = new_ignore_whitespace;
    *concat = seq_new(span_here(p));
    return true;
}

/* ParserI::pop_group */
static bool pop_group(parser *p, chn_ast_seq *group_concat) {
    if (p->groups_len == 0) return fail(p, CHN_AST_ERR_GROUP_UNOPENED, span_char(p));
    group_state top = p->groups[--p->groups_len];
    bool has_alt = false;
    chn_ast_seq alt;
    memset(&alt, 0, sizeof(alt));
    if (top.is_alternation) {
        if (p->groups_len == 0 || p->groups[p->groups_len - 1].is_alternation) {
            if (p->groups_len) --p->groups_len;
            return fail(p, CHN_AST_ERR_GROUP_UNOPENED, span_char(p));
        }
        has_alt = true;
        alt = top.alternation;
        top = p->groups[--p->groups_len];
    }
    p->ignore_whitespace = top.ignore_whitespace;
    group_concat->span.end = p->pos;
    bump(p);
    chn_ast *group = top.group;
    group->group.span.end = p->pos;
    if (has_alt) {
        alt.span.end = group_concat->span.end;
        chn_ast *last = seq_into_ast(p, group_concat, CHN_AST_CONCAT);
        if (!last || !ast_vec_push(p, &alt.asts, last)) return false;
        group->group.ast = seq_into_ast(p, &alt, CHN_AST_ALTERNATION);
    } else {
        group->group.ast = seq_into_ast(p, group_concat, CHN_AST_CONCAT);
    }
    if (!group->group.ast) return false;
    chn_ast_seq prior = top.concat;
    if (!ast_vec_push(p, &prior.asts, group)) return false;
    *group_concat = prior;
    return true;
}

/* ParserI::push_or_add_alternation */
static bool push_or_add_alternation(parser *p, chn_ast_seq *concat) {
    const chn_ast_position start = concat->span.start;
    chn_ast *ast = seq_into_ast(p, concat, CHN_AST_CONCAT);
    if (!ast) return false;
    if (p->groups_len && p->groups[p->groups_len - 1].is_alternation) {
        return ast_vec_push(p, &p->groups[p->groups_len - 1].alternation.asts, ast);
    }
    group_state g;
    memset(&g, 0, sizeof(g));
    g.is_alternation = true;
    g.alternation = seq_new(span_new(start, p->pos));
    if (!ast_vec_push(p, &g.alternation.asts, ast)) return false;
    return group_push(p, &g);
}

/* ParserI::push_alternate */
static bool push_alternate(parser *p, chn_ast_seq *concat) {
    concat->span.end = p->pos;
    if (!push_or_add_alternation(p, concat)) return false;
    bump(p);
    *concat = seq_new(span_here(p));
    return true;
}

/* ParserI::pop_group_end */
static bool pop_group_end(parser *p, chn_ast_seq *concat, chn_ast **out) {
    concat->span.end = p->pos;
    if (p->groups_len == 0) {
        *out = seq_into_ast(p, concat, CHN_AST_CONCAT);
        return *out != NULL;
    }
    group_state top = p->groups[--p->groups_len];
    if (!top.is_alternation) return fail(p, CHN_AST_ERR_GROUP_UNCLOSED, top.group->group.span);
    top.alternation.span.end = p->pos;
    chn_ast *last = seq_into_ast(p, concat, CHN_AST_CONCAT);
    if (!last || !ast_vec_push(p, &top.alternation.asts, last)) return false;
    chn_ast *alt = new_node(p, CHN_AST_ALTERNATION);
    if (!alt) return nomem(p);
    alt->alternation = top.alternation;
    if (p->groups_len > 0) {
        /* An Alternation cannot follow an Alternation; a Group is unclosed. */
        return fail(p, CHN_AST_ERR_GROUP_UNCLOSED, p->groups[p->groups_len - 1].group->group.span);
    }
    *out = alt;
    return true;
}

/* ---- repetitions ---- */

/* Pops the operand of a repetition operator (RepetitionMissing when there is none). */
static bool pop_operand(parser *p, chn_ast_seq *concat, chn_ast **ast) {
    if (concat->asts.len == 0) return fail(p, CHN_AST_ERR_REPETITION_MISSING, span_here(p));
    *ast = concat->asts.ptr[--concat->asts.len];
    if ((*ast)->kind == CHN_AST_EMPTY || (*ast)->kind == CHN_AST_FLAGS) {
        return fail(p, CHN_AST_ERR_REPETITION_MISSING, span_here(p));
    }
    return true;
}

/* ParserI::parse_uncounted_repetition */
static bool parse_uncounted_repetition(parser *p, chn_ast_seq *concat, chn_ast_repetition_kind kind) {
    const chn_ast_position op_start = p->pos;
    chn_ast *ast;
    if (!pop_operand(p, concat, &ast)) return false;
    bool greedy = true;
    if (bump(p) && cur(p) == '?') {
        greedy = false;
        bump(p);
    }
    chn_ast *rep = new_node(p, CHN_AST_REPETITION);
    if (!rep) return nomem(p);
    rep->repetition.span = span_new(ast_span(ast).start, p->pos);
    rep->repetition.op_span = span_new(op_start, p->pos);
    rep->repetition.kind = kind;
    rep->repetition.greedy = greedy;
    rep->repetition.ast = ast;
    return ast_vec_push(p, &concat->asts, rep);
}

/* ParserI::parse_decimal, specialized as parse_counted_repetition does (DecimalEmpty
 * becomes RepetitionCountDecimalEmpty). */
static bool parse_decimal(parser *p, uint32_t *value) {
    while (!is_eof(p) && is_whitespace(cur(p))) bump(p);
    const chn_ast_position start = p->pos;
    uint64_t n = 0;
    bool any = false, overflow = false;
    while (!is_eof(p) && cur(p) >= '0' && cur(p) <= '9') {
        any = true;
        if (!overflow) {
            n = n * 10 + (cur(p) - '0');
            if (n > UINT32_MAX) overflow = true;
        }
        bump_and_bump_space(p);
    }
    const chn_ast_span span = span_new(start, p->pos);
    while (!is_eof(p) && is_whitespace(cur(p))) bump_and_bump_space(p);
    if (!any) return fail(p, CHN_AST_ERR_REPETITION_COUNT_DECIMAL_EMPTY, span);
    if (overflow) return fail(p, CHN_AST_ERR_DECIMAL_INVALID, span);
    *value = (uint32_t)n;
    return true;
}

/* ParserI::parse_counted_repetition */
static bool parse_counted_repetition(parser *p, chn_ast_seq *concat) {
    const chn_ast_position start = p->pos;
    chn_ast *ast;
    if (!pop_operand(p, concat, &ast)) return false;
    if (!bump_and_bump_space(p)) return fail(p, CHN_AST_ERR_REPETITION_COUNT_UNCLOSED, span_new(start, p->pos));
    uint32_t count_start, count_end = 0;
    if (!parse_decimal(p, &count_start)) return false;
    chn_ast_repetition_kind kind = CHN_AST_REP_EXACTLY;
    if (is_eof(p)) return fail(p, CHN_AST_ERR_REPETITION_COUNT_UNCLOSED, span_new(start, p->pos));
    if (cur(p) == ',') {
        if (!bump_and_bump_space(p)) {
            return fail(p, CHN_AST_ERR_REPETITION_COUNT_UNCLOSED, span_new(start, p->pos));
        }
        if (cur(p) != '}') {
            if (!parse_decimal(p, &count_end)) return false;
            kind = CHN_AST_REP_BOUNDED;
        } else {
            kind = CHN_AST_REP_AT_LEAST;
        }
    }
    if (is_eof(p) || cur(p) != '}') {
        return fail(p, CHN_AST_ERR_REPETITION_COUNT_UNCLOSED, span_new(start, p->pos));
    }
    bool greedy = true;
    if (bump_and_bump_space(p) && cur(p) == '?') {
        greedy = false;
        bump(p);
    }
    const chn_ast_span op_span = span_new(start, p->pos);
    if (kind == CHN_AST_REP_BOUNDED && count_start > count_end) {
        return fail(p, CHN_AST_ERR_REPETITION_COUNT_INVALID, op_span);
    }
    chn_ast *rep = new_node(p, CHN_AST_REPETITION);
    if (!rep) return nomem(p);
    rep->repetition.span = span_new(ast_span(ast).start, p->pos);
    rep->repetition.op_span = op_span;
    rep->repetition.kind = kind;
    rep->repetition.m = count_start;
    rep->repetition.n = count_end;
    rep->repetition.greedy = greedy;
    rep->repetition.ast = ast;
    return ast_vec_push(p, &concat->asts, rep);
}

/* ---- escapes and primitives ---- */

/* ParserI::parse_hex_digits */
static bool parse_hex_digits(parser *p, chn_ast_hex_kind kind, chn_ast_literal *lit) {
    const int digits = kind == CHN_AST_HEX_X ? 2 : (kind == CHN_AST_HEX_UNICODE_SHORT ? 4 : 8);
    const chn_ast_position start = p->pos;
    uint32_t value = 0;
    for (int i = 0; i < digits; ++i) {
        if (i > 0 && !bump_and_bump_space(p)) return fail(p, CHN_AST_ERR_ESCAPE_UNEXPECTED_EOF, span_here(p));
        if (!is_hex(cur(p))) return fail(p, CHN_AST_ERR_ESCAPE_HEX_INVALID_DIGIT, span_char(p));
        value = (value << 4) | hex_value(cur(p));
    }
    bump_and_bump_space(p);
    const chn_ast_position end = p->pos;
    if (value > 0x10FFFF || (value >= 0xD800 && value <= 0xDFFF)) {
        return fail(p, CHN_AST_ERR_ESCAPE_HEX_INVALID, span_new(start, end));
    }
    lit->span = span_new(start, end);
    lit->kind = CHN_AST_LIT_HEX_FIXED;
    lit->hex = kind;
    lit->c = value;
    return true;
}

/* ParserI::parse_hex_brace */
static bool parse_hex_brace(parser *p, chn_ast_hex_kind kind, chn_ast_literal *lit) {
    const chn_ast_position brace_pos = p->pos;
    const chn_ast_position start = span_char(p).end;
    uint64_t value = 0;
    bool any = false, overflow = false;
    while (bump_and_bump_space(p) && cur(p) != '}') {
        if (!is_hex(cur(p))) return fail(p, CHN_AST_ERR_ESCAPE_HEX_INVALID_DIGIT, span_char(p));
        any = true;
        if (!overflow) {
            value = (value << 4) | hex_value(cur(p));
            if (value > UINT32_MAX) overflow = true;
        }
    }
    if (is_eof(p)) return fail(p, CHN_AST_ERR_ESCAPE_UNEXPECTED_EOF, span_new(brace_pos, p->pos));
    const chn_ast_position end = p->pos;
    bump_and_bump_space(p);
    if (!any) return fail(p, CHN_AST_ERR_ESCAPE_HEX_EMPTY, span_new(brace_pos, p->pos));
    if (overflow || value > 0x10FFFF || (value >= 0xD800 && value <= 0xDFFF)) {
        return fail(p, CHN_AST_ERR_ESCAPE_HEX_INVALID, span_new(start, end));
    }
    lit->span = span_new(start, p->pos);
    lit->kind = CHN_AST_LIT_HEX_BRACE;
    lit->hex = kind;
    lit->c = (uint32_t)value;
    return true;
}

/* ParserI::parse_hex */
static bool parse_hex(parser *p, chn_ast_literal *lit) {
    const uint32_t c = cur(p);
    const chn_ast_hex_kind kind = c == 'x' ? CHN_AST_HEX_X : (c == 'u' ? CHN_AST_HEX_UNICODE_SHORT : CHN_AST_HEX_UNICODE_LONG);
    if (!bump_and_bump_space(p)) return fail(p, CHN_AST_ERR_ESCAPE_UNEXPECTED_EOF, span_here(p));
    if (cur(p) == '{') return parse_hex_brace(p, kind, lit);
    return parse_hex_digits(p, kind, lit);
}

/* Splits a \p{...} body at its first `!=`, `:` or `=` (in that priority). */
static bool unicode_named(parser *p, const char *body, size_t len, chn_ast_class_unicode *cls) {
    size_t i;
    for (i = 0; i + 1 < len; ++i) {
        if (body[i] == '!' && body[i + 1] == '=') {
            cls->kind = CHN_AST_UNICODE_NAMED_VALUE;
            cls->op = CHN_AST_UNICODE_OP_NOT_EQUAL;
            return arena_str(p, body, i, &cls->name) && arena_str(p, body + i + 2, len - i - 2, &cls->value);
        }
    }
    const char ops[2] = {':', '='};
    for (int k = 0; k < 2; ++k) {
        const char *at = (const char *)memchr(body, ops[k], len);
        if (at) {
            i = (size_t)(at - body);
            cls->kind = CHN_AST_UNICODE_NAMED_VALUE;
            cls->op = k == 0 ? CHN_AST_UNICODE_OP_COLON : CHN_AST_UNICODE_OP_EQUAL;
            return arena_str(p, body, i, &cls->name) && arena_str(p, body + i + 1, len - i - 1, &cls->value);
        }
    }
    cls->kind = CHN_AST_UNICODE_NAMED;
    return arena_str(p, body, len, &cls->name);
}

/* ParserI::parse_unicode_class (the name is not checked here). */
static bool parse_unicode_class(parser *p, chn_ast_class_unicode *cls) {
    memset(cls, 0, sizeof(*cls));
    cls->negated = cur(p) == 'P';
    if (!bump_and_bump_space(p)) return fail(p, CHN_AST_ERR_ESCAPE_UNEXPECTED_EOF, span_here(p));
    chn_ast_position start;
    if (cur(p) == '{') {
        start = span_char(p).end;
        /* scratch: the body's chars (spaces dropped under x) */
        char small[128];
        char *buf = small;
        size_t len = 0, cap = sizeof(small);
        bool ok = true;
        while (bump_and_bump_space(p) && cur(p) != '}') {
            const size_t n = len_utf8(cur(p));
            if (cap - len < n) {
                const size_t new_cap = cap * 2;
                char *grown = (char *)malloc(new_cap);
                if (!grown) {
                    ok = false;
                    break;
                }
                memcpy(grown, buf, len);
                if (buf != small) free(buf);
                buf = grown;
                cap = new_cap;
            }
            memcpy(buf + len, p->pattern + p->pos.offset, n);
            len += n;
        }
        if (ok && !is_eof(p)) {
            bump(p);
            ok = unicode_named(p, buf, len, cls);
            if (buf != small) free(buf);
            if (!ok) return false;
        } else {
            if (buf != small) free(buf);
            if (!ok) return nomem(p);
            return fail(p, CHN_AST_ERR_ESCAPE_UNEXPECTED_EOF, span_here(p));
        }
    } else {
        start = p->pos;
        const uint32_t c = cur(p);
        if (c == '\\') return fail(p, CHN_AST_ERR_UNICODE_CLASS_INVALID, span_char(p));
        bump_and_bump_space(p);
        cls->kind = CHN_AST_UNICODE_ONE_LETTER;
        cls->letter = c;
    }
    cls->span = span_new(start, p->pos);
    return true;
}

/* ParserI::parse_perl_class */
static void parse_perl_class(parser *p, chn_ast_class_perl *cls) {
    const uint32_t c = cur(p);
    cls->span = span_char(p);
    bump(p);
    cls->negated = c == 'D' || c == 'S' || c == 'W';
    cls->kind = (c == 'd' || c == 'D') ? CHN_AST_PERL_DIGIT : ((c == 's' || c == 'S') ? CHN_AST_PERL_SPACE : CHN_AST_PERL_WORD);
}

/* ParserI::parse_escape */
static bool parse_escape(parser *p, primitive *prim) {
    memset(prim, 0, sizeof(*prim));
    const chn_ast_position start = p->pos;
    if (!bump(p)) return fail(p, CHN_AST_ERR_ESCAPE_UNEXPECTED_EOF, span_new(start, p->pos));
    const uint32_t c = cur(p);
    if (c >= '0' && c <= '9') {
        /* '0'..='7' with octal off, and '8'..='9' */
        return fail(p, CHN_AST_ERR_UNSUPPORTED_BACKREFERENCE, span_new(start, span_char(p).end));
    }
    if (c == 'x' || c == 'u' || c == 'U') {
        prim->kind = PRIM_LITERAL;
        if (!parse_hex(p, &prim->literal)) return false;
        prim->literal.span.start = start;
        return true;
    }
    if (c == 'p' || c == 'P') {
        prim->kind = PRIM_UNICODE;
        if (!parse_unicode_class(p, &prim->unicode)) return false;
        prim->unicode.span.start = start;
        return true;
    }
    if (c == 'd' || c == 's' || c == 'w' || c == 'D' || c == 'S' || c == 'W') {
        prim->kind = PRIM_PERL;
        parse_perl_class(p, &prim->perl);
        prim->perl.span.start = start;
        return true;
    }
    bump(p);
    const chn_ast_span span = span_new(start, p->pos);
    if (is_meta_character(c) || is_escapeable_character(c)) {
        prim->kind = PRIM_LITERAL;
        prim->literal.span = span;
        prim->literal.kind = is_meta_character(c) ? CHN_AST_LIT_META : CHN_AST_LIT_SUPERFLUOUS;
        prim->literal.c = c;
        return true;
    }
    uint32_t special = 0;
    switch (c) {
    case 'a': special = 0x07; break;
    case 'f': special = 0x0C; break;
    case 't': special = '\t'; break;
    case 'n': special = '\n'; break;
    case 'r': special = '\r'; break;
    case 'v': special = 0x0B; break;
    case 'A': case 'z': case 'b': case 'B':
        prim->kind = PRIM_ASSERTION;
        prim->assertion.span = span;
        prim->assertion.kind = c == 'A' ? CHN_AST_ASSERT_START_TEXT
            : (c == 'z' ? CHN_AST_ASSERT_END_TEXT
                        : (c == 'b' ? CHN_AST_ASSERT_WORD_BOUNDARY : CHN_AST_ASSERT_NOT_WORD_BOUNDARY));
        return true;
    default:
        return fail(p, CHN_AST_ERR_ESCAPE_UNRECOGNIZED, span);
    }
    prim->kind = PRIM_LITERAL;
    prim->literal.span = span;
    prim->literal.kind = CHN_AST_LIT_SPECIAL;
    prim->literal.c = special;
    return true;
}

/* ParserI::parse_primitive */
static bool parse_primitive(parser *p, primitive *prim) {
    const uint32_t c = cur(p);
    if (c == '\\') return parse_escape(p, prim);
    memset(prim, 0, sizeof(*prim));
    if (c == '.') {
        prim->kind = PRIM_DOT;
        prim->dot = span_char(p);
    } else if (c == '^' || c == '$') {
        prim->kind = PRIM_ASSERTION;
        prim->assertion.span = span_char(p);
        prim->assertion.kind = c == '^' ? CHN_AST_ASSERT_START_LINE : CHN_AST_ASSERT_END_LINE;
    } else {
        prim->kind = PRIM_LITERAL;
        prim->literal.span = span_char(p);
        prim->literal.kind = CHN_AST_LIT_VERBATIM;
        prim->literal.c = c;
    }
    bump(p);
    return true;
}

/* Primitive::span */
static chn_ast_span prim_span(const primitive *prim) {
    switch (prim->kind) {
    case PRIM_LITERAL: return prim->literal.span;
    case PRIM_ASSERTION: return prim->assertion.span;
    case PRIM_DOT: return prim->dot;
    case PRIM_PERL: return prim->perl.span;
    case PRIM_UNICODE: return prim->unicode.span;
    }
    return prim->dot;
}

/* Primitive::into_ast */
static chn_ast *prim_into_ast(parser *p, const primitive *prim) {
    static const chn_ast_kind kinds[] = {CHN_AST_LITERAL, CHN_AST_ASSERTION, CHN_AST_DOT, CHN_AST_CLASS_PERL,
        CHN_AST_CLASS_UNICODE};
    chn_ast *node = new_node(p, kinds[prim->kind]);
    if (!node) {
        nomem(p);
        return NULL;
    }
    switch (prim->kind) {
    case PRIM_LITERAL: node->literal = prim->literal; break;
    case PRIM_ASSERTION: node->assertion = prim->assertion; break;
    case PRIM_DOT: node->dot = prim->dot; break;
    case PRIM_PERL: node->class_perl = prim->perl; break;
    case PRIM_UNICODE: node->class_unicode = prim->unicode; break;
    }
    return node;
}

/* Primitive::into_class_set_item */
static bool prim_into_class_set_item(parser *p, const primitive *prim, chn_ast_class_item *item) {
    memset(item, 0, sizeof(*item));
    switch (prim->kind) {
    case PRIM_LITERAL:
        item->kind = CHN_AST_ITEM_LITERAL;
        item->literal = prim->literal;
        return true;
    case PRIM_PERL:
        item->kind = CHN_AST_ITEM_PERL;
        item->perl = prim->perl;
        return true;
    case PRIM_UNICODE:
        item->kind = CHN_AST_ITEM_UNICODE;
        item->unicode = prim->unicode;
        return true;
    default:
        return fail(p, CHN_AST_ERR_CLASS_ESCAPE_INVALID, prim_span(prim));
    }
}

/* ---- character classes ---- */

/* ParserI::unclosed_class_error */
static bool unclosed_class_error(parser *p) {
    for (size_t i = p->classes_len; i-- > 0;) {
        if (!p->classes[i].is_op) return fail(p, CHN_AST_ERR_CLASS_UNCLOSED, p->classes[i].set->span);
    }
    return fail(p, CHN_AST_ERR_CLASS_UNCLOSED, span_here(p)); /* unreachable, as in Rust */
}

/* ParserI::pop_class_op */
static bool pop_class_op(parser *p, const chn_ast_class_set *rhs, chn_ast_class_set *out) {
    class_state *top = &p->classes[p->classes_len - 1];
    if (!top->is_op) {
        *out = *rhs;
        return true;
    }
    p->classes_len -= 1;
    chn_ast_class_binop *op = (chn_ast_class_binop *)chn_arena_alloc(&p->arena, sizeof(chn_ast_class_binop));
    if (!op) return nomem(p);
    op->kind = top->kind;
    op->lhs = top->lhs;
    op->rhs = *rhs;
    op->span = span_new(set_span(&op->lhs).start, set_span(rhs).end);
    out->is_op = true;
    out->op = op;
    return true;
}

/* ParserI::push_class_op */
static bool push_class_op(parser *p, chn_ast_binop_kind kind, chn_ast_class_union *next_union) {
    chn_ast_class_set item, lhs;
    memset(&item, 0, sizeof(item));
    item.item = union_into_item(next_union);
    if (!pop_class_op(p, &item, &lhs)) return false;
    class_state op;
    memset(&op, 0, sizeof(op));
    op.is_op = true;
    op.kind = kind;
    op.lhs = lhs;
    if (!class_push(p, &op)) return false;
    *next_union = union_new(span_here(p));
    return true;
}

/* ParserI::parse_set_class_open: the opened class in *set, the union that follows in *u. */
static bool parse_set_class_open(parser *p, chn_ast_class_bracketed **set, chn_ast_class_union *u) {
    const chn_ast_position start = p->pos;
    if (!bump_and_bump_space(p)) return fail(p, CHN_AST_ERR_CLASS_UNCLOSED, span_new(start, p->pos));
    bool negated = false;
    if (cur(p) == '^') {
        if (!bump_and_bump_space(p)) return fail(p, CHN_AST_ERR_CLASS_UNCLOSED, span_new(start, p->pos));
        negated = true;
    }
    *u = union_new(span_here(p));
    while (cur(p) == '-') {
        chn_ast_class_item item;
        memset(&item, 0, sizeof(item));
        item.kind = CHN_AST_ITEM_LITERAL;
        item.literal.span = span_char(p);
        item.literal.kind = CHN_AST_LIT_VERBATIM;
        item.literal.c = '-';
        if (!union_push(p, u, &item)) return false;
        if (!bump_and_bump_space(p)) return fail(p, CHN_AST_ERR_CLASS_UNCLOSED, span_new(start, start));
    }
    if (u->items.len == 0 && cur(p) == ']') {
        chn_ast_class_item item;
        memset(&item, 0, sizeof(item));
        item.kind = CHN_AST_ITEM_LITERAL;
        item.literal.span = span_char(p);
        item.literal.kind = CHN_AST_LIT_VERBATIM;
        item.literal.c = ']';
        if (!union_push(p, u, &item)) return false;
        if (!bump_and_bump_space(p)) return fail(p, CHN_AST_ERR_CLASS_UNCLOSED, span_new(start, p->pos));
    }
    chn_ast_class_bracketed *b = (chn_ast_class_bracketed *)chn_arena_alloc(&p->arena, sizeof(chn_ast_class_bracketed));
    if (!b) return nomem(p);
    memset(b, 0, sizeof(*b));
    b->span = span_new(start, p->pos);
    b->negated = negated;
    b->kind.is_op = false;
    b->kind.item.kind = CHN_AST_ITEM_UNION;
    b->kind.item.union_ = union_new(span_new(u->span.start, u->span.start));
    *set = b;
    return true;
}

/* ParserI::push_class_open */
static bool push_class_open(parser *p, chn_ast_class_union *u) {
    class_state open;
    memset(&open, 0, sizeof(open));
    chn_ast_class_union nested;
    if (!parse_set_class_open(p, &open.set, &nested)) return false;
    open.union_ = *u;
    if (!class_push(p, &open)) return false;
    *u = nested;
    return true;
}

/* ParserI::pop_class: *done and *out when the top-level class closed, else *u is the
 * parent union with the nested class added. */
static bool pop_class(parser *p, chn_ast_class_union *u, bool *done, chn_ast **out) {
    chn_ast_class_set item, prevset;
    memset(&item, 0, sizeof(item));
    item.item = union_into_item(u);
    if (!pop_class_op(p, &item, &prevset)) return false;
    const class_state open = p->classes[--p->classes_len];
    bump(p);
    chn_ast_class_bracketed *set = open.set;
    set->span.end = p->pos;
    set->kind = prevset;
    if (p->classes_len == 0) {
        chn_ast *node = new_node(p, CHN_AST_CLASS_BRACKETED);
        if (!node) return nomem(p);
        node->class_bracketed = *set;
        *done = true;
        *out = node;
        return true;
    }
    *u = open.union_;
    chn_ast_class_item nested;
    memset(&nested, 0, sizeof(nested));
    nested.kind = CHN_AST_ITEM_BRACKETED;
    nested.bracketed = set;
    *done = false;
    return union_push(p, u, &nested);
}

/* The body of ParserI::maybe_parse_ascii_class; on false the caller backs up. */
static bool parse_ascii_class(parser *p, chn_ast_position start, chn_ast_class_ascii *cls) {
    static const char *const names[] = {"alnum", "alpha", "ascii", "blank", "cntrl", "digit", "graph",
        "lower", "print", "punct", "space", "upper", "word", "xdigit"};
    bool negated = false;
    if (!bump(p) || cur(p) != ':') return false;
    if (!bump(p)) return false;
    if (cur(p) == '^') {
        negated = true;
        if (!bump(p)) return false;
    }
    const size_t name_start = p->pos.offset;
    while (cur(p) != ':' && bump(p)) {
    }
    if (is_eof(p)) return false;
    const size_t name_len = p->pos.offset - name_start;
    if (!bump_if(p, ":]")) return false;
    for (size_t k = 0; k < sizeof(names) / sizeof(names[0]); ++k) {
        if (strlen(names[k]) == name_len && memcmp(names[k], p->pattern + name_start, name_len) == 0) {
            cls->span = span_new(start, p->pos);
            cls->kind = (chn_ast_ascii_kind)k;
            cls->negated = negated;
            return true;
        }
    }
    return false;
}

/* ParserI::maybe_parse_ascii_class: never fails; backs up to the `[` when no ASCII class
 * is there. */
static bool maybe_parse_ascii_class(parser *p, chn_ast_class_ascii *cls) {
    const chn_ast_position start = p->pos;
    if (parse_ascii_class(p, start, cls)) return true;
    p->pos = start;
    return false;
}

/* ParserI::parse_set_class_item */
static bool parse_set_class_item(parser *p, primitive *prim) {
    if (cur(p) == '\\') return parse_escape(p, prim);
    memset(prim, 0, sizeof(*prim));
    prim->kind = PRIM_LITERAL;
    prim->literal.span = span_char(p);
    prim->literal.kind = CHN_AST_LIT_VERBATIM;
    prim->literal.c = cur(p);
    bump(p);
    return true;
}

/* ParserI::parse_set_class_range */
static bool parse_set_class_range(parser *p, chn_ast_class_item *item) {
    primitive prim1, prim2;
    if (!parse_set_class_item(p, &prim1)) return false;
    bump_space(p);
    if (is_eof(p)) return unclosed_class_error(p);
    if (cur(p) != '-' || peek_space(p) == ']' || peek_space(p) == '-') {
        return prim_into_class_set_item(p, &prim1, item);
    }
    if (!bump_and_bump_space(p)) return unclosed_class_error(p);
    if (!parse_set_class_item(p, &prim2)) return false;
    const chn_ast_span span = span_new(prim_span(&prim1).start, prim_span(&prim2).end);
    if (prim1.kind != PRIM_LITERAL) return fail(p, CHN_AST_ERR_CLASS_RANGE_LITERAL, prim_span(&prim1));
    if (prim2.kind != PRIM_LITERAL) return fail(p, CHN_AST_ERR_CLASS_RANGE_LITERAL, prim_span(&prim2));
    if (prim1.literal.c > prim2.literal.c) return fail(p, CHN_AST_ERR_CLASS_RANGE_INVALID, span);
    memset(item, 0, sizeof(*item));
    item->kind = CHN_AST_ITEM_RANGE;
    item->range.span = span;
    item->range.start = prim1.literal;
    item->range.end = prim2.literal;
    return true;
}

/* ParserI::parse_set_class */
static bool parse_set_class(parser *p, chn_ast **out) {
    chn_ast_class_union u = union_new(span_here(p));
    for (;;) {
        bump_space(p);
        if (is_eof(p)) return unclosed_class_error(p);
        const uint32_t c = cur(p);
        if (c == '[') {
            if (p->classes_len > 0) {
                chn_ast_class_item item;
                memset(&item, 0, sizeof(item));
                if (maybe_parse_ascii_class(p, &item.ascii)) {
                    item.kind = CHN_AST_ITEM_ASCII;
                    if (!union_push(p, &u, &item)) return false;
                    continue;
                }
            }
            if (!push_class_open(p, &u)) return false;
        } else if (c == ']') {
            bool done;
            if (!pop_class(p, &u, &done, out)) return false;
            if (done) return true;
        } else if ((c == '&' || c == '-' || c == '~') && peek(p) == c) {
            bump(p);
            bump(p);
            const chn_ast_binop_kind kind = c == '&' ? CHN_AST_BINOP_INTERSECTION
                : (c == '-' ? CHN_AST_BINOP_DIFFERENCE : CHN_AST_BINOP_SYMMETRIC_DIFFERENCE);
            if (!push_class_op(p, kind, &u)) return false;
        } else {
            chn_ast_class_item item;
            if (!parse_set_class_range(p, &item)) return false;
            if (!union_push(p, &u, &item)) return false;
        }
    }
}

/* ---- the nest limit (NestLimiter over ast::visit's depth-first order) ---- */

typedef struct nest_limiter {
    parser *p;
    uint32_t depth;
} nest_limiter;

static bool nest_increment(nest_limiter *n, chn_ast_span span) {
    if (n->depth == UINT32_MAX) {
        n->p->err.limit = UINT32_MAX;
        return fail(n->p, CHN_AST_ERR_NEST_LIMIT_EXCEEDED, span);
    }
    const uint32_t depth = n->depth + 1;
    if (depth > n->p->nest_limit) {
        n->p->err.limit = n->p->nest_limit;
        return fail(n->p, CHN_AST_ERR_NEST_LIMIT_EXCEEDED, span);
    }
    n->depth = depth;
    return true;
}

static bool nest_class_set(nest_limiter *n, const chn_ast_class_set *set);

/* visit_class_set_item_pre / post around an item's children. */
static bool nest_class_item(nest_limiter *n, const chn_ast_class_item *item) {
    if (item->kind == CHN_AST_ITEM_BRACKETED) {
        if (!nest_increment(n, item->bracketed->span)) return false;
        if (!nest_class_set(n, &item->bracketed->kind)) return false;
        n->depth -= 1;
    } else if (item->kind == CHN_AST_ITEM_UNION) {
        if (!nest_increment(n, item->union_.span)) return false;
        for (size_t i = 0; i < item->union_.items.len; ++i) {
            if (!nest_class_item(n, &item->union_.items.ptr[i])) return false;
        }
        n->depth -= 1;
    }
    return true;
}

static bool nest_class_set(nest_limiter *n, const chn_ast_class_set *set) {
    if (!set->is_op) return nest_class_item(n, &set->item);
    if (!nest_increment(n, set->op->span)) return false;
    if (!nest_class_set(n, &set->op->lhs) || !nest_class_set(n, &set->op->rhs)) return false;
    n->depth -= 1;
    return true;
}

static bool nest_check(nest_limiter *n, const chn_ast *ast) {
    switch (ast->kind) {
    case CHN_AST_CLASS_BRACKETED:
        if (!nest_increment(n, ast->class_bracketed.span)) return false;
        if (!nest_class_set(n, &ast->class_bracketed.kind)) return false;
        break;
    case CHN_AST_REPETITION:
        if (!nest_increment(n, ast->repetition.span)) return false;
        if (!nest_check(n, ast->repetition.ast)) return false;
        break;
    case CHN_AST_GROUP:
        if (!nest_increment(n, ast->group.span)) return false;
        if (!nest_check(n, ast->group.ast)) return false;
        break;
    case CHN_AST_ALTERNATION:
    case CHN_AST_CONCAT: {
        const chn_ast_seq *seq = ast->kind == CHN_AST_CONCAT ? &ast->concat : &ast->alternation;
        if (!nest_increment(n, seq->span)) return false;
        for (size_t i = 0; i < seq->asts.len; ++i) {
            if (!nest_check(n, seq->asts.ptr[i])) return false;
        }
        break;
    }
    default:
        return true;
    }
    n->depth -= 1;
    return true;
}

/* ---- the parse loop ---- */

/* ParserI::parse_with_comments (comments are not kept). */
static bool parse_ast(parser *p, chn_ast **out) {
    chn_ast_seq concat = seq_new(span_here(p));
    for (;;) {
        bump_space(p);
        if (is_eof(p)) break;
        bool ok = true;
        switch (cur(p)) {
        case '(': ok = push_group(p, &concat); break;
        case ')': ok = pop_group(p, &concat); break;
        case '|': ok = push_alternate(p, &concat); break;
        case '[': {
            chn_ast *cls;
            ok = parse_set_class(p, &cls) && ast_vec_push(p, &concat.asts, cls);
            break;
        }
        case '?': ok = parse_uncounted_repetition(p, &concat, CHN_AST_REP_ZERO_OR_ONE); break;
        case '*': ok = parse_uncounted_repetition(p, &concat, CHN_AST_REP_ZERO_OR_MORE); break;
        case '+': ok = parse_uncounted_repetition(p, &concat, CHN_AST_REP_ONE_OR_MORE); break;
        case '{': ok = parse_counted_repetition(p, &concat); break;
        default: {
            primitive prim;
            ok = parse_primitive(p, &prim);
            if (ok) {
                chn_ast *node = prim_into_ast(p, &prim);
                ok = node != NULL && ast_vec_push(p, &concat.asts, node);
            }
            break;
        }
        }
        if (!ok) return false;
    }
    if (!pop_group_end(p, &concat, out)) return false;
    nest_limiter n = {p, 0};
    return nest_check(&n, *out);
}

chn_hir_status chn_hir_parse(const char *pattern, size_t pattern_len, uint32_t nest_limit,
    chn_hir *out, chn_hir_string *error) {
    chn_hir_make_empty(out);
    error->ptr = NULL;
    error->len = 0;
    parser p;
    memset(&p, 0, sizeof(p));
    p.pattern = pattern;
    p.len = pattern_len;
    p.pos.offset = 0;
    p.pos.line = 1;
    p.pos.column = 1;
    p.nest_limit = nest_limit;
    chn_arena_init(&p.arena);
    chn_ast *ast = NULL;
    bool ok = parse_ast(&p, &ast);
    free(p.groups);
    free(p.classes);
    free(p.names);
    if (ok) ok = chn_translate(ast, out, &p.err);
    chn_arena_free(&p.arena);
    if (ok) return CHN_HIR_OK;
    if (p.err.kind == CHN_SYNTAX_ERR_NOMEM) return CHN_HIR_ERROR_NOMEM;
    return chn_syntax_error_display(pattern, pattern_len, &p.err, error) == CHN_HIR_OK ? CHN_HIR_ERROR_SYNTAX
                                                                                      : CHN_HIR_ERROR_NOMEM;
}
