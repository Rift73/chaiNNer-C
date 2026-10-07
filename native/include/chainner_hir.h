/* Port of regex-syntax 0.7.2 (src/hir/mod.rs, src/parser.rs, src/utf8.rs, src/lib.rs), MIT OR Apache-2.0. */
/* The HIR contract between the regex front end (X1: parse, ast, error, translate, hir,
 * interval, unicode, utf8 and unicode_tables/) and the engine (X2: compile, prog, pikevm,
 * backtrack, exec, iter, prefilter, position, dfa). It is private to chainner_regex.lib:
 * chainner_ext and _chainner_graph use chainner_regex.h only.
 *
 * Conventions:
 * - Every type, field and function is named after its Rust origin, given in its comment.
 * - Moves follow Rust. A parameter documented as "consumed" is owned by the callee from
 *   the call on, on success and on failure alike; the caller keeps only the storage,
 *   which the callee leaves empty (a chn_hir becomes Hir::empty(), a vector or string
 *   becomes {NULL, 0}).
 * - A chn_hir held by value (a Vec<Hir> element, a parse result, a constructor's output)
 *   is released with chn_hir_drop. A chn_hir behind a Box<Hir> field (Capture::sub,
 *   Repetition::sub) was malloc'd and is released with chn_hir_free.
 * - Every heap block here comes from malloc/realloc and goes back through free, inside
 *   chainner_regex.lib only.
 * - Rust aborts on allocation failure. C returns CHN_HIR_ERROR_NOMEM instead, frees what
 *   it consumed and leaves every output empty.
 * - The regex 1.8.4 engine reads only the Properties marked "engine:" below; the rest
 *   exist because X1 computes Properties inductively, exactly as Rust does. */
#ifndef CHAINNER_HIR_H
#define CHAINNER_HIR_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef enum chn_hir_status {
    CHN_HIR_OK = 0,
    CHN_HIR_ERROR_SYNTAX = 1,        /* regex_syntax::Error: Parse(ast::Error) | Translate(hir::Error) */
    CHN_HIR_ERROR_NOMEM = 2,         /* allocation failure (Rust: abort) */
    CHN_HIR_ERROR_UNIMPLEMENTED = 3  /* X0 scaffold stub; never returned once X1 lands */
} chn_hir_status;

/* String / Box<str>: UTF-8, len bytes, malloc'd with one trailing NUL that len does not
 * count. It may contain U+0000 (a pattern may). ptr == NULL with len == 0 is the empty
 * value, and None where an Option<Box<str>> is mirrored. */
typedef struct chn_hir_string {
    char *ptr;
    size_t len;
} chn_hir_string;

/* Option<usize>. Some(usize::MAX) is a real value (saturating arithmetic produces it). */
typedef struct chn_hir_option_usize {
    bool some;
    size_t value;
} chn_hir_option_usize;

/* Option<u32>. */
typedef struct chn_hir_option_u32 {
    bool some;
    uint32_t value;
} chn_hir_option_u32;

/* Look. Each value is Look::as_repr(), one bit of a LookSet. */
typedef enum chn_hir_look {
    CHN_HIR_LOOK_START = 1 << 0,              /* Look::Start */
    CHN_HIR_LOOK_END = 1 << 1,                /* Look::End */
    CHN_HIR_LOOK_START_LF = 1 << 2,           /* Look::StartLF */
    CHN_HIR_LOOK_END_LF = 1 << 3,             /* Look::EndLF */
    CHN_HIR_LOOK_START_CRLF = 1 << 4,         /* Look::StartCRLF */
    CHN_HIR_LOOK_END_CRLF = 1 << 5,           /* Look::EndCRLF */
    CHN_HIR_LOOK_WORD_ASCII = 1 << 6,         /* Look::WordAscii */
    CHN_HIR_LOOK_WORD_ASCII_NEGATE = 1 << 7,  /* Look::WordAsciiNegate */
    CHN_HIR_LOOK_WORD_UNICODE = 1 << 8,       /* Look::WordUnicode */
    CHN_HIR_LOOK_WORD_UNICODE_NEGATE = 1 << 9 /* Look::WordUnicodeNegate */
} chn_hir_look;

/* LookSet { bits: u16 }. */
typedef struct chn_hir_look_set {
    uint16_t bits; /* LookSet::bits */
} chn_hir_look_set;

/* LookSet::empty */
static inline chn_hir_look_set chn_hir_look_set_empty(void) {
    chn_hir_look_set set = {0};
    return set;
}

/* LookSet::full (every bit, !0) */
static inline chn_hir_look_set chn_hir_look_set_full(void) {
    chn_hir_look_set set = {UINT16_MAX};
    return set;
}

/* LookSet::singleton */
static inline chn_hir_look_set chn_hir_look_set_singleton(chn_hir_look look) {
    chn_hir_look_set set = {(uint16_t)look};
    return set;
}

/* LookSet::is_empty */
static inline bool chn_hir_look_set_is_empty(chn_hir_look_set set) {
    return set.bits == 0;
}

/* LookSet::contains */
static inline bool chn_hir_look_set_contains(chn_hir_look_set set, chn_hir_look look) {
    return (set.bits & (uint16_t)look) != 0;
}

/* LookSet::contains_word (Unicode or ASCII, plain or negated) */
static inline bool chn_hir_look_set_contains_word(chn_hir_look_set set) {
    return (set.bits & (CHN_HIR_LOOK_WORD_ASCII | CHN_HIR_LOOK_WORD_ASCII_NEGATE
        | CHN_HIR_LOOK_WORD_UNICODE | CHN_HIR_LOOK_WORD_UNICODE_NEGATE)) != 0;
}

/* LookSet::union */
static inline chn_hir_look_set chn_hir_look_set_union(chn_hir_look_set a, chn_hir_look_set b) {
    chn_hir_look_set set = {(uint16_t)(a.bits | b.bits)};
    return set;
}

/* LookSet::intersect */
static inline chn_hir_look_set chn_hir_look_set_intersect(chn_hir_look_set a, chn_hir_look_set b) {
    chn_hir_look_set set = {(uint16_t)(a.bits & b.bits)};
    return set;
}

/* Literal(pub Box<[u8]>). Never empty inside a Hir: Hir::literal turns an empty one
 * into Hir::empty(). */
typedef struct chn_hir_literal {
    uint8_t *bytes; /* Literal.0 (Box<[u8]> pointer) */
    size_t len;     /* Literal.0 (Box<[u8]> length) */
} chn_hir_literal;

/* ClassUnicodeRange { start: char, end: char }: Unicode scalar values, start <= end. */
typedef struct chn_hir_class_unicode_range {
    uint32_t start; /* ClassUnicodeRange::start */
    uint32_t end;   /* ClassUnicodeRange::end */
} chn_hir_class_unicode_range;

/* ClassBytesRange { start: u8, end: u8 }, start <= end. */
typedef struct chn_hir_class_bytes_range {
    uint8_t start; /* ClassBytesRange::start */
    uint8_t end;   /* ClassBytesRange::end */
} chn_hir_class_bytes_range;

/* ClassUnicode { set: IntervalSet<ClassUnicodeRange> } with
 * IntervalSet { ranges: Vec<I>, folded: bool } flattened into it. ranges is canonical
 * (sorted, non-overlapping, non-adjacent), as IntervalSet::canonicalize leaves it. */
typedef struct chn_hir_class_unicode {
    chn_hir_class_unicode_range *ranges; /* ClassUnicode::set.ranges (Vec pointer) */
    size_t len;                          /* ClassUnicode::set.ranges (Vec length) */
    size_t cap;                          /* ClassUnicode::set.ranges (Vec capacity) */
    bool folded;                         /* ClassUnicode::set.folded */
} chn_hir_class_unicode;

/* ClassBytes { set: IntervalSet<ClassBytesRange> }, flattened as above. */
typedef struct chn_hir_class_bytes {
    chn_hir_class_bytes_range *ranges; /* ClassBytes::set.ranges (Vec pointer) */
    size_t len;                        /* ClassBytes::set.ranges (Vec length) */
    size_t cap;                        /* ClassBytes::set.ranges (Vec capacity) */
    bool folded;                       /* ClassBytes::set.folded */
} chn_hir_class_bytes;

/* Class's discriminant. */
typedef enum chn_hir_class_kind {
    CHN_HIR_CLASS_UNICODE = 0, /* Class::Unicode(ClassUnicode) */
    CHN_HIR_CLASS_BYTES = 1    /* Class::Bytes(ClassBytes) */
} chn_hir_class_kind;

/* Class. An empty class (len == 0) matches nothing; Hir::fail() is an empty Bytes class. */
typedef struct chn_hir_class {
    chn_hir_class_kind kind; /* Class (variant) */
    union {
        chn_hir_class_unicode unicode; /* Class::Unicode.0 */
        chn_hir_class_bytes bytes;     /* Class::Bytes.0 */
    };
} chn_hir_class;

typedef struct chn_hir chn_hir;

/* Capture { index: u32, name: Option<Box<str>>, sub: Box<Hir> }. */
typedef struct chn_hir_capture {
    uint32_t index;      /* Capture::index (1-based; 0 is the implicit whole match) */
    chn_hir_string name; /* Capture::name; ptr == NULL is None (the parser rejects empty names) */
    chn_hir *sub;        /* Capture::sub (Box<Hir>: malloc'd, never NULL) */
} chn_hir_capture;

/* Repetition { min: u32, max: Option<u32>, greedy: bool, sub: Box<Hir> }. */
typedef struct chn_hir_repetition {
    uint32_t min;           /* Repetition::min */
    chn_hir_option_u32 max; /* Repetition::max (None: unbounded) */
    bool greedy;            /* Repetition::greedy */
    chn_hir *sub;           /* Repetition::sub (Box<Hir>: malloc'd, never NULL) */
} chn_hir_repetition;

/* Vec<Hir>: len values stored contiguously; cap is the allocation in elements. */
typedef struct chn_hir_vec {
    chn_hir *ptr; /* Vec pointer */
    size_t len;   /* Vec length */
    size_t cap;   /* Vec capacity */
} chn_hir_vec;

/* Properties(Box<PropertiesI>): PropertiesI is kept by value in each chn_hir. */
typedef struct chn_hir_properties {
    chn_hir_option_usize minimum_len;                  /* PropertiesI::minimum_len */
    chn_hir_option_usize maximum_len;                  /* PropertiesI::maximum_len */
    chn_hir_look_set look_set;                         /* PropertiesI::look_set (engine: compile.rs, exec.rs) */
    chn_hir_look_set look_set_prefix;                  /* PropertiesI::look_set_prefix (engine: compile.rs is_anchored_start, exec.rs) */
    chn_hir_look_set look_set_suffix;                  /* PropertiesI::look_set_suffix (engine: compile.rs is_anchored_end, exec.rs) */
    chn_hir_look_set look_set_prefix_any;              /* PropertiesI::look_set_prefix_any (engine: exec.rs) */
    chn_hir_look_set look_set_suffix_any;              /* PropertiesI::look_set_suffix_any (engine: exec.rs) */
    bool utf8;                                         /* PropertiesI::utf8, Properties::is_utf8 (engine: exec.rs) */
    size_t explicit_captures_len;                      /* PropertiesI::explicit_captures_len */
    chn_hir_option_usize static_explicit_captures_len; /* PropertiesI::static_explicit_captures_len (engine: compile.rs) */
    bool literal;                                      /* PropertiesI::literal, Properties::is_literal */
    bool alternation_literal;                          /* PropertiesI::alternation_literal (engine: exec.rs alternation_literals) */
} chn_hir_properties;

/* HirKind's discriminant. */
typedef enum chn_hir_kind {
    CHN_HIR_EMPTY = 0,      /* HirKind::Empty */
    CHN_HIR_LITERAL = 1,    /* HirKind::Literal(Literal) */
    CHN_HIR_CLASS = 2,      /* HirKind::Class(Class) */
    CHN_HIR_LOOK = 3,       /* HirKind::Look(Look) */
    CHN_HIR_REPETITION = 4, /* HirKind::Repetition(Repetition) */
    CHN_HIR_CAPTURE = 5,    /* HirKind::Capture(Capture) */
    CHN_HIR_CONCAT = 6,     /* HirKind::Concat(Vec<Hir>), always >= 2 subs */
    CHN_HIR_ALTERNATION = 7 /* HirKind::Alternation(Vec<Hir>), always >= 2 subs */
} chn_hir_kind;

/* Hir { kind: HirKind, props: Properties }. Build it only through the chn_hir_make_*
 * smart constructors (as Rust requires): they keep the invariants and compute props. */
struct chn_hir {
    chn_hir_kind kind; /* Hir::kind (variant) */
    union {
        chn_hir_literal literal;       /* HirKind::Literal.0 */
        chn_hir_class cls;             /* HirKind::Class.0 */
        chn_hir_look look;             /* HirKind::Look.0 */
        chn_hir_repetition repetition; /* HirKind::Repetition.0 */
        chn_hir_capture capture;       /* HirKind::Capture.0 */
        chn_hir_vec concat;            /* HirKind::Concat.0 */
        chn_hir_vec alternation;       /* HirKind::Alternation.0 */
    };
    chn_hir_properties props; /* Hir::props */
};

/* Dot: the flavours of `.` taken by Hir::dot. */
typedef enum chn_hir_dot {
    CHN_HIR_DOT_ANY_CHAR = 0,             /* Dot::AnyChar */
    CHN_HIR_DOT_ANY_BYTE = 1,             /* Dot::AnyByte */
    CHN_HIR_DOT_ANY_CHAR_EXCEPT_LF = 2,   /* Dot::AnyCharExceptLF */
    CHN_HIR_DOT_ANY_CHAR_EXCEPT_CRLF = 3, /* Dot::AnyCharExceptCRLF */
    CHN_HIR_DOT_ANY_BYTE_EXCEPT_LF = 4,   /* Dot::AnyByteExceptLF */
    CHN_HIR_DOT_ANY_BYTE_EXCEPT_CRLF = 5  /* Dot::AnyByteExceptCRLF */
} chn_hir_dot;

/* Smart constructors (hir.c). Each writes a complete Hir to *out, overwriting it without
 * releasing what it held. */

/* Hir::empty */
void chn_hir_make_empty(chn_hir *out);
/* Hir::fail: an empty Class::Bytes with Properties::class's props. */
void chn_hir_make_fail(chn_hir *out);
/* Hir::literal. bytes (malloc'd; NULL when len == 0) is consumed; len == 0 gives Hir::empty(). */
void chn_hir_make_literal(chn_hir *out, uint8_t *bytes, size_t len);
/* Hir::class. *cls is consumed. Empty -> Hir::fail(); one element -> Hir::literal of it
 * (which allocates, hence the status). */
chn_hir_status chn_hir_make_class(chn_hir *out, chn_hir_class *cls);
/* Hir::look */
void chn_hir_make_look(chn_hir *out, chn_hir_look look);
/* Hir::repetition. rep->sub is consumed: {0, Some(0)} drops it for Hir::empty();
 * {1, Some(1)} moves *rep->sub into *out and frees the box. */
void chn_hir_make_repetition(chn_hir *out, chn_hir_repetition *rep);
/* Hir::capture. cap->name and cap->sub are consumed. */
void chn_hir_make_capture(chn_hir *out, chn_hir_capture *cap);
/* Hir::concat. *subs and every element are consumed (literal merging, one-level flattening,
 * Empty skipping, the 0- and 1-element rewrites). */
chn_hir_status chn_hir_make_concat(chn_hir *out, chn_hir_vec *subs);
/* Hir::alternation. *subs and every element are consumed (flattening, singleton and class
 * merging, lift_common_prefix, the 0- and 1-element rewrites). */
chn_hir_status chn_hir_make_alternation(chn_hir *out, chn_hir_vec *subs);
/* Hir::dot */
chn_hir_status chn_hir_make_dot(chn_hir *out, chn_hir_dot dot);

/* Drop for Hir: releases everything *hir owns and leaves it as Hir::empty(). */
void chn_hir_drop(chn_hir *hir);
/* Drop for Box<Hir>: chn_hir_drop then free. NULL is allowed. */
void chn_hir_free(chn_hir *hir);

/* ParserBuilder::new().nest_limit(nest_limit).build().parse(pattern) (parse.c), every
 * other option at regex 1.8.4's RegexOptions default: unicode and utf8 on; octal,
 * case_insensitive, multi_line, dot_matches_new_line, swap_greed, ignore_whitespace and
 * crlf off. pattern is UTF-8 (Rust &str), pattern_len bytes.
 * - CHN_HIR_OK: *out holds the HIR; *error is left empty.
 * - CHN_HIR_ERROR_SYNTAX: *error holds the error's Display text, exactly
 *   `e.to_string()` (error.rs's caret-formatted message; no "Invalid regex: " prefix);
 *   *out is Hir::empty().
 * - CHN_HIR_ERROR_NOMEM: *out is Hir::empty(), *error empty.
 * The caller drops *out with chn_hir_drop and frees error->ptr with free. */
chn_hir_status chn_hir_parse(const char *pattern, size_t pattern_len, uint32_t nest_limit,
    chn_hir *out, chn_hir_string *error);

/* regex_syntax::is_word_byte: [_0-9a-zA-Z]. */
static inline bool chn_hir_is_word_byte(uint8_t c) {
    return c == '_' || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

/* regex_syntax::is_word_character (unicode.c, over the perl_word table): UTS#18 Annex C
 * \w. c is a Unicode scalar value. */
bool chn_hir_is_word_character(uint32_t c);

/* Utf8Range { start: u8, end: u8 }. */
typedef struct chn_hir_utf8_range {
    uint8_t start; /* Utf8Range::start */
    uint8_t end;   /* Utf8Range::end */
} chn_hir_utf8_range;

/* Utf8Sequence::{One, Two, Three, Four}: len is the variant's arity, ranges[0..len] its
 * payload (Utf8Sequence::as_slice). */
typedef struct chn_hir_utf8_sequence {
    size_t len;                   /* the variant: 1 One, 2 Two, 3 Three, 4 Four */
    chn_hir_utf8_range ranges[4]; /* the variant's [Utf8Range; len] */
} chn_hir_utf8_sequence;

/* ScalarRange { start: u32, end: u32 } (utf8.rs, private there). */
typedef struct chn_hir_utf8_scalar_range {
    uint32_t start; /* ScalarRange::start */
    uint32_t end;   /* ScalarRange::end */
} chn_hir_utf8_scalar_range;

/* Utf8Sequences { range_stack: Vec<ScalarRange> }. */
typedef struct chn_hir_utf8_sequences {
    chn_hir_utf8_scalar_range *range_stack; /* Utf8Sequences::range_stack (Vec pointer) */
    size_t len;                             /* Utf8Sequences::range_stack (Vec length) */
    size_t cap;                             /* Utf8Sequences::range_stack (Vec capacity) */
} chn_hir_utf8_sequences;

/* Utf8Sequences::new(start, end) on uninitialized storage (start/end: scalar values). */
chn_hir_status chn_hir_utf8_sequences_init(chn_hir_utf8_sequences *it, uint32_t start, uint32_t end);
/* Utf8Sequences::reset */
chn_hir_status chn_hir_utf8_sequences_reset(chn_hir_utf8_sequences *it, uint32_t start, uint32_t end);
/* Iterator::next: 1 when *out holds the next sequence, 0 when exhausted, -1 when the
 * range stack could not grow (out of memory). */
int chn_hir_utf8_sequences_next(chn_hir_utf8_sequences *it, chn_hir_utf8_sequence *out);
/* Drop: frees range_stack and leaves *it empty. */
void chn_hir_utf8_sequences_drop(chn_hir_utf8_sequences *it);

#ifdef __cplusplus
}
#endif
#endif
