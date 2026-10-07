/* Port of regex-syntax 0.7.2 (src/ast/mod.rs, src/hir/mod.rs Error), MIT OR Apache-2.0. */
/* The regex front end's private types: the AST (ast/mod.rs), its arena, and the errors
 * that parse.c and translate.c report and error.c formats. Private to X1's files. */
#ifndef CHAINNER_REGEX_AST_H
#define CHAINNER_REGEX_AST_H
#include "chainner_hir.h"

/* Position { offset, line, column }: a byte offset, a 1-based line and a 1-based column
 * counted in chars. Ordered by offset only (Ord for Position). */
typedef struct chn_ast_position {
    size_t offset;
    size_t line;
    size_t column;
} chn_ast_position;

/* Span { start, end }, ordered by (start, end). */
typedef struct chn_ast_span {
    chn_ast_position start;
    chn_ast_position end;
} chn_ast_span;

/* ast::ErrorKind, then hir::ErrorKind, then the C-only allocation failure. */
typedef enum chn_syntax_error_kind {
    CHN_AST_ERR_CAPTURE_LIMIT_EXCEEDED,
    CHN_AST_ERR_CLASS_ESCAPE_INVALID,
    CHN_AST_ERR_CLASS_RANGE_INVALID,
    CHN_AST_ERR_CLASS_RANGE_LITERAL,
    CHN_AST_ERR_CLASS_UNCLOSED,
    CHN_AST_ERR_DECIMAL_EMPTY,
    CHN_AST_ERR_DECIMAL_INVALID,
    CHN_AST_ERR_ESCAPE_HEX_EMPTY,
    CHN_AST_ERR_ESCAPE_HEX_INVALID,
    CHN_AST_ERR_ESCAPE_HEX_INVALID_DIGIT,
    CHN_AST_ERR_ESCAPE_UNEXPECTED_EOF,
    CHN_AST_ERR_ESCAPE_UNRECOGNIZED,
    CHN_AST_ERR_FLAG_DANGLING_NEGATION,
    CHN_AST_ERR_FLAG_DUPLICATE,          /* aux: original */
    CHN_AST_ERR_FLAG_REPEATED_NEGATION,  /* aux: original */
    CHN_AST_ERR_FLAG_UNEXPECTED_EOF,
    CHN_AST_ERR_FLAG_UNRECOGNIZED,
    CHN_AST_ERR_GROUP_NAME_DUPLICATE,    /* aux: original */
    CHN_AST_ERR_GROUP_NAME_EMPTY,
    CHN_AST_ERR_GROUP_NAME_INVALID,
    CHN_AST_ERR_GROUP_NAME_UNEXPECTED_EOF,
    CHN_AST_ERR_GROUP_UNCLOSED,
    CHN_AST_ERR_GROUP_UNOPENED,
    CHN_AST_ERR_NEST_LIMIT_EXCEEDED,     /* limit */
    CHN_AST_ERR_REPETITION_COUNT_INVALID,
    CHN_AST_ERR_REPETITION_COUNT_DECIMAL_EMPTY,
    CHN_AST_ERR_REPETITION_COUNT_UNCLOSED,
    CHN_AST_ERR_REPETITION_MISSING,
    CHN_AST_ERR_UNICODE_CLASS_INVALID,
    CHN_AST_ERR_UNSUPPORTED_BACKREFERENCE,
    CHN_AST_ERR_UNSUPPORTED_LOOK_AROUND,
    CHN_HIR_ERR_UNICODE_NOT_ALLOWED,
    CHN_HIR_ERR_INVALID_UTF8,
    CHN_HIR_ERR_UNICODE_PROPERTY_NOT_FOUND,
    CHN_HIR_ERR_UNICODE_PROPERTY_VALUE_NOT_FOUND,
    CHN_SYNTAX_ERR_NOMEM
} chn_syntax_error_kind;

/* ast::Error / hir::Error (kind, span) with ast::Error::auxiliary_span; the pattern is
 * the caller's. */
typedef struct chn_syntax_error {
    chn_syntax_error_kind kind;
    uint32_t limit;      /* NestLimitExceeded(limit) */
    chn_ast_span span;
    bool has_aux;
    chn_ast_span aux;    /* FlagDuplicate, FlagRepeatedNegation, GroupNameDuplicate: original */
} chn_syntax_error;

/* Display for ast::ErrorKind and hir::ErrorKind: appends the message to buf (cap bytes,
 * NUL-terminated) and returns its length. ast.c (ast kinds), hir.c (hir kinds). */
size_t chn_ast_error_kind_display(const chn_syntax_error *err, char *buf, size_t cap);
size_t chn_hir_error_kind_display(const chn_syntax_error *err, char *buf, size_t cap);

/* Display for regex_syntax::Error (error.rs Formatter): the caret-formatted text. */
chn_hir_status chn_syntax_error_display(const char *pattern, size_t pattern_len,
    const chn_syntax_error *err, chn_hir_string *out);

/* A bump arena that owns every AST node and AST vector of one parse (ast.c). */
typedef struct chn_arena_chunk chn_arena_chunk;
typedef struct chn_arena {
    chn_arena_chunk *head;
    char *next;
    size_t left;
} chn_arena;

void chn_arena_init(chn_arena *arena);
void *chn_arena_alloc(chn_arena *arena, size_t size); /* NULL when out of memory */
void chn_arena_free(chn_arena *arena);

/* A string slice owned by the arena or borrowed from the pattern. */
typedef struct chn_ast_str {
    const char *ptr;
    size_t len;
} chn_ast_str;

typedef struct chn_ast chn_ast;

/* Vec<Ast> (pointers into the arena). */
typedef struct chn_ast_vec {
    chn_ast **ptr;
    size_t len;
    size_t cap;
} chn_ast_vec;

typedef enum chn_ast_literal_kind {
    CHN_AST_LIT_VERBATIM,
    CHN_AST_LIT_META,
    CHN_AST_LIT_SUPERFLUOUS,
    CHN_AST_LIT_OCTAL,
    CHN_AST_LIT_HEX_FIXED,
    CHN_AST_LIT_HEX_BRACE,
    CHN_AST_LIT_SPECIAL
} chn_ast_literal_kind;

typedef enum chn_ast_hex_kind {
    CHN_AST_HEX_X,
    CHN_AST_HEX_UNICODE_SHORT,
    CHN_AST_HEX_UNICODE_LONG
} chn_ast_hex_kind;

/* Literal { span, kind, c } (SpecialLiteralKind is not kept: nothing reads it). */
typedef struct chn_ast_literal {
    chn_ast_span span;
    chn_ast_literal_kind kind;
    chn_ast_hex_kind hex; /* HexFixed(_) / HexBrace(_) */
    uint32_t c;
} chn_ast_literal;

typedef enum chn_ast_assertion_kind {
    CHN_AST_ASSERT_START_LINE,
    CHN_AST_ASSERT_END_LINE,
    CHN_AST_ASSERT_START_TEXT,
    CHN_AST_ASSERT_END_TEXT,
    CHN_AST_ASSERT_WORD_BOUNDARY,
    CHN_AST_ASSERT_NOT_WORD_BOUNDARY
} chn_ast_assertion_kind;

typedef struct chn_ast_assertion {
    chn_ast_span span;
    chn_ast_assertion_kind kind;
} chn_ast_assertion;

typedef enum chn_ast_perl_kind {
    CHN_AST_PERL_DIGIT,
    CHN_AST_PERL_SPACE,
    CHN_AST_PERL_WORD
} chn_ast_perl_kind;

typedef struct chn_ast_class_perl {
    chn_ast_span span;
    chn_ast_perl_kind kind;
    bool negated;
} chn_ast_class_perl;

typedef enum chn_ast_ascii_kind {
    CHN_AST_ASCII_ALNUM,
    CHN_AST_ASCII_ALPHA,
    CHN_AST_ASCII_ASCII,
    CHN_AST_ASCII_BLANK,
    CHN_AST_ASCII_CNTRL,
    CHN_AST_ASCII_DIGIT,
    CHN_AST_ASCII_GRAPH,
    CHN_AST_ASCII_LOWER,
    CHN_AST_ASCII_PRINT,
    CHN_AST_ASCII_PUNCT,
    CHN_AST_ASCII_SPACE,
    CHN_AST_ASCII_UPPER,
    CHN_AST_ASCII_WORD,
    CHN_AST_ASCII_XDIGIT
} chn_ast_ascii_kind;

typedef struct chn_ast_class_ascii {
    chn_ast_span span;
    chn_ast_ascii_kind kind;
    bool negated;
} chn_ast_class_ascii;

typedef enum chn_ast_unicode_kind {
    CHN_AST_UNICODE_ONE_LETTER,
    CHN_AST_UNICODE_NAMED,
    CHN_AST_UNICODE_NAMED_VALUE
} chn_ast_unicode_kind;

typedef enum chn_ast_unicode_op {
    CHN_AST_UNICODE_OP_EQUAL,
    CHN_AST_UNICODE_OP_COLON,
    CHN_AST_UNICODE_OP_NOT_EQUAL
} chn_ast_unicode_op;

/* ClassUnicode { span, negated, kind: OneLetter(c) | Named(name) | NamedValue { op, name, value } }. */
typedef struct chn_ast_class_unicode {
    chn_ast_span span;
    bool negated;
    chn_ast_unicode_kind kind;
    chn_ast_unicode_op op;
    uint32_t letter;    /* OneLetter */
    chn_ast_str name;   /* Named, NamedValue */
    chn_ast_str value;  /* NamedValue */
} chn_ast_class_unicode;

typedef struct chn_ast_class_bracketed chn_ast_class_bracketed;
typedef struct chn_ast_class_binop chn_ast_class_binop;
typedef struct chn_ast_class_item chn_ast_class_item;

/* Vec<ClassSetItem>. */
typedef struct chn_ast_item_vec {
    chn_ast_class_item *ptr;
    size_t len;
    size_t cap;
} chn_ast_item_vec;

/* ClassSetUnion { span, items }. */
typedef struct chn_ast_class_union {
    chn_ast_span span;
    chn_ast_item_vec items;
} chn_ast_class_union;

typedef enum chn_ast_item_kind {
    CHN_AST_ITEM_EMPTY,
    CHN_AST_ITEM_LITERAL,
    CHN_AST_ITEM_RANGE,
    CHN_AST_ITEM_ASCII,
    CHN_AST_ITEM_UNICODE,
    CHN_AST_ITEM_PERL,
    CHN_AST_ITEM_BRACKETED,
    CHN_AST_ITEM_UNION
} chn_ast_item_kind;

/* ClassSetRange { span, start, end }. */
typedef struct chn_ast_class_range {
    chn_ast_span span;
    chn_ast_literal start;
    chn_ast_literal end;
} chn_ast_class_range;

/* ClassSetItem. */
struct chn_ast_class_item {
    chn_ast_item_kind kind;
    union {
        chn_ast_span empty;                  /* Empty(span) */
        chn_ast_literal literal;             /* Literal */
        chn_ast_class_range range;           /* Range */
        chn_ast_class_ascii ascii;           /* Ascii */
        chn_ast_class_unicode unicode;       /* Unicode */
        chn_ast_class_perl perl;             /* Perl */
        chn_ast_class_bracketed *bracketed;  /* Bracketed(Box<ClassBracketed>) */
        chn_ast_class_union union_;          /* Union */
    };
};

/* ClassSet: Item(ClassSetItem) | BinaryOp(ClassSetBinaryOp). */
typedef struct chn_ast_class_set {
    bool is_op;
    union {
        chn_ast_class_item item;
        chn_ast_class_binop *op;
    };
} chn_ast_class_set;

typedef enum chn_ast_binop_kind {
    CHN_AST_BINOP_INTERSECTION,
    CHN_AST_BINOP_DIFFERENCE,
    CHN_AST_BINOP_SYMMETRIC_DIFFERENCE
} chn_ast_binop_kind;

/* ClassSetBinaryOp { span, kind, lhs, rhs }. */
struct chn_ast_class_binop {
    chn_ast_span span;
    chn_ast_binop_kind kind;
    chn_ast_class_set lhs;
    chn_ast_class_set rhs;
};

/* ClassBracketed { span, negated, kind }. */
struct chn_ast_class_bracketed {
    chn_ast_span span;
    bool negated;
    chn_ast_class_set kind;
};

typedef enum chn_ast_flag {
    CHN_AST_FLAG_CASE_INSENSITIVE,
    CHN_AST_FLAG_MULTI_LINE,
    CHN_AST_FLAG_DOT_MATCHES_NEW_LINE,
    CHN_AST_FLAG_SWAP_GREED,
    CHN_AST_FLAG_UNICODE,
    CHN_AST_FLAG_CRLF,
    CHN_AST_FLAG_IGNORE_WHITESPACE,
    CHN_AST_FLAG_NEGATION /* FlagsItemKind::Negation */
} chn_ast_flag;

/* FlagsItem { span, kind }. */
typedef struct chn_ast_flags_item {
    chn_ast_span span;
    chn_ast_flag kind;
} chn_ast_flags_item;

/* Flags { span, items }: add_item rejects duplicates, so at most 7 flags and 1 negation. */
typedef struct chn_ast_flags {
    chn_ast_span span;
    size_t len;
    chn_ast_flags_item items[8];
} chn_ast_flags;

typedef enum chn_ast_repetition_kind {
    CHN_AST_REP_ZERO_OR_ONE,
    CHN_AST_REP_ZERO_OR_MORE,
    CHN_AST_REP_ONE_OR_MORE,
    CHN_AST_REP_EXACTLY,  /* Range(Exactly(m)) */
    CHN_AST_REP_AT_LEAST, /* Range(AtLeast(m)) */
    CHN_AST_REP_BOUNDED   /* Range(Bounded(m, n)) */
} chn_ast_repetition_kind;

/* Repetition { span, op: RepetitionOp { span, kind }, greedy, ast }. */
typedef struct chn_ast_repetition {
    chn_ast_span span;
    chn_ast_span op_span;
    chn_ast_repetition_kind kind;
    uint32_t m;
    uint32_t n;
    bool greedy;
    chn_ast *ast;
} chn_ast_repetition;

typedef enum chn_ast_group_kind {
    CHN_AST_GROUP_CAPTURE_INDEX,
    CHN_AST_GROUP_CAPTURE_NAME,
    CHN_AST_GROUP_NON_CAPTURING
} chn_ast_group_kind;

/* Group { span, kind, ast }: CaptureIndex(index) | CaptureName { name: CaptureName {
 * span, name, index } } | NonCapturing(flags). */
typedef struct chn_ast_group {
    chn_ast_span span;
    chn_ast_group_kind kind;
    uint32_t index;
    chn_ast_span name_span;
    chn_ast_str name;
    chn_ast_flags flags;
    chn_ast *ast;
} chn_ast_group;

/* Alternation / Concat { span, asts }. */
typedef struct chn_ast_seq {
    chn_ast_span span;
    chn_ast_vec asts;
} chn_ast_seq;

typedef enum chn_ast_kind {
    CHN_AST_EMPTY,
    CHN_AST_FLAGS,
    CHN_AST_LITERAL,
    CHN_AST_DOT,
    CHN_AST_ASSERTION,
    CHN_AST_CLASS_UNICODE,
    CHN_AST_CLASS_PERL,
    CHN_AST_CLASS_BRACKETED,
    CHN_AST_REPETITION,
    CHN_AST_GROUP,
    CHN_AST_ALTERNATION,
    CHN_AST_CONCAT
} chn_ast_kind;

/* Ast. Class(Unicode | Perl | Bracketed) is flattened into three kinds. */
struct chn_ast {
    chn_ast_kind kind;
    union {
        chn_ast_span empty;                    /* Empty(span) */
        struct {
            chn_ast_span span;
            chn_ast_flags flags;
        } set_flags;                           /* Flags(SetFlags { span, flags }) */
        chn_ast_literal literal;               /* Literal */
        chn_ast_span dot;                      /* Dot(span) */
        chn_ast_assertion assertion;           /* Assertion */
        chn_ast_class_unicode class_unicode;   /* Class(Unicode) */
        chn_ast_class_perl class_perl;         /* Class(Perl) */
        chn_ast_class_bracketed class_bracketed; /* Class(Bracketed) */
        chn_ast_repetition repetition;         /* Repetition */
        chn_ast_group group;                   /* Group */
        chn_ast_seq alternation;               /* Alternation */
        chn_ast_seq concat;                    /* Concat */
    };
};

/* Translator::translate (translate.c), with TranslatorBuilder's defaults (utf8 on, every
 * flag unset): *out gets the HIR, or *err the error (CHN_SYNTAX_ERR_NOMEM included). */
bool chn_translate(const chn_ast *ast, chn_hir *out, chn_syntax_error *err);

#endif
