/* Port of regex-syntax 0.7.2 src/unicode.rs, MIT OR Apache-2.0. */
/* Unicode class queries, the Perl classes and simple case folding (unicode.c), with
 * every unicode-* feature enabled as regex 1.8.4 builds regex-syntax. Private to X1's
 * files. */
#ifndef CHAINNER_REGEX_UNICODE_H
#define CHAINNER_REGEX_UNICODE_H
#include "chainner_hir.h"

/* unicode::Error (PerlClassNotFound cannot happen with unicode-perl), plus NOMEM. */
typedef enum chn_unicode_error {
    CHN_UNICODE_OK = 0,
    CHN_UNICODE_PROPERTY_NOT_FOUND,
    CHN_UNICODE_PROPERTY_VALUE_NOT_FOUND,
    CHN_UNICODE_NOMEM
} chn_unicode_error;

/* ClassQuery's variants. */
typedef enum chn_unicode_query_kind {
    CHN_UNICODE_QUERY_ONE_LETTER, /* OneLetter(letter) */
    CHN_UNICODE_QUERY_BINARY,     /* Binary(name) */
    CHN_UNICODE_QUERY_BY_VALUE    /* ByValue { property_name: name, property_value: value } */
} chn_unicode_query_kind;

/* unicode::class(query): *out is a new class on CHN_UNICODE_OK, empty otherwise. */
chn_unicode_error chn_unicode_class(chn_unicode_query_kind kind, uint32_t letter,
    const char *name, size_t name_len, const char *value, size_t value_len,
    chn_hir_class_unicode *out);

/* unicode::perl_word, perl_space, perl_digit: a new class in *out. */
chn_hir_status chn_unicode_perl_word(chn_hir_class_unicode *out);
chn_hir_status chn_unicode_perl_space(chn_hir_class_unicode *out);
chn_hir_status chn_unicode_perl_digit(chn_hir_class_unicode *out);

/* SimpleCaseFolder::overlaps */
bool chn_unicode_case_fold_overlaps(uint32_t start, uint32_t end);

/* impl Interval for ClassUnicodeRange: case_fold_simple. Appends (unsorted) every simple
 * case folding of every codepoint in [start, end], as SimpleCaseFolder::mapping yields
 * them in increasing codepoint order. */
chn_hir_status chn_unicode_case_fold_range(chn_hir_class_unicode *set, uint32_t start, uint32_t end);

/* char::is_alphabetic and char::is_numeric as the real module's core has them (Unicode
 * 15.0.0, the version of regex-syntax's tables): parse.c's capture name check. */
bool chn_unicode_is_alphabetic(uint32_t c);
bool chn_unicode_is_numeric(uint32_t c);

#endif
