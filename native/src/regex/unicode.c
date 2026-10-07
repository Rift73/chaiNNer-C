/* Port of regex-syntax 0.7.2 src/unicode.rs (and src/lib.rs's is_word_character), MIT OR Apache-2.0. */
/* Unicode class lookup by canonicalized property names and values, the Perl classes,
 * simple case folding and the \w test. Every table is binary searched, as in Rust. */
#include "unicode.h"
#include "interval.h"
#include "unicode_tables/unicode_tables.h"
#include <stdlib.h>
#include <string.h>

/* Ord for &str: bytes, then length. b is NUL-terminated. */
static int str_cmp(const char *a, size_t a_len, const char *b) {
    const size_t b_len = strlen(b);
    const int c = memcmp(a, b, a_len < b_len ? a_len : b_len);
    if (c != 0) return c;
    return a_len < b_len ? -1 : (a_len > b_len ? 1 : 0);
}

/* binary_search_by_key over a BY_NAME table. */
static const chn_ut_named_ranges *find_named(const chn_ut_named_ranges *table, size_t len,
    const char *name) {
    size_t lo = 0, hi = len;
    const size_t name_len = strlen(name);
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const int c = str_cmp(name, name_len, table[mid].name);
        if (c == 0) return &table[mid];
        if (c > 0) lo = mid + 1;
        else hi = mid;
    }
    return NULL;
}

/* binary_search_by_key over (alias, canonical) pairs: the canonical name or NULL. */
static const char *find_pair(const chn_ut_name_pair *table, size_t len, const char *name, size_t name_len) {
    size_t lo = 0, hi = len;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const int c = str_cmp(name, name_len, table[mid].name);
        if (c == 0) return table[mid].canonical;
        if (c > 0) lo = mid + 1;
        else hi = mid;
    }
    return NULL;
}

/* hir_class: ClassUnicode::new over a generated table. */
static chn_unicode_error hir_class(const chn_ut_named_ranges *set, chn_hir_class_unicode *out) {
    return chn_class_unicode_new(out, set->ranges, set->len) == CHN_HIR_OK ? CHN_UNICODE_OK : CHN_UNICODE_NOMEM;
}

/* property_set(name_map, canonical).map(hir_class).ok_or(missing) */
static chn_unicode_error property_set(const chn_ut_named_ranges *table, size_t len, const char *name,
    chn_unicode_error missing, chn_hir_class_unicode *out) {
    const chn_ut_named_ranges *set = find_named(table, len, name);
    if (!set) return missing;
    return hir_class(set, out);
}

chn_hir_status chn_unicode_perl_word(chn_hir_class_unicode *out) {
    return chn_class_unicode_new(out, chn_ut_perl_word, chn_ut_perl_word_len);
}

chn_hir_status chn_unicode_perl_space(chn_hir_class_unicode *out) {
    /* unicode-bool: property_bool::WHITE_SPACE */
    const chn_ut_named_ranges *set = find_named(chn_ut_property_bool_by_name, chn_ut_property_bool_by_name_len, "White_Space");
    return chn_class_unicode_new(out, set->ranges, set->len);
}

chn_hir_status chn_unicode_perl_digit(chn_hir_class_unicode *out) {
    /* unicode-gencat: general_category::DECIMAL_NUMBER */
    const chn_ut_named_ranges *set = find_named(chn_ut_general_category_by_name, chn_ut_general_category_by_name_len, "Decimal_Number");
    return chn_class_unicode_new(out, set->ranges, set->len);
}

bool chn_hir_is_word_character(uint32_t c) {
    if (c <= 0xFF && chn_hir_is_word_byte((uint8_t)c)) return true;
    size_t lo = 0, hi = chn_ut_perl_word_len;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const chn_hir_class_unicode_range r = chn_ut_perl_word[mid];
        if (r.start <= c && c <= r.end) return true;
        if (r.start > c) hi = mid;
        else lo = mid + 1;
    }
    return false;
}

/* Whether c lies in a generated range table (binary search). */
static bool in_ranges(const chn_ut_named_ranges *set, uint32_t c) {
    size_t lo = 0, hi = set->len;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (set->ranges[mid].end < c) lo = mid + 1;
        else if (set->ranges[mid].start > c) hi = mid;
        else return true;
    }
    return false;
}

/* core's Alphabetic. The real module's core is at Unicode 15.0.0 (it rejects the
 * Extension I ideographs 15.1.0 added in capture names), as regex-syntax's table is. */
bool chn_unicode_is_alphabetic(uint32_t c) {
    return in_ranges(find_named(chn_ut_property_bool_by_name, chn_ut_property_bool_by_name_len, "Alphabetic"), c);
}

/* core's N (Nd | Nl | No), Unicode 15.0.0. */
bool chn_unicode_is_numeric(uint32_t c) {
    return in_ranges(find_named(chn_ut_general_category_by_name, chn_ut_general_category_by_name_len, "Number"), c);
}

/* The index of the first CASE_FOLDING_SIMPLE key >= c. */
static size_t case_fold_lower_bound(uint32_t c) {
    size_t lo = 0, hi = chn_ut_case_folding_simple_len;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (chn_ut_case_folding_simple[mid].c < c) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

bool chn_unicode_case_fold_overlaps(uint32_t start, uint32_t end) {
    const size_t i = case_fold_lower_bound(start);
    return i < chn_ut_case_folding_simple_len && chn_ut_case_folding_simple[i].c <= end;
}

/* Rust walks every codepoint of [start, end] through SimpleCaseFolder::mapping, which
 * yields a table entry's values for each key it meets; walking the keys within
 * [start, end] pushes the same values in the same order. */
chn_hir_status chn_unicode_case_fold_range(chn_hir_class_unicode *set, uint32_t start, uint32_t end) {
    for (size_t i = case_fold_lower_bound(start); i < chn_ut_case_folding_simple_len; ++i) {
        const chn_ut_case_fold entry = chn_ut_case_folding_simple[i];
        if (entry.c > end) break;
        for (uint32_t k = 0; k < entry.len; ++k) {
            const uint32_t folded = chn_ut_case_folding_simple_values[entry.offset + k];
            const chn_hir_status status = chn_class_unicode_append(set, folded, folded);
            if (status != CHN_HIR_OK) return status;
        }
    }
    return CHN_HIR_OK;
}

/* symbolic_name_normalize (UAX44-LM3): drops ' ', '_', '-' and non-ASCII bytes, lowercases
 * ASCII, ignores an "is" prefix (and maps a resulting "c" back to "isc"). out holds at
 * least max(len, 3) + 1 bytes; the result is NUL-terminated and its length returned. */
static size_t symbolic_name_normalize(const char *name, size_t len, char *out) {
    size_t start = 0, next_write = 0;
    bool starts_with_is = false;
    if (len >= 2) {
        starts_with_is = (name[0] == 'i' || name[0] == 'I') && (name[1] == 's' || name[1] == 'S');
        if (starts_with_is) start = 2;
    }
    for (size_t i = start; i < len; ++i) {
        const unsigned char b = (unsigned char)name[i];
        if (b == ' ' || b == '_' || b == '-') continue;
        if (b >= 'A' && b <= 'Z') out[next_write++] = (char)(b + ('a' - 'A'));
        else if (b <= 0x7F) out[next_write++] = (char)b;
    }
    if (starts_with_is && next_write == 1 && out[0] == 'c') {
        out[0] = 'i';
        out[1] = 's';
        out[2] = 'c';
        next_write = 3;
    }
    out[next_write] = '\0';
    return next_write;
}

/* property_values(canonical_property_name) */
static const chn_ut_property_value_set *property_values(const char *canonical) {
    size_t lo = 0, hi = chn_ut_property_values_len;
    const size_t len = strlen(canonical);
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const int c = str_cmp(canonical, len, chn_ut_property_values[mid].property);
        if (c == 0) return &chn_ut_property_values[mid];
        if (c > 0) lo = mid + 1;
        else hi = mid;
    }
    return NULL;
}

/* canonical_prop */
static const char *canonical_prop(const char *norm, size_t len) {
    return find_pair(chn_ut_property_names, chn_ut_property_names_len, norm, len);
}

/* canonical_value */
static const char *canonical_value(const chn_ut_property_value_set *vals, const char *norm, size_t len) {
    return find_pair(vals->values, vals->len, norm, len);
}

/* canonical_gencat */
static const char *canonical_gencat(const char *norm, size_t len) {
    if (str_cmp(norm, len, "any") == 0) return "Any";
    if (str_cmp(norm, len, "assigned") == 0) return "Assigned";
    if (str_cmp(norm, len, "ascii") == 0) return "ASCII";
    return canonical_value(property_values("General_Category"), norm, len);
}

/* canonical_script */
static const char *canonical_script(const char *norm, size_t len) {
    return canonical_value(property_values("Script"), norm, len);
}

/* gencat(canonical_name) */
static chn_unicode_error gencat(const char *name, chn_hir_class_unicode *out) {
    static const chn_hir_class_unicode_range ascii = {0, 0x7F}, any = {0, 0x10FFFF};
    if (strcmp(name, "Decimal_Number") == 0) {
        return chn_unicode_perl_digit(out) == CHN_HIR_OK ? CHN_UNICODE_OK : CHN_UNICODE_NOMEM;
    }
    if (strcmp(name, "ASCII") == 0) {
        return chn_class_unicode_new(out, &ascii, 1) == CHN_HIR_OK ? CHN_UNICODE_OK : CHN_UNICODE_NOMEM;
    }
    if (strcmp(name, "Any") == 0) {
        return chn_class_unicode_new(out, &any, 1) == CHN_HIR_OK ? CHN_UNICODE_OK : CHN_UNICODE_NOMEM;
    }
    if (strcmp(name, "Assigned") == 0) {
        const chn_unicode_error err = gencat("Unassigned", out);
        if (err != CHN_UNICODE_OK) return err;
        if (chn_class_unicode_negate(out) != CHN_HIR_OK) {
            chn_class_unicode_drop(out);
            return CHN_UNICODE_NOMEM;
        }
        return CHN_UNICODE_OK;
    }
    return property_set(chn_ut_general_category_by_name, chn_ut_general_category_by_name_len, name,
        CHN_UNICODE_PROPERTY_VALUE_NOT_FOUND, out);
}

/* bool_property(canonical_name) */
static chn_unicode_error bool_property(const char *name, chn_hir_class_unicode *out) {
    if (strcmp(name, "Decimal_Number") == 0) {
        return chn_unicode_perl_digit(out) == CHN_HIR_OK ? CHN_UNICODE_OK : CHN_UNICODE_NOMEM;
    }
    if (strcmp(name, "White_Space") == 0) {
        return chn_unicode_perl_space(out) == CHN_HIR_OK ? CHN_UNICODE_OK : CHN_UNICODE_NOMEM;
    }
    return property_set(chn_ut_property_bool_by_name, chn_ut_property_bool_by_name_len, name,
        CHN_UNICODE_PROPERTY_NOT_FOUND, out);
}

/* ages(canonical_age), unioned as unicode::class does for ByValue { "Age", .. }. */
static chn_unicode_error ages(const char *canonical_age, chn_hir_class_unicode *out) {
    static const char *const AGES[] = {
        "V1_1", "V2_0", "V2_1", "V3_0", "V3_1", "V3_2", "V4_0", "V4_1", "V5_0", "V5_1",
        "V5_2", "V6_0", "V6_1", "V6_2", "V6_3", "V7_0", "V8_0", "V9_0", "V10_0", "V11_0",
        "V12_0", "V12_1", "V13_0", "V14_0", "V15_0",
    };
    const size_t count = sizeof(AGES) / sizeof(AGES[0]);
    size_t pos = count;
    for (size_t i = 0; i < count; ++i) {
        if (strcmp(AGES[i], canonical_age) == 0) {
            pos = i;
            break;
        }
    }
    if (pos == count) return CHN_UNICODE_PROPERTY_VALUE_NOT_FOUND;
    chn_class_unicode_init(out);
    for (size_t i = 0; i <= pos; ++i) {
        chn_hir_class_unicode set;
        const chn_ut_named_ranges *table = find_named(chn_ut_age_by_name, chn_ut_age_by_name_len, AGES[i]);
        chn_hir_status status = chn_class_unicode_new(&set, table->ranges, table->len);
        if (status == CHN_HIR_OK) status = chn_class_unicode_union(out, &set);
        chn_class_unicode_drop(&set);
        if (status != CHN_HIR_OK) {
            chn_class_unicode_drop(out);
            return CHN_UNICODE_NOMEM;
        }
    }
    return CHN_UNICODE_OK;
}

/* ClassQuery::canonical_binary then the matching lookup. */
static chn_unicode_error canonical_binary_class(const char *norm, size_t len, chn_hir_class_unicode *out) {
    if (str_cmp(norm, len, "cf") != 0 && str_cmp(norm, len, "sc") != 0 && str_cmp(norm, len, "lc") != 0) {
        const char *canon = canonical_prop(norm, len);
        if (canon) return bool_property(canon, out);
    }
    const char *canon = canonical_gencat(norm, len);
    if (canon) return gencat(canon, out);
    canon = canonical_script(norm, len);
    if (canon) {
        return property_set(chn_ut_script_by_name, chn_ut_script_by_name_len, canon,
            CHN_UNICODE_PROPERTY_VALUE_NOT_FOUND, out);
    }
    return CHN_UNICODE_PROPERTY_NOT_FOUND;
}

/* ClassQuery::ByValue: canonicalize, then the matching lookup. */
static chn_unicode_error by_value_class(const char *name, size_t name_len, const char *value,
    size_t value_len, chn_hir_class_unicode *out) {
    const char *canon_name = canonical_prop(name, name_len);
    if (!canon_name) return CHN_UNICODE_PROPERTY_NOT_FOUND;
    if (strcmp(canon_name, "General_Category") == 0) {
        const char *canon = canonical_gencat(value, value_len);
        if (!canon) return CHN_UNICODE_PROPERTY_VALUE_NOT_FOUND;
        return gencat(canon, out);
    }
    if (strcmp(canon_name, "Script") == 0) {
        const char *canon = canonical_script(value, value_len);
        if (!canon) return CHN_UNICODE_PROPERTY_VALUE_NOT_FOUND;
        return property_set(chn_ut_script_by_name, chn_ut_script_by_name_len, canon,
            CHN_UNICODE_PROPERTY_VALUE_NOT_FOUND, out);
    }
    const chn_ut_property_value_set *vals = property_values(canon_name);
    if (!vals) return CHN_UNICODE_PROPERTY_VALUE_NOT_FOUND;
    const char *canon_val = canonical_value(vals, value, value_len);
    if (!canon_val) return CHN_UNICODE_PROPERTY_VALUE_NOT_FOUND;
    if (strcmp(canon_name, "Age") == 0) return ages(canon_val, out);
    if (strcmp(canon_name, "Script_Extensions") == 0) {
        return property_set(chn_ut_script_extension_by_name, chn_ut_script_extension_by_name_len,
            canon_val, CHN_UNICODE_PROPERTY_VALUE_NOT_FOUND, out);
    }
    if (strcmp(canon_name, "Grapheme_Cluster_Break") == 0) {
        return property_set(chn_ut_grapheme_cluster_break_by_name, chn_ut_grapheme_cluster_break_by_name_len,
            canon_val, CHN_UNICODE_PROPERTY_VALUE_NOT_FOUND, out);
    }
    if (strcmp(canon_name, "Sentence_Break") == 0) {
        return property_set(chn_ut_sentence_break_by_name, chn_ut_sentence_break_by_name_len,
            canon_val, CHN_UNICODE_PROPERTY_VALUE_NOT_FOUND, out);
    }
    if (strcmp(canon_name, "Word_Break") == 0) {
        return property_set(chn_ut_word_break_by_name, chn_ut_word_break_by_name_len,
            canon_val, CHN_UNICODE_PROPERTY_VALUE_NOT_FOUND, out);
    }
    return CHN_UNICODE_PROPERTY_NOT_FOUND;
}

chn_unicode_error chn_unicode_class(chn_unicode_query_kind kind, uint32_t letter,
    const char *name, size_t name_len, const char *value, size_t value_len,
    chn_hir_class_unicode *out) {
    chn_class_unicode_init(out);
    if (kind == CHN_UNICODE_QUERY_BY_VALUE) {
        /* Normalizing never lengthens a name beyond max(len, 3). */
        char small[64];
        char *buf = small;
        const size_t need = name_len + value_len + 8;
        if (need > sizeof(small)) {
            buf = (char *)malloc(need);
            if (!buf) return CHN_UNICODE_NOMEM;
        }
        const size_t norm_name_len = symbolic_name_normalize(name, name_len, buf);
        char *norm_value = buf + (name_len > 3 ? name_len : 3) + 1;
        const size_t norm_value_len = symbolic_name_normalize(value, value_len, norm_value);
        const chn_unicode_error err = by_value_class(buf, norm_name_len, norm_value, norm_value_len, out);
        if (buf != small) free(buf);
        return err;
    }
    char letter_utf8[4];
    if (kind == CHN_UNICODE_QUERY_ONE_LETTER) {
        /* c.to_string() */
        if (letter < 0x80) {
            letter_utf8[0] = (char)letter;
            name_len = 1;
        } else if (letter < 0x800) {
            letter_utf8[0] = (char)(0xC0 | (letter >> 6));
            letter_utf8[1] = (char)(0x80 | (letter & 0x3F));
            name_len = 2;
        } else if (letter < 0x10000) {
            letter_utf8[0] = (char)(0xE0 | (letter >> 12));
            letter_utf8[1] = (char)(0x80 | ((letter >> 6) & 0x3F));
            letter_utf8[2] = (char)(0x80 | (letter & 0x3F));
            name_len = 3;
        } else {
            letter_utf8[0] = (char)(0xF0 | (letter >> 18));
            letter_utf8[1] = (char)(0x80 | ((letter >> 12) & 0x3F));
            letter_utf8[2] = (char)(0x80 | ((letter >> 6) & 0x3F));
            letter_utf8[3] = (char)(0x80 | (letter & 0x3F));
            name_len = 4;
        }
        name = letter_utf8;
    }
    char small[64];
    char *buf = small;
    if (name_len + 4 > sizeof(small)) {
        buf = (char *)malloc(name_len + 4);
        if (!buf) return CHN_UNICODE_NOMEM;
    }
    const size_t norm_len = symbolic_name_normalize(name, name_len, buf);
    const chn_unicode_error err = canonical_binary_class(buf, norm_len, out);
    if (buf != small) free(buf);
    return err;
}
