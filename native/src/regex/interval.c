/* Port of regex-syntax 0.7.2 src/hir/interval.rs (with the Interval impls of src/hir/mod.rs), MIT OR Apache-2.0. */
/* IntervalSet<ClassUnicodeRange> and IntervalSet<ClassBytesRange>: interval_template.h
 * instantiated for each, plus the two Interval::case_fold_simple impls. */
#include "interval.h"
#include "unicode.h"
#include <stdlib.h>
#include <string.h>

/* Bound for char: increment and decrement skip the surrogate block. */
static uint32_t char_increment(uint32_t c) {
    return c == 0xD7FF ? 0xE000 : c + 1;
}

static uint32_t char_decrement(uint32_t c) {
    return c == 0xE000 ? 0xD7FF : c - 1;
}

/* impl Interval for ClassBytesRange: case_fold_simple (ASCII only). */
static chn_hir_status bytes_range_case_fold(chn_hir_class_bytes *set, chn_hir_class_bytes_range r);

#define SET chn_hir_class_unicode
#define RANGE chn_hir_class_unicode_range
#define BOUND uint32_t
#define FN(name) chn_class_unicode_##name
#define BOUND_MIN 0u
#define BOUND_MAX 0x10FFFFu
#define BOUND_INC(b) char_increment(b)
#define BOUND_DEC(b) char_decrement(b)
#define RANGE_CASE_FOLD(set, r) chn_unicode_case_fold_range((set), (r).start, (r).end)
#include "interval_template.h"
#undef SET
#undef RANGE
#undef BOUND
#undef FN
#undef BOUND_MIN
#undef BOUND_MAX
#undef BOUND_INC
#undef BOUND_DEC
#undef RANGE_CASE_FOLD

#define SET chn_hir_class_bytes
#define RANGE chn_hir_class_bytes_range
#define BOUND uint8_t
#define FN(name) chn_class_bytes_##name
#define BOUND_MIN ((uint8_t)0)
#define BOUND_MAX ((uint8_t)0xFF)
#define BOUND_INC(b) ((uint8_t)((b) + 1))
#define BOUND_DEC(b) ((uint8_t)((b) - 1))
#define RANGE_CASE_FOLD(set, r) bytes_range_case_fold((set), (r))
#include "interval_template.h"

static chn_hir_status bytes_range_case_fold(chn_hir_class_bytes *set, chn_hir_class_bytes_range r) {
    chn_hir_status status = CHN_HIR_OK;
    const chn_hir_class_bytes_range lower_az = {'a', 'z'}, upper_az = {'A', 'Z'};
    if (!chn_class_bytes_is_intersection_empty(lower_az, r)) {
        const uint8_t lower = r.start > 'a' ? r.start : (uint8_t)'a';
        const uint8_t upper = r.end < 'z' ? r.end : (uint8_t)'z';
        status = chn_class_bytes_append_range(set, chn_class_bytes_create((uint8_t)(lower - 32), (uint8_t)(upper - 32)));
        if (status != CHN_HIR_OK) return status;
    }
    if (!chn_class_bytes_is_intersection_empty(upper_az, r)) {
        const uint8_t lower = r.start > 'A' ? r.start : (uint8_t)'A';
        const uint8_t upper = r.end < 'Z' ? r.end : (uint8_t)'Z';
        status = chn_class_bytes_append_range(set, chn_class_bytes_create((uint8_t)(lower + 32), (uint8_t)(upper + 32)));
    }
    return status;
}

chn_hir_status chn_class_unicode_push(chn_hir_class_unicode *set, uint32_t start, uint32_t end) {
    return chn_class_unicode_push_range(set, chn_class_unicode_create(start, end));
}

chn_hir_status chn_class_unicode_append(chn_hir_class_unicode *set, uint32_t start, uint32_t end) {
    return chn_class_unicode_append_range(set, chn_class_unicode_create(start, end));
}

bool chn_class_unicode_is_ascii(const chn_hir_class_unicode *set) {
    return set->len == 0 || set->ranges[set->len - 1].end <= 0x7F;
}

chn_hir_status chn_class_bytes_push(chn_hir_class_bytes *set, uint8_t start, uint8_t end) {
    return chn_class_bytes_push_range(set, chn_class_bytes_create(start, end));
}

bool chn_class_bytes_is_ascii(const chn_hir_class_bytes *set) {
    return set->len == 0 || set->ranges[set->len - 1].end <= 0x7F;
}
