/* Port of regex-syntax 0.7.2 (src/hir/interval.rs, the Interval impls and class methods of src/hir/mod.rs), MIT OR Apache-2.0. */
/* IntervalSet<ClassUnicodeRange> and IntervalSet<ClassBytesRange> (interval.c): the
 * ClassUnicode and ClassBytes of chainner_hir.h. Every operation keeps Rust's results
 * and its `folded` bookkeeping. Statuses are CHN_HIR_OK or CHN_HIR_ERROR_NOMEM; on
 * NOMEM the set is left valid (possibly unchanged). Private to X1's files. */
#ifndef CHAINNER_REGEX_INTERVAL_H
#define CHAINNER_REGEX_INTERVAL_H
#include "chainner_hir.h"

/* ClassUnicode::empty (folded: an empty set is case folded). */
void chn_class_unicode_init(chn_hir_class_unicode *set);
/* ClassUnicode::new(ranges): copies, canonicalizes; folded when empty. */
chn_hir_status chn_class_unicode_new(chn_hir_class_unicode *set,
    const chn_hir_class_unicode_range *ranges, size_t len);
/* ClassUnicode::push (canonicalizes; folded = false). */
chn_hir_status chn_class_unicode_push(chn_hir_class_unicode *set, uint32_t start, uint32_t end);
chn_hir_status chn_class_unicode_clone(chn_hir_class_unicode *dst, const chn_hir_class_unicode *src);
void chn_class_unicode_drop(chn_hir_class_unicode *set);
chn_hir_status chn_class_unicode_union(chn_hir_class_unicode *set, const chn_hir_class_unicode *other);
chn_hir_status chn_class_unicode_intersect(chn_hir_class_unicode *set, const chn_hir_class_unicode *other);
chn_hir_status chn_class_unicode_difference(chn_hir_class_unicode *set, const chn_hir_class_unicode *other);
chn_hir_status chn_class_unicode_symmetric_difference(chn_hir_class_unicode *set,
    const chn_hir_class_unicode *other);
chn_hir_status chn_class_unicode_negate(chn_hir_class_unicode *set);
/* ClassUnicode::case_fold_simple (unicode-case is always available). */
chn_hir_status chn_class_unicode_case_fold_simple(chn_hir_class_unicode *set);
bool chn_class_unicode_is_ascii(const chn_hir_class_unicode *set);
/* IntervalSet::eq (ranges only; folded is not identity). */
bool chn_class_unicode_eq(const chn_hir_class_unicode *a, const chn_hir_class_unicode *b);
/* Appends without canonicalizing (Interval::case_fold_simple's pushes; unicode.c). */
chn_hir_status chn_class_unicode_append(chn_hir_class_unicode *set, uint32_t start, uint32_t end);

void chn_class_bytes_init(chn_hir_class_bytes *set);
chn_hir_status chn_class_bytes_new(chn_hir_class_bytes *set,
    const chn_hir_class_bytes_range *ranges, size_t len);
chn_hir_status chn_class_bytes_push(chn_hir_class_bytes *set, uint8_t start, uint8_t end);
chn_hir_status chn_class_bytes_clone(chn_hir_class_bytes *dst, const chn_hir_class_bytes *src);
void chn_class_bytes_drop(chn_hir_class_bytes *set);
chn_hir_status chn_class_bytes_union(chn_hir_class_bytes *set, const chn_hir_class_bytes *other);
chn_hir_status chn_class_bytes_intersect(chn_hir_class_bytes *set, const chn_hir_class_bytes *other);
chn_hir_status chn_class_bytes_difference(chn_hir_class_bytes *set, const chn_hir_class_bytes *other);
chn_hir_status chn_class_bytes_symmetric_difference(chn_hir_class_bytes *set,
    const chn_hir_class_bytes *other);
chn_hir_status chn_class_bytes_negate(chn_hir_class_bytes *set);
chn_hir_status chn_class_bytes_case_fold_simple(chn_hir_class_bytes *set);
bool chn_class_bytes_is_ascii(const chn_hir_class_bytes *set);
bool chn_class_bytes_eq(const chn_hir_class_bytes *a, const chn_hir_class_bytes *b);

#endif
