/* The cross merge of cn_filter_morphology (filter_ops.c's morph_merge). Shared by
 * that baseline unit and the /arch:AVX2 unit morphology_avx2.c: a static helper
 * compiled into each including unit, never inline, used by both (C4505). */
#ifndef CHAINNER_MERGE_SHARED_H
#define CHAINNER_MERGE_SHARED_H
#include "chainner.h"

/* Element i: a[i] replaces b[i] when strictly greater (maximum) or less, B3's
 * statement verbatim. B3 executes it as comiss and a conditional integer move of a[i]
 * into b[i] (VERIFIED(dumpbin) 0x18000FAFA-0x18000FB0D), so a kept b[i] keeps its
 * bits, and an unordered compare keeps b[i]; it never calls update. */
static void merge_one(const float *a, float *b, size_t i, int maximum)
{
    if (maximum ? a[i] > b[i] : a[i] < b[i]) b[i] = a[i];
}

/* morphology_avx2.c: merge_one over [begin, end), inside one chunk, for ISA level
 * avx2 or above: groups of 8 as vcmpps(a, b, _CMP_GT_OQ or _CMP_LT_OQ) and
 * vblendvps(b, a, mask), which stores each lane's original bits; the rest merge_one. */
void cn_morphology_merge_avx2(const float *a, float *b, size_t begin, size_t end, int maximum);
#endif
