/* Port of regex-syntax 0.7.2 src/hir/interval.rs, MIT OR Apache-2.0. */
/* IntervalSet<I>'s generic body, instantiated twice by interval.c (as Rust monomorphizes
 * it): before each inclusion interval.c defines
 *   SET, RANGE, BOUND        the set, interval and bound types;
 *   FN(name)                 the public name of each method;
 *   BOUND_MIN, BOUND_MAX     Bound::min_value / max_value;
 *   BOUND_INC(b), BOUND_DEC(b) Bound::increment / decrement (never over- or underflowing
 *                            where they are called, as in Rust);
 *   RANGE_CASE_FOLD(set, r)  Interval::case_fold_simple, appending to set->ranges.
 * The "append to the end, then drain the prefix" scheme of interval.rs is kept, so the
 * only allocations are a set's own growth. */

/* Interval::create */
static RANGE FN(create)(BOUND lower, BOUND upper) {
    RANGE r;
    if (lower <= upper) {
        r.start = lower;
        r.end = upper;
    } else {
        r.start = upper;
        r.end = lower;
    }
    return r;
}

/* Ord for the interval: (start, end) lexicographically. */
static int FN(cmp)(RANGE a, RANGE b) {
    if (a.start != b.start) return a.start < b.start ? -1 : 1;
    if (a.end != b.end) return a.end < b.end ? -1 : 1;
    return 0;
}

static int FN(qsort_cmp)(const void *a, const void *b) {
    return FN(cmp)(*(const RANGE *)a, *(const RANGE *)b);
}

/* Interval::is_contiguous (overlapping or adjacent). */
static bool FN(is_contiguous)(RANGE a, RANGE b) {
    const uint32_t lower = (uint32_t)(a.start > b.start ? a.start : b.start);
    const uint32_t upper = (uint32_t)(a.end < b.end ? a.end : b.end);
    return lower <= upper + 1u;
}

/* Interval::is_intersection_empty */
static bool FN(is_intersection_empty)(RANGE a, RANGE b) {
    const BOUND lower = a.start > b.start ? a.start : b.start;
    const BOUND upper = a.end < b.end ? a.end : b.end;
    return lower > upper;
}

/* Interval::is_subset (a within b) */
static bool FN(is_subset)(RANGE a, RANGE b) {
    return b.start <= a.start && a.start <= b.end && b.start <= a.end && a.end <= b.end;
}

/* Grows the vector so that `extra` more intervals fit. */
static chn_hir_status FN(reserve)(SET *set, size_t extra) {
    if (set->cap - set->len >= extra) return CHN_HIR_OK;
    if (extra > SIZE_MAX / sizeof(RANGE) - set->len) return CHN_HIR_ERROR_NOMEM;
    const size_t need = set->len + extra;
    size_t cap = set->cap ? set->cap : 4;
    while (cap < need) cap = cap > SIZE_MAX / sizeof(RANGE) / 2 ? need : cap * 2;
    RANGE *ranges = (RANGE *)realloc(set->ranges, cap * sizeof(RANGE));
    if (!ranges) return CHN_HIR_ERROR_NOMEM;
    set->ranges = ranges;
    set->cap = cap;
    return CHN_HIR_OK;
}

/* Vec::push */
static chn_hir_status FN(append_range)(SET *set, RANGE r) {
    const chn_hir_status status = FN(reserve)(set, 1);
    if (status != CHN_HIR_OK) return status;
    set->ranges[set->len++] = r;
    return CHN_HIR_OK;
}

/* self.ranges.drain(..drain_end) */
static void FN(drain_prefix)(SET *set, size_t drain_end) {
    memmove(set->ranges, set->ranges + drain_end, (set->len - drain_end) * sizeof(RANGE));
    set->len -= drain_end;
}

/* IntervalSet::is_canonical */
static bool FN(is_canonical)(const SET *set) {
    for (size_t i = 1; i < set->len; ++i) {
        if (FN(cmp)(set->ranges[i - 1], set->ranges[i]) >= 0) return false;
        if (FN(is_contiguous)(set->ranges[i - 1], set->ranges[i])) return false;
    }
    return true;
}

/* The merge loop of IntervalSet::canonicalize over sorted intervals, in place. */
static void FN(merge_sorted)(SET *set) {
    size_t w = 0;
    for (size_t i = 0; i < set->len; ++i) {
        const RANGE r = set->ranges[i];
        if (w > 0 && FN(is_contiguous)(set->ranges[w - 1], r)) {
            RANGE *last = &set->ranges[w - 1];
            const BOUND lower = last->start < r.start ? last->start : r.start;
            const BOUND upper = last->end > r.end ? last->end : r.end;
            *last = FN(create)(lower, upper);
            continue;
        }
        set->ranges[w++] = r;
    }
    set->len = w;
}

/* IntervalSet::canonicalize */
static void FN(canonicalize)(SET *set) {
    if (FN(is_canonical)(set)) return;
    qsort(set->ranges, set->len, sizeof(RANGE), FN(qsort_cmp));
    FN(merge_sorted)(set);
}

void FN(init)(SET *set) {
    set->ranges = NULL;
    set->len = 0;
    set->cap = 0;
    set->folded = true;
}

void FN(drop)(SET *set) {
    free(set->ranges);
    FN(init)(set);
}

chn_hir_status FN(new)(SET *set, const RANGE *ranges, size_t len) {
    FN(init)(set);
    if (len) {
        const chn_hir_status status = FN(reserve)(set, len);
        if (status != CHN_HIR_OK) return status;
        memcpy(set->ranges, ranges, len * sizeof(RANGE));
        set->len = len;
    }
    set->folded = len == 0;
    FN(canonicalize)(set);
    return CHN_HIR_OK;
}

chn_hir_status FN(clone)(SET *dst, const SET *src) {
    const chn_hir_status status = FN(new)(dst, src->ranges, src->len);
    dst->folded = src->folded;
    return status;
}

/* IntervalSet::push. The set is canonical before the push, so inserting the interval at
 * its sorted place and merging gives what Rust's sort and merge give. */
static chn_hir_status FN(push_range)(SET *set, RANGE r) {
    const chn_hir_status status = FN(reserve)(set, 1);
    if (status != CHN_HIR_OK) return status;
    size_t lo = 0, hi = set->len;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (FN(cmp)(set->ranges[mid], r) < 0) lo = mid + 1;
        else hi = mid;
    }
    memmove(set->ranges + lo + 1, set->ranges + lo, (set->len - lo) * sizeof(RANGE));
    set->ranges[lo] = r;
    set->len += 1;
    FN(merge_sorted)(set);
    set->folded = false;
    return CHN_HIR_OK;
}

bool FN(eq)(const SET *a, const SET *b) {
    if (a->len != b->len) return false;
    for (size_t i = 0; i < a->len; ++i) {
        if (a->ranges[i].start != b->ranges[i].start || a->ranges[i].end != b->ranges[i].end) return false;
    }
    return true;
}

/* IntervalSet::case_fold_simple */
chn_hir_status FN(case_fold_simple)(SET *set) {
    if (set->folded) return CHN_HIR_OK;
    const size_t len = set->len;
    for (size_t i = 0; i < len; ++i) {
        const RANGE r = set->ranges[i];
        const chn_hir_status status = RANGE_CASE_FOLD(set, r);
        if (status != CHN_HIR_OK) {
            FN(canonicalize)(set);
            return status;
        }
    }
    FN(canonicalize)(set);
    set->folded = true;
    return CHN_HIR_OK;
}

/* IntervalSet::union. Both sides are canonical, so merging the two sorted runs (from the
 * back, into the reserved tail) equals Rust's extend-and-sort. */
chn_hir_status FN(union)(SET *set, const SET *other) {
    if (other->len == 0 || FN(eq)(set, other)) return CHN_HIR_OK;
    const chn_hir_status status = FN(reserve)(set, other->len);
    if (status != CHN_HIR_OK) return status;
    size_t i = set->len, j = other->len, k = set->len + other->len;
    while (j > 0) {
        if (i > 0 && FN(cmp)(set->ranges[i - 1], other->ranges[j - 1]) > 0) {
            set->ranges[--k] = set->ranges[--i];
        } else {
            set->ranges[--k] = other->ranges[--j];
        }
    }
    set->len += other->len;
    FN(merge_sorted)(set);
    set->folded = set->folded && other->folded;
    return CHN_HIR_OK;
}

/* IntervalSet::intersect */
chn_hir_status FN(intersect)(SET *set, const SET *other) {
    if (set->len == 0) return CHN_HIR_OK;
    if (other->len == 0) {
        set->len = 0;
        set->folded = true;
        return CHN_HIR_OK;
    }
    const size_t drain_end = set->len;
    const chn_hir_status status = FN(reserve)(set, drain_end + other->len);
    if (status != CHN_HIR_OK) return status;
    size_t a = 0, b = 0;
    for (;;) {
        const RANGE ra = set->ranges[a], rb = other->ranges[b];
        const BOUND lower = ra.start > rb.start ? ra.start : rb.start;
        const BOUND upper = ra.end < rb.end ? ra.end : rb.end;
        if (lower <= upper) set->ranges[set->len++] = FN(create)(lower, upper);
        if (ra.end < rb.end) {
            if (++a == drain_end) break;
        } else {
            if (++b == other->len) break;
        }
    }
    FN(drain_prefix)(set, drain_end);
    set->folded = set->folded && other->folded;
    return CHN_HIR_OK;
}

/* Interval::difference: *n of out[0..2] is the number of pieces. */
static void FN(range_difference)(RANGE self, RANGE other, RANGE out[2], int *n) {
    *n = 0;
    if (FN(is_subset)(self, other)) return;
    if (FN(is_intersection_empty)(self, other)) {
        out[(*n)++] = self;
        return;
    }
    if (other.start > self.start) out[(*n)++] = FN(create)(self.start, BOUND_DEC(other.start));
    if (other.end < self.end) out[(*n)++] = FN(create)(BOUND_INC(other.end), self.end);
}

/* IntervalSet::difference */
chn_hir_status FN(difference)(SET *set, const SET *other) {
    if (set->len == 0 || other->len == 0) return CHN_HIR_OK;
    const size_t drain_end = set->len;
    size_t a = 0, b = 0;
    chn_hir_status status;
    while (a < drain_end && b < other->len) {
        if (other->ranges[b].end < set->ranges[a].start) {
            b += 1;
            continue;
        }
        if (set->ranges[a].end < other->ranges[b].start) {
            if ((status = FN(append_range)(set, set->ranges[a])) != CHN_HIR_OK) return status;
            a += 1;
            continue;
        }
        RANGE range = set->ranges[a];
        bool lost = false;
        while (b < other->len && !FN(is_intersection_empty)(range, other->ranges[b])) {
            const RANGE old_range = range;
            RANGE pieces[2];
            int n;
            FN(range_difference)(range, other->ranges[b], pieces, &n);
            if (n == 0) {
                lost = true;
                break;
            }
            if (n == 2) {
                if ((status = FN(append_range)(set, pieces[0])) != CHN_HIR_OK) return status;
                range = pieces[1];
            } else {
                range = pieces[0];
            }
            if (other->ranges[b].end > old_range.end) break;
            b += 1;
        }
        if (lost) {
            a += 1;
            continue;
        }
        if ((status = FN(append_range)(set, range)) != CHN_HIR_OK) return status;
        a += 1;
    }
    while (a < drain_end) {
        if ((status = FN(append_range)(set, set->ranges[a])) != CHN_HIR_OK) return status;
        a += 1;
    }
    FN(drain_prefix)(set, drain_end);
    set->folded = set->folded && other->folded;
    return CHN_HIR_OK;
}

/* IntervalSet::symmetric_difference */
chn_hir_status FN(symmetric_difference)(SET *set, const SET *other) {
    SET intersection;
    chn_hir_status status = FN(clone)(&intersection, set);
    if (status == CHN_HIR_OK) status = FN(intersect)(&intersection, other);
    if (status == CHN_HIR_OK) status = FN(union)(set, other);
    if (status == CHN_HIR_OK) status = FN(difference)(set, &intersection);
    FN(drop)(&intersection);
    return status;
}

/* IntervalSet::negate */
chn_hir_status FN(negate)(SET *set) {
    if (set->len == 0) {
        const chn_hir_status status = FN(append_range)(set, FN(create)(BOUND_MIN, BOUND_MAX));
        if (status != CHN_HIR_OK) return status;
        set->folded = true;
        return CHN_HIR_OK;
    }
    const size_t drain_end = set->len;
    const chn_hir_status status = FN(reserve)(set, drain_end + 1);
    if (status != CHN_HIR_OK) return status;
    if (set->ranges[0].start > BOUND_MIN) {
        set->ranges[set->len++] = FN(create)(BOUND_MIN, BOUND_DEC(set->ranges[0].start));
    }
    for (size_t i = 1; i < drain_end; ++i) {
        set->ranges[set->len++] = FN(create)(BOUND_INC(set->ranges[i - 1].end), BOUND_DEC(set->ranges[i].start));
    }
    if (set->ranges[drain_end - 1].end < BOUND_MAX) {
        set->ranges[set->len++] = FN(create)(BOUND_INC(set->ranges[drain_end - 1].end), BOUND_MAX);
    }
    FN(drain_prefix)(set, drain_end);
    return CHN_HIR_OK;
}
