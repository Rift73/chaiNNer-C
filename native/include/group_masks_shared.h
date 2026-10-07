/* The lane masks of the AVX-512 units' 16-lane groups (convolution_avx512.c,
 * separable_avx512.c): static helpers compiled into each including unit, never
 * inline. Every including unit uses every helper (C4505). */
#ifndef CHAINNER_GROUP_MASKS_SHARED_H
#define CHAINNER_GROUP_MASKS_SHARED_H
#include <immintrin.h>
#include <stddef.h>

/* Lanes [low, high) of a group, 0 <= low < high <= 16. */
static __mmask16 lane_mask(size_t low, size_t high)
{
    return (__mmask16)((1u << high) - (1u << low));
}

/* The groups [column, stop) (16-lane grid) of a row segment's vector outputs
 * [first, end), where only the first group can hold lanes below first and only the
 * last group lanes from end on (first < column + 16, end > stop - 16); those lanes
 * are masked off. *head is the first group's mask, or 0 when it is whole; *tail the
 * last group's, or 0 when it is whole or is the first group (then *head holds both
 * ends). [*whole, *whole_end), relative to column, are the whole groups between. */
static void group_masks(size_t column, size_t stop, size_t first, size_t end, __mmask16 *head,
    __mmask16 *tail, size_t *whole, size_t *whole_end)
{
    size_t count = stop - column, last = count - 16;
    size_t low = first > column ? first - column : 0, high = end < stop ? end - column : count;
    *head = *tail = 0;
    *whole = 0;
    *whole_end = count;
    if (low > 0 || high < 16) {
        *head = lane_mask(low, high < 16 ? high : 16);
        *whole = 16;
    }
    /* *whole <= *whole_end: count >= 16, and the end moves to last only when last >= *whole. */
    if (high < count && last >= *whole) {
        *tail = lane_mask(0, high - last);
        *whole_end = last;
    }
}
#endif
