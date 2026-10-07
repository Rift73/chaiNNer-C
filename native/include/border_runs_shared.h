/* The border runs of the row functions in the ISA units of the separable filters
 * (separable_avx2.c, separable_avx512.c) and the spatial convolution
 * (convolution_avx2.c, convolution_avx512.c): a static helper compiled into each
 * including unit, never inline. Every including unit uses it (C4505). */
#ifndef CHAINNER_BORDER_RUNS_SHARED_H
#define CHAINNER_BORDER_RUNS_SHARED_H
#include <stddef.h>

/* A row segment's vector groups [column, stop) of `lanes` lanes (column and stop on
 * the lane grid). Every lane of a group inside [inner, outer) (inner rounded up and
 * outer down to the grid, both clamped into [column, stop)) has all its taps inside
 * the row, without reflection. The border runs [column, *head) and [*tail, stop)
 * hold the other groups, each widened into the interior to `block` elements (a block
 * of groups) where the segment holds that many; [*head, *tail) is the interior. */
static void border_runs(size_t column, size_t stop, size_t inner, size_t outer, size_t lanes, size_t block,
    size_t *head, size_t *tail)
{
    inner = (inner + lanes - 1) / lanes * lanes;
    outer = outer / lanes * lanes;
    inner = inner < column ? column : inner < stop ? inner : stop;
    outer = outer < inner ? inner : outer < stop ? outer : stop;
    *head = inner;
    if (inner > column) *head = column + block < outer ? column + block : outer;
    if (*head < inner) *head = inner;
    *tail = outer;
    if (stop > outer) *tail = stop - *head < block ? *head : stop - block < outer ? stop - block : outer;
}
#endif
