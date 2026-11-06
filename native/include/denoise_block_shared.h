/* The block's sums over every displacement, compiled into each level after its row functions (the notices at the head
 * of denoise_ops.cpp cover this file):
 * nlm_column_add: columns[i] += the squared distance of row's pixel i and the pixel a displacement away
 * nlm_column_slide: columns[i] += that distance in row down minus that distance in row up
 * nlm_window_pass: out[i] = in[i] + in[i + span], plus columns[i + 2 * span] when odd; in may be out
 * nlm_accumulate: from distances[i]'s weight, a_sums' lane i gets a_pixels' pixel i, and b_sums' lane i b_pixels' pixel i
 * when b_sums is given; a_sums may be null too. A plane of the sums is rows * row_lanes after the one before it. */
#ifndef CHAINNER_DENOISE_BLOCK_SHARED_H
#define CHAINNER_DENOISE_BLOCK_SHARED_H
#include "denoise_shared.h"

/* distances[i] = the sum of the patch's columns from i, for lanes pixels: each pass doubles the window, plus one on a
 * set bit. */
static void nlm_patch_distances(const nlm_block *b, size_t lanes)
{
    size_t size = (size_t)b->patch * 2 + 1, total = lanes + size - 1, span = 1;
    const uint32_t *in = b->columns;
    int bit = 0;
    while ((size >> (bit + 1)) != 1) ++bit;
    for (; bit >= 0; --bit) {
        int odd = (int)(size >> bit) & 1;
        size_t next = span * 2 + (size_t)odd;
        nlm_window_pass(b->distances, in, b->columns, total - next + 1, span, odd);
        in = b->distances;
        span = next;
    }
}

/* columns[i] = the column's distance over the patch's rows from top. */
static void nlm_columns_from(const nlm_block *b, const uint8_t *top, ptrdiff_t displacement, size_t count)
{
    memset(b->columns, 0, (count + NLM_COLUMN_SLACK) * sizeof(uint32_t));
    for (size_t py = 0; py < (size_t)b->patch * 2 + 1; ++py)
        nlm_column_add(b->columns, top + py * b->stride, displacement, count, b->channels);
}

/* A pixel p and the pixel q a displacement (dy, dx) away have one patch distance, whichever is looked from, so one image
 * of distances serves a displacement and its opposite. Only dy > 0, or dy = 0 and dx > 0, is computed, over every pair
 * with p or q in the block. Row r has p in the block's row r - dy, q in its row r; lane i has p in the block's column
 * i - left, q in its column i - right, and a lane outside the block lands in its row's margins. */
static void nlm_block_sums(const nlm_block *b)
{
    size_t c = b->channels, patch = (size_t)b->patch, size = patch * 2 + 1, search = (size_t)b->search;
    size_t edge = (size_t)b->border - patch;
    const uint8_t *top = b->padded + (b->first + edge) * b->stride + (b->column + edge) * c;
    const uint8_t *center = top + patch * b->stride + patch * c;
    uint32_t *sums = b->sums + search;
    memset(b->distances, 0, b->width * sizeof(uint32_t));
    for (size_t y = 0; y < b->rows; ++y)
        nlm_accumulate(b, b->width, sums + y * b->row_lanes, center + y * b->stride, NULL, NULL);
    for (size_t dy = 0; dy <= search; ++dy) {
        for (int dx = dy ? -b->search : 1; dx <= b->search; ++dx) {
            ptrdiff_t displacement = (ptrdiff_t)dy * (ptrdiff_t)b->stride + (ptrdiff_t)dx * (ptrdiff_t)c;
            size_t left = dx > 0 ? (size_t)dx : 0, right = dx < 0 ? (size_t)-dx : 0;
            size_t lanes = b->width + left + right, count = lanes + patch * 2, end = b->rows + dy;
            const uint8_t *start = top - dy * b->stride - left * c;
            const uint8_t *p = center - dy * b->stride - left * c, *q = center - right * c;
            nlm_columns_from(b, start, displacement, count);
            for (size_t r = 0;;) {
                if (r < b->rows || r >= dy) {
                    nlm_patch_distances(b, lanes);
                    nlm_accumulate(b, lanes, r >= dy ? sums + (r - dy) * b->row_lanes - left : NULL, q + r * b->stride,
                        r < b->rows ? sums + r * b->row_lanes - right : NULL, p + r * b->stride);
                }
                if (++r == end) break;
                if (r == b->rows && r < dy && (dy - r) * 2 > size) {
                    /* The rows that neither side reads cost more to slide over than to start again after. */
                    r = dy;
                    nlm_columns_from(b, start + r * b->stride, displacement, count);
                } else {
                    const uint8_t *up = start + (r - 1) * b->stride;
                    nlm_column_slide(b->columns, up + size * b->stride, up, displacement, count, c);
                }
            }
        }
    }
}
#endif
