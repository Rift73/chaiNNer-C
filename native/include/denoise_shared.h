/* The non-local means block: one tile's run of image columns, which denoise_ops.cpp's tile loop hands to a level's
 * functions (denoise_avx2.c, denoise_avx512.c; the scalar level in denoise_ops.cpp). Every level adds the same integers
 * to the same sums, in its own order, so their outputs are identical.
 *
 * The vector levels have no scalar tails: a row function rounds its count up to whole vectors. The lanes past the count
 * read the padded image's next bytes and the slack of columns and distances, and land only in that slack and in the
 * margins of the sums' rows, which nothing reads. No lane inside the count reads them, and the weight lookup compares
 * unsigned, so whatever a slack lane holds stays inside the table. */
#ifndef CHAINNER_DENOISE_SHARED_H
#define CHAINNER_DENOISE_SHARED_H
#include "chainner.h"
#include <string.h>

/* A vector level's widest step: bytes past the padded image's end, and lanes past a row in columns and distances. */
#define NLM_PADDED_SLACK 64
#define NLM_COLUMN_SLACK 32
/* A row of the sums holds search lanes, its width, search lanes and one vector step, rounded up to this many lanes. */
#define NLM_ROW_ROUND 16

typedef struct nlm_block {
    const uint8_t *padded;  /* the image with border reflected pixels on every side */
    size_t stride;          /* bytes per padded row */
    size_t channels;
    size_t first, rows;     /* the tile's image rows */
    size_t column, width;   /* the block's image columns */
    size_t row_lanes;       /* lanes per row of the sums */
    int patch, search, border, shift;
    uint32_t limit;         /* patch distances from here on weigh 0 */
    const int32_t *weights; /* weights[distance >> shift], distance < limit */
    uint32_t *columns;      /* width + search + 2 * patch + NLM_COLUMN_SLACK: each column's distance over the patch's rows */
    uint32_t *distances;    /* the same size: each pixel's patch distance */
    uint32_t *sums;         /* channels + 1 planes of rows * row_lanes: the weights, then the channels' weighted sums */
} nlm_block;

#if CN_ISA_UNITS()
#ifdef __cplusplus
extern "C" {
#endif
void cn_nlm_block_avx2(const nlm_block *block);
void cn_nlm_block_avx512(const nlm_block *block);
#ifdef __cplusplus
}
#endif
#endif

#endif
