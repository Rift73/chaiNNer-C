/* The direct window folds of morphology_avx2.c, which replace the deques of the two deque units (filter_ops.c's
 * morph_lines, morphology_ops.c's ellipse_horizontal) at avx2 and above for windows of at most 64 on each side over
 * images without exceptional words (no NaN). Each output is its clipped window's last extremal element, as the
 * deques give it, found by folding the window in index order and taking every value that ties or beats the extremum
 * so far (vcmpps, then vblendvps, which moves bits). */
#ifndef CHAINNER_MORPH_FOLD_H
#define CHAINNER_MORPH_FOLD_H
#include <stddef.h>

#define CN_MORPH_FOLD_RADIUS 64

/* A row's outputs along the row: in and out are that row, width pixels of channels samples. */
void cn_morphology_row_avx2(const float *in, float *out, size_t width, size_t channels, size_t radius, int maximum);
/* Row y's outputs down the columns: in and out are whole images of height rows of row samples. */
void cn_morphology_column_avx2(const float *in, float *out, size_t height, size_t row, size_t y, size_t radius,
                               int maximum);
#endif
