/* Median Blur's 3x3 and 5x5 comparator networks and its AVX2 unit (neighborhood_avx2.c), reached only through
 * small_median_range's dispatch on the call's ISA level (neighborhood_ops.c). The networks are adapted from OpenCV
 * 5.0.0 median_blur.simd.hpp. Copyright (C) 2000-2008, 2018 Intel Corporation; (C) 2009 Willow Garage; (C) 2014-2015
 * Itseez. BSD notice is in chainner_native.LICENSE.txt. */
#ifndef CHAINNER_MEDIAN_SHARED_H
#define CHAINNER_MEDIAN_SHARED_H
#include <stddef.h>
#include <stdint.h>

/* A window's samples in row-major order (dy, then dx); after the pairs (min to the first, max to the second), the
 * middle sample holds the median. */
static const uint8_t cn_median_square3[][2] = {
    {1,2},{4,5},{7,8},{0,1},{3,4},{6,7},{1,2},{4,5},{7,8},
    {0,3},{5,8},{4,7},{3,6},{1,4},{2,5},{4,7},{4,2},{6,4},{4,2}
};
static const uint8_t cn_median_square5[][2] = {
    {1,2},{0,1},{1,2},{4,5},{3,4},{4,5},{0,3},{2,5},{2,3},{1,4},
    {1,2},{3,4},{7,8},{6,7},{7,8},{10,11},{9,10},{10,11},{6,9},{8,11},
    {8,9},{7,10},{7,8},{9,10},{0,6},{4,10},{4,6},{2,8},{2,4},{6,8},
    {1,7},{5,11},{5,7},{3,9},{3,5},{7,9},{1,2},{3,4},{5,6},{7,8},
    {9,10},{13,14},{12,13},{13,14},{16,17},{15,16},{16,17},{12,15},{14,17},{14,15},
    {13,16},{13,14},{15,16},{19,20},{18,19},{19,20},{21,22},{23,24},{21,23},{22,24},
    {22,23},{18,21},{20,23},{20,21},{19,22},{22,24},{19,20},{21,22},{23,24},{12,18},
    {16,22},{16,18},{14,20},{20,24},{14,16},{18,20},{22,24},{13,19},{17,23},{17,19},
    {15,21},{15,17},{19,21},{13,14},{15,16},{17,18},{19,20},{21,22},{23,24},{0,12},
    {8,20},{8,12},{4,16},{16,24},{12,16},{2,14},{10,22},{10,14},{6,18},{6,10},
    {10,12},{1,13},{9,21},{9,13},{5,17},{13,17},{3,15},{11,23},{11,15},{7,19},
    {7,11},{11,13},{11,12}
};

/* out[first, last) of row y (flat samples of a w x c row; last - first >= 8) for radius 1 or 2: the samples whose
 * window lies inside the row (x within [radius, w - radius)), rows clamped to [0, h), eight at a time through the
 * network by vminps/vmaxps; the last group overlaps back. Only for an image without NaN and -0, where the median is
 * one value whatever the comparisons choose on equal inputs. */
void cn_median_interior_avx2(const float *src, float *out, size_t h, size_t w, size_t c, size_t radius, size_t y,
                             size_t first, size_t last);
/* 1 when src[0, count) holds a NaN or -0 (bit tests, so only without DAZ, under which the entry's == 0 also takes
 * the negative denormals), else 0. */
int cn_median_special_avx2(const float *src, size_t count);
#endif
