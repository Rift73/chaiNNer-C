#ifndef CHAINNER_NOISE_RNG_H
#define CHAINNER_NOISE_RNG_H

#include <stddef.h>
#include <stdint.h>

/* NumPy 1.24.4 PCG64 XSL-RR state, including its cached upper uint32 half. */
typedef struct {
    uint64_t high, low, increment_high, increment_low;
    uint32_t cached_uint32;
    int has_uint32;
} cn_rng;

/* Integer seed words are little-endian base-2^32, at least one word. */
void cn_rng_seed(cn_rng *rng, const uint32_t *words, size_t count);
uint64_t cn_rng_next64(cn_rng *rng);
uint32_t cn_rng_next32(cn_rng *rng);
double cn_rng_double(cn_rng *rng);
/* Inclusive masked rejection, identical to NumPy ndarray shuffle. */
uint64_t cn_rng_interval(cn_rng *rng, uint64_t maximum);
void cn_rng_shuffle_i64(cn_rng *rng, int64_t *values, size_t count);
/* Advance by whole 64-bit draws and discard any cached uint32 half. */
void cn_rng_advance(cn_rng *rng, uint64_t delta);

#endif
