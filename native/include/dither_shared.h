/* The palette dither's job (neighborhood_ops.c), shared with the /arch:AVX2 unit
 * dither_avx2.c: types and a prototype only, no static bodies. */
#ifndef CHAINNER_DITHER_SHARED_H
#define CHAINNER_DITHER_SHARED_H
#include "chainner.h"
#include "isa.h"

/* Palettes below 300 unique colors, and those holding a nonfinite color, have no kd
 * tree; their channel-major copy holds at most 299 entries rounded up to groups of 8,
 * so a nonfinite palette takes it up to 304 colors. */
#define DITHER_SOA_ENTRIES 304

typedef struct palette_entry { float color[4], key; size_t ordinal; } palette_entry;

typedef struct palette_tree {
    size_t color, left, right;
    float low[4], high[4];
} palette_tree;

/* isa: the level the C entry read once for the call. soa: with no kd tree and at most
 * DITHER_SOA_ENTRIES colors, cn_neighborhood_palette_apply's copy of the sorted palette
 * by channel, palette[i].color[k] at soa[k * soa_stride + i], soa_stride a multiple of
 * 8 at least palette_count (the lanes past it are zero); NULL elsewhere. */
typedef struct dither_job {
    const float *src;
    float *out, *error_memory;
    size_t height, width, channels, colors, palette_count, map_size;
    int mode, algorithm;
    float factor, inverse, thresholds[256];
    /* Ordered dithering's threshold rows: thresholds (map_stride = map_size) for maps up
     * to 16, else a map_stride = min(width, map_size) table of min(height, map_size) rows,
     * the only cells an image reads. */
    const float *map;
    size_t map_stride;
    palette_entry *palette;
    palette_tree *tree;
    size_t tree_root;
    cn_isa_level isa;
    const float *soa;
    size_t soa_stride;
} dither_job;

/* dither_avx2.c, for ISA level avx2 or above with job->soa: nearest_palette's linear
 * scan as a brute-force search. Each entry's distance keeps palette_distance's
 * sequence (0 + d0 d0, then + dk dk in channel order); the result is index 0 when the
 * first distance is NaN, else the first index whose distance compares equal to the
 * minimum over the non-NaN distances, as the scan's strict < gives. Lanes past
 * palette_count are masked, so there is no scalar tail. */
size_t cn_dither_nearest_avx2(const dither_job *job, const float *color);
#endif
