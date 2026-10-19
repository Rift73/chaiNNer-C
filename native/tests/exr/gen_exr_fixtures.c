/* Writes the EXR fixtures of native/tests/test_exr_decode.py with the vendored OpenEXRCore 3.5.2 writer (GPL-3.0).
 *
 * Pixel values follow formulas the test recomputes with NumPy, so its references do not come from OpenEXR:
 *   exact  (lossless codecs): half bits ((x*7919 + y*104729 + c*15485863) mod 0x7C00), negative where (x+y+c) % 3 == 0,
 *          +inf, NaN and -inf at (0,0), (1,0), (2,0) of channel 0 (x, y: data-window-relative; c: the channel's
 *          position in the file's sorted channel list); float32 ((n mod 200001) - 100000) / 64; uint32 n mod 2^32
 *   smooth (lossy codecs): half of 0.1 + 3*(x/W)*(y/H) + 0.5*c/3 (float64 rounded once, nearest even)
 *
 * Build (after native\Build.ps1; the static libraries are ThinLTO bitcode, so lld-link links them):
 *   clang-cl /O2 /MT -flto=thin /I third_party\openexr\OpenEXRCore /I third_party\openexr\chainner
 *     /I third_party\imath\chainner /I third_party\imath\chainner\Imath /I third_party\imath\Imath
 *     tests\exr\gen_exr_fixtures.c build\chainner_openexr.lib build\chainner_openjph.lib build\chainner_zstd.lib
 *     build\chainner_deflate.lib /Fe:<scratch>\gen_exr_fixtures.exe -fuse-ld=lld
 * Run: gen_exr_fixtures <directory> [4k]  (4k writes 3840x2160 RGBA half files for the decode-speed measurement). */
#include <openexr.h>

#define IMATH_HALF_NO_LOOKUP_TABLE
#include <half.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct channel_spec {
    const char *name;
    exr_pixel_type_t type;
    int xs, ys;
} channel_spec;

typedef struct image_spec {
    const char *file;
    exr_storage_t storage;
    exr_compression_t compression;
    int32_t min_x, min_y, width, height;
    int display; /* 1: display window (0,0)-(40,50) instead of the data window */
    uint32_t tile_x, tile_y;
    exr_tile_level_mode_t levels;
    int smooth;
    int channels;
    channel_spec channel[8];
    int chromaticities; /* 1: a non-default chromaticities attribute */
    int second_part;    /* 1: a second part follows with other values */
} image_spec;

static int fail(const char *what, exr_result_t rv) {
    fprintf(stderr, "%s: %s\n", what, exr_get_default_error_message(rv));
    return 1;
}

static size_t type_bytes(exr_pixel_type_t t) { return t == EXR_PIXEL_HALF ? 2 : 4; }

/* The sample of channel position c at data-window-relative (x, y), in the channel's type. */
static void sample(const image_spec *s, int c, int64_t x, int64_t y, int level, uint8_t *out) {
    const exr_pixel_type_t t = s->channel[c].type;
    const uint64_t n = (uint64_t)(x * 7919 + y * 104729 + (int64_t)c * 15485863 + level * 31);
    if (t == EXR_PIXEL_UINT) {
        const uint32_t v = (uint32_t)n;
        memcpy(out, &v, 4);
    } else if (t == EXR_PIXEL_FLOAT) {
        const float v = (float)((int64_t)(n % 200001) - 100000) / 64.0f;
        memcpy(out, &v, 4);
    } else if (s->smooth) {
        const double v = 0.1 + 3.0 * ((double)x / s->width) * ((double)y / s->height) + 0.5 * c / 3.0;
        const uint16_t h = imath_float_to_half((float)v); /* exact: v is formed in float64, then float32 rounds it */
        memcpy(out, &h, 2);
    } else {
        uint16_t h = (uint16_t)(n % 0x7C00);
        if ((x + y + c) % 3 == 0) h |= 0x8000;
        if (c == 0 && y == 0 && x < 3 && level == 0) h = x == 0 ? 0x7C00 : x == 1 ? 0x7E00 : 0xFC00;
        memcpy(out, &h, 2);
    }
}

static int write_part(exr_context_t f, int part, const image_spec *s, int second) {
    int32_t levels_x = 1, levels_y = 1;
    if (s->storage == EXR_STORAGE_TILED) {
        exr_result_t rv = exr_get_tile_levels(f, part, &levels_x, &levels_y);
        if (rv) return fail("tile levels", rv);
    }
    for (int ly = 0; ly < levels_y; ++ly)
        for (int lx = 0; lx < levels_x; ++lx) {
            if (s->levels == EXR_TILE_MIPMAP_LEVELS && lx != ly) continue;
            int32_t w = s->width, h = s->height;
            if (s->storage == EXR_STORAGE_TILED) {
                exr_result_t rv = exr_get_level_sizes(f, part, lx, ly, &w, &h);
                if (rv) return fail("level sizes", rv);
            }
            /* One plane per channel at its sampled resolution. */
            uint8_t *plane[8];
            int32_t pw[8];
            for (int c = 0; c < s->channels; ++c) {
                const int xs = s->channel[c].xs, ys = s->channel[c].ys;
                pw[c] = (w + xs - 1) / xs;
                const int32_t ph = (h + ys - 1) / ys;
                plane[c] = (uint8_t *)malloc((size_t)pw[c] * ph * type_bytes(s->channel[c].type));
                for (int32_t y = 0; y < ph; ++y)
                    for (int32_t x = 0; x < pw[c]; ++x)
                        sample(s, c + (second ? 1 : 0), (int64_t)x * xs, (int64_t)y * ys, lx + ly + second * 7,
                               plane[c] + ((size_t)y * pw[c] + x) * type_bytes(s->channel[c].type));
            }
            int32_t chunks_x = 1, chunks_y;
            int32_t lines = 1;
            if (s->storage == EXR_STORAGE_TILED) {
                exr_result_t rv = exr_get_tile_counts(f, part, lx, ly, &chunks_x, &chunks_y);
                if (rv) return fail("tile counts", rv);
            } else {
                exr_result_t rv = exr_get_scanlines_per_chunk(f, part, &lines);
                if (rv) return fail("lines per chunk", rv);
                chunks_y = (h + lines - 1) / lines;
            }
            exr_encode_pipeline_t encoder;
            memset(&encoder, 0, sizeof encoder);
            int started = 0;
            for (int32_t cy = 0; cy < chunks_y; ++cy)
                for (int32_t cx = 0; cx < chunks_x; ++cx) {
                    exr_chunk_info_t info;
                    int64_t x0 = 0, y0;
                    exr_result_t rv;
                    if (s->storage == EXR_STORAGE_TILED) {
                        rv = exr_write_tile_chunk_info(f, part, cx, cy, lx, ly, &info);
                        x0 = (int64_t)cx * s->tile_x;
                        y0 = (int64_t)cy * s->tile_y;
                    } else {
                        rv = exr_write_scanline_chunk_info(f, part, s->min_y + cy * lines, &info);
                        y0 = (int64_t)cy * lines;
                    }
                    if (rv) return fail("chunk info", rv);
                    rv = started ? exr_encoding_update(f, part, &info, &encoder)
                                 : exr_encoding_initialize(f, part, &info, &encoder);
                    if (rv) return fail("encoder", rv);
                    for (int c = 0; c < encoder.channel_count; ++c) {
                        const size_t b = type_bytes(s->channel[c].type);
                        encoder.channels[c].encode_from_ptr =
                            plane[c] + ((size_t)(y0 / s->channel[c].ys) * pw[c] + (size_t)(x0 / s->channel[c].xs)) * b;
                        encoder.channels[c].user_pixel_stride = (int32_t)b;
                        encoder.channels[c].user_line_stride = (int32_t)(pw[c] * b);
                        encoder.channels[c].user_bytes_per_element = (int16_t)b;
                        encoder.channels[c].user_data_type = (uint16_t)s->channel[c].type;
                    }
                    if (!started) {
                        rv = exr_encoding_choose_default_routines(f, part, &encoder);
                        if (rv) return fail("encoder routines", rv);
                        started = 1;
                    }
                    rv = exr_encoding_run(f, part, &encoder);
                    if (rv) return fail("encode", rv);
                }
            exr_encoding_destroy(f, &encoder);
            for (int c = 0; c < s->channels; ++c) free(plane[c]);
        }
    return 0;
}

static int add_part(exr_context_t f, const image_spec *s, const char *name, int *part) {
    exr_result_t rv = exr_add_part(f, name, s->storage, part);
    if (rv) return fail("add part", rv);
    const exr_attr_box2i_t data = {{s->min_x, s->min_y}, {s->min_x + s->width - 1, s->min_y + s->height - 1}};
    const exr_attr_box2i_t display = s->display ? (exr_attr_box2i_t){{0, 0}, {40, 50}} : data;
    const exr_attr_v2f_t center = {0.f, 0.f};
    rv = exr_initialize_required_attr(f, *part, &display, &data, 1.f, &center, 1.f, EXR_LINEORDER_INCREASING_Y,
                                      s->compression);
    if (rv) return fail("required attributes", rv);
    for (int c = 0; c < s->channels; ++c) {
        rv = exr_add_channel(f, *part, s->channel[c].name, s->channel[c].type, EXR_PERCEPTUALLY_LOGARITHMIC,
                             s->channel[c].xs, s->channel[c].ys);
        if (rv) return fail("channel", rv);
    }
    if (s->storage == EXR_STORAGE_TILED) {
        rv = exr_set_tile_descriptor(f, *part, s->tile_x, s->tile_y, s->levels, EXR_TILE_ROUND_DOWN);
        if (rv) return fail("tiles", rv);
    }
    if (s->chromaticities) {
        const exr_attr_chromaticities_t c = {0.708f, 0.292f, 0.170f, 0.797f, 0.131f, 0.046f, 0.3127f, 0.3290f};
        rv = exr_attr_set_chromaticities(f, *part, "chromaticities", &c);
        if (rv) return fail("chromaticities", rv);
    }
    return 0;
}

static int write_image(const char *directory, const image_spec *s) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", directory, s->file);
    exr_context_t f = NULL;
    exr_context_initializer_t init = EXR_DEFAULT_CONTEXT_INITIALIZER;
    exr_result_t rv = exr_start_write(&f, path, EXR_WRITE_FILE_DIRECTLY, &init);
    if (rv) return fail(path, rv);
    int part0 = 0, part1 = 0;
    if (add_part(f, s, "first", &part0) || (s->second_part && add_part(f, s, "second", &part1))) return 1;
    rv = exr_write_header(f);
    if (rv) return fail("header", rv);
    if (write_part(f, part0, s, 0) || (s->second_part && write_part(f, part1, s, 1))) return 1;
    rv = exr_finish(&f);
    if (rv) return fail("finish", rv);
    return 0;
}

#define H EXR_PIXEL_HALF
#define F EXR_PIXEL_FLOAT
#define U EXR_PIXEL_UINT
#define RGBA_HALF 4, {{"A", H, 1, 1}, {"B", H, 1, 1}, {"G", H, 1, 1}, {"R", H, 1, 1}}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: gen_exr_fixtures <directory> [4k]\n");
        return 2;
    }
    const char *dir = argv[1];
    if (argc > 2 && strcmp(argv[2], "4k") == 0) {
        static const struct {
            const char *file;
            exr_compression_t compression;
        } bench[] = {{"4k_none.exr", EXR_COMPRESSION_NONE}, {"4k_zip.exr", EXR_COMPRESSION_ZIP},
                     {"4k_piz.exr", EXR_COMPRESSION_PIZ},   {"4k_dwaa.exr", EXR_COMPRESSION_DWAA},
                     {"4k_htj2k.exr", EXR_COMPRESSION_HTJ2K256}};
        for (size_t i = 0; i < sizeof bench / sizeof bench[0]; ++i) {
            const image_spec s = {bench[i].file, EXR_STORAGE_SCANLINE, bench[i].compression, 0, 0, 3840, 2160, 0, 0, 0,
                                  EXR_TILE_ONE_LEVEL, 1, RGBA_HALF};
            if (write_image(dir, &s)) return 1;
        }
        return 0;
    }
    static const struct {
        const char *name;
        exr_compression_t compression;
        int lossy;
    } codecs[] = {{"none", EXR_COMPRESSION_NONE, 0},      {"rle", EXR_COMPRESSION_RLE, 0},
                  {"zips", EXR_COMPRESSION_ZIPS, 0},      {"zip", EXR_COMPRESSION_ZIP, 0},
                  {"piz", EXR_COMPRESSION_PIZ, 0},        {"pxr24", EXR_COMPRESSION_PXR24, 0},
                  {"b44", EXR_COMPRESSION_B44, 1},        {"b44a", EXR_COMPRESSION_B44A, 1},
                  {"dwaa", EXR_COMPRESSION_DWAA, 1},      {"dwab", EXR_COMPRESSION_DWAB, 1},
                  {"htj2k256", EXR_COMPRESSION_HTJ2K256, 0}, {"htj2k32", EXR_COMPRESSION_HTJ2K32, 0},
                  {"lj2k", EXR_COMPRESSION_LJ2K, 1},      {"zstd", EXR_COMPRESSION_ZSTD, 0}};
    char names[sizeof codecs / sizeof codecs[0]][64];
    for (size_t i = 0; i < sizeof codecs / sizeof codecs[0]; ++i) {
        snprintf(names[i], sizeof names[i], "rgba_half_%s.exr", codecs[i].name);
        const image_spec s = {names[i], EXR_STORAGE_SCANLINE, codecs[i].compression, 0, 0, 23, 37, 0, 0, 0,
                              EXR_TILE_ONE_LEVEL, codecs[i].lossy, RGBA_HALF};
        if (write_image(dir, &s)) return 1;
    }
    const image_spec specs[] = {
        {"rgb_float_zip.exr", EXR_STORAGE_SCANLINE, EXR_COMPRESSION_ZIP, 0, 0, 23, 37, 0, 0, 0, EXR_TILE_ONE_LEVEL, 0,
         3, {{"B", F, 1, 1}, {"G", F, 1, 1}, {"R", F, 1, 1}}},
        {"rgb_float_piz.exr", EXR_STORAGE_SCANLINE, EXR_COMPRESSION_PIZ, 0, 0, 23, 37, 0, 0, 0, EXR_TILE_ONE_LEVEL, 0,
         3, {{"B", F, 1, 1}, {"G", F, 1, 1}, {"R", F, 1, 1}}},
        {"rgb_float_pxr24.exr", EXR_STORAGE_SCANLINE, EXR_COMPRESSION_PXR24, 0, 0, 23, 37, 0, 0, 0, EXR_TILE_ONE_LEVEL,
         0, 3, {{"B", F, 1, 1}, {"G", F, 1, 1}, {"R", F, 1, 1}}},
        {"rgb_uint_zip.exr", EXR_STORAGE_SCANLINE, EXR_COMPRESSION_ZIP, 0, 0, 23, 37, 0, 0, 0, EXR_TILE_ONE_LEVEL, 0,
         3, {{"B", U, 1, 1}, {"G", U, 1, 1}, {"R", U, 1, 1}}},
        {"rgba_mixed_rle.exr", EXR_STORAGE_SCANLINE, EXR_COMPRESSION_RLE, 0, 0, 23, 37, 0, 0, 0, EXR_TILE_ONE_LEVEL, 0,
         4, {{"A", H, 1, 1}, {"B", F, 1, 1}, {"G", U, 1, 1}, {"R", H, 1, 1}}},
        {"rg_half_zip.exr", EXR_STORAGE_SCANLINE, EXR_COMPRESSION_ZIP, 0, 0, 23, 37, 0, 0, 0, EXR_TILE_ONE_LEVEL, 0, 2,
         {{"G", H, 1, 1}, {"R", H, 1, 1}}},
        {"rgba_extra_zip.exr", EXR_STORAGE_SCANLINE, EXR_COMPRESSION_ZIP, 0, 0, 23, 37, 0, 0, 0, EXR_TILE_ONE_LEVEL, 0,
         6, {{"A", H, 1, 1}, {"B", H, 1, 1}, {"G", H, 1, 1}, {"R", H, 1, 1}, {"Z", F, 1, 1}, {"diffuse.R", H, 1, 1}}},
        {"y_half_piz.exr", EXR_STORAGE_SCANLINE, EXR_COMPRESSION_PIZ, 0, 0, 23, 37, 0, 0, 0, EXR_TILE_ONE_LEVEL, 0, 1,
         {{"Y", H, 1, 1}}},
        {"ya_half_zip.exr", EXR_STORAGE_SCANLINE, EXR_COMPRESSION_ZIP, 0, 0, 23, 37, 0, 0, 0, EXR_TILE_ONE_LEVEL, 0, 2,
         {{"A", H, 1, 1}, {"Y", H, 1, 1}}},
        {"z_float_none.exr", EXR_STORAGE_SCANLINE, EXR_COMPRESSION_NONE, 0, 0, 23, 37, 0, 0, 0, EXR_TILE_ONE_LEVEL, 0,
         1, {{"Z", F, 1, 1}}},
        {"yc_half_zip.exr", EXR_STORAGE_SCANLINE, EXR_COMPRESSION_ZIP, 0, 0, 24, 36, 0, 0, 0, EXR_TILE_ONE_LEVEL, 1, 3,
         {{"BY", H, 2, 2}, {"RY", H, 2, 2}, {"Y", H, 1, 1}}},
        {"yca_half_piz_chroma.exr", EXR_STORAGE_SCANLINE, EXR_COMPRESSION_PIZ, -4, 6, 24, 36, 0, 0, 0,
         EXR_TILE_ONE_LEVEL, 1, 4, {{"A", H, 1, 1}, {"BY", H, 2, 2}, {"RY", H, 2, 2}, {"Y", H, 1, 1}}, 1},
        {"multi3_float_zip.exr", EXR_STORAGE_SCANLINE, EXR_COMPRESSION_ZIP, 0, 0, 23, 37, 0, 0, 0, EXR_TILE_ONE_LEVEL,
         0, 3, {{"0", F, 1, 1}, {"1", F, 1, 1}, {"2", F, 1, 1}}},
        {"tiled_rgba_zip.exr", EXR_STORAGE_TILED, EXR_COMPRESSION_ZIP, 0, 0, 23, 37, 0, 16, 8, EXR_TILE_ONE_LEVEL, 0,
         RGBA_HALF},
        {"tiled_mip_rgb_piz.exr", EXR_STORAGE_TILED, EXR_COMPRESSION_PIZ, 0, 0, 23, 37, 0, 8, 8, EXR_TILE_MIPMAP_LEVELS,
         0, 3, {{"B", H, 1, 1}, {"G", H, 1, 1}, {"R", H, 1, 1}}},
        {"tiled_rip_rgba_dwab.exr", EXR_STORAGE_TILED, EXR_COMPRESSION_DWAB, 0, 0, 23, 37, 0, 16, 16,
         EXR_TILE_RIPMAP_LEVELS, 1, RGBA_HALF},
        {"window_rgba_zip.exr", EXR_STORAGE_SCANLINE, EXR_COMPRESSION_ZIP, -5, 7, 23, 37, 1, 0, 0, EXR_TILE_ONE_LEVEL,
         0, RGBA_HALF},
        {"window_tiled_rgba_rle.exr", EXR_STORAGE_TILED, EXR_COMPRESSION_RLE, 3, -9, 23, 37, 1, 16, 8,
         EXR_TILE_ONE_LEVEL, 0, RGBA_HALF},
        {"multipart_rgb_zip.exr", EXR_STORAGE_SCANLINE, EXR_COMPRESSION_ZIP, 0, 0, 23, 37, 0, 0, 0, EXR_TILE_ONE_LEVEL,
         0, 3, {{"B", H, 1, 1}, {"G", H, 1, 1}, {"R", H, 1, 1}}, 0, 1},
        {"other_channels_zip.exr", EXR_STORAGE_SCANLINE, EXR_COMPRESSION_ZIP, 0, 0, 23, 37, 0, 0, 0,
         EXR_TILE_ONE_LEVEL, 0, 2, {{"diffuse.G", H, 1, 1}, {"diffuse.R", H, 1, 1}}},
    };
    for (size_t i = 0; i < sizeof specs / sizeof specs[0]; ++i)
        if (write_image(dir, &specs[i])) return 1;
    return 0;
}
