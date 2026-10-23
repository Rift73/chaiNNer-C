/* PNG through libpng 1.6.57 with OpenCV 5.0's PngDecoder (grfmt_png.cpp) under IMREAD_UNCHANGED, for still images:
 * its own chunk walk to the first IDAT (readHeader), then libpng from the first byte with readData's transforms
 * (GPL-3.0). An animated PNG (acTL with more than one frame) takes OpenCV's other path and is not ported.
 * The common files skip libpng's row loop and zlib: png_decode_direct. */
#include "image_codecs.h"

#include <setjmp.h>
#include <stdlib.h>
#include <string.h>

#include "libdeflate.h"
#include "png.h"

static uint32_t be32(const uint8_t *p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }

#define CHUNK(a, b, c, d) ((uint32_t)(a) << 24 | (uint32_t)(b) << 16 | (uint32_t)(c) << 8 | (uint32_t)(d))

/* readHeader's walk with read_chunk's size rules: false where OpenCV returns false; the walk records whether the
 * file is animated (acTL frames > 1) and the last fcTL before IDAT, which png_read_info's size is checked against. */
typedef struct png_walk {
    int fctl;
    uint32_t w0, h0, x0, y0;
} png_walk;

static int opencv_png_walk(const uint8_t *data, size_t size, png_walk *walk) {
    size_t at = 8;
    int first = 1;
    memset(walk, 0, sizeof *walk);
    for (;;) {
        if (size - at < 8) return 0;
        const size_t length = (size_t)be32(data + at) + 12;
        const uint32_t id = be32(data + at + 4);
        if (id == CHUNK('I', 'H', 'D', 'R')) {
            if (length != 8 + 13 + 4) return 0;
        } else if (id == CHUNK('a', 'c', 'T', 'L')) {
            if (length != 8 + 8 + 4) return 0;
        } else if (id == CHUNK('f', 'c', 'T', 'L')) {
            if (length != 8 + 26 + 4) return 0;
        } else if (id == CHUNK('b', 'K', 'G', 'D')) {
            if (length != 8 + 1 + 4 && length != 8 + 2 + 4 && length != 8 + 6 + 4) return 0;
        } else if (id != CHUNK('f', 'd', 'A', 'T') && id != CHUNK('I', 'D', 'A', 'T') && id != CHUNK('I', 'E', 'N', 'D') &&
                   id != CHUNK('P', 'L', 'T', 'E') && id != CHUNK('t', 'E', 'X', 't') && id != CHUNK('t', 'R', 'N', 'S')) {
            if (length > PNG_USER_CHUNK_MALLOC_MAX) return 0;
        }
        if (size - at < length) return 0;
        const uint8_t *chunk = data + at;
        at += length;
        if (first) {
            if (id != CHUNK('I', 'H', 'D', 'R')) return 0; /* CgBI or anything else first */
            first = 0;
            continue;
        }
        if (!id) return 0;
        if (id == CHUNK('I', 'D', 'A', 'T')) return 1;
        if (id == CHUNK('a', 'c', 'T', 'L')) {
            const uint32_t frames = be32(chunk + 8);
            if (frames != 1) return 0; /* 0 fails in OpenCV; more is an animation, not ported */
        }
        if (id == CHUNK('f', 'c', 'T', 'L')) {
            walk->fctl = 1;
            walk->w0 = be32(chunk + 12);
            walk->h0 = be32(chunk + 16);
            walk->x0 = be32(chunk + 20);
            walk->y0 = be32(chunk + 24);
            if (chunk[32] > 2 || chunk[33] > 1) return 0;
        }
    }
}

typedef struct png_source {
    const uint8_t *data;
    size_t size, at;
} png_source;

/* readDataFromBuf: reading past the end is a libpng error ("PNG input buffer is incomplete"). */
static void png_read_buffer(png_structp png, png_bytep out, size_t count) {
    png_source *source = (png_source *)png_get_io_ptr(png);
    if (source->size - source->at < count) png_error(png, "PNG input buffer is incomplete");
    memcpy(out, source->data + source->at, count);
    source->at += count;
}

static void png_fail(png_structp png, png_const_charp message) {
    (void)message;
    png_longjmp(png, 1);
}

static void png_quiet(png_structp png, png_const_charp message) {
    (void)png;
    (void)message;
}

#define OPENCV_MAX_PIXELS CN_MAX_IMAGE_PIXELS

/* The direct path's largest filtered image (its one extra buffer): 16 megapixels of RGBA. */
#define DIRECT_MAX_BYTES (UINT64_C(1) << 26)

static uint8_t png_paeth(int a, int b, int c) {
    const int pa = abs(b - c), pb = abs(a - c), pc = abs(a + b - 2 * c);
    return (uint8_t)(pa <= pb && pa <= pc ? a : pb <= pc ? b : c);
}

/* Sub (1), Average (3) or Paeth (4) from a row's second pixel on, each byte's left and upper-left neighbours carried
 * in locals instead of re-read from the bytes just written; bpp is a constant at each call. */
static inline void png_unfilter_left(uint8_t filter, uint8_t *row, const uint8_t *prior, size_t length, size_t bpp) {
    uint8_t left[8], upper_left[8];
    for (size_t k = 0; k < bpp; ++k) {
        left[k] = row[k];
        upper_left[k] = prior[k];
    }
    for (size_t i = bpp; i < length; i += bpp) {
        for (size_t k = 0; k < bpp; ++k) {
            const uint8_t up = prior[i + k];
            const int predictor = filter == 1   ? left[k]
                                  : filter == 3 ? (left[k] + up) >> 1
                                                : png_paeth(left[k], up, upper_left[k]);
            left[k] = row[i + k] = (uint8_t)(row[i + k] + predictor);
            upper_left[k] = up;
        }
    }
}

/* PNG's five filters (method 0) undone in place on one row of length bytes, prior the previous unfiltered row (zeros
 * before the first); 0 for any other filter type, which libpng rejects ("bad adaptive filter value"). */
static int png_unfilter(uint8_t filter, uint8_t *row, const uint8_t *prior, size_t length, size_t bpp) {
    if (filter > 4) return 0;
    if (filter == 2) {
        for (size_t i = 0; i < length; ++i) row[i] = (uint8_t)(row[i] + prior[i]);
    } else if (filter != 0) {
        /* The first pixel's left neighbours are 0: Average adds half the byte above, Paeth the byte above. */
        for (size_t k = 0; k < bpp && filter != 1; ++k)
            row[k] = (uint8_t)(row[k] + (filter == 3 ? prior[k] >> 1 : prior[k]));
        switch (bpp) {
        case 1: png_unfilter_left(filter, row, prior, length, 1); break;
        case 2: png_unfilter_left(filter, row, prior, length, 2); break;
        case 3: png_unfilter_left(filter, row, prior, length, 3); break;
        case 4: png_unfilter_left(filter, row, prior, length, 4); break;
        case 6: png_unfilter_left(filter, row, prior, length, 6); break;
        default: png_unfilter_left(filter, row, prior, length, 8); break;
        }
    }
    return 1;
}

/* An indexed image's expansion, readData's png_set_palette_to_rgb (with png_set_tRNS_to_alpha at 4 channels): its
 * PLTE colours, its tRNS alpha values (255 past them) and its bit depth (1, 2, 4 or 8). */
typedef struct png_palette {
    const png_color *colors;
    const png_byte *alpha;
    int color_count, alpha_count, depth;
} png_palette;

/* The rows of a non-interlaced image, without libpng's row loop: 8- or 16-bit grey, RGB or RGBA without tRNS, or
 * indexed (palette not NULL). libpng has read the chunks before the first IDAT (its data at idat); this path takes
 * the file when the IDAT chunks that follow have valid CRCs and lengths below 2^31 and an empty IEND with a valid
 * CRC comes right after them, their zlib stream declares the 32 KiB window (so zlib and libdeflate accept the same
 * distances) and inflates, Adler-32 included, to exactly the filtered rows, and every index is inside the palette.
 * The rows are unfiltered and stored as readData's transforms give them: BGR(A) order (png_set_bgr), 16-bit samples
 * little-endian (png_set_swap), indices expanded to their colours. For such a file libpng gives the same pixels and
 * no error; for every other file this returns 0 having written only pixels, and libpng decodes from idat as before. */
static int png_decode_direct(const uint8_t *data, size_t size, size_t idat, uint32_t width, uint32_t height,
                             uint32_t channels, uint32_t sample_bytes, const png_palette *palette, uint8_t *pixels) {
    static const uint8_t iend[12] = {0, 0, 0, 0, 'I', 'E', 'N', 'D', 0xAE, 0x42, 0x60, 0x82};
    const size_t pixel = (size_t)channels * sample_bytes, out_row = width * pixel;
    const size_t row = palette ? ((size_t)width * (size_t)palette->depth + 7) / 8 : out_row;
    const size_t filtered = (row + 1) * height;
    if (idat < 8 || filtered > DIRECT_MAX_BYTES) return 0;
    size_t at = idat - 8, total = 0, chunks = 0;
    while (size - at >= 12 && memcmp(data + at + 4, "IDAT", 4) == 0) {
        const uint32_t length = be32(data + at);
        if (length >= UINT32_C(1) << 31 || size - at - 12 < length ||
            libdeflate_crc32(0, data + at + 4, (size_t)length + 4) != be32(data + at + 8 + length))
            return 0;
        total += length;
        ++chunks;
        at += 12 + (size_t)length;
    }
    if (size - at < sizeof iend || memcmp(data + at, iend, sizeof iend) != 0) return 0;
    uint8_t *joined = NULL;
    const uint8_t *stream = data + idat;
    if (chunks > 1) {
        joined = (uint8_t *)malloc(total ? total : 1);
        if (!joined) return 0;
        size_t done = 0;
        for (size_t chunk = idat - 8; done < total;) {
            const size_t length = be32(data + chunk);
            memcpy(joined + done, data + chunk + 8, length);
            done += length;
            chunk += 12 + length;
        }
        stream = joined;
    }
    int ok = 0;
    uint8_t *const rows = (uint8_t *)malloc(row + filtered);
    struct libdeflate_decompressor *const inflater = libdeflate_alloc_decompressor();
    size_t used = 0;
    if (rows && inflater && total >= 2 && stream[0] >> 4 == 7 &&
        libdeflate_zlib_decompress_ex(inflater, stream, total, rows + row, filtered, &used, NULL) ==
            LIBDEFLATE_SUCCESS &&
        used == total) {
        memset(rows, 0, row);
        ok = 1;
        for (uint32_t y = 0; ok && y < height; ++y) {
            uint8_t *const line = rows + row + (size_t)y * (row + 1);
            const uint8_t *const prior = y ? line + 1 - (row + 1) : rows;
            ok = png_unfilter(line[0], line + 1, prior, row, palette ? 1 : pixel);
            if (!ok) break;
            const uint8_t *source = line + 1;
            uint8_t *target = pixels + (size_t)y * out_row;
            if (palette) {
                const uint32_t depth = (uint32_t)palette->depth, mask = (1u << depth) - 1;
                for (uint32_t x = 0; x < width; ++x, target += channels) {
                    const uint32_t bit = x * depth;
                    const uint32_t index = (uint32_t)(source[bit >> 3] >> (8 - depth - (bit & 7))) & mask;
                    if ((int)index >= palette->color_count) {
                        ok = 0;
                        break;
                    }
                    target[0] = palette->colors[index].blue;
                    target[1] = palette->colors[index].green;
                    target[2] = palette->colors[index].red;
                    if (channels == 4) target[3] = (int)index < palette->alpha_count ? palette->alpha[index] : 255;
                }
            } else if (sample_bytes == 2) {
                for (uint32_t x = 0; x < width; ++x, source += pixel, target += pixel) {
                    for (uint32_t k = 0; k < channels; ++k) {
                        const uint32_t from = channels > 1 && k < 3 ? 2 - k : k;
                        target[2 * k] = source[2 * from + 1];
                        target[2 * k + 1] = source[2 * from];
                    }
                }
            } else if (channels == 1) {
                memcpy(target, source, row);
            } else {
                for (uint32_t x = 0; x < width; ++x, source += channels, target += channels) {
                    target[0] = source[2];
                    target[1] = source[1];
                    target[2] = source[0];
                    if (channels == 4) target[3] = source[3];
                }
            }
        }
    }
    libdeflate_free_decompressor(inflater);
    free(rows);
    free(joined);
    return ok;
}

cn_codec_status cn_png_decode(const uint8_t *data, size_t size, cn_codec_allocate allocate, void *context) {
    png_walk walk;
    if (size < 8 || !opencv_png_walk(data, size, &walk)) return CN_CODEC_FALLBACK;
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, png_fail, png_quiet);
    if (!png) return CN_CODEC_FALLBACK;
    png_infop info = png_create_info_struct(png);
    png_source source = {data, size, 0};
    png_bytep *volatile rows = NULL;
    volatile cn_codec_status status = CN_CODEC_FALLBACK;
    if (!info || setjmp(png_jmpbuf(png))) {
        png_free(png, rows);
        png_destroy_read_struct(&png, &info, NULL);
        return status;
    }
    png_set_read_fn(png, &source, png_read_buffer);
    png_read_info(png, info);
    png_uint_32 width, height;
    int depth, colour, interlace;
    png_get_IHDR(png, info, &width, &height, &depth, &colour, &interlace, NULL, NULL);
    if (walk.fctl && ((uint64_t)walk.x0 + walk.w0 > width || (uint64_t)walk.y0 + walk.h0 > height)) png_longjmp(png, 1);
    if (!width || !height || width > (UINT32_C(1) << 20) || height > (UINT32_C(1) << 20) ||
        (uint64_t)width * height > OPENCV_MAX_PIXELS)
        png_longjmp(png, 1);
    /* readHeader's type. An indexed image's colours and tRNS values are copied for the direct path now:
       png_read_update_info's expansion clears the info's tRNS count. */
    uint32_t channels = 1;
    png_color colors[PNG_MAX_PALETTE_LENGTH];
    png_byte alpha[PNG_MAX_PALETTE_LENGTH];
    png_palette palette = {colors, alpha, 0, 0, depth};
    if (colour == PNG_COLOR_TYPE_RGB || colour == PNG_COLOR_TYPE_PALETTE) {
        png_bytep trans;
        int count = 0;
        png_color_16p values;
        png_get_tRNS(png, info, &trans, &count, &values);
        channels = count > 0 ? 4 : 3;
        png_colorp plte;
        if (colour == PNG_COLOR_TYPE_PALETTE && png_get_PLTE(png, info, &plte, &palette.color_count) &&
            palette.color_count <= PNG_MAX_PALETTE_LENGTH && count <= PNG_MAX_PALETTE_LENGTH) {
            memcpy(colors, plte, (size_t)palette.color_count * sizeof *plte);
            if (count > 0) memcpy(alpha, trans, (size_t)count);
            palette.alpha_count = count;
        } else {
            palette.color_count = 0;
        }
    } else if (colour == PNG_COLOR_TYPE_GRAY_ALPHA || colour == PNG_COLOR_TYPE_RGB_ALPHA) {
        channels = 4;
    }
    const uint32_t sample_bytes = depth == 16 ? 2 : 1;
    /* readData's transforms */
    png_set_swap(png);
    if (channels < 4) png_set_strip_alpha(png);
    else png_set_tRNS_to_alpha(png);
    if (colour == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if ((colour & PNG_COLOR_MASK_COLOR) == 0 && depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if ((colour & PNG_COLOR_MASK_COLOR) && channels > 1) png_set_bgr(png);
    else if (channels > 1) png_set_gray_to_rgb(png);
    else png_set_rgb_to_gray(png, 1, 0.299, 0.587);
    png_set_interlace_handling(png);
    png_read_update_info(png, info);
    const size_t stride = (size_t)width * channels * sample_bytes;
    if (png_get_rowbytes(png, info) != stride) png_longjmp(png, 1);
    uint8_t *const pixels = allocate(context, height, width, channels, sample_bytes);
    if (!pixels) png_longjmp(png, 1);
    /* The direct path's files: indexed (any depth, with or without tRNS), or 8- or 16-bit grey, RGB or RGBA without
       tRNS; none interlaced. */
    int direct = interlace == PNG_INTERLACE_NONE;
    if (colour == PNG_COLOR_TYPE_PALETTE) {
        direct = direct && palette.color_count > 0;
    } else {
        direct = direct && (depth == 8 || depth == 16) && colour != PNG_COLOR_TYPE_GRAY_ALPHA &&
                 !png_get_valid(png, info, PNG_INFO_tRNS);
    }
    if (!direct || !png_decode_direct(data, size, source.at, width, height, channels, sample_bytes,
                                      colour == PNG_COLOR_TYPE_PALETTE ? &palette : NULL, pixels)) {
        rows = (png_bytep *)png_malloc(png, sizeof(png_bytep) * height);
        for (uint32_t y = 0; y < height; ++y) rows[y] = pixels + (size_t)y * stride;
        png_read_image(png, rows);
        png_read_end(png, info);
    }
    status = CN_CODEC_OK;
    png_free(png, rows);
    png_destroy_read_struct(&png, &info, NULL);
    return status;
}

/* Encoding: OpenCV 5.0's PngEncoder::write without parameters, into memory (writeDataToBuf). */
static void png_write_buffer(png_structp png, png_bytep bytes, size_t count) {
    cn_codec_output *out = (cn_codec_output *)png_get_io_ptr(png);
    cn_codec_output_append(out, bytes, count);
    if (out->failed) png_error(png, "out of memory");
}

static void png_flush_none(png_structp png) { (void)png; }

cn_codec_status cn_png_encode(const cn_codec_image *image, cn_codec_output *out) {
    const uint32_t channels = image->channels;
    if ((image->sample_bytes != 1 && image->sample_bytes != 2) || (channels != 1 && channels != 3 && channels != 4))
        return CN_CODEC_FALLBACK;
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, png_fail, png_quiet);
    if (!png) return CN_CODEC_FALLBACK;
    png_infop info = png_create_info_struct(png);
    png_bytep *volatile rows = NULL;
    volatile cn_codec_status status = CN_CODEC_FALLBACK;
    if (!info || setjmp(png_jmpbuf(png))) {
        png_free(png, rows);
        png_destroy_write_struct(&png, &info);
        return status;
    }
    /* The filtered rows plus framing: level 1 with Z_RLE rarely gives more. */
    cn_codec_output_reserve(out, ((size_t)image->width * channels * image->sample_bytes + 1) * image->height + 4096);
    png_set_write_fn(png, out, png_write_buffer, png_flush_none);
    png_set_filter(png, PNG_FILTER_TYPE_BASE, PNG_FILTER_SUB);
    png_set_compression_level(png, 1);   /* Z_BEST_SPEED */
    png_set_compression_strategy(png, 3); /* Z_RLE */
    png_set_IHDR(png, info, image->width, image->height, image->sample_bytes == 1 ? 8 : 16,
                 channels == 1 ? PNG_COLOR_TYPE_GRAY : channels == 3 ? PNG_COLOR_TYPE_RGB : PNG_COLOR_TYPE_RGBA,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    png_set_bgr(png);
    png_set_swap(png);
    rows = (png_bytep *)png_malloc(png, sizeof(png_bytep) * image->height);
    for (uint32_t y = 0; y < image->height; ++y) rows[y] = (png_bytep)(image->data + (size_t)y * image->stride);
    png_write_image(png, rows);
    png_write_end(png, info);
    status = CN_CODEC_OK;
    png_free(png, rows);
    png_destroy_write_struct(&png, &info);
    return status;
}
