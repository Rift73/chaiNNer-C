/* PNG through libpng 1.6.57 with OpenCV 5.0's PngDecoder (grfmt_png.cpp) under IMREAD_UNCHANGED, for still images:
 * its own chunk walk to the first IDAT (readHeader), then libpng from the first byte with readData's transforms
 * (GPL-3.0). An animated PNG (acTL with more than one frame) takes OpenCV's other path and is not ported. */
#include "image_codecs.h"

#include <setjmp.h>
#include <string.h>

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

#define OPENCV_MAX_PIXELS (UINT64_C(1) << 30)

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
    int depth, colour;
    png_get_IHDR(png, info, &width, &height, &depth, &colour, NULL, NULL, NULL);
    if (walk.fctl && ((uint64_t)walk.x0 + walk.w0 > width || (uint64_t)walk.y0 + walk.h0 > height)) png_longjmp(png, 1);
    if (!width || !height || width > (UINT32_C(1) << 20) || height > (UINT32_C(1) << 20) ||
        (uint64_t)width * height > OPENCV_MAX_PIXELS)
        png_longjmp(png, 1);
    /* readHeader's type */
    uint32_t channels = 1;
    if (colour == PNG_COLOR_TYPE_RGB || colour == PNG_COLOR_TYPE_PALETTE) {
        png_bytep trans;
        int count = 0;
        png_color_16p values;
        png_get_tRNS(png, info, &trans, &count, &values);
        channels = count > 0 ? 4 : 3;
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
    rows = (png_bytep *)png_malloc(png, sizeof(png_bytep) * height);
    for (uint32_t y = 0; y < height; ++y) rows[y] = pixels + (size_t)y * stride;
    png_read_image(png, rows);
    png_read_end(png, info);
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
