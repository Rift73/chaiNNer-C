/* libFuzzer harness over the PNG decoder's inflate paths (GPL-3.0; README.md here). The input's first byte picks the
 * mode, the rest is the file:
 *
 * Even: a zlib stream for cn_png_inflate_parallel, checked against serial libdeflate. A stream with a wrong Adler-32
 * never decodes, so the harness writes the right one first when the DEFLATE data ends where the stream does. Wherever
 * the parallel path says yes, the serial one must accept the same stream at the same size and give the same bytes;
 * a size off by one and a damaged Adler-32 must be declined.
 *
 * Odd: a PNG file for cn_png_decode, with every chunk's CRC repaired first, since one wrong CRC ends a file before any
 * inflate (the IDAT walk of png_decode_direct, libpng's read of a critical chunk). The result is checked against libpng
 * alone with the same transforms: the same success and the same bytes, except for the codes zlib refuses and libdeflate
 * reads (png_codec.c's png_decode_direct), which must then inflate serially too.
 *
 * Output buffers have exactly the size asked for, so that AddressSanitizer sees any write past them. */
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "image_codecs.h"
#include "libdeflate.h"
#include "png.h"

/* Larger images and outputs are skipped, to keep each run short and small. */
#define MAX_BYTES ((size_t)4 << 20)

static void fail(const char *what) {
    fprintf(stderr, "fuzz_png_inflate: %s\n", what);
    abort();
}

static uint32_t be32(const uint8_t *p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }

static void put_be32(uint8_t *p, uint32_t value) {
    p[0] = (uint8_t)(value >> 24);
    p[1] = (uint8_t)(value >> 16);
    p[2] = (uint8_t)(value >> 8);
    p[3] = (uint8_t)value;
}

/* The parallel path on a copy of the stream, into a buffer of exactly out_size; its result, and in *out_ret the output
 * (NULL when it declined). */
static int parallel(const uint8_t *stream, size_t size, size_t out_size, uint8_t **out_ret) {
    uint8_t *const out = (uint8_t *)malloc(out_size ? out_size : 1);
    if (!out) fail("out of memory");
    const int taken = cn_png_inflate_parallel(stream, size, out, out_size);
    if (out_ret && taken) *out_ret = out;
    else free(out);
    return taken;
}

static void check_stream(const uint8_t *data, size_t size) {
    if (size < 6) return;
    uint8_t *const stream = (uint8_t *)malloc(size);
    uint8_t *const expected = (uint8_t *)malloc(MAX_BYTES);
    struct libdeflate_decompressor *const inflater = libdeflate_alloc_decompressor();
    if (!stream || !expected || !inflater) fail("out of memory");
    memcpy(stream, data, size);
    size_t used = 0, length = 0;
    /* Where the DEFLATE data ends four bytes before the input, the Adler-32 is made right. */
    if (libdeflate_deflate_decompress_ex(inflater, stream + 2, size - 2, expected, MAX_BYTES, &used, &length) ==
            LIBDEFLATE_SUCCESS &&
        used == size - 6)
        put_be32(stream + size - 4, libdeflate_adler32(1, expected, length));
    size_t zlib_used = 0, zlib_length = 0;
    const int serial = libdeflate_zlib_decompress_ex(inflater, stream, size, expected, MAX_BYTES, &zlib_used,
                                                     &zlib_length) == LIBDEFLATE_SUCCESS &&
                       zlib_used == size;
    /* An invalid stream gets a size in the parallel path's reach: at most five times the input. */
    const size_t out_size = serial ? zlib_length : size * (1 + data[size / 2] % 5);
    uint8_t *out = NULL;
    if (out_size <= MAX_BYTES && parallel(stream, size, out_size, &out)) {
        if (!serial) fail("the parallel path took a stream the serial one refuses");
        if (memcmp(out, expected, out_size) != 0) fail("the parallel path gave other bytes");
        free(out);
    }
    if (serial && zlib_length) {
        if (parallel(stream, size, zlib_length - 1, NULL)) fail("the parallel path took a size one too small");
        if (parallel(stream, size, zlib_length / 2, NULL)) fail("the parallel path took half the size");
        if (zlib_length < MAX_BYTES && parallel(stream, size, zlib_length + 1, NULL))
            fail("the parallel path took a size one too large");
        stream[size - 1] ^= 1;
        if (parallel(stream, size, zlib_length, NULL)) fail("the parallel path took a wrong Adler-32");
    }
    libdeflate_free_decompressor(inflater);
    free(expected);
    free(stream);
}

typedef struct image {
    uint8_t *pixels;
    size_t bytes;
} image;

static uint8_t *image_allocate(void *context, uint32_t height, uint32_t width, uint32_t channels,
                               uint32_t sample_bytes) {
    image *const target = (image *)context;
    const uint64_t bytes = (uint64_t)height * width * channels * sample_bytes;
    if (bytes > MAX_BYTES) return NULL;
    free(target->pixels);
    target->bytes = (size_t)bytes;
    target->pixels = (uint8_t *)malloc(target->bytes);
    if (!target->pixels) fail("out of memory");
    return target->pixels;
}

typedef struct png_input {
    const uint8_t *data;
    size_t size, at;
} png_input;

static void reference_read(png_structp png, png_bytep out, size_t count) {
    png_input *const input = (png_input *)png_get_io_ptr(png);
    if (input->size - input->at < count) png_error(png, "the input ended");
    memcpy(out, input->data + input->at, count);
    input->at += count;
}

static void reference_error(png_structp png, png_const_charp message) {
    (void)message;
    png_longjmp(png, 1);
}

static void reference_warning(png_structp png, png_const_charp message) {
    (void)png;
    (void)message;
}

/* cn_png_decode's libpng path alone, with its transforms. 1 with the image, or 0 where libpng fails. */
static int reference_decode(const uint8_t *data, size_t size, image *result) {
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, reference_error, reference_warning);
    if (!png) fail("out of memory");
    png_infop info = png_create_info_struct(png);
    png_input input = {data, size, 0};
    png_bytep *volatile rows = NULL;
    volatile int ok = 0;
    if (!info) fail("out of memory");
    if (setjmp(png_jmpbuf(png))) {
        png_free(png, rows);
        png_destroy_read_struct(&png, &info, NULL);
        return ok;
    }
    png_set_read_fn(png, &input, reference_read);
    png_read_info(png, info);
    png_uint_32 width, height;
    int depth, colour, interlace;
    png_get_IHDR(png, info, &width, &height, &depth, &colour, &interlace, NULL, NULL);
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
    if (png_get_rowbytes(png, info) != stride || !image_allocate(result, height, width, channels, sample_bytes))
        png_longjmp(png, 1);
    rows = (png_bytep *)png_malloc(png, sizeof(png_bytep) * height);
    for (uint32_t y = 0; y < height; ++y) rows[y] = result->pixels + (size_t)y * stride;
    png_read_image(png, rows);
    png_read_end(png, info);
    ok = 1;
    png_longjmp(png, 1);
}

/* Whether the IDAT chunks of a file with good CRCs, joined, inflate serially. */
static int stream_inflates(const uint8_t *file, size_t size) {
    uint8_t *const joined = (uint8_t *)malloc(size);
    uint8_t *const out = (uint8_t *)malloc(MAX_BYTES);
    struct libdeflate_decompressor *const inflater = libdeflate_alloc_decompressor();
    if (!joined || !out || !inflater) fail("out of memory");
    size_t total = 0, used = 0, length = 0;
    for (size_t at = 8; size - at >= 12;) {
        const size_t chunk = be32(file + at);
        if (size - at - 12 < chunk) break;
        if (memcmp(file + at + 4, "IDAT", 4) == 0) {
            memcpy(joined + total, file + at + 8, chunk);
            total += chunk;
        }
        at += 12 + chunk;
    }
    const int ok = libdeflate_zlib_decompress_ex(inflater, joined, total, out, MAX_BYTES, &used, &length) ==
                       LIBDEFLATE_SUCCESS &&
                   used == total;
    libdeflate_free_decompressor(inflater);
    free(out);
    free(joined);
    return ok;
}

static void check_file(const uint8_t *data, size_t size) {
    if (size < 8) return;
    uint8_t *const file = (uint8_t *)malloc(size);
    if (!file) fail("out of memory");
    memcpy(file, data, size);
    for (size_t at = 8; size - at >= 12;) {
        const size_t chunk = be32(file + at);
        if (size - at - 12 < chunk) break;
        put_be32(file + at + 8 + chunk, libdeflate_crc32(0, file + at + 4, chunk + 4));
        at += 12 + chunk;
    }
    image ours = {NULL, 0}, theirs = {NULL, 0};
    const int decoded = cn_png_decode(file, size, image_allocate, &ours) == CN_CODEC_OK;
    if (decoded) {
        if (reference_decode(file, size, &theirs)) {
            if (ours.bytes != theirs.bytes || memcmp(ours.pixels, theirs.pixels, ours.bytes) != 0)
                fail("the decoder and libpng gave other bytes");
        } else if (!stream_inflates(file, size)) {
            fail("the decoder took a file libpng and the serial inflate refuse");
        }
    }
    free(theirs.pixels);
    free(ours.pixels);
    free(file);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (!size) return 0;
    if (data[0] & 1) check_file(data + 1, size - 1);
    else check_stream(data + 1, size - 1);
    return 0;
}
