/* chaiNNer-C's default PNG encoder (decision 2026-10-21; GPL-3.0): every row filtered by libpng's MinSum rule
 * (png_encode_shared.h), the rows cut into bands that libdeflate 1.26 (vendored, third_party/libdeflate) compresses at
 * level 6 in parallel on the native pool, each band's DEFLATE data a piece of one zlib stream: a band ends with an
 * empty non-final stored block (the vendored libdeflate_deflate_compress_nonfinal patch), the last band ends the
 * stream, and the stream's Adler-32 is the bands' Adler-32 values combined. Each band is one IDAT chunk. The bytes
 * depend on the image alone: the band count on its size, never on the CPU count, and every ISA level gives the same
 * filtered rows. libdeflate's own SIMD (Adler-32, CRC-32, match finder setup) dispatches on its CPUID check and gives
 * the same values on every CPU. */
#include "png_encode_shared.h"
#include "isa.h"
#include "parallel.h"
#include "run_kernels.h"
#include <libdeflate.h>
#include <stdlib.h>
#include <string.h>

#define PNG_LEVEL 6
/* Bands: at most PNG_MAX_BANDS, each with at least PNG_MIN_BAND_BYTES of filtered rows and PNG_MIN_BAND_ROWS rows,
 * so a small image takes the bands it can fill (one below 32 KiB), and at most PNG_MAX_BAND_BYTES each (with a row's
 * bytes added by the split), so a band's data stays far below a chunk's limit of 2^31 - 1 bytes. 16 KiB (measured
 * 2026-10-21 on 16 threads): larger minimums leave images of 128^2 to 768^2 pixels with fewer bands than the pool can
 * run (up to 3.5x slower at 256 KiB), smaller ones gain nothing more (the pool's dispatch and libdeflate's set-up per
 * band dominate); each band costs some compression, about 1 % at 512^2 RGB against 256 KiB bands. */
#define PNG_MAX_BANDS 32
#define PNG_MIN_BAND_BYTES ((size_t)16 * 1024)
#define PNG_MIN_BAND_ROWS 8
#define PNG_MAX_BAND_BYTES ((size_t)1 << 29)
_Static_assert(CN_PNG_MAX_ROW_SAMPLES < PNG_MAX_BAND_BYTES, "a row (with its filter byte) fits in a band");

/* One band's IDAT chunk: its length and type, its data (band 0's begins with the zlib header, the last band's ends
 * with the stream's Adler-32, written after the join), its CRC. */
typedef struct png_band {
    uint8_t *chunk;
    size_t size;      /* the chunk's bytes */
    size_t offset;    /* the chunk's place in the file */
    size_t filtered;  /* the band's filtered bytes, the data its DEFLATE piece holds */
    uint32_t adler;   /* Adler-32 of the band's filtered bytes */
    uint32_t crc;     /* CRC-32 of the chunk's type and data, but for the last band's Adler-32 */
    cn_status status;
} png_band;

typedef struct png_job {
    const uint8_t *source;
    ptrdiff_t row_stride, pixel_stride, channel_stride;
    size_t height, width, channels, row_bytes, bands;
    int bgr;    /* channels 0-2 are blue, green, red, written as red, green, blue */
    int packed; /* pixel_stride == channels and channel_stride == 1 (for more than one channel) */
    int direct; /* packed in the PNG's channel order: the filters read the source rows */
    cn_isa_level isa;
    png_band *band;
} png_job;

static void put32(uint8_t *p, uint32_t value)
{
    p[0] = (uint8_t)(value >> 24);
    p[1] = (uint8_t)(value >> 16);
    p[2] = (uint8_t)(value >> 8);
    p[3] = (uint8_t)value;
}

/* The Adler-32 of A followed by B from adler(A), adler(B) and B's length (RFC 1950's sums s1 and s2 mod 65521):
 * s1(AB) = s1(A) + s1(B) - 1 and s2(AB) = s2(A) + s2(B) + length(B) * (s1(A) - 1). */
static uint32_t adler32_combine(uint32_t a, uint32_t b, uint64_t b_length)
{
    const uint64_t base = 65521, a1 = a & 0xFFFF, b1 = b & 0xFFFF;
    const uint64_t s1 = (a1 + b1 + base - 1) % base;
    const uint64_t s2 = ((a >> 16) + (b >> 16) + (b_length % base) * ((a1 + base - 1) % base)) % base;
    return (uint32_t)(s2 << 16 | s1);
}

static size_t band_count(size_t height, size_t filtered)
{
    size_t bands = filtered / PNG_MIN_BAND_BYTES;
    if (bands > PNG_MAX_BANDS) bands = PNG_MAX_BANDS;
    if (bands > height / PNG_MIN_BAND_ROWS) bands = height / PNG_MIN_BAND_ROWS;
    /* At most height, since a row holds at most PNG_MAX_BAND_BYTES. */
    const size_t needed = filtered / PNG_MAX_BAND_BYTES + (filtered % PNG_MAX_BAND_BYTES != 0);
    if (bands < needed) bands = needed;
    return bands ? bands : 1;
}

/* Row y as packed bytes in the PNG's channel order: the source row itself, or its copy in buffer. */
static const uint8_t *packed_row(const png_job *job, size_t y, uint8_t *buffer)
{
    const uint8_t *source = job->source + (ptrdiff_t)y * job->row_stride;
    if (job->direct) return source;
    if (job->packed) {
#if defined(_MSC_VER) && defined(_M_X64)
        if (job->isa >= CN_ISA_AVX2) {
            cn_png_swap_row_avx2(source, job->width, job->channels, buffer);
            return buffer;
        }
#endif
        png_swap_pixels(source, 0, job->width, job->channels, buffer);
        return buffer;
    }
    uint8_t *out = buffer;
    for (size_t x = 0; x < job->width; ++x) {
        const uint8_t *pixel = source + (ptrdiff_t)x * job->pixel_stride;
        for (size_t k = 0; k < job->channels; ++k) {
            const size_t from = job->bgr && k < 3 ? 2 - k : k;
            *out++ = pixel[(ptrdiff_t)from * job->channel_stride];
        }
    }
    return buffer;
}

static void filter_row(cn_isa_level isa, const uint8_t *row, const uint8_t *up, size_t length, size_t bpp,
                       uint8_t *out)
{
#if defined(_MSC_VER) && defined(_M_X64)
    if (isa >= CN_ISA_AVX2) {
        cn_png_filter_row_avx2(row, up, length, bpp, out);
        return;
    }
#else
    (void)isa;
#endif
    uint64_t sums[5] = {0, 0, 0, 0, 0};
    png_costs(row, up, 0, length, bpp, sums);
    const int type = png_choice(sums);
    out[0] = (uint8_t)type;
    png_filter_bytes(type, row, up, 0, length, bpp, out + 1);
}

/* Band `index`: rows [height * index / bands, height * (index + 1) / bands), filtered, then compressed into its
 * chunk. */
static cn_status encode_band(const png_job *job, size_t index, png_band *band)
{
    const size_t y0 = job->height * index / job->bands, y1 = job->height * (index + 1) / job->bands;
    const size_t n = job->row_bytes, length = (y1 - y0) * (n + 1);
    const int first = index == 0, last = index + 1 == job->bands;
    /* The filtered rows, then two packed rows (unless the source rows are read in place), then the zeros above the
     * image's first row. */
    const size_t rows = job->direct ? 0 : 2 * n, zeros = first ? n : 0;
    uint8_t *work = malloc(length + rows + zeros);
    if (!work) return CN_ALLOCATION_FAILED;
    uint8_t *filtered = work, *buffers[2] = {work + length, work + length + n};
    const uint8_t *up;
    if (first) {
        memset(work + length + rows, 0, zeros);
        up = work + length + rows;
    } else {
        up = packed_row(job, y0 - 1, buffers[1]);
    }
    for (size_t y = y0; y < y1; ++y) {
        uint8_t *buffer = buffers[(y - y0) & 1];
        const uint8_t *row = packed_row(job, y, buffer);
        filter_row(job->isa, row, up, n, job->channels, filtered + (y - y0) * (n + 1));
        up = row;
    }
    band->filtered = length;
    band->adler = libdeflate_adler32(1, filtered, length);

    struct libdeflate_compressor *compressor = libdeflate_alloc_compressor(PNG_LEVEL);
    if (!compressor) {
        free(work);
        return CN_ALLOCATION_FAILED;
    }
    const size_t header = first ? 2 : 0, trailer = last ? 4 : 0;
    const size_t bound = libdeflate_deflate_compress_bound(compressor, length) + 5;
    uint8_t *chunk = malloc(8 + header + bound + trailer + 4);
    size_t written = 0;
    if (chunk) {
        uint8_t *data = chunk + 8 + header;
        written = last ? libdeflate_deflate_compress(compressor, filtered, length, data, bound)
                       : libdeflate_deflate_compress_nonfinal(compressor, filtered, length, data, bound);
    }
    libdeflate_free_compressor(compressor);
    free(work);
    if (!chunk) return CN_ALLOCATION_FAILED;
    if (!written) {
        free(chunk); /* not expected: the bound holds every output */
        return CN_SIZE_OVERFLOW;
    }
    if (first) {
        chunk[8] = 0x78; /* zlib: DEFLATE with a 32 KiB window */
        chunk[9] = 0x9C; /* no dictionary, level hint 2 (libdeflate's for level 6), check bits */
    }
    const size_t data = header + written + trailer;
    put32(chunk, (uint32_t)data);
    memcpy(chunk + 4, "IDAT", 4);
    band->crc = libdeflate_crc32(0, chunk + 4, 4 + header + written);
    if (!last) put32(chunk + 8 + data, band->crc);
    band->chunk = chunk;
    band->size = 12 + data;
    return CN_OK;
}

static void encode_bands(void *context, size_t begin, size_t end)
{
    png_job *job = context;
    for (size_t i = begin; i < end; ++i) job->band[i].status = encode_band(job, i, &job->band[i]);
}

typedef struct png_copy {
    uint8_t *out;
    const png_band *band;
} png_copy;

/* The bands' chunks into the file, in parallel: the file's pages are first touched (and faulted in) here. */
static void copy_bands(void *context, size_t begin, size_t end)
{
    const png_copy *copy = context;
    for (size_t i = begin; i < end; ++i) memcpy(copy->out + copy->band[i].offset, copy->band[i].chunk, copy->band[i].size);
}

/* (count - 1) * |stride| within a quarter of PTRDIFF_MAX, so a sample's offset (three such terms) fits. */
static int span_fits(size_t count, ptrdiff_t stride)
{
    const size_t magnitude = stride < 0 ? (size_t)0 - (size_t)stride : (size_t)stride;
    return magnitude == 0 || count - 1 <= (size_t)(PTRDIFF_MAX / 4) / magnitude;
}

CN_EXPORT cn_status cn_png_encode_u8(const uint8_t *source, size_t height, size_t width, size_t channels,
    ptrdiff_t row_stride, ptrdiff_t pixel_stride, ptrdiff_t channel_stride, int bgr, cn_png_allocate allocate,
    void *context)
{
    if (!source || !allocate || channels < 1 || channels > 4 || height < 1 || width < 1 || height > CN_PNG_MAX_SIDE
        || width > CN_PNG_MAX_SIDE || width > CN_PNG_MAX_ROW_SAMPLES / channels)
        return CN_INVALID_ARGUMENT;
    if (!span_fits(height, row_stride) || !span_fits(width, pixel_stride) || !span_fits(channels, channel_stride))
        return CN_SIZE_OVERFLOW;
    const size_t row_bytes = width * channels, filtered = height * (row_bytes + 1);
    const size_t bands = band_count(height, filtered);
    png_band *band = calloc(bands, sizeof *band);
    if (!band) return CN_ALLOCATION_FAILED;
    const int packed = pixel_stride == (ptrdiff_t)channels && (channels == 1 || channel_stride == 1);
    png_job job = {source, row_stride, pixel_stride, channel_stride, height, width, channels, row_bytes, bands,
                   bgr && channels >= 3, packed, packed && !(bgr && channels >= 3), cn_isa_current(), band};
    cn_status status = cn_parallel_for(bands, 1, encode_bands, &job);
    for (size_t i = 0; i < bands && status == CN_OK; ++i) status = band[i].status;
    if (status == CN_OK) {
        uint32_t adler = band[0].adler;
        for (size_t i = 1; i < bands; ++i) adler = adler32_combine(adler, band[i].adler, band[i].filtered);
        png_band *last = &band[bands - 1];
        put32(last->chunk + last->size - 8, adler);
        put32(last->chunk + last->size - 4, libdeflate_crc32(last->crc, last->chunk + last->size - 8, 4));
        size_t total = 8 + 25;
        for (size_t i = 0; i < bands; ++i) {
            band[i].offset = total;
            total += band[i].size;
        }
        total += 12;
        uint8_t *out = allocate(context, total);
        if (!out) {
            status = CN_ALLOCATION_FAILED;
        } else {
            static const uint8_t signature[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
            static const uint8_t colour_types[4] = {0, 4, 2, 6}; /* grey, grey + alpha, RGB, RGBA */
            static const uint8_t end[12] = {0, 0, 0, 0, 'I', 'E', 'N', 'D', 0xAE, 0x42, 0x60, 0x82};
            memcpy(out, signature, 8);
            uint8_t *ihdr = out + 8;
            put32(ihdr, 13);
            memcpy(ihdr + 4, "IHDR", 4);
            put32(ihdr + 8, (uint32_t)width);
            put32(ihdr + 12, (uint32_t)height);
            ihdr[16] = 8; /* bit depth */
            ihdr[17] = colour_types[channels - 1];
            ihdr[18] = ihdr[19] = ihdr[20] = 0; /* deflate, adaptive filtering, no interlace */
            put32(ihdr + 21, libdeflate_crc32(0, ihdr + 4, 17));
            png_copy copy = {out, band};
            status = cn_parallel_for(bands, 1, copy_bands, &copy);
            memcpy(out + total - 12, end, 12);
        }
    }
    for (size_t i = 0; i < bands; ++i) free(band[i].chunk);
    free(band);
    return status;
}
