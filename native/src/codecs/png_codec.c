/* PNG through libpng 1.6.57 with OpenCV 5.0's PngDecoder (grfmt_png.cpp) under IMREAD_UNCHANGED, for still images:
 * its own chunk walk to the first IDAT (readHeader), then libpng from the first byte with readData's transforms
 * (GPL-3.0). An animated PNG (acTL with more than one frame) takes OpenCV's other path and is not ported.
 * The common files skip libpng's row loop and zlib: png_decode_direct. */
#include "image_codecs.h"

#include <immintrin.h>
#include <setjmp.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "adler32_combine.h"
#include "isa.h"
#include "libdeflate.h"
#include "parallel.h"
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

/* The direct path's largest filtered image (its one extra buffer, a quarter of the float32 image the node returns;
 * if it cannot be allocated libpng decodes the file): 500 megapixels of RGBA. */
#define DIRECT_MAX_BYTES (UINT64_C(1) << 31)
/* The pixels' bytes of a store chunk, at least: the pixels' pages are first touched there, and with fresh pages 64 KiB
 * chunks took 1.8 ms at 3840x2160 RGB on 16 threads against 1.1 ms for 512 KiB, one thread 3.9 (measured 2026-11-04).
 */
#define DIRECT_STORE_GRAIN ((size_t)512 * 1024)

static uint8_t png_paeth(int a, int b, int c) {
    const int pa = abs(b - c), pb = abs(a - c), pc = abs(a + b - 2 * c);
    return (uint8_t)(pa <= pb && pa <= pc ? a : pb <= pc ? b : c);
}

/* Sub (1), Average (3) or Paeth (4) on the bytes [from, to) of a row (from past its first pixel), each byte's left and
 * upper-left neighbours carried in locals instead of re-read from the bytes just written; bpp is a constant at each
 * call. */
static inline void png_unfilter_left(uint8_t filter, uint8_t *row, const uint8_t *prior, size_t from, size_t to,
                                     size_t bpp) {
    uint8_t left[8], upper_left[8];
    for (size_t k = 0; k < bpp; ++k) {
        left[k] = row[from - bpp + k];
        upper_left[k] = prior[from - bpp + k];
    }
    for (size_t i = from; i < to; i += bpp) {
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

#if CN_ISA_UNITS()
/* png_unfilter_avx2.c: Paeth on whole pixels from byte from on; returns the first byte it left to the scalar code. */
size_t cn_png_unpaeth_avx2(uint8_t *row, const uint8_t *prior, size_t from, size_t to, size_t bpp);
#endif

/* PNG's filters (method 0, types 0 to 4) undone in place on the bytes [from, to) of a row, whole pixels, the row's bytes
 * before them undone and prior the previous unfiltered row (zeros before the first). */
static void png_unfilter(uint8_t filter, uint8_t *row, const uint8_t *prior, size_t from, size_t to, size_t bpp,
                         cn_isa_level isa) {
    if (filter == 2) {
        for (size_t i = from; i < to; ++i) row[i] = (uint8_t)(row[i] + prior[i]);
    } else if (filter != 0) {
        if (from == 0) {
            /* The first pixel's left neighbours are 0: Average adds half the byte above, Paeth the byte above. */
            for (size_t k = 0; k < bpp && filter != 1; ++k)
                row[k] = (uint8_t)(row[k] + (filter == 3 ? prior[k] >> 1 : prior[k]));
            from = bpp;
        }
#if CN_ISA_UNITS()
        if (filter == 4 && isa >= CN_ISA_AVX2) from = cn_png_unpaeth_avx2(row, prior, from, to, bpp);
#else
        (void)isa;
#endif
        switch (bpp) {
        case 1: png_unfilter_left(filter, row, prior, from, to, 1); break;
        case 2: png_unfilter_left(filter, row, prior, from, to, 2); break;
        case 3: png_unfilter_left(filter, row, prior, from, to, 3); break;
        case 4: png_unfilter_left(filter, row, prior, from, to, 4); break;
        case 6: png_unfilter_left(filter, row, prior, from, to, 6); break;
        default: png_unfilter_left(filter, row, prior, from, to, 8); break;
        }
    }
}

/* A wave across the rows. A byte needs only its left, upper and upper-left neighbours, so bands of rows run at once when
 * each stays behind the band above it, block of columns by block. Bands of 16 rows keep most of a block's upper
 * neighbours in the worker's own cache; twice as many blocks as workers keep every worker running. Workers are half
 * the CPUs, up to 16: a 4K image gains nothing past 16, and with CPUs free a worker the system takes off one goes on
 * at once on another, while the wave waits for it. */
#define WAVE_BAND_ROWS 16
#define WAVE_MAX_BANDS 256
#define WAVE_MAX_WORKERS 16
#define WAVE_MIN_BLOCK_BYTES 128
#define WAVE_MIN_BYTES (512 * 1024)
/* About ten blocks' time. A worker that waits longer lets go of its band, so that one taken off its CPU while waiting
 * holds up nothing; only one inside a block does. With every CPU busy with other work, 4K decodes whose workers kept
 * their bands took up to 1.9 s, against 0.3 s with no wave; letting go gives 0.4 s. */
#define WAVE_SPINS 1024

typedef struct png_wave {
    uint8_t *rows; /* a zero row, then each row after its filter type */
    size_t row, height, bpp, band_rows, bands, block_bytes, blocks;
    cn_isa_level isa;
    /* Each band's blocks done and whether a worker holds it, a cache line apart from the next band's at any alignment. */
    struct {
        atomic_size_t blocks_done;
        atomic_int held;
        uint8_t apart[64];
    } band_state[WAVE_MAX_BANDS];
} png_wave;

static void png_unfilter_tile(const png_wave *wave, size_t y0, size_t y1, size_t from, size_t to) {
    const size_t stride = wave->row + 1;
    for (size_t y = y0; y < y1; ++y) {
        uint8_t *const line = wave->rows + wave->row + y * stride;
        png_unfilter(line[0], line + 1, y ? line + 1 - stride : wave->rows, from, to, wave->bpp, wave->isa);
    }
}

/* A held band's blocks, until it ends or the band above stays level with it for WAVE_SPINS. */
static void png_wave_band(png_wave *wave, size_t band) {
    const size_t y0 = band * wave->band_rows;
    const size_t y1 = wave->height - y0 < wave->band_rows ? wave->height : y0 + wave->band_rows;
    size_t block = atomic_load_explicit(&wave->band_state[band].blocks_done, memory_order_relaxed);
    for (int spins = 0; block < wave->blocks;) {
        if (band && atomic_load_explicit(&wave->band_state[band - 1].blocks_done, memory_order_acquire) <= block) {
            if (++spins == WAVE_SPINS) return;
            _mm_pause();
            continue;
        }
        spins = 0;
        png_unfilter_tile(wave, y0, y1, block * wave->block_bytes,
                          block + 1 == wave->blocks ? wave->row : (block + 1) * wave->block_bytes);
        atomic_store_explicit(&wave->band_state[band].blocks_done, ++block, memory_order_release);
    }
}

/* One worker: holds the first free band with a block ready, again and again until every band is done. */
static void png_wave_worker(void *context, size_t begin, size_t end) {
    png_wave *const wave = (png_wave *)context;
    (void)begin;
    (void)end;
    size_t first = 0;
    for (;;) {
        while (first < wave->bands &&
               atomic_load_explicit(&wave->band_state[first].blocks_done, memory_order_relaxed) == wave->blocks)
            ++first;
        if (first == wave->bands) return;
        for (size_t band = first; band < wave->bands; ++band) {
            const size_t done = atomic_load_explicit(&wave->band_state[band].blocks_done, memory_order_relaxed);
            if (done < wave->blocks &&
                (!band || atomic_load_explicit(&wave->band_state[band - 1].blocks_done, memory_order_relaxed) > done) &&
                !atomic_load_explicit(&wave->band_state[band].held, memory_order_relaxed) &&
                !atomic_exchange_explicit(&wave->band_state[band].held, 1, memory_order_acquire)) {
                png_wave_band(wave, band);
                atomic_store_explicit(&wave->band_state[band].held, 0, memory_order_release);
                break;
            }
            if (!done) break; /* the bands below it have nothing ready */
        }
        _mm_pause();
    }
}

/* Every row's filter undone in place: rows is a zero row of row bytes, then height rows, each after its filter type.
 * 0 when a filter type is above 4, which libpng rejects ("bad adaptive filter value"), with nothing undone, or when
 * the pool fails; libpng then decodes the file. */
static int png_unfilter_rows(uint8_t *rows, size_t row, size_t height, size_t bpp) {
    for (size_t y = 0; y < height; ++y)
        if (rows[row + y * (row + 1)] > 4) return 0;
    int isa[3];
    cn_isa_get(isa);
    png_wave wave;
    wave.rows = rows;
    wave.row = row;
    wave.height = height;
    wave.bpp = bpp;
    wave.isa = (cn_isa_level)isa[0];
    wave.band_rows = (height + WAVE_MAX_BANDS - 1) / WAVE_MAX_BANDS;
    if (wave.band_rows < WAVE_BAND_ROWS) wave.band_rows = WAVE_BAND_ROWS;
    wave.bands = (height + wave.band_rows - 1) / wave.band_rows;
    size_t workers = cn_parallel_capacity() / 2;
    if (workers > WAVE_MAX_WORKERS) workers = WAVE_MAX_WORKERS;
    if (workers > wave.bands) workers = wave.bands;
    wave.blocks = 2 * workers < row / WAVE_MIN_BLOCK_BYTES ? 2 * workers : row / WAVE_MIN_BLOCK_BYTES;
    if ((row + 1) * height < WAVE_MIN_BYTES || workers < 2 || wave.blocks < 2) {
        png_unfilter_tile(&wave, 0, height, 0, row);
        return 1;
    }
    wave.block_bytes = row / wave.blocks / bpp * bpp;
    for (size_t band = 0; band < wave.bands; ++band) {
        atomic_init(&wave.band_state[band].blocks_done, 0);
        atomic_init(&wave.band_state[band].held, 0);
    }
    return cn_parallel_for(workers, 1, png_wave_worker, &wave) == CN_OK;
}

/* An indexed image's expansion, readData's png_set_palette_to_rgb (with png_set_tRNS_to_alpha at 4 channels): its
 * PLTE colours, its tRNS alpha values (255 past them) and its bit depth (1, 2, 4 or 8). */
typedef struct png_palette {
    const png_color *colors;
    const png_byte *alpha;
    int color_count, alpha_count, depth;
} png_palette;

/* One IDAT chunk's piece of the zlib stream and the part of the filtered rows it must give. */
typedef struct png_piece {
    const uint8_t *data;
    size_t size;
    uint8_t *out;
    size_t bytes;
    uint32_t adler;
} png_piece;

typedef struct png_pieces {
    png_piece *piece;
    size_t count;
    atomic_int failed;
} png_pieces;

/* Pieces with one decompressor. The last piece ends the DEFLATE stream, at its last byte; the others are non-final. */
static void png_inflate_range(void *context, size_t begin, size_t end) {
    png_pieces *const job = (png_pieces *)context;
    struct libdeflate_decompressor *const inflater = libdeflate_alloc_decompressor();
    if (!inflater) atomic_store(&job->failed, 1);
    for (size_t i = begin; i < end && !atomic_load(&job->failed); ++i) {
        png_piece *const piece = &job->piece[i];
        size_t used = piece->size;
        const enum libdeflate_result result =
            i + 1 < job->count
                ? libdeflate_deflate_decompress_nonfinal(inflater, piece->data, piece->size, piece->out, piece->bytes,
                                                         NULL)
                : libdeflate_deflate_decompress_ex(inflater, piece->data, piece->size, piece->out, piece->bytes, &used,
                                                   NULL);
        if (result != LIBDEFLATE_SUCCESS || used != piece->size) atomic_store(&job->failed, 1);
        else piece->adler = libdeflate_adler32(1, piece->out, piece->bytes);
    }
    libdeflate_free_decompressor(inflater);
}

/* A zlib stream cut into independent pieces, one per IDAT chunk, as png_encode.c writes it (other encoders' full
 * flushes may too), inflated in parallel into the filtered rows, without joining the chunks: piece i must give rows
 * [height * i / chunks, height * (i + 1) / chunks) of line bytes each. The caller has seen every chunk but the last end
 * in an empty stored block's bytes, which only costs a wrong file a failed try. Why the rows are those of the whole
 * stream: piece 0 starts at a block boundary, after the zlib header. A piece's parse does not depend on the output
 * before it, and a match reaching before the piece's slot is an error, so what a piece gives alone it gives in the
 * whole stream; ending at its last byte on an empty non-final stored block, it leaves a decoder of the whole stream at
 * a block boundary at the next piece's first byte. By induction that decoder gives these rows, and it would accept the
 * stream: 1 is returned only with the zlib header libdeflate's wrapper takes, exactly the rows' bytes, and the Adler-32
 * in the last chunk's last four bytes. 0 for anything else, having written only the rows: the whole stream then
 * decides. */
static int png_inflate_pieces(const uint8_t *data, size_t idat, size_t chunks, size_t height, size_t line,
                              uint8_t *filtered) {
    const uint8_t *const header = data + idat;
    if (be32(data + idat - 8) < 6 || header[0] != 0x78 || (0x7800u | header[1]) % 31 != 0 || header[1] & 0x20) return 0;
    png_piece *const piece = (png_piece *)malloc(chunks * sizeof *piece);
    if (!piece) return 0;
    size_t at = idat - 8;
    for (size_t i = 0; i < chunks; ++i) {
        const size_t y0 = height * i / chunks, y1 = height * (i + 1) / chunks, length = be32(data + at);
        piece[i] = (png_piece){data + at + 8, length, filtered + y0 * line, (y1 - y0) * line, 0};
        at += 12 + length;
    }
    int ok = piece[chunks - 1].size >= 4;
    if (ok) {
        const uint32_t expected = be32(piece[chunks - 1].data + piece[chunks - 1].size - 4);
        piece[0].data += 2;
        piece[0].size -= 2;
        piece[chunks - 1].size -= 4;
        png_pieces job = {.piece = piece, .count = chunks};
        atomic_init(&job.failed, 0);
        ok = cn_parallel_for(chunks, 1, png_inflate_range, &job) == CN_OK && !atomic_load(&job.failed);
        if (ok) {
            uint32_t adler = piece[0].adler;
            for (size_t i = 1; i < chunks; ++i) adler = adler32_combine(adler, piece[i].adler, piece[i].bytes);
            ok = adler == expected;
        }
    }
    free(piece);
    return ok;
}

/* The unfiltered rows (each after its filter byte, line bytes apart) on their way to the pixels. */
typedef struct png_store {
    const uint8_t *lines;
    size_t line;
    uint8_t *pixels;
    size_t out_row;
    uint32_t width, channels, sample_bytes;
    const png_palette *palette;
    atomic_int failed;
} png_store;

/* Rows as readData's transforms give them: BGR(A) order (png_set_bgr), 16-bit samples little-endian (png_set_swap),
 * indices expanded to their colours; an index outside the palette fails the job. */
static void png_store_range(void *context, size_t begin, size_t end) {
    png_store *const job = (png_store *)context;
    const uint32_t width = job->width, channels = job->channels, sample_bytes = job->sample_bytes;
    const png_palette *const palette = job->palette;
    const size_t pixel = (size_t)channels * sample_bytes;
    for (size_t y = begin; y < end; ++y) {
        const uint8_t *source = job->lines + y * job->line + 1;
        uint8_t *target = job->pixels + y * job->out_row;
        if (palette) {
            const uint32_t depth = (uint32_t)palette->depth, mask = (1u << depth) - 1;
            for (uint32_t x = 0; x < width; ++x, target += channels) {
                const uint32_t bit = x * depth;
                const uint32_t index = (uint32_t)(source[bit >> 3] >> (8 - depth - (bit & 7))) & mask;
                if ((int)index >= palette->color_count) {
                    atomic_store(&job->failed, 1);
                    return;
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
            memcpy(target, source, job->out_row);
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

/* The rows of a non-interlaced image, without libpng's row loop: 8- or 16-bit grey, RGB or RGBA without tRNS, or
 * indexed (palette not NULL). libpng has read the chunks before the first IDAT (its data at idat); this path takes
 * the file when the IDAT chunks that follow have valid CRCs and lengths below 2^31 and an empty IEND with a valid
 * CRC comes right after them, their zlib stream declares the 32 KiB window (so zlib and libdeflate accept the same
 * distances) and inflates, Adler-32 included, to exactly the filtered rows, and every index is inside the palette.
 * The rows are unfiltered, then stored (png_store_range). For such a file libpng gives the same pixels and no error,
 * except for a crafted stream with codes zlib refuses and libdeflate reads (length symbols 286 and 287, distance
 * symbols 30 and 31, too many codes), which opens here where libpng fails; for every other file this returns 0 having
 * written only pixels, and libpng decodes from idat as before. */
static int png_decode_direct(const uint8_t *data, size_t size, size_t idat, uint32_t width, uint32_t height,
                             uint32_t channels, uint32_t sample_bytes, const png_palette *palette, uint8_t *pixels) {
    static const uint8_t iend[12] = {0, 0, 0, 0, 'I', 'E', 'N', 'D', 0xAE, 0x42, 0x60, 0x82};
    static const uint8_t stored_end[4] = {0, 0, 0xFF, 0xFF};
    const size_t pixel = (size_t)channels * sample_bytes, out_row = width * pixel;
    const size_t row = palette ? ((size_t)width * (size_t)palette->depth + 7) / 8 : out_row;
    const size_t filtered = (row + 1) * height;
    if (idat < 8 || filtered > DIRECT_MAX_BYTES) return 0;
    /* flushed: how many chunks, from the first on, end in an empty stored block's bytes. */
    size_t at = idat - 8, total = 0, chunks = 0, flushed = 0;
    while (size - at >= 12 && memcmp(data + at + 4, "IDAT", 4) == 0) {
        const uint32_t length = be32(data + at);
        if (length >= UINT32_C(1) << 31 || size - at - 12 < length ||
            libdeflate_crc32(0, data + at + 4, (size_t)length + 4) != be32(data + at + 8 + length))
            return 0;
        if (flushed == chunks && length >= 4 && memcmp(data + at + 8 + length - 4, stored_end, 4) == 0) ++flushed;
        total += length;
        ++chunks;
        at += 12 + (size_t)length;
    }
    if (size - at < sizeof iend || memcmp(data + at, iend, sizeof iend) != 0) return 0;
    uint8_t *const rows = (uint8_t *)malloc(row + filtered);
    if (!rows) return 0;
    uint8_t *const lines = rows + row;
    int ok = chunks >= 2 && chunks <= height && flushed + 1 >= chunks &&
             png_inflate_pieces(data, idat, chunks, height, row + 1, lines);
    if (!ok && total >= 2) {
        uint8_t *joined = NULL;
        const uint8_t *stream = data + idat;
        if (chunks > 1) {
            stream = joined = (uint8_t *)malloc(total);
            for (size_t chunk = idat - 8, done = 0; joined && done < total;) {
                const size_t length = be32(data + chunk);
                memcpy(joined + done, data + chunk + 8, length);
                done += length;
                chunk += 12 + length;
            }
        }
        struct libdeflate_decompressor *const inflater = libdeflate_alloc_decompressor();
        size_t used = 0;
        ok = stream && inflater && stream[0] >> 4 == 7 &&
             (cn_png_inflate_parallel(stream, total, lines, filtered) ||
              (libdeflate_zlib_decompress_ex(inflater, stream, total, lines, filtered, &used, NULL) ==
                   LIBDEFLATE_SUCCESS &&
               used == total));
        libdeflate_free_decompressor(inflater);
        free(joined);
    }
    if (ok) {
        memset(rows, 0, row);
        ok = png_unfilter_rows(rows, row, height, palette ? 1 : pixel);
    }
    if (ok) {
        png_store store = {.lines = lines, .line = row + 1, .pixels = pixels, .out_row = out_row, .width = width,
                           .channels = channels, .sample_bytes = sample_bytes, .palette = palette};
        atomic_init(&store.failed, 0);
        ok = cn_parallel_for(height, 1 + DIRECT_STORE_GRAIN / out_row, png_store_range, &store) == CN_OK &&
             !atomic_load(&store.failed);
    }
    free(rows);
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
