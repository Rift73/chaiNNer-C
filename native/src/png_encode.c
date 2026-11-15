/* chaiNNer-C's default PNG encoder (GPL-3.0): every row filtered by libpng's MinSum rule
 * (png_encode_shared.h), the rows cut into bands that libdeflate 1.26 (vendored, third_party/libdeflate) compresses at
 * level 6 in parallel on the native pool, each band's DEFLATE data a piece of one zlib stream: a band ends with an
 * empty non-final stored block (the vendored libdeflate_deflate_compress_nonfinal patch), the last band ends the
 * stream, and the stream's Adler-32 is the bands' Adler-32 values combined. Each band is one IDAT chunk. The bytes
 * depend on the image alone: the band count on its size, never on the CPU count, and every ISA level gives the same
 * filtered rows. libdeflate's own SIMD (Adler-32, CRC-32, match finder setup) dispatches on its CPUID check and gives
 * the same values on every CPU.
 *
 * That is effort 0 (cn_png_encode_u8). The higher efforts (cn_png_encode_u8_effort, png_efforts) trade time for size:
 * a higher libdeflate level; a filter rule that estimates the compressed size better than MinSum (each row's order-0
 * entropy, or a trial compression of the row after the two before it); and a history for each band, the 32 KiB of
 * filtered rows before it, which its matches may reach (libdeflate_deflate_compress_history, a second vendored patch),
 * with all rows filtered before any band compresses. At efforts 1 and 2 the near-optimal parser also fits its costs on
 * that history first; effort 2 compresses in at most 24 bands (each band restarts the parser's cost model, which costs
 * about 0.3 % of the file at 32 bands). Effort 3 writes one stream: its matches are found ahead, by finder tasks a few
 * chunks before the parse (libdeflate_find_matches, each chunk primed with the 32 KiB before it, which gives the same
 * matches as one search; png_ring), while libdeflate's near-optimal parser runs over the whole image on one thread
 * (libdeflate_deflate_compress_matches); its bytes are libdeflate level 12's on the filtered rows; above 6 MiB of
 * filtered rows it parses segments of about 3 MiB in parallel, each after a 256 KiB lead-in. Effort 4 is effort 3 with
 * one more step for an image of one segment: once the serial parse has fixed the blocks and their costs, every block
 * is parsed again on the pool from its costs changed at random (4 restarts of up to 30 %, seeded by the block, the
 * restart and the round), keeping its smallest parse; in each of 3 more rounds a block also starts from the costs its
 * predecessor's best parse came from, as the serial chain did; then neighbouring blocks that are smaller as one block
 * are merged (libdeflate_plan_blocks, _refine_block and _write_blocks). The bytes depend on the image alone, never on
 * the thread count. Above effort 0 the bands' data form one stream of IDAT chunks of 4 MiB. The trial rule's choice
 * depends on the rows before it in its filter band, so the rows are filtered in band_count's bands at every effort,
 * whatever bands compress. The entropy rule compares double
 * sums of log2 values: the same bytes wherever the C runtime's log2 is the same.
 *
 * Compressors and band buffers stay between calls in slots that an atomic exchange takes (cn_png_encode_release frees
 * them, at the end of each run), so a batch's files reuse warm memory where each band otherwise took the heap's lock and
 * the page faults of 2 MiB at 4K: a call finding no idle one allocates and never waits, and no thread owns any. They
 * never change the bytes: every kept byte is written before it is read, and a compression resets libdeflate's hash
 * tables, through which alone it reaches anything of an earlier one. */
#include "png_encode_shared.h"
#include "adler32_combine.h"
#include "isa.h"
#include "parallel.h"
#include "run_kernels.h"
#include <libdeflate.h>
#include <immintrin.h>
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#define PNG_LEVEL 6
/* The filter rules: libpng's MinSum (png_encode_shared.h), filter_row_entropy's, filter_row_trial's. */
enum { PNG_RULE_MINSUM, PNG_RULE_ENTROPY, PNG_RULE_TRIAL };
/* A band's history: none (it starts with an empty window); the 32 KiB of filtered rows before it as its window, with
 * the parser's costs fitted on it first (libdeflate_deflate_compress_history); one stream (PNG_HISTORY_STREAM). */
enum { PNG_HISTORY_NONE, PNG_HISTORY_WARM, PNG_HISTORY_STREAM, PNG_HISTORY_REFINED };
/* PNG_HISTORY_REFINED: PNG_HISTORY_STREAM, and an image of one segment has its blocks' parses refined on the pool
 * (libdeflate_refine_block): png_restarts restarts per block from costs changed by up to png_noise per mille, in
 * rounds (measured 2026-11-16 on a 1440x1080 frame: 4 restarts of 30 % in 4 rounds, then the merges, gave 0.23 % of
 * the file over effort 3 in about 0.8 s on 32 threads; more restarts in fewer rounds gained less). The rounds come from
 * the plan's size alone, never a clock, so the bytes are the same on any machine (refine_rounds). */
#define PNG_RESTARTS 4
#define PNG_NOISE 300
#define PNG_MAX_ROUNDS 5
/* PNG_HISTORY_STREAM's segments: about PNG_SEGMENT_BYTES of filtered rows each, each led in by up to PNG_LEAD_BYTES
 * (measured 2026-11-16: on 4K frames 8 segments with 256 KiB lead-ins matched one stream's size at a quarter of its
 * time; longer lead-ins gained nothing). */
#define PNG_SEGMENT_BYTES ((size_t)3 << 20)
#define PNG_LEAD_BYTES ((size_t)256 << 10)
/* The sizes in use: PNG_SEGMENT_BYTES and PNG_LEAD_BYTES but in tests (cn_png_encode_test_segments). */
static size_t png_segment_bytes = PNG_SEGMENT_BYTES, png_lead_bytes = PNG_LEAD_BYTES;
static unsigned png_restarts = PNG_RESTARTS, png_noise = PNG_NOISE, png_rounds = 0; /* 0: refine_rounds' */
/* Bands: at most PNG_MAX_BANDS, each with at least PNG_MIN_BAND_BYTES of filtered rows and PNG_MIN_BAND_ROWS rows,
 * so a small image takes the bands it can fill (one below 32 KiB), and at most PNG_MAX_BAND_BYTES each (with a row's
 * bytes added by the split), so a band's data stays far below a chunk's limit of 2^31 - 1 bytes. 16 KiB (measured
 * on 16 threads): larger minimums leave images of 128^2 to 768^2 pixels with fewer bands than the pool can
 * run (up to 3.5x slower at 256 KiB), smaller ones gain nothing more (the pool's dispatch and libdeflate's set-up per
 * band dominate); each band costs some compression, about 1 % at 512^2 RGB against 256 KiB bands. */
#define PNG_MAX_BANDS 32
#define PNG_MIN_BAND_BYTES ((size_t)16 * 1024)
#define PNG_MIN_BAND_ROWS 8
#define PNG_MAX_BAND_BYTES ((size_t)1 << 29)
/* Above effort 0 an IDAT chunk holds at most PNG_IDAT_BYTES: decoders such as OpenCV 4.11's warn above 8,000,000. */
#define PNG_IDAT_BYTES ((size_t)4 << 20)
_Static_assert(CN_PNG_MAX_ROW_SAMPLES < PNG_MAX_BAND_BYTES, "a row (with its filter byte) fits in a band");
/* Kept between calls: a compressor per band running at once; a buffer per chunk of the eight calls an item window
 * overlaps (item_window.py's largest automatic size) and per band running at once; no buffer above 8 MiB (a band of an
 * image of about 85 megapixels), so a larger image allocates as it did. A kept buffer takes an eighth more than asked,
 * which covers the bands' row counts and both kinds of buffer. */
#define PNG_KEPT_COMPRESSORS PNG_MAX_BANDS
#define PNG_KEPT_BUFFERS (9 * PNG_MAX_BANDS)
#define PNG_KEEP_BYTES ((size_t)8 << 20)

typedef struct png_buffer {
    size_t capacity;
    uint8_t data[];
} png_buffer;

static _Atomic(void *) kept_compressors[PNG_KEPT_COMPRESSORS], kept_buffers[PNG_KEPT_BUFFERS];

/* An idle block of slots, looking from `start` (a call's bands start apart); NULL when none is idle. */
static void *take_kept(_Atomic(void *) *slots, size_t count, size_t start)
{
    for (size_t i = 0; i < count; ++i) {
        _Atomic(void *) *slot = &slots[(start + i) % count];
        if (!atomic_load_explicit(slot, memory_order_relaxed)) continue;
        void *block = atomic_exchange(slot, NULL);
        if (block) return block;
    }
    return NULL;
}

/* Puts block into an empty slot; 0 when every slot holds one. */
static int give_kept(_Atomic(void *) *slots, size_t count, size_t start, void *block)
{
    for (size_t i = 0; i < count; ++i) {
        _Atomic(void *) *slot = &slots[(start + i) % count];
        void *empty = NULL;
        if (!atomic_load_explicit(slot, memory_order_relaxed) && atomic_compare_exchange_strong(slot, &empty, block))
            return 1;
    }
    return 0;
}

/* A buffer of at least `bytes`, not zeroed: an idle one that holds them, else a new one; NULL when memory runs out. */
static png_buffer *take_buffer(size_t start, size_t bytes)
{
    size_t capacity = bytes;
    if (bytes <= PNG_KEEP_BYTES) {
        png_buffer *kept = take_kept(kept_buffers, PNG_KEPT_BUFFERS, start);
        if (kept && kept->capacity >= bytes) return kept;
        free(kept);
        capacity += bytes / 8;
    }
    png_buffer *buffer = malloc(sizeof *buffer + capacity);
    if (buffer) buffer->capacity = capacity;
    return buffer;
}

static void give_buffer(size_t start, png_buffer *buffer)
{
    if (buffer->capacity > PNG_KEEP_BYTES || !give_kept(kept_buffers, PNG_KEPT_BUFFERS, start, buffer)) free(buffer);
}

CN_EXPORT void cn_png_encode_release(void)
{
    for (size_t i = 0; i < PNG_KEPT_COMPRESSORS; ++i)
        libdeflate_free_compressor(atomic_exchange(&kept_compressors[i], NULL));
    for (size_t i = 0; i < PNG_KEPT_BUFFERS; ++i) free(atomic_exchange(&kept_buffers[i], NULL));
}

/* One band's IDAT chunk: its length and type, its data (band 0's begins with the zlib header, the last band's ends
 * with the stream's Adler-32, written after the join), its CRC. */
typedef struct png_band {
    png_buffer *chunk;
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
    int level, rule, history;
    uint8_t *whole;            /* with a history: the image's filtered rows, filtered before any band compresses */
    size_t filters;            /* with a history: the bands the rows are filtered in (band_count's) */
    cn_status *filter_status;  /* their results */
    struct png_ring *rings; /* PNG_HISTORY_STREAM: each band's match lists */
    size_t finders;         /* finder tasks per band */
    /* PNG_HISTORY_REFINED with one segment: the blocks planned for the one band */
    struct libdeflate_block_plan *plans;
    size_t plan_count;
    uint32_t *plan_cache;
    struct png_refined *refined;
    struct libdeflate_compressor *planner; /* PNG_HISTORY_REFINED: plans the one band's blocks in stream_tasks */
    size_t plan_max, plan_cache_max;
    struct libdeflate_compressor *finder; /* the search's settings for the rings' finders, shared */
    unsigned round;     /* PNG_HISTORY_REFINED: the round of refinement */
    _Atomic(int) task_failed; /* a pool task could not allocate: the call fails */
    uint32_t *starts;   /* from round 1: every block's costs as round 0 left them */
    png_band *band;
} png_job;

static void put32(uint8_t *p, uint32_t value)
{
    p[0] = (uint8_t)(value >> 24);
    p[1] = (uint8_t)(value >> 16);
    p[2] = (uint8_t)(value >> 8);
    p[3] = (uint8_t)value;
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
#if CN_ISA_UNITS()
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
#if CN_ISA_UNITS()
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

/* The entropy rule: the filter whose filtered row has the least order-0 entropy (its bytes' counts h giving
 * -sum h * log2(h / length) bits); a tie (rows of a few bytes, whose bytes all differ under every filter) goes to
 * the lower MinSum cost, then to the lower type. scratch holds 5 * length bytes. */
static void filter_row_entropy(const uint8_t *row, const uint8_t *up, size_t length, size_t bpp, uint8_t *scratch,
                               uint8_t *out)
{
    double best = 0;
    uint64_t best_sum = 0;
    int type = 0;
    for (int f = 0; f < 5; ++f) {
        uint8_t *t = scratch + (size_t)f * length;
        png_filter_bytes(f, row, up, 0, length, bpp, t);
        uint32_t h[256] = {0};
        for (size_t i = 0; i < length; ++i) ++h[t[i]];
        double cost = 0;
        uint64_t sum = 0;
        for (int v = 0; v < 256; ++v)
            if (h[v]) {
                cost -= h[v] * log2((double)h[v] / (double)length);
                sum += (uint64_t)h[v] * png_cost((uint8_t)v);
            }
        if (f == 0 || cost < best || (cost == best && sum < best_sum)) {
            best = cost;
            best_sum = sum;
            type = f;
        }
    }
    out[0] = (uint8_t)type;
    memcpy(out + 1, scratch + (size_t)type * length, length);
}

/* The trial rule: the filter whose row (filter byte and bytes), after the up to PNG_TRIAL_ROWS rows of its band
 * already filtered (from band_begin), compresses smallest at libdeflate level PNG_TRIAL_LEVEL, ties to the lower
 * type; but a trial of at most PNG_TRIAL_SHORT bytes, which libdeflate stores at that level (inputs of up to 51
 * bytes) or nearly, so that every filter ties, takes the lowest MinSum cost. trial holds (PNG_TRIAL_ROWS + 1) *
 * (length + 1) bytes, then `bound` bytes for the compressed trial. */
#define PNG_TRIAL_ROWS 2
#define PNG_TRIAL_LEVEL 1
#define PNG_TRIAL_SHORT 64
static void filter_row_trial(struct libdeflate_compressor *compressor, const uint8_t *row, const uint8_t *up,
                             size_t length, size_t bpp, const uint8_t *band_begin, uint8_t *trial, size_t bound,
                             uint8_t *out)
{
    const size_t stride = length + 1;
    size_t prior = (size_t)(out - band_begin) / stride;
    if (prior > PNG_TRIAL_ROWS) prior = PNG_TRIAL_ROWS;
    memcpy(trial, out - prior * stride, prior * stride);
    uint8_t *candidate = trial + prior * stride, *packed = trial + (PNG_TRIAL_ROWS + 1) * stride;
    size_t best = SIZE_MAX;
    uint64_t best_sum = 0;
    const int short_trial = (prior + 1) * stride <= PNG_TRIAL_SHORT;
    int type = 0;
    for (int f = 0; f < 5; ++f) {
        candidate[0] = (uint8_t)f;
        png_filter_bytes(f, row, up, 0, length, bpp, candidate + 1);
        const size_t size = libdeflate_deflate_compress(compressor, trial, (prior + 1) * stride, packed, bound);
        uint64_t sum = 0;
        if (short_trial)
            for (size_t i = 1; i <= length; ++i) sum += png_cost(candidate[i]);
        if (size && (size < best || (size == best && sum < best_sum))) {
            best = size;
            best_sum = sum;
            type = f;
        }
    }
    out[0] = (uint8_t)type;
    png_filter_bytes(type, row, up, 0, length, bpp, out + 1);
}

/* Rows [y0, y1), the first rows of a band, filtered into out ((y1 - y0) * (row_bytes + 1) bytes) by the job's rule;
 * work holds two packed rows (unless the source rows are read in place) and, for y0 = 0, the zeros above the image's
 * first row after them. */
static cn_status filter_rows(const png_job *job, size_t y0, size_t y1, uint8_t *work, uint8_t *out)
{
    const size_t n = job->row_bytes, rows = job->direct ? 0 : 2 * n;
    uint8_t *buffers[2] = {work, work + n};
    const uint8_t *up;
    if (y0 == 0) {
        memset(work + rows, 0, n);
        up = work + rows;
    } else {
        up = packed_row(job, y0 - 1, buffers[1]);
    }
    struct libdeflate_compressor *trial = NULL;
    size_t trial_bound = 0, scratch_bytes = 0;
    if (job->rule == PNG_RULE_TRIAL) {
        trial = libdeflate_alloc_compressor(PNG_TRIAL_LEVEL);
        if (!trial) return CN_ALLOCATION_FAILED;
        trial_bound = libdeflate_deflate_compress_bound(trial, (PNG_TRIAL_ROWS + 1) * (n + 1));
        scratch_bytes = (PNG_TRIAL_ROWS + 1) * (n + 1) + trial_bound;
    } else if (job->rule == PNG_RULE_ENTROPY) {
        scratch_bytes = 5 * n;
    }
    uint8_t *scratch = scratch_bytes ? malloc(scratch_bytes) : NULL;
    if (scratch_bytes && !scratch) {
        libdeflate_free_compressor(trial);
        return CN_ALLOCATION_FAILED;
    }
    for (size_t y = y0; y < y1; ++y) {
        const uint8_t *row = packed_row(job, y, buffers[(y - y0) & 1]);
        uint8_t *filtered = out + (y - y0) * (n + 1);
        if (job->rule == PNG_RULE_TRIAL)
            filter_row_trial(trial, row, up, n, job->channels, out, scratch, trial_bound, filtered);
        else if (job->rule == PNG_RULE_ENTROPY)
            filter_row_entropy(row, up, n, job->channels, scratch, filtered);
        else
            filter_row(job->isa, row, up, n, job->channels, filtered);
        up = row;
    }
    free(scratch);
    libdeflate_free_compressor(trial);
    return CN_OK;
}

/* Band `index`'s rows. */
static void band_rows(const png_job *job, size_t index, size_t *y0, size_t *y1)
{
    *y0 = job->height * index / job->bands;
    *y1 = job->height * (index + 1) / job->bands;
}

/* With a history, the image's rows filtered into job->whole in job->filters bands, before any band compresses. */
static void filter_bands(void *context, size_t begin, size_t end)
{
    png_job *job = context;
    const size_t n = job->row_bytes;
    for (size_t i = begin; i < end; ++i) {
        const size_t y0 = job->height * i / job->filters, y1 = job->height * (i + 1) / job->filters;
        png_buffer *work = take_buffer(i, 3 * n);
        job->filter_status[i] = work ? filter_rows(job, y0, y1, work->data, job->whole + y0 * (n + 1))
                                     : CN_ALLOCATION_FAILED;
        if (work) give_buffer(i, work);
    }
}

/* PNG_HISTORY_STREAM: a band's match lists in a ring of PNG_RING_CHUNKS chunks of PNG_RING_BYTES positions, from its
 * lead-in to its end. A chunk is found (libdeflate_find_matches, primed with the 32 KiB before it, which gives the
 * lists of one search over the image) by a finder task running ahead of the band's parse, or by the parse itself when
 * it reaches a chunk no finder took; a finder takes a chunk only while the parse has left its slot, and stops when no
 * chunk is left or, before the parse starts, when the ring is full, so that no task waits for one that may not run.
 * The lists never depend on which task found them. Memory: PNG_RING_CHUNKS chunks of about 30 bytes per position. */
#define PNG_RING_BYTES ((size_t)128 << 10)
#define PNG_MATCHES_PER_BYTE 6
/* Effort 4's plan keeps every block's lists: room for PNG_PLAN_MATCHES_PER_BYTE matches per position on average (the
 * test images need 1.3 to 3.4; about 175 MB for the 6 MiB of one segment); without room, or too short to plan, the
 * image gets effort 3's stream. */
#define PNG_PLAN_MATCHES_PER_BYTE 6
#define PNG_RING_CHUNKS 4
/* At most PNG_RING_FINDERS finders per band: more only wait (measured: 3 keep one parse fed). */
#define PNG_RING_FINDERS 3
#define PNG_RING_ENTRIES (PNG_RING_BYTES * PNG_MATCHES_PER_BYTE + 260)
typedef struct png_ring {
    const png_job *job;
    size_t in;       /* the band's start in the filtered rows */
    ptrdiff_t first; /* its first position, relative to in (minus its lead-in) */
    ptrdiff_t end;   /* its length */
    size_t chunks;
    _Atomic(int) *state;      /* per chunk: 0 free, 1 being found, 2 found */
    _Atomic(size_t) reading;  /* the chunk the parse reads */
    _Atomic(int) parsing;     /* 1 while the parse runs, 2 once it ended */
    uint32_t *start[PNG_RING_CHUNKS];
    uint16_t *count[PNG_RING_CHUNKS];
    uint8_t *list[PNG_RING_CHUNKS];
    size_t capacity[PNG_RING_CHUNKS]; /* entries */
    _Atomic(int) failed;              /* a slot's list could not grow: the lists left empty, the call fails */
} png_ring;

/* Chunk k into its slot; a chunk whose lists outgrow the slot's list grows it. Once an allocation failed (here or
 * in libdeflate_find_matches) the ring has failed for good: this and every later chunk get empty lists (no match),
 * nothing grows again, the finders stop, the parse ends quickly, and the call returns CN_ALLOCATION_FAILED. */
static void ring_find(png_ring *r, size_t k)
{
    const png_job *job = r->job;
    const size_t slot = k % PNG_RING_CHUNKS, filtered = job->height * (job->row_bytes + 1);
    const ptrdiff_t b = r->first + (ptrdiff_t)(k * PNG_RING_BYTES);
    const ptrdiff_t e = b + (ptrdiff_t)PNG_RING_BYTES < r->end ? b + (ptrdiff_t)PNG_RING_BYTES : r->end;
    while (!atomic_load(&r->failed)) {
        const size_t found = libdeflate_find_matches(job->finder, job->whole, filtered,
                                                     (size_t)((ptrdiff_t)r->in + b), (size_t)((ptrdiff_t)r->in + e),
                                                     32768, r->start[slot], r->count[slot], r->list[slot], 0,
                                                     r->capacity[slot]);
        if (found == SIZE_MAX) break;
        if (found) return;
        uint8_t *grown = malloc(r->capacity[slot] * 2 * 4);
        if (!grown) break;
        free(r->list[slot]);
        r->list[slot] = grown;
        r->capacity[slot] *= 2;
    }
    atomic_store(&r->failed, 1);
    memset(r->start[slot], 0, (size_t)(e - b) * sizeof *r->start[slot]);
    memset(r->count[slot], 0, (size_t)(e - b) * sizeof *r->count[slot]);
}

/* libdeflate's window for position p: the chunk holding it, found or waited for. */
static void ring_window(void *context, ptrdiff_t p, const uint32_t **start, const uint16_t **count, const void **list,
                        ptrdiff_t *base, ptrdiff_t *end)
{
    png_ring *r = context;
    const size_t k = (size_t)(p - r->first) / PNG_RING_BYTES, slot = k % PNG_RING_CHUNKS;
    const ptrdiff_t b = r->first + (ptrdiff_t)(k * PNG_RING_BYTES);
    atomic_store(&r->reading, k);
    int expected = 0;
    if (atomic_compare_exchange_strong(&r->state[k], &expected, 1)) {
        ring_find(r, k);
        atomic_store(&r->state[k], 2);
    } else {
        while (atomic_load(&r->state[k]) != 2) _mm_pause();
    }
    *start = r->start[slot];
    *count = r->count[slot];
    *base = b;
    *list = r->list[slot];
    *end = b + (ptrdiff_t)PNG_RING_BYTES < r->end ? b + (ptrdiff_t)PNG_RING_BYTES : r->end;
}

/* A finder of band r's ring: the lowest free chunk within the ring, while there is one. */
static void ring_finder(png_ring *r)
{
    for (;;) {
        const size_t reading = atomic_load(&r->reading);
        size_t k = reading, limit = reading + PNG_RING_CHUNKS < r->chunks ? reading + PNG_RING_CHUNKS : r->chunks;
        int expected = 0;
        while (k < limit && (atomic_load(&r->state[k]) != 0 || !atomic_compare_exchange_strong(&r->state[k],
                                                                                                &expected, 1))) {
            expected = 0;
            ++k;
        }
        if (k < limit) {
            ring_find(r, k);
            atomic_store(&r->state[k], 2);
            continue;
        }
        if (limit == r->chunks || atomic_load(&r->failed)) return; /* every chunk left is taken, or failed */
        if (atomic_load(&r->parsing) != 1) return; /* not started (the ring is full) or ended */
        _mm_pause();
    }
}

/* PNG_HISTORY_REFINED: task t refines block t / PNG_RESTART_GROUPS with its group t % PNG_RESTART_GROUPS of the
 * restarts, the block's seed its index; each result goes to its own slot, so that the choice after does not depend on
 * which task ran first. */
#define PNG_RESTART_GROUPS 8
typedef struct png_refined {
    int better;
    uint32_t bits;
    uint32_t costs[LIBDEFLATE_COST_WORDS];
} png_refined;

static void refine_blocks(void *context, size_t begin, size_t end)
{
    png_job *job = context;
    struct libdeflate_compressor *c = libdeflate_alloc_compressor(job->level);
    if (!c) {
        atomic_store(&job->task_failed, 1);
        return;
    }
    const unsigned per_group = (png_restarts + PNG_RESTART_GROUPS - 1) / PNG_RESTART_GROUPS;
    for (size_t t = begin; t < end; ++t) {
        const size_t block = t / PNG_RESTART_GROUPS;
        const unsigned group = (unsigned)(t % PNG_RESTART_GROUPS), first = group * per_group;
        const unsigned count = first >= png_restarts ? 0 : png_restarts - first < per_group ? png_restarts - first
                                                                                              : per_group;
        png_refined *r = &job->refined[t];
        /* Round 0 from the block's own costs; a later round from those its predecessor's best parse came from. */
        const uint32_t *start = job->round && block ? job->starts + (block - 1) * LIBDEFLATE_COST_WORDS : NULL;
        r->better = count && libdeflate_refine_block(c, &job->plans[block], job->plan_cache, start,
                                                     (uint64_t)block + ((uint64_t)job->round << 32), first, count,
                                                     png_noise, &r->bits, r->costs);
    }
    libdeflate_free_compressor(c);
}

/* PNG_HISTORY_REFINED, after the rounds: pair i is blocks i and i + 1 parsed as one block, from either one's costs
 * (to convergence), within libdeflate's limits on a block (PNG_MERGE_LENGTH bytes, PNG_MERGE_CACHE cache entries). */
#define PNG_MERGE_LENGTH 300000
#define PNG_MERGE_CACHE 1500000
/* Effort 4's rounds for a plan of `entries` cache entries (positions plus their matches): one per PNG_ROUND_ENTRIES
 * that the plan stays below PNG_BUDGET_ENTRIES, at most PNG_MAX_ROUNDS, none above it (the merges still run). Fitted
 * on 2026-11-16 to seven frames of 0.8 to 2.1 megapixels so that each takes about 1 s on 32 threads (the plan, the
 * merges and the writing grow with the entries, each round by about 5 ns per entry). */
#define PNG_BUDGET_ENTRIES ((size_t)32000000)
#define PNG_ROUND_ENTRIES ((size_t)2500000)
static unsigned refine_rounds(size_t entries)
{
    const size_t fit = entries < PNG_BUDGET_ENTRIES ? (PNG_BUDGET_ENTRIES - entries) / PNG_ROUND_ENTRIES : 0;
    return fit > PNG_MAX_ROUNDS ? PNG_MAX_ROUNDS : (unsigned)fit;
}

static void merge_pairs(void *context, size_t begin, size_t end)
{
    png_job *job = context;
    struct libdeflate_compressor *c = libdeflate_alloc_compressor(job->level);
    if (!c) {
        atomic_store(&job->task_failed, 1);
        return;
    }
    for (size_t i = begin; i < end; ++i) {
        const struct libdeflate_block_plan *a = &job->plans[i], *b = &job->plans[i + 1];
        png_refined *r = &job->refined[i];
        r->better = 0;
        if (a->mode || b->mode || a->length + b->length > PNG_MERGE_LENGTH
            || b->cache_end - a->cache_begin > PNG_MERGE_CACHE)
            continue;
        struct libdeflate_block_plan both = *a;
        both.length = a->length + b->length;
        both.cache_end = b->cache_end;
        both.bits = a->bits + b->bits; /* a merge is kept only when it is smaller than the two */
        r->better = libdeflate_refine_block(c, &both, job->plan_cache, NULL, 0, 0, 1, 0, &r->bits, r->costs);
        both.bits = r->better ? r->bits : both.bits;
        r->better |= libdeflate_refine_block(c, &both, job->plan_cache, b->costs, 0, 0, 1, 0, &r->bits, r->costs);
    }
    libdeflate_free_compressor(c);
}

/* Band `index`: its rows filtered (with a history, already filtered in job->whole), then compressed into its chunk;
 * with a history, its matches may reach into the 32 KiB of filtered rows before it (and the parser
 * also fits its costs on them first). */
static cn_status encode_band(const png_job *job, size_t index, png_band *band)
{
    size_t y0, y1;
    band_rows(job, index, &y0, &y1);
    const size_t n = job->row_bytes, length = (y1 - y0) * (n + 1);
    const int first = index == 0, last = index + 1 == job->bands;
    png_buffer *work_buffer = NULL;
    const uint8_t *filtered;
    size_t history = 0;
    if (job->whole) {
        filtered = job->whole + y0 * (n + 1);
        history = y0 * (n + 1) < 32768 ? y0 * (n + 1) : 32768;
    } else {
        /* The filtered rows, then the filter's work rows. */
        work_buffer = take_buffer(index, length + 3 * n);
        if (!work_buffer) return CN_ALLOCATION_FAILED;
        const cn_status status = filter_rows(job, y0, y1, work_buffer->data + length, work_buffer->data);
        if (status != CN_OK) {
            give_buffer(index, work_buffer);
            return status;
        }
        filtered = work_buffer->data;
    }
    band->filtered = length;
    band->adler = libdeflate_adler32(1, filtered, length);

    struct libdeflate_compressor *compressor = take_kept(kept_compressors, PNG_KEPT_COMPRESSORS, index);
    if (compressor && job->level != PNG_LEVEL) {
        libdeflate_free_compressor(compressor);
        compressor = NULL;
    }
    if (!compressor) compressor = libdeflate_alloc_compressor(job->level);
    if (!compressor) {
        if (work_buffer) give_buffer(index, work_buffer);
        return CN_ALLOCATION_FAILED;
    }
    const size_t header = first ? 2 : 0, trailer = last ? 4 : 0;
    const size_t bound = libdeflate_deflate_compress_bound(compressor, length) + 5;
    png_buffer *chunk_buffer = take_buffer(index, 8 + header + bound + trailer + 4);
    size_t written = 0;
    if (chunk_buffer) {
        uint8_t *data = chunk_buffer->data + 8 + header;
        if (job->plans) {
            written = libdeflate_write_blocks(compressor, filtered, length, job->plans, job->plan_count,
                                              job->plan_cache, data, bound, last);
        } else if (job->history == PNG_HISTORY_STREAM) {
            /* The band's matches from its ring, from its lead-in on. */
            const size_t begin = y0 * (n + 1), lead = begin < png_lead_bytes ? begin : png_lead_bytes;
            const struct libdeflate_matches matches = {NULL, NULL, NULL, ring_window, &job->rings[index]};
            written = libdeflate_deflate_compress_matches(compressor, filtered, length, lead, 1, &matches, data,
                                                          bound, last);
        }
        else if (job->history)
            written = libdeflate_deflate_compress_history(compressor, filtered, length, history, 1, data, bound,
                                                          last);
        else
            written = last ? libdeflate_deflate_compress(compressor, filtered, length, data, bound)
                           : libdeflate_deflate_compress_nonfinal(compressor, filtered, length, data, bound);
    }
    if (job->level != PNG_LEVEL || !give_kept(kept_compressors, PNG_KEPT_COMPRESSORS, index, compressor))
        libdeflate_free_compressor(compressor);
    if (work_buffer) give_buffer(index, work_buffer);
    if (!chunk_buffer) return CN_ALLOCATION_FAILED;
    band->chunk = chunk_buffer; /* the call gives it back, whatever happens to the others */
    if (!written) return CN_SIZE_OVERFLOW; /* not expected: the bound holds every output */
    uint8_t *chunk = chunk_buffer->data;
    if (first) {
        chunk[8] = 0x78; /* zlib: DEFLATE with a 32 KiB window */
        /* no dictionary, the level hint libdeflate's zlib wrapper writes (2 to level 6, 3 above), check bits */
        chunk[9] = job->level <= 6 ? 0x9C : 0xDA;
    }
    const size_t data = header + written + trailer;
    put32(chunk, (uint32_t)data);
    memcpy(chunk + 4, "IDAT", 4);
    band->crc = libdeflate_crc32(0, chunk + 4, 4 + header + written);
    if (!last) put32(chunk + 8 + data, band->crc);
    band->size = 12 + data;
    return CN_OK;
}

/* Whether a band's ring failed to allocate (ring_find). */
static int rings_failed(const png_job *job)
{
    for (size_t i = 0; i < job->bands; ++i)
        if (atomic_load(&job->rings[i].failed)) return 1;
    return 0;
}

/* PNG_HISTORY_STREAM: tasks [0, bands) parse the bands (or plan the one band), the rest are the rings' finders. */
static void stream_tasks(void *context, size_t begin, size_t end)
{
    png_job *job = context;
    for (size_t t = begin; t < end; ++t) {
        if (t < job->bands) {
            png_ring *r = &job->rings[t];
            atomic_store(&r->parsing, 1);
            if (job->planner)
                job->plan_count = libdeflate_plan_blocks(job->planner, job->whole, job->height * (job->row_bytes + 1),
                                                         &(struct libdeflate_matches){NULL, NULL, NULL, ring_window,
                                                                                      r},
                                                         job->plans, job->plan_max, job->plan_cache,
                                                         job->plan_cache_max);
            else
                job->band[t].status = encode_band(job, t, &job->band[t]);
            atomic_store(&r->parsing, 2);
        } else {
            ring_finder(&job->rings[(t - job->bands) % job->bands]);
        }
    }
}

static void encode_bands(void *context, size_t begin, size_t end)
{
    png_job *job = context;
    for (size_t i = begin; i < end; ++i) job->band[i].status = encode_band(job, i, &job->band[i]);
}

typedef struct png_copy {
    uint8_t *out;
    const png_band *band;
    int one_chunk; /* the bands' data only, into IDAT chunks of PNG_IDAT_BYTES (the last one shorter) */
} png_copy;

/* The bands' chunks into the file, in parallel: the file's pages are first touched (and faulted in) here. */
static void copy_bands(void *context, size_t begin, size_t end)
{
    const png_copy *copy = context;
    for (size_t i = begin; i < end; ++i) {
        if (!copy->one_chunk) {
            memcpy(copy->out + copy->band[i].offset, copy->band[i].chunk->data, copy->band[i].size);
            continue;
        }
        /* The band's data from data offset `at` on, a chunk's 12 bytes of framing before each PNG_IDAT_BYTES. */
        const uint8_t *from = copy->band[i].chunk->data + 8;
        size_t at = copy->band[i].offset, left = copy->band[i].size - 12;
        while (left) {
            const size_t room = PNG_IDAT_BYTES - at % PNG_IDAT_BYTES, n = left < room ? left : room;
            memcpy(copy->out + 8 + 25 + 8 + at + at / PNG_IDAT_BYTES * 12, from, n);
            from += n;
            at += n;
            left -= n;
        }
    }
}

/* Efforts above 0: the IDAT chunks' lengths, types and CRCs around the data copy_bands placed. */
typedef struct png_chunks {
    uint8_t *first; /* the first chunk */
    size_t data;    /* the data's bytes */
} png_chunks;

static void frame_chunks(void *context, size_t begin, size_t end)
{
    const png_chunks *chunks = context;
    for (size_t i = begin; i < end; ++i) {
        uint8_t *chunk = chunks->first + i * (PNG_IDAT_BYTES + 12);
        const size_t rest = chunks->data - i * PNG_IDAT_BYTES, n = rest < PNG_IDAT_BYTES ? rest : PNG_IDAT_BYTES;
        put32(chunk, (uint32_t)n);
        memcpy(chunk + 4, "IDAT", 4);
        put32(chunk + 8 + n, libdeflate_crc32(0, chunk + 4, 4 + n));
    }
}

/* (count - 1) * |stride| within a quarter of PTRDIFF_MAX, so a sample's offset (three such terms) fits. */
static int span_fits(size_t count, ptrdiff_t stride)
{
    const size_t magnitude = stride < 0 ? (size_t)0 - (size_t)stride : (size_t)stride;
    return magnitude == 0 || count - 1 <= (size_t)(PTRDIFF_MAX / 4) / magnitude;
}

/* Each effort's compression level, filter rule, band cap (0: band_count's bands) and history. */
static const struct {
    int level, rule;
    size_t bands;
    int history;
} png_efforts[CN_PNG_EFFORTS] = {
    {PNG_LEVEL, PNG_RULE_MINSUM, 0, PNG_HISTORY_NONE},
    {10, PNG_RULE_ENTROPY, 0, PNG_HISTORY_WARM},
    {11, PNG_RULE_TRIAL, 24, PNG_HISTORY_WARM},
    {12, PNG_RULE_TRIAL, 0, PNG_HISTORY_STREAM},
    {12, PNG_RULE_TRIAL, 0, PNG_HISTORY_REFINED},
};

CN_EXPORT cn_status cn_png_encode_u8_effort(const uint8_t *source, size_t height, size_t width, size_t channels,
    ptrdiff_t row_stride, ptrdiff_t pixel_stride, ptrdiff_t channel_stride, int bgr, int effort,
    cn_png_allocate allocate, void *context)
{
    if (!source || !allocate || channels < 1 || channels > 4 || height < 1 || width < 1 || height > CN_PNG_MAX_SIDE
        || width > CN_PNG_MAX_SIDE || width > CN_PNG_MAX_ROW_SAMPLES / channels || effort < 0
        || effort >= CN_PNG_EFFORTS)
        return CN_INVALID_ARGUMENT;
    if (!span_fits(height, row_stride) || !span_fits(width, pixel_stride) || !span_fits(channels, channel_stride))
        return CN_SIZE_OVERFLOW;
    const size_t row_bytes = width * channels, filtered = height * (row_bytes + 1);
    /* The rows are filtered in band_count's bands; an effort's cap lowers the bands that compress, never below one
     * band per PNG_MAX_BAND_BYTES. */
    const size_t automatic = band_count(height, filtered);
    const size_t needed = filtered / PNG_MAX_BAND_BYTES + (filtered % PNG_MAX_BAND_BYTES != 0);
    size_t bands = automatic;
    if (png_efforts[effort].bands && png_efforts[effort].bands < bands) bands = png_efforts[effort].bands;
    /* PNG_HISTORY_STREAM: one stream up to 2 PNG_SEGMENT_BYTES, else a segment per PNG_SEGMENT_BYTES (at most
     * band_count's), each parsed on its own thread after a lead-in that fits the parser's costs. */
    if (png_efforts[effort].history >= PNG_HISTORY_STREAM) {
        bands = filtered / png_segment_bytes;
        if (bands > automatic) bands = automatic;
        if (bands < 1) bands = 1;
    }
    if (bands < needed) bands = needed;
    const int refined = png_efforts[effort].history == PNG_HISTORY_REFINED;
    const int history = refined ? PNG_HISTORY_STREAM : png_efforts[effort].history;
    png_band *band = calloc(bands, sizeof *band);
    cn_status *filter_status = history ? calloc(automatic, sizeof *filter_status) : NULL;
    if (!band || (history && !filter_status)) {
        free(band);
        free(filter_status);
        return CN_ALLOCATION_FAILED;
    }
    const int packed = pixel_stride == (ptrdiff_t)channels && (channels == 1 || channel_stride == 1);
    png_job job = {source, row_stride, pixel_stride, channel_stride, height, width, channels, row_bytes, bands,
                   bgr && channels >= 3, packed, packed && !(bgr && channels >= 3), cn_isa_current(),
                   png_efforts[effort].level, png_efforts[effort].rule, history, NULL, automatic, filter_status,
                   .band = band};
    png_buffer *whole = NULL;
    cn_status status = CN_OK;
    if (history) {
        whole = take_buffer(0, filtered);
        if (!whole) {
            status = CN_ALLOCATION_FAILED;
        } else {
            job.whole = whole->data;
            status = cn_parallel_for(automatic, 1, filter_bands, &job);
            for (size_t i = 0; i < automatic && status == CN_OK; ++i) status = filter_status[i];
        }
    }
    int streamed = 0;
    if (status == CN_OK && history == PNG_HISTORY_STREAM) {
        /* Each band's ring, its parse (or the one band's plan) and its finders in one call; the finders per band fill
         * the pool's other threads (the lists, and so the bytes, do not depend on how many). */
        job.finder = libdeflate_alloc_compressor(job.level);
        job.rings = calloc(bands, sizeof *job.rings);
        if (!job.finder || !job.rings) status = CN_ALLOCATION_FAILED;
        for (size_t i = 0; i < bands && status == CN_OK; ++i) {
            png_ring *r = &job.rings[i];
            size_t y0, y1;
            band_rows(&job, i, &y0, &y1);
            const size_t begin = y0 * (row_bytes + 1), lead = begin < png_lead_bytes ? begin : png_lead_bytes;
            r->job = &job;
            r->in = begin;
            r->first = -(ptrdiff_t)lead;
            r->end = (ptrdiff_t)((y1 - y0) * (row_bytes + 1));
            r->chunks = (size_t)(r->end - r->first + (ptrdiff_t)PNG_RING_BYTES - 1) / PNG_RING_BYTES;
            r->state = calloc(r->chunks, sizeof *r->state);
            if (!r->state) status = CN_ALLOCATION_FAILED;
            for (size_t k = 0; k < PNG_RING_CHUNKS && status == CN_OK; ++k) {
                r->start[k] = malloc(PNG_RING_BYTES * sizeof *r->start[k]);
                r->count[k] = malloc(PNG_RING_BYTES * sizeof *r->count[k]);
                r->capacity[k] = PNG_RING_ENTRIES;
                r->list[k] = malloc(PNG_RING_ENTRIES * 4);
                if (!r->start[k] || !r->count[k] || !r->list[k]) status = CN_ALLOCATION_FAILED;
            }
        }
        const unsigned threads = cn_parallel_capacity();
        job.finders = threads > bands ? (threads - bands) / bands : 0;
        if (job.finders > PNG_RING_FINDERS) job.finders = PNG_RING_FINDERS;
        /* One segment by the segment rule (below 2 PNG_SEGMENT_BYTES of filtered rows; an image of one band for its
         * few rows may be of any size), refined: plan its blocks here (with the finders), refine them on the pool,
         * write them below. Anything larger gets effort 3's bytes. */
        if (status == CN_OK && refined && bands == 1 && filtered / png_segment_bytes < 2) {
            job.plan_max = filtered / 1024 + 16;
            job.plan_cache_max = filtered * (1 + PNG_PLAN_MATCHES_PER_BYTE);
            job.plans = malloc(job.plan_max * sizeof *job.plans);
            job.plan_cache = malloc(job.plan_cache_max * sizeof *job.plan_cache);
            job.planner = libdeflate_alloc_compressor(job.level);
            if (!job.plans || !job.plan_cache || !job.planner) status = CN_ALLOCATION_FAILED;
        }
        if (status == CN_OK) status = cn_parallel_for(bands * (1 + job.finders), 1, stream_tasks, &job);
        if (status == CN_OK && rings_failed(&job)) status = CN_ALLOCATION_FAILED;
        libdeflate_free_compressor(job.planner);
        job.planner = NULL;
        streamed = !job.plans;
        if (status == CN_OK && job.plans) {
            const size_t tasks = job.plan_count * PNG_RESTART_GROUPS;
            job.refined = job.plan_count ? calloc(tasks, sizeof *job.refined) : NULL;
            job.starts = job.plan_count ? malloc(job.plan_count * sizeof job.plans->costs) : NULL;
            if (job.plan_count && (!job.refined || !job.starts)) status = CN_ALLOCATION_FAILED;
            if (status == CN_OK && job.plan_count) {
                const unsigned rounds = png_rounds ? png_rounds
                                                   : refine_rounds(job.plans[job.plan_count - 1].cache_end);
                for (job.round = 0; job.round < rounds && status == CN_OK; ++job.round) {
                    for (size_t b = 0; b < job.plan_count; ++b)
                        memcpy(job.starts + b * LIBDEFLATE_COST_WORDS, job.plans[b].costs, sizeof job.plans->costs);
                    memset(job.refined, 0, tasks * sizeof *job.refined);
                    status = cn_parallel_for(tasks, 1, refine_blocks, &job);
                    /* Each block: the smallest result, the lowest group among equal ones. */
                    for (size_t t = 0; t < tasks; ++t) {
                        struct libdeflate_block_plan *plan = &job.plans[t / PNG_RESTART_GROUPS];
                        if (job.refined[t].better && job.refined[t].bits < plan->bits) {
                            plan->bits = job.refined[t].bits;
                            memcpy(plan->costs, job.refined[t].costs, sizeof plan->costs);
                        }
                    }
                }
                /* Neighbours that are smaller as one block are merged, left to right, each block at most once. */
                if (status == CN_OK && job.plan_count > 1) {
                    memset(job.refined, 0, (job.plan_count - 1) * sizeof *job.refined);
                    status = cn_parallel_for(job.plan_count - 1, 1, merge_pairs, &job);
                    size_t kept = 0;
                    for (size_t i = 0; i < job.plan_count; ++i) {
                        job.plans[kept] = job.plans[i];
                        if (i + 1 < job.plan_count && job.refined[i].better) {
                            job.plans[kept].length += job.plans[i + 1].length;
                            job.plans[kept].cache_end = job.plans[i + 1].cache_end;
                            job.plans[kept].bits = job.refined[i].bits;
                            memcpy(job.plans[kept].costs, job.refined[i].costs, sizeof job.plans->costs);
                            ++i;
                        }
                        ++kept;
                    }
                    job.plan_count = kept;
                }
                if (status == CN_OK && atomic_load(&job.task_failed)) status = CN_ALLOCATION_FAILED;
            }
            /* No plan (too short, or out of room, which the input alone decides): the stream as effort 3 writes it. */
            if (status == CN_OK && !job.plan_count) {
                free(job.plans);
                job.plans = NULL;
                for (size_t i = 0; i < bands; ++i) {
                    png_ring *r = &job.rings[i];
                    for (size_t k = 0; k < r->chunks; ++k) atomic_store(&r->state[k], 0);
                    atomic_store(&r->reading, 0);
                    atomic_store(&r->parsing, 0);
                }
                status = cn_parallel_for(bands * (1 + job.finders), 1, stream_tasks, &job);
                if (status == CN_OK && rings_failed(&job)) status = CN_ALLOCATION_FAILED;
                streamed = 1;
            }
        }
    }
    if (status == CN_OK && !streamed) status = cn_parallel_for(bands, 1, encode_bands, &job);
    if (job.rings) {
        for (size_t i = 0; i < bands; ++i) {
            for (size_t k = 0; k < PNG_RING_CHUNKS; ++k) {
                free(job.rings[i].start[k]);
                free(job.rings[i].count[k]);
                free(job.rings[i].list[k]);
            }
            free(job.rings[i].state);
        }
        free(job.rings);
    }
    libdeflate_free_compressor(job.finder);
    free(job.plans);
    free(job.plan_cache);
    free(job.refined);
    free(job.starts);
    if (whole) give_buffer(0, whole);
    free(filter_status);
    for (size_t i = 0; i < bands && status == CN_OK; ++i) status = band[i].status;
    if (status == CN_OK) {
        uint32_t adler = band[0].adler;
        for (size_t i = 1; i < bands; ++i) adler = adler32_combine(adler, band[i].adler, band[i].filtered);
        png_band *last = &band[bands - 1];
        uint8_t *end_of_chunk = last->chunk->data + last->size;
        put32(end_of_chunk - 8, adler);
        put32(end_of_chunk - 4, libdeflate_crc32(last->crc, end_of_chunk - 8, 4));
        /* Efforts above 0 write the bands' data as one stream of IDAT chunks of PNG_IDAT_BYTES each but the last (12
         * bytes less per band than a chunk per band); effort 0 writes a chunk per band. */
        size_t data = 0;
        for (size_t i = 0; i < bands; ++i) data += band[i].size - 12;
        const int one_chunk = effort > 0;
        const size_t chunks = data / PNG_IDAT_BYTES + (data % PNG_IDAT_BYTES != 0);
        size_t total = 8 + 25;
        if (one_chunk) {
            size_t at = 0;
            for (size_t i = 0; i < bands; ++i) {
                band[i].offset = at; /* in the data */
                at += band[i].size - 12;
            }
            total += data + 12 * chunks;
        } else {
            for (size_t i = 0; i < bands; ++i) {
                band[i].offset = total;
                total += band[i].size;
            }
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
            png_copy copy = {out, band, one_chunk};
            status = cn_parallel_for(bands, 1, copy_bands, &copy);
            if (one_chunk && status == CN_OK) {
                png_chunks framing = {out + 8 + 25, data};
                status = cn_parallel_for(chunks, 1, frame_chunks, &framing);
            }
            memcpy(out + total - 12, end, 12);
        }
    }
    for (size_t i = 0; i < bands; ++i)
        if (band[i].chunk) give_buffer(i, band[i].chunk);
    free(band);
    return status;
}

CN_EXPORT cn_status cn_png_encode_u8(const uint8_t *source, size_t height, size_t width, size_t channels,
    ptrdiff_t row_stride, ptrdiff_t pixel_stride, ptrdiff_t channel_stride, int bgr, cn_png_allocate allocate,
    void *context)
{
    return cn_png_encode_u8_effort(source, height, width, channels, row_stride, pixel_stride, channel_stride, bgr, 0,
                                   allocate, context);
}

CN_EXPORT void cn_png_encode_test_refine(unsigned restarts, unsigned noise, unsigned rounds)
{
    png_restarts = restarts;
    png_noise = noise > 1000 ? 1000 : noise;
    png_rounds = rounds;
}

CN_EXPORT void cn_png_encode_test_segments(size_t segment_bytes, size_t lead_bytes)
{
    png_segment_bytes = segment_bytes ? segment_bytes : PNG_SEGMENT_BYTES;
    png_lead_bytes = segment_bytes ? lead_bytes : PNG_LEAD_BYTES;
}
