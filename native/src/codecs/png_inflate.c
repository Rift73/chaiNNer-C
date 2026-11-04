/* One zlib stream inflated on many threads, for the PNG decoder's direct path. Files from most encoders hold a single
 * DEFLATE stream with matches across the whole file and nothing that says where its blocks start. Four steps:
 *
 * 1. Search. The compressed bytes are cut at even offsets, and each cut moves forward to the first bit where a
 *    non-final dynamic Huffman header validates. Stored and fixed blocks are not searched for (they carry too little to
 *    validate): the stretch before them simply decodes through them.
 * 2. Decode. Each stretch decodes from its start to the first block boundary at or past the next stretch's start, into
 *    its own buffers. The output before it is unknown, so it starts as 16-bit symbols with a marker for each of the
 *    32 KiB before it, which matches copy like any other value. Once a block boundary has 32 KiB without a marker
 *    behind it, no later match can reach one, and the rest decodes as plain bytes with those 32 KiB as history.
 * 3. Chain. The bits parse the same whatever the history holds, so a stretch's length and end are right if its start
 *    is. The first stretch starts at the stream's real start, and each stretch's end must be exactly the next one's
 *    start: an unbroken chain from the real start IS the serial parse, and a false candidate breaks it. The chain also
 *    gives every stretch its place in the output.
 * 4. Place. The last 32 KiB of a stretch are the history of the next: those that still hold markers resolve first, in
 *    order, and then each stretch resolves its markers, copies its bytes and hashes its piece of the output in
 *    parallel. The whole output's Adler-32 is checked against the stream's.
 *
 * Any failure returns 0 having written only inside out, and the caller inflates from scratch: this path takes a stream
 * only where the serial one gives the same bytes and no error. */
#include "image_codecs.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "adler32_combine.h"
#include "libdeflate.h"
#include "parallel.h"

/* DEFLATE's longest match distance. */
#define HISTORY ((size_t)32768)

/* Symbols below this are bytes; this plus a position in the 32 KiB before a stretch is a marker of that position. */
#define FIRST_MARKER 256u

/* Below this much compressed data per stretch, and below this many stretches, the search, the buffers and the placing
 * cost more than the threads save (measured: about 1.5 times faster at 1 MiB, slower at 0.5). The fuzz build lowers the
 * size, so that small inputs reach this path. */
#ifndef PNG_INFLATE_MIN_STRETCH_BYTES
#define PNG_INFLATE_MIN_STRETCH_BYTES ((size_t)256 << 10)
#endif
#define MIN_STRETCHES 4
#define MAX_STRETCHES 64

/* The serial path's time goes with the compressed bytes, the placing's with the output, and in a stream compressed
 * further than this most matches lead back to the history, so that nearly all of it decodes as symbols: at 4K, Pillow's
 * level 6 measured 2.2 times faster at 3:1 and 2 times slower at 15:1, however many threads. */
#define MAX_RATIO 5

/* The largest output tried (about 180 megapixels of RGB): a hostile stream can hold six times the output in buffers. */
#define MAX_OUT_BYTES ((size_t)512 << 20)

/* A stretch's buffers: from its first guess to the whole output, four times the size at a time, in symbols then
 * bytes. */
#define MAX_PARTS 12

/* How far one search goes (zlib's blocks hold a few tens of KiB, libdeflate's a few hundred at most), and how many
 * headers that pass the precode check it decodes (about one in 300 bytes of real streams), so that a file made of
 * near-misses stays cheap. */
#define SEARCH_BYTES ((size_t)128 << 10)
#define HEADER_ATTEMPTS 1024

/* The most a dynamic Huffman header reads, its last 8-byte load included: 74 bits, then 316 lengths of at most 14. */
#define HEADER_MAX_BYTES ((size_t)600)

static inline uint64_t bits_at(const uint8_t *in, size_t bit) {
    uint64_t word;
    memcpy(&word, in + (bit >> 3), sizeof word);
    return word >> (bit & 7); /* at least 57 valid bits */
}

/* Whether a dynamic Huffman header at bit gives complete codes with an end-of-block symbol. Stricter than the decoder
 * (a code libdeflate would take but this does not only costs a cut); the chain, not this, decides what is true. */
static int dynamic_header_valid(const uint8_t *in, size_t bit, int *attempts) {
    static const uint8_t order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
    const uint64_t head = bits_at(in, bit);
    const unsigned litlen = 257 + ((head >> 3) & 31), offsets = 1 + ((head >> 8) & 31);
    const unsigned listed = 4 + ((head >> 13) & 15);
    if (litlen > 286 || offsets > 30) return 0;
    const uint64_t precode = bits_at(in, bit + 17);
    unsigned space = 0;
    for (unsigned i = 0; i < listed; ++i) {
        const unsigned length = (precode >> (3 * i)) & 7;
        if (length) space += 128u >> length;
        if (space > 128) return 0;
    }
    if (space != 128 || !*attempts) return 0;
    --*attempts;
    /* The precode's table, indexed by the next 7 bits; complete, so every entry gets filled. */
    uint8_t precode_lengths[19] = {0};
    unsigned counts[8] = {0}, next[8] = {0};
    for (unsigned i = 0; i < listed; ++i) {
        const unsigned length = (precode >> (3 * i)) & 7;
        precode_lengths[order[i]] = (uint8_t)length;
        ++counts[length];
    }
    counts[0] = 0;
    for (unsigned length = 1; length < 8; ++length) next[length] = (next[length - 1] + counts[length - 1]) << 1;
    uint8_t table[128]; /* symbol << 3 | codeword length */
    for (unsigned symbol = 0; symbol < 19; ++symbol) {
        const unsigned length = precode_lengths[symbol];
        if (!length) continue;
        const unsigned codeword = next[length]++;
        unsigned reversed = 0;
        for (unsigned b = 0; b < length; ++b) reversed |= ((codeword >> b) & 1) << (length - 1 - b);
        for (unsigned fill = reversed; fill < 128; fill += 1u << length) table[fill] = (uint8_t)(symbol << 3 | length);
    }
    uint8_t lengths[286 + 30];
    const unsigned total = litlen + offsets;
    size_t at = bit + 17 + 3 * (size_t)listed;
    for (unsigned i = 0; i < total;) {
        uint64_t word = bits_at(in, at);
        const unsigned entry = table[word & 127], symbol = entry >> 3;
        word >>= entry & 7;
        at += entry & 7;
        if (symbol < 16) {
            lengths[i++] = (uint8_t)symbol;
            continue;
        }
        if (symbol == 16 && !i) return 0;
        const unsigned extra = symbol == 16 ? 2 : symbol == 17 ? 3 : 7;
        const unsigned repeat = (symbol == 18 ? 11 : 3) + (unsigned)(word & ((1u << extra) - 1));
        if (repeat > total - i) return 0;
        memset(lengths + i, symbol == 16 ? lengths[i - 1] : 0, repeat);
        i += repeat;
        at += extra;
    }
    if (!lengths[256]) return 0;
    uint32_t used = 0;
    for (unsigned i = 0; i < litlen; ++i)
        if (lengths[i]) used += UINT32_C(32768) >> lengths[i];
    if (used != 32768) return 0;
    used = 0;
    unsigned codewords = 0;
    for (unsigned i = litlen; i < total; ++i) {
        if (lengths[i]) {
            used += UINT32_C(32768) >> lengths[i];
            ++codewords;
        }
    }
    return used == 32768 || !codewords || (codewords == 1 && used == 16384);
}

/* The first bit in [from, limit) with a plausible non-final dynamic block, or SIZE_MAX: BFINAL 0 and BTYPE 2 are the
 * three bits 0, 0, 1, found 48 positions at a time. */
static size_t find_block(const uint8_t *in, size_t from, size_t limit) {
    int attempts = HEADER_ATTEMPTS;
    for (size_t bit = from; bit < limit; bit += 48) {
        const uint64_t word = bits_at(in, bit);
        uint64_t hits = ~word & ~(word >> 1) & (word >> 2) & ((UINT64_C(1) << 48) - 1);
        for (; hits; hits &= hits - 1) {
            const size_t candidate = bit + (size_t)__builtin_ctzll(hits);
            if (candidate < limit && dynamic_header_valid(in, candidate, &attempts)) return candidate;
        }
        if (!attempts) break;
    }
    return SIZE_MAX;
}

/* A piece of a later stretch's output: HISTORY items, then count output items, as symbols or as bytes. */
typedef struct inflate_part {
    uint16_t *symbols;
    uint8_t *bytes;
    size_t count;
} inflate_part;

typedef struct inflate_stretch {
    size_t start_bit, end_bit;
    size_t length, at; /* output bytes, and their place in out */
    inflate_part parts[MAX_PARTS]; /* none for the first stretch, already in place */
    size_t part_count;
    uint32_t adler;
    int ok, final; /* it decoded up to its end; that end closed the final block */
} inflate_stretch;

typedef struct inflate_job {
    const uint8_t *in;
    size_t in_size;
    uint8_t *out;
    size_t out_size;
    size_t count;
    atomic_size_t scratch; /* buffer bytes reserved, mostly never touched: at most six times out_size, so that a false
                              start or a crafted stream that decodes without end soon stops */
    inflate_stretch stretches[MAX_STRETCHES];
} inflate_job;

static void search_range(void *context, size_t begin, size_t end) {
    inflate_job *const job = (inflate_job *)context;
    for (size_t k = begin; k < end; ++k) {
        const size_t from = job->in_size / job->count * k, next = job->in_size / job->count * (k + 1);
        size_t limit = from + SEARCH_BYTES < next ? from + SEARCH_BYTES : next;
        if (limit > job->in_size - HEADER_MAX_BYTES) limit = job->in_size - HEADER_MAX_BYTES;
        job->stretches[k].start_bit = !k ? 0 : limit > from ? find_block(job->in, from * 8, limit * 8) : SIZE_MAX;
    }
}

/* Opens a part for capacity items after its history: the last HISTORY items of the prior part, whose symbols then hold
 * no marker if this part is bytes, or markers of the 32 KiB before the stretch if there is no prior part. */
static int part_open(inflate_job *job, inflate_part *part, const inflate_part *prior, int plain, size_t capacity) {
    const size_t size = (HISTORY + capacity) * (plain ? 1 : sizeof *part->symbols);
    if (atomic_fetch_add(&job->scratch, size) + size > job->out_size * 6) return 0;
    if (plain) {
        part->bytes = (uint8_t *)malloc(size);
        if (!part->bytes) return 0;
        if (prior->bytes) {
            memcpy(part->bytes, prior->bytes + prior->count, HISTORY);
        } else {
            for (size_t i = 0; i < HISTORY; ++i) part->bytes[i] = (uint8_t)prior->symbols[prior->count + i];
        }
    } else {
        part->symbols = (uint16_t *)malloc(size);
        if (!part->symbols) return 0;
        if (prior) {
            memcpy(part->symbols, prior->symbols + prior->count, HISTORY * sizeof *part->symbols);
        } else {
            for (size_t i = 0; i < HISTORY; ++i) part->symbols[i] = (uint16_t)(FIRST_MARKER + i);
        }
    }
    return 1;
}

/* A later stretch: each part runs until the output outgrows it, the next one (four times the size) picking up at the
 * last block boundary reached so that nothing is copied, or until the symbols hold no marker and the next part is
 * bytes. */
static int decode_later_stretch(struct libdeflate_decompressor *inflater, inflate_job *job, inflate_stretch *s,
                                size_t stop_bit, size_t capacity) {
    size_t bit = s->start_bit;
    int plain = 0;
    while (s->part_count < MAX_PARTS) {
        inflate_part *const part = &s->parts[s->part_count];
        ++s->part_count;
        if (!part_open(job, part, s->part_count > 1 ? part - 1 : NULL, plain, capacity)) return 0;
        struct libdeflate_range range = {bit, stop_bit, HISTORY, 0, 0, 0};
        const enum libdeflate_result result =
            plain ? libdeflate_deflate_decompress_range(inflater, job->in, job->in_size, part->bytes + HISTORY,
                                                        capacity, &range)
                  : libdeflate_deflate_decompress_symbols(inflater, job->in, job->in_size, part->symbols + HISTORY,
                                                          capacity, &range);
        part->count = range.end_out;
        s->length += range.end_out;
        s->end_bit = bit = range.end_bit;
        s->final = range.final;
        if (result == LIBDEFLATE_SUCCESS) {
            if (range.final || bit >= stop_bit) return 1;
            plain = 1;
        } else if (result != LIBDEFLATE_INSUFFICIENT_SPACE || capacity == job->out_size) {
            return 0;
        } else {
            capacity = capacity * 4 < job->out_size ? capacity * 4 : job->out_size;
        }
    }
    return 0;
}

static void decode_range(void *context, size_t begin, size_t end) {
    inflate_job *const job = (inflate_job *)context;
    struct libdeflate_decompressor *const inflater = libdeflate_alloc_decompressor();
    if (!inflater) return;
    for (size_t k = begin; k < end; ++k) {
        inflate_stretch *const s = &job->stretches[k];
        const size_t stop_bit = k + 1 < job->count ? job->stretches[k + 1].start_bit : SIZE_MAX;
        if (k) {
            /* Output near its share of the compressed bytes, as a first guess. */
            const size_t bytes = ((stop_bit == SIZE_MAX ? job->in_size * 8 : stop_bit) - s->start_bit) / 8;
            const double share = 1.25 * ((double)bytes / (double)job->in_size);
            size_t capacity = (size_t)((double)job->out_size * share) + ((size_t)64 << 10);
            if (capacity > job->out_size) capacity = job->out_size;
            s->ok = decode_later_stretch(inflater, job, s, stop_bit, capacity);
        } else {
            /* The real start: the real output, in place, with no history to reach, as the serial path has it. */
            struct libdeflate_range range = {s->start_bit, stop_bit, 0, 0, 0, 0};
            s->ok = libdeflate_deflate_decompress_range(inflater, job->in, job->in_size, job->out, job->out_size,
                                                        &range) == LIBDEFLATE_SUCCESS;
            s->length = range.end_out;
            s->end_bit = range.end_bit;
            s->final = range.final;
        }
    }
    libdeflate_free_decompressor(inflater);
}

/* Whether the ends chain from the stream's real start to its real end, with the output's exact size, and every stretch
 * long enough to hold the next one's whole history. Sets each stretch's place in out. */
static int chain_holds(inflate_job *job) {
    size_t at = 0;
    for (size_t k = 0; k < job->count; ++k) {
        inflate_stretch *const s = &job->stretches[k];
        const int last = k + 1 == job->count;
        if (!s->ok || s->length < HISTORY || s->length > job->out_size - at) return 0;
        if (last ? !s->final || (s->end_bit + 7) / 8 != job->in_size || s->length != job->out_size - at
                 : s->final || s->end_bit != s[1].start_bit)
            return 0;
        s->at = at;
        at += s->length;
    }
    return 1;
}

/* A later stretch's last 32 KiB where they are plain bytes in its last part, or NULL: the first stretch's are in out,
 * and a stretch that ended in symbols has its own written there by place_tails. Either way no order is needed after. */
static const uint8_t *plain_tail(const inflate_stretch *s) {
    if (!s->part_count) return NULL;
    const inflate_part *const part = &s->parts[s->part_count - 1];
    return part->bytes ? part->bytes + part->count : NULL;
}

/* In runs: most of them hold no marker, and narrowing a run is what the compiler vectorizes. */
static void resolve(uint8_t *target, const uint16_t *symbols, size_t count, const uint8_t *history) {
    for (size_t run = 0; run < count; run += 64) {
        const size_t stop = count - run < 64 ? count : run + 64;
        unsigned seen = 0;
        for (size_t i = run; i < stop; ++i) {
            seen |= symbols[i];
            target[i] = (uint8_t)symbols[i];
        }
        if (seen < FIRST_MARKER) continue;
        for (size_t i = run; i < stop; ++i) {
            if (symbols[i] >= FIRST_MARKER) target[i] = history[(symbols[i] - FIRST_MARKER) & (HISTORY - 1)];
        }
    }
}

/* A later stretch's output bytes in [from, to), in their place in out. */
static void place(const inflate_job *job, const inflate_stretch *s, size_t from, size_t to) {
    const uint8_t *const tail = plain_tail(s - 1);
    const uint8_t *const history = tail ? tail : job->out + s->at - HISTORY;
    uint8_t *const target = job->out + s->at;
    size_t at = 0;
    for (size_t p = 0; p < s->part_count; ++p) {
        const inflate_part *const part = &s->parts[p];
        const size_t first = from > at ? from : at, stop = to < at + part->count ? to : at + part->count;
        if (first < stop && part->bytes) {
            memcpy(target + first, part->bytes + HISTORY + (first - at), stop - first);
        } else if (first < stop) {
            resolve(target + first, part->symbols + HISTORY + (first - at), stop - first, history);
        }
        at += part->count;
    }
}

/* The last HISTORY bytes of each stretch that ended still in symbols: in order, since each may hold markers of the one
 * before. */
static void place_tails(const inflate_job *job) {
    for (size_t k = 1; k < job->count; ++k) {
        const inflate_stretch *const s = &job->stretches[k];
        if (!plain_tail(s)) place(job, s, s->length - HISTORY, s->length);
    }
}

/* Each stretch's piece of the output, but the tails place_tails has written. */
static void place_range(void *context, size_t begin, size_t end) {
    inflate_job *const job = (inflate_job *)context;
    for (size_t k = begin; k < end; ++k) {
        inflate_stretch *const s = &job->stretches[k];
        if (k) place(job, s, 0, plain_tail(s) ? s->length : s->length - HISTORY);
        s->adler = libdeflate_adler32(1, job->out + s->at, s->length);
    }
}

/* Returning this much memory to the system takes long enough for threads to pay. A failed cn_parallel_for ran
 * nothing. */
static void free_range(void *context, size_t begin, size_t end) {
    inflate_job *const job = (inflate_job *)context;
    for (size_t k = begin; k < end; ++k) {
        for (size_t p = 0; p < job->stretches[k].part_count; ++p) {
            free(job->stretches[k].parts[p].symbols);
            free(job->stretches[k].parts[p].bytes);
        }
    }
}

int cn_png_inflate_parallel(const uint8_t *stream, size_t size, uint8_t *out, size_t out_size) {
    const unsigned capacity = cn_parallel_capacity();
    if (size < 6 + HEADER_MAX_BYTES) return 0;
    const size_t in_size = size - 6;
    size_t count = in_size / PNG_INFLATE_MIN_STRETCH_BYTES;
    if (count > capacity) count = capacity;
    if (count > MAX_STRETCHES) count = MAX_STRETCHES;
    /* The zlib header as libdeflate checks it. A stream that opens with anything but a non-final dynamic block is taken
       to go on so (level 0, incompressible data, Z_FIXED, one huge block): there is nothing to find in it. */
    if (count < MIN_STRETCHES || out_size > MAX_OUT_BYTES || out_size / MAX_RATIO > in_size ||
        (stream[0] << 8 | stream[1]) % 31 ||
        (stream[0] & 15) != 8 || stream[0] >> 4 > 7 || stream[1] & 32 || (stream[2] & 7) != 4)
        return 0;
    inflate_job job = {.in = stream + 2, .in_size = in_size, .out = out, .out_size = out_size, .count = count};
    atomic_init(&job.scratch, 0);
    if (cn_parallel_for(count, 1, search_range, &job) != CN_OK) return 0;
    job.count = 1;
    for (size_t k = 1; k < count; ++k) {
        if (job.stretches[k].start_bit != SIZE_MAX) job.stretches[job.count++].start_bit = job.stretches[k].start_bit;
    }
    if (job.count < 2) return 0;
    int ok = 0;
    if (cn_parallel_for(job.count, 1, decode_range, &job) == CN_OK && chain_holds(&job)) {
        place_tails(&job);
        if (cn_parallel_for(job.count, 1, place_range, &job) == CN_OK) {
            uint32_t adler = job.stretches[0].adler;
            for (size_t k = 1; k < job.count; ++k)
                adler = adler32_combine(adler, job.stretches[k].adler, job.stretches[k].length);
            const uint8_t *const end = stream + size;
            ok = adler == ((uint32_t)end[-4] << 24 | (uint32_t)end[-3] << 16 | (uint32_t)end[-2] << 8 | end[-1]);
        }
    }
    if (cn_parallel_for(job.count, 1, free_range, &job) != CN_OK) free_range(&job, 0, job.count);
    return ok;
}
