/* Port of regex-syntax 0.7.2 src/utf8.rs, MIT OR Apache-2.0. */
/* Utf8Sequences: the alternation of byte-range sequences that matches exactly the UTF-8
 * encodings of a scalar value range. */
#include "chainner_hir.h"
#include <stdlib.h>

enum { MAX_UTF8_BYTES = 4 };

static chn_hir_status push(chn_hir_utf8_sequences *it, uint32_t start, uint32_t end) {
    if (it->len == it->cap) {
        const size_t cap = it->cap ? it->cap * 2 : 8;
        chn_hir_utf8_scalar_range *stack =
            (chn_hir_utf8_scalar_range *)realloc(it->range_stack, cap * sizeof(*stack));
        if (!stack) return CHN_HIR_ERROR_NOMEM;
        it->range_stack = stack;
        it->cap = cap;
    }
    it->range_stack[it->len].start = start;
    it->range_stack[it->len].end = end;
    it->len += 1;
    return CHN_HIR_OK;
}

chn_hir_status chn_hir_utf8_sequences_init(chn_hir_utf8_sequences *it, uint32_t start, uint32_t end) {
    it->range_stack = NULL;
    it->len = 0;
    it->cap = 0;
    return push(it, start, end);
}

chn_hir_status chn_hir_utf8_sequences_reset(chn_hir_utf8_sequences *it, uint32_t start, uint32_t end) {
    it->len = 0;
    return push(it, start, end);
}

static uint32_t max_scalar_value(int nbytes) {
    switch (nbytes) {
    case 1: return 0x007F;
    case 2: return 0x07FF;
    case 3: return 0xFFFF;
    default: return 0x0010FFFF;
    }
}

/* char::encode_utf8 */
static size_t encode(uint32_t c, uint8_t out[MAX_UTF8_BYTES]) {
    if (c < 0x80) {
        out[0] = (uint8_t)c;
        return 1;
    }
    if (c < 0x800) {
        out[0] = (uint8_t)(0xC0 | (c >> 6));
        out[1] = (uint8_t)(0x80 | (c & 0x3F));
        return 2;
    }
    if (c < 0x10000) {
        out[0] = (uint8_t)(0xE0 | (c >> 12));
        out[1] = (uint8_t)(0x80 | ((c >> 6) & 0x3F));
        out[2] = (uint8_t)(0x80 | (c & 0x3F));
        return 3;
    }
    out[0] = (uint8_t)(0xF0 | (c >> 18));
    out[1] = (uint8_t)(0x80 | ((c >> 12) & 0x3F));
    out[2] = (uint8_t)(0x80 | ((c >> 6) & 0x3F));
    out[3] = (uint8_t)(0x80 | (c & 0x3F));
    return 4;
}

int chn_hir_utf8_sequences_next(chn_hir_utf8_sequences *it, chn_hir_utf8_sequence *out) {
    while (it->len > 0) {
        chn_hir_utf8_scalar_range r = it->range_stack[--it->len];
        for (;;) {
            /* ScalarRange::split */
            if (r.start < 0xE000 && r.end > 0xD7FF) {
                if (push(it, 0xE000, r.end) != CHN_HIR_OK) return -1;
                r.end = 0xD7FF;
                continue;
            }
            if (r.start > r.end) break; /* !is_valid: continue 'TOP */
            bool split = false;
            for (int i = 1; i < MAX_UTF8_BYTES; ++i) {
                const uint32_t max = max_scalar_value(i);
                if (r.start <= max && max < r.end) {
                    if (push(it, max + 1, r.end) != CHN_HIR_OK) return -1;
                    r.end = max;
                    split = true;
                    break;
                }
            }
            if (split) continue;
            if (r.end <= 0x7F) { /* as_ascii */
                out->len = 1;
                out->ranges[0].start = (uint8_t)r.start;
                out->ranges[0].end = (uint8_t)r.end;
                return 1;
            }
            for (int i = 1; i < MAX_UTF8_BYTES; ++i) {
                const uint32_t m = (1u << (6 * i)) - 1;
                if ((r.start & ~m) != (r.end & ~m)) {
                    if ((r.start & m) != 0) {
                        if (push(it, (r.start | m) + 1, r.end) != CHN_HIR_OK) return -1;
                        r.end = r.start | m;
                        split = true;
                        break;
                    }
                    if ((r.end & m) != m) {
                        if (push(it, r.end & ~m, r.end) != CHN_HIR_OK) return -1;
                        r.end = (r.end & ~m) - 1;
                        split = true;
                        break;
                    }
                }
            }
            if (split) continue;
            uint8_t start[MAX_UTF8_BYTES], end[MAX_UTF8_BYTES];
            const size_t n = encode(r.start, start);
            encode(r.end, end);
            out->len = n;
            for (size_t i = 0; i < n; ++i) {
                out->ranges[i].start = start[i];
                out->ranges[i].end = end[i];
            }
            return 1;
        }
    }
    return 0;
}

void chn_hir_utf8_sequences_drop(chn_hir_utf8_sequences *it) {
    free(it->range_stack);
    it->range_stack = NULL;
    it->len = 0;
    it->cap = 0;
}
