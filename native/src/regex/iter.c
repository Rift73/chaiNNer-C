/* Port of regex 1.8.4 src/re_trait.rs (Matches, CaptureMatches) and src/re_unicode.rs (Split), and chainner_ext 0.3.10 crates/regex-py/src/lib.rs (split, split_without_captures), MIT OR Apache-2.0. */
#include "prog.h"
#include <stdlib.h>

void chn_regex_iter_init(chn_regex_iter *it, const chn_regex *re, const char *text, size_t text_len) {
    it->re = re;
    it->text = text;
    it->text_len = text_len;
    it->last_end = 0;
    it->last_match = CHN_REGEX_NO_SLOT;
}

/* The search position never passes the text end here (Matches::next returns None first),
 * so no search panics; a panic status is still passed on rather than dropped. */
static chn_regex_status drop_panic(chn_regex_status st, chn_regex_string *panic) {
    chn_regex_string_drop(panic);
    return st;
}

chn_regex_status chn_regex_iter_next(chn_regex_iter *it, size_t *start, size_t *end) {
    const uint8_t *text = (const uint8_t *)it->text;
    for (;;) {
        *start = *end = CHN_REGEX_NO_SLOT;
        if (it->last_end > it->text_len) return CHN_REGEX_NO_MATCH;
        chn_regex_string panic;
        size_t s, e;
        const chn_regex_status st = chn_exec_find_at(it->re, text, it->text_len, it->last_end, &s, &e, &panic);
        if (st != CHN_REGEX_OK) return drop_panic(st, &panic);
        if (s == e) {
            /* An empty match: progress by next_after_empty, and never directly after a
             * match. */
            it->last_end = chn_next_utf8(text, it->text_len, e);
            if (e == it->last_match) continue;
        } else {
            it->last_end = e;
        }
        it->last_match = e;
        *start = s;
        *end = e;
        return CHN_REGEX_OK;
    }
}

chn_regex_status chn_regex_iter_next_captures(chn_regex_iter *it, size_t *slots, size_t slots_len) {
    const uint8_t *text = (const uint8_t *)it->text;
    for (;;) {
        if (it->last_end > it->text_len) {
            for (size_t i = 0; i < slots_len; ++i) slots[i] = CHN_REGEX_NO_SLOT;
            return CHN_REGEX_NO_MATCH;
        }
        chn_regex_string panic;
        const chn_regex_status st = chn_exec_captures_read_at(it->re, slots, slots_len, text, it->text_len,
            it->last_end, &panic);
        if (st != CHN_REGEX_OK) return drop_panic(st, &panic);
        const size_t s = slots[0], e = slots[1];
        if (s == e) {
            it->last_end = chn_next_utf8(text, it->text_len, e);
            if (e == it->last_match) continue;
        } else {
            it->last_end = e;
        }
        it->last_match = e;
        return CHN_REGEX_OK;
    }
}

void chn_regex_spans_drop(chn_regex_spans *spans) {
    free(spans->ptr);
    spans->ptr = NULL;
    spans->len = 0;
    spans->cap = 0;
}

static bool spans_push(chn_regex_spans *spans, size_t start, size_t end) {
    if (spans->len == spans->cap) {
        const size_t cap = spans->cap ? spans->cap * 2 : 8;
        chn_regex_span *ptr = (chn_regex_span *)realloc(spans->ptr, cap * sizeof(*ptr));
        if (!ptr) return false;
        spans->ptr = ptr;
        spans->cap = cap;
    }
    spans->ptr[spans->len].start = start;
    spans->ptr[spans->len].end = end;
    ++spans->len;
    return true;
}

chn_regex_status chn_regex_split(const chn_regex *re, const char *text, size_t text_len,
    chn_regex_spans *out) {
    out->ptr = NULL;
    out->len = out->cap = 0;
    const size_t nslots = 2 * chn_regex_captures_len(re);
    size_t *slots = (size_t *)malloc(nslots * sizeof(size_t));
    if (!slots) return CHN_REGEX_ERROR_NOMEM;
    chn_regex_iter it;
    chn_regex_iter_init(&it, re, text, text_len);
    size_t last = 0;
    chn_regex_status st = CHN_REGEX_NO_MATCH;
    bool ok = true;
    while (ok && (st = chn_regex_iter_next_captures(&it, slots, nslots)) == CHN_REGEX_OK) {
        const size_t start = slots[0], end = slots[1];
        if (start > last) ok = spans_push(out, last, start);
        last = end;
        for (size_t g = 1; ok && 2 * g < nslots; ++g) {
            if (slots[2 * g] != CHN_REGEX_NO_SLOT && slots[2 * g + 1] != CHN_REGEX_NO_SLOT) {
                ok = spans_push(out, slots[2 * g], slots[2 * g + 1]);
            }
        }
    }
    free(slots);
    if (ok && st != CHN_REGEX_NO_MATCH) {
        chn_regex_spans_drop(out);
        return st;
    }
    if (ok && last < text_len) ok = spans_push(out, last, text_len);
    if (!ok) {
        chn_regex_spans_drop(out);
        return CHN_REGEX_ERROR_NOMEM;
    }
    return CHN_REGEX_OK;
}

chn_regex_status chn_regex_split_without_captures(const chn_regex *re, const char *text,
    size_t text_len, chn_regex_spans *out) {
    out->ptr = NULL;
    out->len = out->cap = 0;
    chn_regex_iter it;
    chn_regex_iter_init(&it, re, text, text_len);
    size_t last = 0;
    for (;;) {
        size_t s, e;
        const chn_regex_status st = chn_regex_iter_next(&it, &s, &e);
        if (st == CHN_REGEX_NO_MATCH) {
            /* Split::next's final piece (last <= text.len() here). */
            if (!spans_push(out, last, text_len)) break;
            return CHN_REGEX_OK;
        }
        if (st != CHN_REGEX_OK) {
            chn_regex_spans_drop(out);
            return st;
        }
        if (!spans_push(out, last, s)) break;
        last = e;
    }
    chn_regex_spans_drop(out);
    return CHN_REGEX_ERROR_NOMEM;
}
