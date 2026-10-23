/* libFuzzer harness over the regex engine (GPL-3.0; README.md here). An input is a pattern, a 0xFF byte, then a text
 * (0xFF never occurs in UTF-8; inputs that are not UTF-8 are skipped, as Python callers never pass them). The pattern
 * is compiled under every engine setting of chn_regex_options, and each compiled regex runs regex-py's operations: a
 * search at every byte position up to one past the end, the capture and match iterators, both splits and the
 * PosTranslator over every span found. Each setting writes a transcript of its results (searches inside a character
 * or past the end apart); transcripts that must agree (LLVMFuzzerTestOneInput) and differ abort, which libFuzzer
 * reports as a crash, and AddressSanitizer reports the memory errors. */
#include "chainner_regex.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct transcript {
    unsigned char *data;
    size_t size, capacity;
} transcript;

static void put(transcript *t, const void *bytes, size_t count) {
    if (t->size + count > t->capacity) {
        size_t capacity = t->capacity ? t->capacity * 2 : 4096;
        while (capacity < t->size + count) capacity *= 2;
        unsigned char *grown = realloc(t->data, capacity);
        if (!grown) abort();
        t->data = grown;
        t->capacity = capacity;
    }
    memcpy(t->data + t->size, bytes, count);
    t->size += count;
}

static void put_size(transcript *t, size_t value) { put(t, &value, sizeof value); }

static void put_string(transcript *t, chn_regex_string *s) {
    put_size(t, s->len);
    put(t, s->ptr, s->len);
    chn_regex_string_drop(s);
}

/* Strict UTF-8 as Rust's str: no overlong forms, surrogates or values above U+10FFFF. */
static int utf8(const unsigned char *p, size_t n) {
    for (size_t i = 0; i < n;) {
        const unsigned char c = p[i];
        size_t length;
        unsigned int value;
        if (c < 0x80) {
            ++i;
            continue;
        }
        if (c >= 0xC2 && c <= 0xDF) length = 2, value = c & 0x1F;
        else if (c >= 0xE0 && c <= 0xEF) length = 3, value = c & 0x0F;
        else if (c >= 0xF0 && c <= 0xF4) length = 4, value = c & 0x07;
        else return 0;
        if (n - i < length) return 0;
        for (size_t k = 1; k < length; ++k) {
            if ((p[i + k] & 0xC0) != 0x80) return 0;
            value = value << 6 | (p[i + k] & 0x3F);
        }
        if ((length == 3 && (value < 0x800 || (value >= 0xD800 && value <= 0xDFFF))) ||
            (length == 4 && (value < 0x10000 || value > 0x10FFFF)))
            return 0;
        i += length;
    }
    return 1;
}

/* regex-py's PosTranslator over the spans a search or an iterator found, start then end. */
static void put_char_positions(transcript *t, const char *text, size_t text_len, const size_t *slots, size_t count) {
    chn_regex_pos_translator translator;
    chn_regex_pos_translator_init(&translator, text, text_len);
    for (size_t i = 0; i < count; ++i) {
        if (slots[i] == CHN_REGEX_NO_SLOT) continue;
        size_t position = 0;
        chn_regex_string panic = {NULL, 0};
        const chn_regex_status status = chn_regex_pos_translator_get_char_pos(&translator, slots[i], &position, &panic);
        put_size(t, (size_t)status);
        put_size(t, position);
        put_string(t, &panic);
    }
    chn_regex_pos_translator_drop(&translator);
}

static void put_spans(transcript *t, chn_regex_status status, chn_regex_spans *spans) {
    put_size(t, (size_t)status);
    put_size(t, spans->len);
    for (size_t i = 0; i < spans->len; ++i) {
        put_size(t, spans->ptr[i].start);
        put_size(t, spans->ptr[i].end);
    }
    chn_regex_spans_drop(spans);
}

static void run(const chn_regex_options *options, const char *pattern, size_t pattern_len, const char *text,
                size_t text_len, transcript *t, transcript *edge) {
    chn_regex *re = NULL;
    chn_regex_string error = {NULL, 0};
    const chn_regex_status compiled = chn_regex_compile(pattern, pattern_len, options, &re, &error);
    put_size(t, (size_t)compiled);
    put_string(t, &error);
    if (compiled != CHN_REGEX_OK) return;
    const size_t slots_len = 2 * chn_regex_captures_len(re);
    size_t *slots = malloc(slots_len * sizeof *slots);
    if (!slots) abort();
    put_size(t, chn_regex_groups(re));
    for (size_t i = 0; i < chn_regex_captures_len(re); ++i) {
        size_t length = 0;
        const char *name = chn_regex_capture_name(re, i, &length);
        put_size(t, name ? length : SIZE_MAX);
        if (name) put(t, name, length);
    }
    for (size_t position = 0; position <= text_len + 1; ++position) {
        /* A search inside a character or past the end goes to `edge`. */
        transcript *into =
            position < text_len ? ((text[position] & 0xC0) != 0x80 ? t : edge) : position == text_len ? t : edge;
        chn_regex_string panic = {NULL, 0};
        const chn_regex_status status = chn_regex_search_at(re, text, text_len, position, slots, slots_len, &panic);
        put_size(into, (size_t)status);
        put(into, slots, slots_len * sizeof *slots);
        put_string(into, &panic);
        if (status == CHN_REGEX_OK) put_char_positions(into, text, text_len, slots, slots_len);
    }
    chn_regex_iter it;
    chn_regex_iter_init(&it, re, text, text_len);
    for (size_t count = 0;; ++count) {
        if (count > text_len + 1) abort(); /* an iterator that does not advance */
        const chn_regex_status status = chn_regex_iter_next_captures(&it, slots, slots_len);
        put_size(t, (size_t)status);
        if (status != CHN_REGEX_OK) break;
        put(t, slots, slots_len * sizeof *slots);
        put_char_positions(t, text, text_len, slots, slots_len);
    }
    chn_regex_iter_init(&it, re, text, text_len);
    for (size_t count = 0;; ++count) {
        if (count > text_len + 1) abort();
        size_t start = 0, end = 0;
        const chn_regex_status status = chn_regex_iter_next(&it, &start, &end);
        put_size(t, (size_t)status);
        if (status != CHN_REGEX_OK) break;
        put_size(t, start);
        put_size(t, end);
    }
    chn_regex_spans spans = {NULL, 0, 0};
    put_spans(t, chn_regex_split(re, text, text_len, &spans), &spans);
    put_spans(t, chn_regex_split_without_captures(re, text, text_len, &spans), &spans);
    free(slots);
    chn_regex_free(re);
}

static void compare(const transcript *a, const transcript *b, const char *what) {
    if (a->size != b->size || memcmp(a->data, b->data, a->size) != 0) {
        fprintf(stderr, "fuzz_regex: %s\n", what);
        abort();
    }
}

/* Whether the pattern may hold a look-around assertion (^, $, \b, \B, \A, \z), where regex 1.8.4's NFA engines and
 * its DFA path may disagree (native/tests/chainner_ext/nfa_boundary_exclusions.json, mechanisms A and B). */
static int may_look(const char *pattern, size_t length) {
    for (size_t i = 0; i < length; ++i)
        if (pattern[i] == '^' || pattern[i] == '$' ||
            (pattern[i] == '\\' && i + 1 < length && strchr("bBAz", pattern[i + 1]) && pattern[i + 1]))
            return 1;
    return 0;
}

int LLVMFuzzerTestOneInput(const unsigned char *data, size_t size) {
    const unsigned char *separator = memchr(data, 0xFF, size);
    const size_t pattern_len = separator ? (size_t)(separator - data) : size;
    const char *pattern = (const char *)data, *text = separator ? (const char *)separator + 1 : "";
    const size_t text_len = separator ? size - pattern_len - 1 : 0;
    if (!utf8(data, pattern_len) || !utf8((const unsigned char *)text, text_len)) return 0;
    for (size_t position = 0; position <= text_len + 1; ++position) (void)chn_regex_to_byte_pos(text, text_len, position);
    /* The production setting, then the others, as chainner_ext_conformance.py compares them with the real module:
     * default (prefilter on or off) everywhere, dfa at character boundaries; nfa and backtrack must equal each other
     * everywhere, and production at character boundaries where the pattern has no look-around. */
    transcript production = {NULL, 0, 0}, production_edge = {NULL, 0, 0}, nfa = {NULL, 0, 0}, nfa_edge = {NULL, 0, 0};
    run(NULL, pattern, pattern_len, text, text_len, &production, &production_edge);
    for (int engine = CHN_REGEX_ENGINE_DEFAULT; engine <= CHN_REGEX_ENGINE_DFA; ++engine)
        for (int prefilter = 1; prefilter >= 0; --prefilter) {
            const chn_regex_options options = {(chn_regex_engine)engine, prefilter != 0};
            transcript other = {NULL, 0, 0}, other_edge = {NULL, 0, 0};
            run(&options, pattern, pattern_len, text, text_len, &other, &other_edge);
            if (engine == CHN_REGEX_ENGINE_DEFAULT) {
                compare(&production, &other, "a default setting differs from production");
                compare(&production_edge, &other_edge, "a default setting differs from production off boundaries");
            } else if (engine == CHN_REGEX_ENGINE_DFA) {
                compare(&production, &other, "a dfa setting differs from production");
            } else if (!nfa.data) {
                nfa = other;
                nfa_edge = other_edge;
                continue;
            } else {
                compare(&nfa, &other, "the backtracker or the PikeVM without prefilter differs from the PikeVM");
                compare(&nfa_edge, &other_edge, "the backtracker or the PikeVM differ off boundaries");
            }
            free(other.data);
            free(other_edge.data);
        }
    if (!may_look(pattern, pattern_len)) compare(&production, &nfa, "the PikeVM differs from production");
    free(production.data);
    free(production_edge.data);
    free(nfa.data);
    free(nfa_edge.data);
    return 0;
}
