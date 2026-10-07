/* Port of regex 1.8.4 src/input.rs and src/utf8.rs (the UTF-8 decoders), MIT OR Apache-2.0. */
#include "prog.h"

/* utf8.rs next_utf8 */
size_t chn_next_utf8(const uint8_t *text, size_t len, size_t i) {
    if (i >= len) return i + 1;
    const uint8_t b = text[i];
    size_t inc;
    if (b <= 0x7F) inc = 1;
    else if (b <= 0xDF) inc = 2;
    else if (b <= 0xEF) inc = 3;
    else inc = 4;
    return i + inc;
}

/* utf8.rs decode_utf8 */
uint32_t chn_decode_utf8(const uint8_t *src, size_t len, size_t *n) {
    if (len == 0) return CHN_CHAR_NONE;
    const uint8_t b0 = src[0];
    if (b0 <= 0x7F) {
        *n = 1;
        return b0;
    }
    if (b0 >= 0xC0 && b0 <= 0xDF) {
        if (len < 2) return CHN_CHAR_NONE;
        const uint8_t b1 = src[1];
        if ((b1 & 0xC0) != 0x80) return CHN_CHAR_NONE;
        const uint32_t cp = ((uint32_t)(b0 & 0x1F) << 6) | (uint32_t)(b1 & 0x3F);
        if (cp < 0x80 || cp > 0x7FF) return CHN_CHAR_NONE;
        *n = 2;
        return cp;
    }
    if (b0 >= 0xE0 && b0 <= 0xEF) {
        if (len < 3) return CHN_CHAR_NONE;
        const uint8_t b1 = src[1], b2 = src[2];
        if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80) return CHN_CHAR_NONE;
        const uint32_t cp = ((uint32_t)(b0 & 0x0F) << 12) | ((uint32_t)(b1 & 0x3F) << 6) | (uint32_t)(b2 & 0x3F);
        if (cp < 0x800 || cp > 0xFFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return CHN_CHAR_NONE;
        *n = 3;
        return cp;
    }
    if (b0 >= 0xF0 && b0 <= 0xF7) {
        if (len < 4) return CHN_CHAR_NONE;
        const uint8_t b1 = src[1], b2 = src[2], b3 = src[3];
        if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80) return CHN_CHAR_NONE;
        const uint32_t cp = ((uint32_t)(b0 & 0x07) << 18) | ((uint32_t)(b1 & 0x3F) << 12)
            | ((uint32_t)(b2 & 0x3F) << 6) | (uint32_t)(b3 & 0x3F);
        if (cp < 0x10000 || cp > 0x10FFFF) return CHN_CHAR_NONE;
        *n = 4;
        return cp;
    }
    return CHN_CHAR_NONE;
}

/* utf8.rs is_start_byte */
static bool is_start_byte(uint8_t b) {
    return (b & 0xC0) != 0x80;
}

/* utf8.rs decode_last_utf8 */
uint32_t chn_decode_last_utf8(const uint8_t *src, size_t len) {
    if (len == 0) return CHN_CHAR_NONE;
    size_t start = len - 1;
    if (src[start] <= 0x7F) return src[start];
    const size_t limit = len >= 4 ? len - 4 : 0;
    while (start > limit) {
        --start;
        if (is_start_byte(src[start])) break;
    }
    size_t n = 0;
    const uint32_t cp = chn_decode_utf8(src + start, len - start, &n);
    if (cp == CHN_CHAR_NONE || n < len - start) return CHN_CHAR_NONE;
    return cp;
}

/* Char::len_utf8 (1 for an absent char) */
static size_t char_len_utf8(uint32_t c) {
    if (c < 0x80 || c == CHN_CHAR_NONE) return 1;
    if (c < 0x800) return 2;
    if (c < 0x10000) return 3;
    return 4;
}

chn_input_at chn_input_at_pos(const chn_input *input, size_t i) {
    chn_input_at at;
    if (i >= input->len) {
        at.pos = input->len;
        at.c = CHN_CHAR_NONE;
        at.byte = -1;
        at.len = 0;
        return at;
    }
    at.pos = i;
    if (input->bytes) {
        /* ByteInput::at */
        at.c = CHN_CHAR_NONE;
        at.byte = input->text[i];
        at.len = 1;
    } else {
        /* CharInput::at */
        const uint8_t b = input->text[i];
        size_t n = 1;
        at.c = b <= 0x7F ? b : chn_decode_utf8(input->text + i, input->len - i, &n);
        at.byte = -1;
        at.len = char_len_utf8(at.c);
    }
    return at;
}

/* Input::next_char */
static uint32_t next_char(const chn_input *input, chn_input_at at) {
    if (!input->bytes) return at.c;
    size_t n;
    return chn_decode_utf8(input->text + at.pos, input->len - at.pos, &n);
}

/* Input::previous_char */
static uint32_t previous_char(const chn_input *input, chn_input_at at) {
    return chn_decode_last_utf8(input->text, at.pos);
}

/* Char::is_word_char */
static bool is_word_char(uint32_t c) {
    return c != CHN_CHAR_NONE && chn_hir_is_word_character(c);
}

/* Char::is_word_byte */
static bool is_word_byte(uint32_t c) {
    return c != CHN_CHAR_NONE && c <= 0x7F && chn_hir_is_word_byte((uint8_t)c);
}

bool chn_input_is_empty_match(const chn_input *input, chn_input_at at, chn_empty_look look) {
    switch (look) {
    case CHN_LOOK_START_LINE: {
        const uint32_t c = previous_char(input, at);
        return at.pos == 0 || c == '\n';
    }
    case CHN_LOOK_END_LINE: {
        const uint32_t c = next_char(input, at);
        return at.pos == input->len || c == '\n';
    }
    case CHN_LOOK_START_TEXT:
        return at.pos == 0;
    case CHN_LOOK_END_TEXT:
        return at.pos == input->len;
    case CHN_LOOK_WORD_BOUNDARY:
        return is_word_char(previous_char(input, at)) != is_word_char(next_char(input, at));
    case CHN_LOOK_NOT_WORD_BOUNDARY:
        return is_word_char(previous_char(input, at)) == is_word_char(next_char(input, at));
    case CHN_LOOK_WORD_BOUNDARY_ASCII:
    case CHN_LOOK_NOT_WORD_BOUNDARY_ASCII: {
        const uint32_t c1 = previous_char(input, at), c2 = next_char(input, at);
        if (input->bytes && input->only_utf8) {
            /* ByteInput: no word boundary at invalid UTF-8. at.is_end() is c and byte
             * both absent. */
            if (c1 == CHN_CHAR_NONE && at.pos != 0) return false;
            if (c2 == CHN_CHAR_NONE && !(at.c == CHN_CHAR_NONE && at.byte < 0)) return false;
        }
        const bool differ = is_word_byte(c1) != is_word_byte(c2);
        return look == CHN_LOOK_WORD_BOUNDARY_ASCII ? differ : !differ;
    }
    }
    return false;
}

bool chn_input_prefix_at(const chn_input *input, const chn_literal_searcher *prefixes,
    chn_input_at at, chn_input_at *out) {
    size_t s, e;
    if (!chn_literal_searcher_find(prefixes, input->text + at.pos, input->len - at.pos, &s, &e)) return false;
    *out = chn_input_at_pos(input, at.pos + s);
    return true;
}
