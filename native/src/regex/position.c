/* Port of chainner_ext 0.3.10 crates/regex-py/src/position.rs (with core::str's slice_error_fail message), MIT OR Apache-2.0. */
#include "chainner_regex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool is_char_boundary(const char *text, size_t len, size_t i) {
    return i == 0 || i >= len || ((unsigned char)text[i] & 0xC0) != 0x80;
}

size_t chn_regex_to_byte_pos(const char *text, size_t text_len, size_t char_pos) {
    if (char_pos == 0) return 0;
    if (char_pos > text_len) return char_pos;
    /* text.char_indices().nth(char_pos), counting chars on the way. */
    size_t count = 0;
    for (size_t i = 0; i < text_len; ++i) {
        if (is_char_boundary(text, text_len, i)) {
            if (count == char_pos) return i;
            ++count;
        }
    }
    /* Out of bounds: char_pos + text.chars().count() - text.len(), wrapping as the
     * release build does. */
    return char_pos + count - text_len;
}

void chn_regex_pos_translator_init(chn_regex_pos_translator *t, const char *text, size_t text_len) {
    t->text = text;
    t->text_len = text_len;
    t->known = NULL;
    t->known_len = 0;
    t->known_cap = 0;
}

/* char's Debug output for the panic message, simplified: the char inside single quotes,
 * with \u{..} for control characters. The message is reached only from a search started
 * inside a character, whose positions never pass a known mid-character position. */
static size_t char_debug(const char *text, size_t start, size_t n, char *out) {
    uint32_t cp = (unsigned char)text[start];
    if (n == 2) cp &= 0x1F;
    else if (n == 3) cp &= 0x0F;
    else if (n == 4) cp &= 0x07;
    for (size_t k = 1; k < n; ++k) cp = (cp << 6) | ((unsigned char)text[start + k] & 0x3F);
    if (cp < 0x20 || (cp >= 0x7F && cp < 0xA0)) {
        return (size_t)snprintf(out, 16, "'\\u{%x}'", (unsigned)cp);
    }
    out[0] = '\'';
    memcpy(out + 1, text + start, n);
    out[n + 1] = '\'';
    return n + 2;
}

/* core::str::slice_error_fail for `&text[begin..]` at a non-boundary begin. */
static chn_regex_status slice_panic(const chn_regex_pos_translator *t, size_t begin, chn_regex_string *panic) {
    size_t trunc = t->text_len < 256 ? t->text_len : 256;
    while (!is_char_boundary(t->text, t->text_len, trunc)) --trunc;
    size_t char_start = begin;
    while (!is_char_boundary(t->text, t->text_len, char_start)) --char_start;
    const unsigned char lead = (unsigned char)t->text[char_start];
    const size_t n = lead < 0x80 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;
    char ch[24];
    const size_t ch_len = char_debug(t->text, char_start, n, ch);
    char head[160];
    const int head_len = snprintf(head, sizeof(head), "byte index %llu is not a char boundary; it is inside ",
        (unsigned long long)begin);
    char mid[96];
    const int mid_len = snprintf(mid, sizeof(mid), " (bytes %llu..%llu) of `", (unsigned long long)char_start,
        (unsigned long long)(char_start + n));
    const char *tail = trunc < t->text_len ? "`[...]" : "`";
    const size_t tail_len = strlen(tail);
    if (head_len < 0 || mid_len < 0) return CHN_REGEX_ERROR_NOMEM;
    const size_t total = (size_t)head_len + ch_len + (size_t)mid_len + trunc + tail_len;
    char *msg = (char *)malloc(total + 1);
    if (!msg) return CHN_REGEX_ERROR_NOMEM;
    size_t w = 0;
    memcpy(msg + w, head, (size_t)head_len);
    w += (size_t)head_len;
    memcpy(msg + w, ch, ch_len);
    w += ch_len;
    memcpy(msg + w, mid, (size_t)mid_len);
    w += (size_t)mid_len;
    memcpy(msg + w, t->text, trunc);
    w += trunc;
    memcpy(msg + w, tail, tail_len + 1);
    panic->ptr = msg;
    panic->len = total;
    return CHN_REGEX_PANIC;
}

chn_regex_status chn_regex_pos_translator_get_char_pos(chn_regex_pos_translator *t,
    size_t byte_pos, size_t *char_pos, chn_regex_string *panic) {
    panic->ptr = NULL;
    panic->len = 0;
    *char_pos = 0;
    size_t start_byte = 0, start_char = 0;
    for (size_t k = t->known_len; k-- > 0;) {
        if (t->known[k].byte_pos <= byte_pos) {
            start_byte = t->known[k].byte_pos;
            start_char = t->known[k].char_pos;
            break;
        }
    }
    if (start_byte == byte_pos) {
        *char_pos = start_char;
        return CHN_REGEX_OK;
    }
    /* &self.text[start_offset.0..] */
    if (!is_char_boundary(t->text, t->text_len, start_byte)) return slice_panic(t, start_byte, panic);
    size_t pos = start_char;
    for (size_t i = start_byte; i < t->text_len; ++i) {
        if (!is_char_boundary(t->text, t->text_len, i)) continue;
        if (i >= byte_pos) break;
        ++pos;
    }
    if (t->known_len == 0 || t->known[t->known_len - 1].byte_pos < byte_pos) {
        if (t->known_len == t->known_cap) {
            const size_t cap = t->known_cap ? t->known_cap * 2 : 8;
            chn_regex_pos_known *known = (chn_regex_pos_known *)realloc(t->known, cap * sizeof(*known));
            if (!known) return CHN_REGEX_ERROR_NOMEM;
            t->known = known;
            t->known_cap = cap;
        }
        t->known[t->known_len].byte_pos = byte_pos;
        t->known[t->known_len].char_pos = pos;
        ++t->known_len;
    }
    *char_pos = pos;
    return CHN_REGEX_OK;
}

void chn_regex_pos_translator_drop(chn_regex_pos_translator *t) {
    free(t->known);
    t->known = NULL;
    t->known_len = 0;
    t->known_cap = 0;
}
