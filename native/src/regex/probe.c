/* chaiNNer-C: regex_probe, the JSON-lines driver for chainner_regex that X4's harness runs.
 * The I/O format is fixed in chainner_regex.h's header comment; this file composes the
 * engine exactly as regex-py does (to_byte_pos, captures_at, PosTranslator). */
#include "chainner_regex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

enum { PROBE_EXIT_FAILURE = 1, PROBE_EXIT_MALFORMED = 2, PROBE_EXIT_STUB = 3 };

/* A growable byte buffer, kept NUL-terminated. */
typedef struct probe_buf {
    char *ptr;
    size_t len;
    size_t cap;
} probe_buf;

static size_t probe_line_number;

static _Noreturn void probe_fail(const char *what) {
    fprintf(stderr, "regex_probe: %s\n", what);
    exit(PROBE_EXIT_FAILURE);
}

static _Noreturn void malformed(const char *what) {
    fprintf(stderr, "regex_probe: malformed input on line %llu: %s\n",
        (unsigned long long)probe_line_number, what);
    exit(PROBE_EXIT_MALFORMED);
}

static void buf_reserve(probe_buf *b, size_t extra) {
    if (b->cap - b->len >= extra) return;
    size_t cap = b->cap ? b->cap : 256;
    while (cap - b->len < extra) {
        if (cap > SIZE_MAX / 2) probe_fail("out of memory");
        cap *= 2;
    }
    char *ptr = (char *)realloc(b->ptr, cap);
    if (!ptr) probe_fail("out of memory");
    b->ptr = ptr;
    b->cap = cap;
}

static void buf_append(probe_buf *b, const char *data, size_t len) {
    buf_reserve(b, len + 1);
    if (len) memcpy(b->ptr + b->len, data, len);
    b->len += len;
    b->ptr[b->len] = '\0';
}

static void buf_clear(probe_buf *b) {
    buf_reserve(b, 1);
    b->len = 0;
    b->ptr[0] = '\0';
}

static void buf_puts(probe_buf *b, const char *s) {
    buf_append(b, s, strlen(s));
}

static void buf_putc(probe_buf *b, char c) {
    buf_append(b, &c, 1);
}

static void buf_put_size(probe_buf *b, size_t value) {
    char digits[24];
    const int n = snprintf(digits, sizeof(digits), "%llu", (unsigned long long)value);
    buf_append(b, digits, (size_t)n);
}

static void buf_put_json_string(probe_buf *b, const char *s, size_t len) {
    static const char hex[] = "0123456789abcdef";
    buf_putc(b, '"');
    for (size_t i = 0; i < len; ++i) {
        const unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"': buf_puts(b, "\\\""); break;
        case '\\': buf_puts(b, "\\\\"); break;
        case '\n': buf_puts(b, "\\n"); break;
        case '\r': buf_puts(b, "\\r"); break;
        case '\t': buf_puts(b, "\\t"); break;
        case '\b': buf_puts(b, "\\b"); break;
        case '\f': buf_puts(b, "\\f"); break;
        default:
            if (c < 0x20) {
                const char escape[6] = {'\\', 'u', '0', '0', hex[c >> 4], hex[c & 15]};
                buf_append(b, escape, sizeof(escape));
            } else {
                buf_putc(b, (char)c);
            }
        }
    }
    buf_putc(b, '"');
}

/* Line input: whole lines of any length, "\n" or "\r\n" ended. */
typedef struct probe_reader {
    FILE *file;
    char chunk[1 << 16];
    size_t pos, len;
    bool eof;
} probe_reader;

static bool read_line(probe_reader *r, probe_buf *line) {
    bool any = false;
    buf_clear(line);
    for (;;) {
        if (r->pos == r->len) {
            if (r->eof) break;
            r->len = fread(r->chunk, 1, sizeof(r->chunk), r->file);
            r->pos = 0;
            if (r->len == 0) {
                if (ferror(r->file)) probe_fail("cannot read stdin");
                r->eof = true;
                break;
            }
        }
        any = true;
        const char *start = r->chunk + r->pos;
        const char *newline = (const char *)memchr(start, '\n', r->len - r->pos);
        const size_t n = newline ? (size_t)(newline - start) : r->len - r->pos;
        buf_append(line, start, n);
        r->pos += n + (newline ? 1 : 0);
        if (newline) break;
    }
    if (line->len && line->ptr[line->len - 1] == '\r') line->ptr[--line->len] = '\0';
    return any;
}

/* Strict UTF-8 (no overlongs, surrogates or values above U+10FFFF), as Rust's &str. */
static bool utf8_valid(const char *s, size_t len) {
    const unsigned char *p = (const unsigned char *)s, *end = p + len;
    while (p < end) {
        const unsigned char c = *p;
        size_t n;
        uint32_t cp;
        if (c < 0x80) { ++p; continue; }
        if (c >= 0xC2 && c <= 0xDF) { n = 1; cp = c & 0x1Fu; }
        else if (c >= 0xE0 && c <= 0xEF) { n = 2; cp = c & 0x0Fu; }
        else if (c >= 0xF0 && c <= 0xF4) { n = 3; cp = c & 0x07u; }
        else return false;
        if ((size_t)(end - p) <= n) return false;
        for (size_t i = 1; i <= n; ++i) {
            if ((p[i] & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (p[i] & 0x3Fu);
        }
        if ((n == 2 && cp < 0x800) || (n == 3 && cp < 0x10000) || cp > 0x10FFFF
            || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
        p += n + 1;
    }
    return true;
}

static void utf8_put(probe_buf *b, uint32_t cp) {
    char bytes[4];
    size_t n;
    if (cp < 0x80) { bytes[0] = (char)cp; n = 1; }
    else if (cp < 0x800) {
        bytes[0] = (char)(0xC0 | (cp >> 6));
        bytes[1] = (char)(0x80 | (cp & 0x3F));
        n = 2;
    } else if (cp < 0x10000) {
        bytes[0] = (char)(0xE0 | (cp >> 12));
        bytes[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        bytes[2] = (char)(0x80 | (cp & 0x3F));
        n = 3;
    } else {
        bytes[0] = (char)(0xF0 | (cp >> 18));
        bytes[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        bytes[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        bytes[3] = (char)(0x80 | (cp & 0x3F));
        n = 4;
    }
    buf_append(b, bytes, n);
}

/* A strict JSON reader over one line. */
typedef struct json_in {
    const char *p;
    const char *end;
} json_in;

static void skip_ws(json_in *in) {
    while (in->p < in->end && (*in->p == ' ' || *in->p == '\t' || *in->p == '\n' || *in->p == '\r')) ++in->p;
}

static bool peek_is(json_in *in, char c) {
    skip_ws(in);
    return in->p < in->end && *in->p == c;
}

static void expect(json_in *in, char c) {
    if (!peek_is(in, c)) {
        char what[32];
        snprintf(what, sizeof(what), "expected '%c'", c);
        malformed(what);
    }
    ++in->p;
}

static uint32_t hex4(json_in *in) {
    if (in->end - in->p < 4) malformed("short \\u escape");
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        const char c = *in->p++;
        uint32_t digit;
        if (c >= '0' && c <= '9') digit = (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') digit = (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') digit = (uint32_t)(c - 'A' + 10);
        else malformed("bad \\u escape");
        value = (value << 4) | digit;
    }
    return value;
}

static void parse_string(json_in *in, probe_buf *out) {
    buf_clear(out);
    expect(in, '"');
    for (;;) {
        if (in->p >= in->end) malformed("unterminated string");
        const unsigned char c = (unsigned char)*in->p++;
        if (c == '"') break;
        if (c < 0x20) malformed("control character in a string");
        if (c != '\\') {
            buf_putc(out, (char)c);
            continue;
        }
        if (in->p >= in->end) malformed("unterminated escape");
        const char e = *in->p++;
        switch (e) {
        case '"': case '\\': case '/': buf_putc(out, e); break;
        case 'b': buf_putc(out, '\b'); break;
        case 'f': buf_putc(out, '\f'); break;
        case 'n': buf_putc(out, '\n'); break;
        case 'r': buf_putc(out, '\r'); break;
        case 't': buf_putc(out, '\t'); break;
        case 'u': {
            uint32_t cp = hex4(in);
            if (cp >= 0xD800 && cp <= 0xDBFF) {
                if (in->end - in->p < 2 || in->p[0] != '\\' || in->p[1] != 'u') malformed("lone surrogate escape");
                in->p += 2;
                const uint32_t low = hex4(in);
                if (low < 0xDC00 || low > 0xDFFF) malformed("lone surrogate escape");
                cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
            } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                malformed("lone surrogate escape");
            }
            utf8_put(out, cp);
            break;
        }
        default: malformed("bad escape");
        }
    }
    if (!utf8_valid(out->ptr, out->len)) malformed("a string is not valid UTF-8");
}

/* A non-negative JSON integer that fits in size_t. */
static size_t parse_size(json_in *in) {
    skip_ws(in);
    if (in->p >= in->end || *in->p < '0' || *in->p > '9') malformed("pos is not a non-negative integer");
    if (*in->p == '0' && in->p + 1 < in->end && in->p[1] >= '0' && in->p[1] <= '9') malformed("leading zero");
    size_t value = 0;
    while (in->p < in->end && *in->p >= '0' && *in->p <= '9') {
        const size_t digit = (size_t)(*in->p++ - '0');
        if (value > (SIZE_MAX - digit) / 10) malformed("pos out of range");
        value = value * 10 + digit;
    }
    if (in->p < in->end && (*in->p == '.' || *in->p == 'e' || *in->p == 'E')) malformed("pos is not an integer");
    return value;
}

static void skip_value(json_in *in, int depth);

static void skip_members(json_in *in, char close, int depth) {
    if (peek_is(in, close)) {
        ++in->p;
        return;
    }
    probe_buf key = {0};
    for (;;) {
        if (close == '}') {
            parse_string(in, &key);
            expect(in, ':');
        }
        skip_value(in, depth + 1);
        if (peek_is(in, ',')) {
            ++in->p;
            continue;
        }
        expect(in, close);
        break;
    }
    free(key.ptr);
}

static void skip_value(json_in *in, int depth) {
    if (depth > 64) malformed("nesting too deep");
    skip_ws(in);
    if (in->p >= in->end) malformed("missing value");
    const char c = *in->p;
    if (c == '"') {
        probe_buf text = {0};
        parse_string(in, &text);
        free(text.ptr);
    } else if (c == '{' || c == '[') {
        ++in->p;
        skip_members(in, c == '{' ? '}' : ']', depth);
    } else if (c == '-' || (c >= '0' && c <= '9')) {
        const char *start = in->p;
        while (in->p < in->end && *in->p && strchr("+-0123456789.eE", *in->p)) ++in->p;
        if (in->p == start) malformed("bad number");
    } else {
        static const char *const words[] = {"true", "false", "null"};
        for (size_t i = 0; i < 3; ++i) {
            const size_t n = strlen(words[i]);
            if ((size_t)(in->end - in->p) >= n && memcmp(in->p, words[i], n) == 0) {
                in->p += n;
                return;
            }
        }
        malformed("bad value");
    }
}

typedef enum probe_op_kind {
    PROBE_COMPILE,
    PROBE_SEARCH,
    PROBE_FINDALL,
    PROBE_SPLIT,
    PROBE_SPLIT_WITHOUT_CAPTURES
} probe_op_kind;

typedef struct probe_op {
    probe_op_kind kind;
    bool has_text;
    probe_buf text;
    size_t pos; /* search's pos; 0 when absent, as RustRegex.search's default */
} probe_op;

typedef struct probe_case {
    probe_buf id, pattern;
    bool has_id, has_pattern, has_ops;
    probe_op *ops;
    size_t ops_len, ops_cap;
} probe_case;

static void parse_op(json_in *in, probe_op *op) {
    probe_buf key = {0}, name = {0};
    bool has_kind = false;
    op->has_text = false;
    op->pos = 0;
    expect(in, '{');
    if (!peek_is(in, '}')) {
        for (;;) {
            parse_string(in, &key);
            expect(in, ':');
            if (strcmp(key.ptr, "op") == 0) {
                parse_string(in, &name);
                has_kind = true;
                if (strcmp(name.ptr, "compile") == 0) op->kind = PROBE_COMPILE;
                else if (strcmp(name.ptr, "search") == 0) op->kind = PROBE_SEARCH;
                else if (strcmp(name.ptr, "findall") == 0) op->kind = PROBE_FINDALL;
                else if (strcmp(name.ptr, "split") == 0) op->kind = PROBE_SPLIT;
                else if (strcmp(name.ptr, "split_without_captures") == 0) op->kind = PROBE_SPLIT_WITHOUT_CAPTURES;
                else malformed("unknown op");
            } else if (strcmp(key.ptr, "text") == 0) {
                parse_string(in, &op->text);
                op->has_text = true;
            } else if (strcmp(key.ptr, "pos") == 0) {
                op->pos = parse_size(in);
            } else {
                skip_value(in, 1);
            }
            if (peek_is(in, ',')) {
                ++in->p;
                continue;
            }
            break;
        }
    }
    expect(in, '}');
    if (!has_kind) malformed("an op without \"op\"");
    if (op->kind != PROBE_COMPILE && !op->has_text) malformed("an op without \"text\"");
    free(key.ptr);
    free(name.ptr);
}

static void parse_case(json_in *in, probe_case *pc) {
    probe_buf key = {0};
    pc->has_id = pc->has_pattern = pc->has_ops = false;
    expect(in, '{');
    if (!peek_is(in, '}')) {
        for (;;) {
            parse_string(in, &key);
            expect(in, ':');
            if (strcmp(key.ptr, "id") == 0) {
                parse_string(in, &pc->id);
                pc->has_id = true;
            } else if (strcmp(key.ptr, "pattern") == 0) {
                parse_string(in, &pc->pattern);
                pc->has_pattern = true;
            } else if (strcmp(key.ptr, "ops") == 0) {
                pc->ops_len = 0;
                pc->has_ops = true;
                expect(in, '[');
                if (!peek_is(in, ']')) {
                    for (;;) {
                        if (pc->ops_len == pc->ops_cap) {
                            const size_t cap = pc->ops_cap ? pc->ops_cap * 2 : 8;
                            probe_op *ops = (probe_op *)realloc(pc->ops, cap * sizeof(*ops));
                            if (!ops) probe_fail("out of memory");
                            memset(ops + pc->ops_cap, 0, (cap - pc->ops_cap) * sizeof(*ops));
                            pc->ops = ops;
                            pc->ops_cap = cap;
                        }
                        parse_op(in, &pc->ops[pc->ops_len++]);
                        if (peek_is(in, ',')) {
                            ++in->p;
                            continue;
                        }
                        break;
                    }
                }
                expect(in, ']');
            } else {
                skip_value(in, 1);
            }
            if (peek_is(in, ',')) {
                ++in->p;
                continue;
            }
            break;
        }
    }
    expect(in, '}');
    skip_ws(in);
    if (in->p != in->end) malformed("trailing characters after the object");
    if (!pc->has_id || !pc->has_pattern || !pc->has_ops) malformed("\"id\", \"pattern\" and \"ops\" are required");
    free(key.ptr);
}

/* Engine state for one input line. */
typedef struct probe_state {
    chn_regex *re;           /* NULL when compile failed */
    chn_regex_status compile_status;
    chn_regex_string compile_error;
    size_t captures_len;
    size_t *slots;           /* 2 * captures_len byte positions */
    size_t *chars;           /* 2 * captures_len char positions */
    bool stub_hit;
} probe_state;

/* A result for a non-OK status: null, {"panic": ...}, or the end of the run. */
static void put_status(probe_buf *res, chn_regex_status status, chn_regex_string *panic, probe_state *s) {
    buf_clear(res);
    switch (status) {
    case CHN_REGEX_NO_MATCH:
        buf_puts(res, "null");
        break;
    case CHN_REGEX_PANIC:
        buf_puts(res, "{\"panic\":");
        buf_put_json_string(res, panic->ptr ? panic->ptr : "", panic->len);
        buf_putc(res, '}');
        chn_regex_string_drop(panic);
        break;
    case CHN_REGEX_UNIMPLEMENTED:
        s->stub_hit = true;
        buf_puts(res, "null");
        break;
    case CHN_REGEX_ERROR_NOMEM:
        probe_fail("the engine ran out of memory");
        break;
    default:
        probe_fail("the engine returned an unexpected status");
    }
}

/* RegexMatch::from_captures (one PosTranslator, groups in index order, start then end),
 * then the match object. Returns CHN_REGEX_OK or the translator's failure. */
static chn_regex_status put_match(probe_buf *res, const probe_state *s,
    chn_regex_pos_translator *t, chn_regex_string *panic) {
    const size_t n = s->captures_len;
    for (size_t i = 0; i < n; ++i) {
        if (s->slots[2 * i] == CHN_REGEX_NO_SLOT || s->slots[2 * i + 1] == CHN_REGEX_NO_SLOT) {
            s->chars[2 * i] = s->chars[2 * i + 1] = CHN_REGEX_NO_SLOT;
            continue;
        }
        for (size_t k = 0; k < 2; ++k) {
            const chn_regex_status status = chn_regex_pos_translator_get_char_pos(
                t, s->slots[2 * i + k], &s->chars[2 * i + k], panic);
            if (status != CHN_REGEX_OK) return status;
        }
    }
    buf_puts(res, "{\"start\":");
    buf_put_size(res, s->chars[0]);
    buf_puts(res, ",\"end\":");
    buf_put_size(res, s->chars[1]);
    buf_puts(res, ",\"len\":");
    buf_put_size(res, s->chars[1] - s->chars[0]);
    buf_puts(res, ",\"groups\":[");
    for (size_t i = 0; i < n; ++i) {
        if (i) buf_putc(res, ',');
        if (s->chars[2 * i] == CHN_REGEX_NO_SLOT) {
            buf_puts(res, "null");
            continue;
        }
        buf_putc(res, '[');
        buf_put_size(res, s->chars[2 * i]);
        buf_putc(res, ',');
        buf_put_size(res, s->chars[2 * i + 1]);
        buf_putc(res, ']');
    }
    buf_puts(res, "],\"names\":{");
    bool first = true;
    for (size_t i = 0; i < n; ++i) {
        size_t len;
        const char *name = chn_regex_capture_name(s->re, i, &len);
        if (!name) continue;
        /* RegexMatch::get_by_name: the first index carrying the name. */
        size_t j = 0;
        for (;; ++j) {
            size_t other_len;
            const char *other = chn_regex_capture_name(s->re, j, &other_len);
            if (other && other_len == len && memcmp(other, name, len) == 0) break;
        }
        if (!first) buf_putc(res, ',');
        first = false;
        buf_put_json_string(res, name, len);
        buf_putc(res, ':');
        if (s->chars[2 * j] == CHN_REGEX_NO_SLOT) {
            buf_puts(res, "null");
        } else {
            buf_putc(res, '[');
            buf_put_size(res, s->chars[2 * j]);
            buf_putc(res, ',');
            buf_put_size(res, s->chars[2 * j + 1]);
            buf_putc(res, ']');
        }
    }
    buf_puts(res, "}}");
    return CHN_REGEX_OK;
}

static void run_compile(probe_buf *res, probe_state *s) {
    buf_clear(res);
    if (s->re) {
        buf_puts(res, "{\"ok\":true,\"groups\":");
        buf_put_size(res, chn_regex_groups(s->re));
        buf_puts(res, ",\"groupindex\":{");
        bool first = true;
        for (size_t i = 0; i < s->captures_len; ++i) {
            size_t len;
            const char *name = chn_regex_capture_name(s->re, i, &len);
            if (!name) continue;
            if (!first) buf_putc(res, ',');
            first = false;
            buf_put_json_string(res, name, len);
            buf_putc(res, ':');
            buf_put_size(res, i);
        }
        buf_puts(res, "}}");
    } else if (s->compile_status == CHN_REGEX_ERROR_INVALID) {
        buf_puts(res, "{\"ok\":false,\"error\":");
        buf_put_json_string(res, s->compile_error.ptr ? s->compile_error.ptr : "", s->compile_error.len);
        buf_putc(res, '}');
    } else {
        buf_puts(res, "{\"ok\":false,\"error\":\"unimplemented: chn_regex_compile\"}");
    }
}

static void run_search(probe_buf *res, probe_state *s, const probe_op *op) {
    chn_regex_string panic = {0};
    const size_t byte_pos = chn_regex_to_byte_pos(op->text.ptr, op->text.len, op->pos);
    chn_regex_status status = chn_regex_search_at(s->re, op->text.ptr, op->text.len, byte_pos,
        s->slots, 2 * s->captures_len, &panic);
    if (status == CHN_REGEX_OK) {
        chn_regex_pos_translator t;
        chn_regex_pos_translator_init(&t, op->text.ptr, op->text.len);
        buf_clear(res);
        status = put_match(res, s, &t, &panic);
        chn_regex_pos_translator_drop(&t);
    }
    if (status != CHN_REGEX_OK) put_status(res, status, &panic, s);
}

static void run_findall(probe_buf *res, probe_state *s, const probe_op *op) {
    chn_regex_string panic = {0};
    chn_regex_iter it;
    chn_regex_pos_translator t;
    chn_regex_status status;
    chn_regex_iter_init(&it, s->re, op->text.ptr, op->text.len);
    chn_regex_pos_translator_init(&t, op->text.ptr, op->text.len);
    buf_clear(res);
    buf_putc(res, '[');
    bool first = true;
    for (;;) {
        status = chn_regex_iter_next_captures(&it, s->slots, 2 * s->captures_len);
        if (status != CHN_REGEX_OK) break;
        if (!first) buf_putc(res, ',');
        first = false;
        status = put_match(res, s, &t, &panic);
        if (status != CHN_REGEX_OK) break;
    }
    chn_regex_pos_translator_drop(&t);
    if (status == CHN_REGEX_NO_MATCH) buf_putc(res, ']');
    else put_status(res, status, &panic, s);
}

static void run_split(probe_buf *res, probe_state *s, const probe_op *op) {
    chn_regex_spans spans;
    const chn_regex_status status = op->kind == PROBE_SPLIT
        ? chn_regex_split(s->re, op->text.ptr, op->text.len, &spans)
        : chn_regex_split_without_captures(s->re, op->text.ptr, op->text.len, &spans);
    if (status != CHN_REGEX_OK) {
        chn_regex_string none = {0};
        put_status(res, status, &none, s);
        return;
    }
    buf_clear(res);
    buf_putc(res, '[');
    for (size_t i = 0; i < spans.len; ++i) {
        if (i) buf_putc(res, ',');
        buf_put_json_string(res, op->text.ptr + spans.ptr[i].start, spans.ptr[i].end - spans.ptr[i].start);
    }
    buf_putc(res, ']');
    chn_regex_spans_drop(&spans);
}

static void run_case(const probe_case *pc, const chn_regex_options *options, probe_buf *out,
    probe_buf *res, bool *stub_hit) {
    probe_state s = {0};
    s.compile_status = chn_regex_compile(pc->pattern.ptr, pc->pattern.len, options, &s.re, &s.compile_error);
    if (s.compile_status == CHN_REGEX_ERROR_NOMEM) probe_fail("the engine ran out of memory");
    if (s.compile_status == CHN_REGEX_UNIMPLEMENTED) s.stub_hit = true;
    if (s.re) {
        s.captures_len = chn_regex_captures_len(s.re);
        s.slots = (size_t *)malloc(2 * s.captures_len * sizeof(size_t));
        s.chars = (size_t *)malloc(2 * s.captures_len * sizeof(size_t));
        if (!s.slots || !s.chars) probe_fail("out of memory");
    }
    buf_clear(out);
    buf_puts(out, "{\"id\":");
    buf_put_json_string(out, pc->id.ptr, pc->id.len);
    buf_puts(out, ",\"results\":[");
    for (size_t i = 0; i < pc->ops_len; ++i) {
        const probe_op *op = &pc->ops[i];
        if (op->kind == PROBE_COMPILE) run_compile(res, &s);
        else if (!s.re) {
            buf_clear(res);
            buf_puts(res, "null");
        } else if (op->kind == PROBE_SEARCH) run_search(res, &s, op);
        else if (op->kind == PROBE_FINDALL) run_findall(res, &s, op);
        else run_split(res, &s, op);
        if (i) buf_putc(out, ',');
        buf_append(out, res->ptr, res->len);
    }
    buf_puts(out, "]}\n");
    free(s.slots);
    free(s.chars);
    chn_regex_string_drop(&s.compile_error);
    chn_regex_free(s.re);
    if (s.stub_hit) *stub_hit = true;
}

static _Noreturn void usage(void) {
    fputs("usage: regex_probe --mode default|nfa|backtrack|dfa --prefilter on|off < in.jsonl > out.jsonl\n", stderr);
    exit(PROBE_EXIT_MALFORMED);
}

int main(int argc, char **argv) {
    static probe_reader reader;
    chn_regex_options options = {CHN_REGEX_ENGINE_DEFAULT, true};
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) usage();
        const char *value = argv[i + 1];
        if (strcmp(argv[i], "--mode") == 0) {
            if (strcmp(value, "default") == 0) options.engine = CHN_REGEX_ENGINE_DEFAULT;
            else if (strcmp(value, "nfa") == 0) options.engine = CHN_REGEX_ENGINE_NFA;
            else if (strcmp(value, "backtrack") == 0) options.engine = CHN_REGEX_ENGINE_BACKTRACK;
            else if (strcmp(value, "dfa") == 0) options.engine = CHN_REGEX_ENGINE_DFA;
            else usage();
        } else if (strcmp(argv[i], "--prefilter") == 0) {
            if (strcmp(value, "on") == 0) options.prefilter = true;
            else if (strcmp(value, "off") == 0) options.prefilter = false;
            else usage();
        } else {
            usage();
        }
    }
#ifdef _WIN32
    if (_setmode(_fileno(stdin), _O_BINARY) == -1 || _setmode(_fileno(stdout), _O_BINARY) == -1) {
        probe_fail("cannot set binary mode");
    }
#endif
    reader.file = stdin;
    probe_buf line = {0}, out = {0}, res = {0};
    probe_case pc = {0};
    bool stub_hit = false;
    while (read_line(&reader, &line)) {
        ++probe_line_number;
        json_in in = {line.ptr, line.ptr + line.len};
        skip_ws(&in);
        if (in.p == in.end) continue;
        parse_case(&in, &pc);
        run_case(&pc, &options, &out, &res, &stub_hit);
        if (fwrite(out.ptr, 1, out.len, stdout) != out.len) probe_fail("cannot write stdout");
    }
    if (fflush(stdout) != 0) probe_fail("cannot write stdout");
    for (size_t i = 0; i < pc.ops_cap; ++i) free(pc.ops[i].text.ptr);
    free(pc.ops);
    free(pc.id.ptr);
    free(pc.pattern.ptr);
    free(line.ptr);
    free(out.ptr);
    free(res.ptr);
    if (stub_hit) {
        fputs("regex_probe: a scaffold stub was hit (CHN_REGEX_UNIMPLEMENTED)\n", stderr);
        return PROBE_EXIT_STUB;
    }
    return 0;
}
