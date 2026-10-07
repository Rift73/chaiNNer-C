/* Port of regex-syntax 0.7.2 src/error.rs, MIT OR Apache-2.0. */
/* The Display of regex_syntax::Error (error::Formatter and Spans): the pattern with its
 * error spans notated by carets, then "error: <kind>". */
#include "ast.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct out_buf {
    char *ptr;
    size_t len;
    size_t cap;
    bool failed;
} out_buf;

static void put(out_buf *b, const char *s, size_t n) {
    if (b->failed) return;
    if (b->cap - b->len < n + 1) {
        size_t cap = b->cap ? b->cap : 128;
        while (cap - b->len < n + 1) cap *= 2;
        char *ptr = (char *)realloc(b->ptr, cap);
        if (!ptr) {
            b->failed = true;
            return;
        }
        b->ptr = ptr;
        b->cap = cap;
    }
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void put_str(out_buf *b, const char *s) {
    put(b, s, strlen(s));
}

static void put_repeat(out_buf *b, char c, size_t n) {
    for (size_t i = 0; i < n; ++i) put(b, &c, 1);
}

static void put_size(out_buf *b, size_t n) {
    char digits[24];
    const int len = snprintf(digits, sizeof(digits), "%llu", (unsigned long long)n);
    put(b, digits, (size_t)len);
}

/* One item of str::lines(): split after each '\n'; a line loses its "\n" and then a
 * "\r" before it (a bare '\r' at the very end stays). */
typedef struct line_iter {
    const char *p;
    const char *end;
} line_iter;

static bool next_line(line_iter *it, const char **line, size_t *len) {
    if (it->p == it->end) return false;
    const char *nl = (const char *)memchr(it->p, '\n', (size_t)(it->end - it->p));
    *line = it->p;
    if (!nl) {
        *len = (size_t)(it->end - it->p);
        it->p = it->end;
        return true;
    }
    size_t n = (size_t)(nl - it->p);
    if (n > 0 && it->p[n - 1] == '\r') --n;
    *len = n;
    it->p = nl + 1;
    return true;
}

static int span_cmp(const chn_ast_span *a, const chn_ast_span *b) {
    if (a->start.offset != b->start.offset) return a->start.offset < b->start.offset ? -1 : 1;
    if (a->end.offset != b->end.offset) return a->end.offset < b->end.offset ? -1 : 1;
    return 0;
}

static size_t digit_count(size_t n) {
    size_t d = 1;
    while (n >= 10) {
        n /= 10;
        ++d;
    }
    return d;
}

chn_hir_status chn_syntax_error_display(const char *pattern, size_t pattern_len,
    const chn_syntax_error *err, chn_hir_string *out) {
    out->ptr = NULL;
    out->len = 0;

    /* Spans::from_formatter */
    size_t line_count = 0;
    {
        line_iter it = {pattern, pattern + pattern_len};
        const char *line;
        size_t len;
        while (next_line(&it, &line, &len)) ++line_count;
    }
    if (pattern_len > 0 && pattern[pattern_len - 1] == '\n') ++line_count;
    const size_t line_number_width = line_count <= 1 ? 0 : digit_count(line_count);
    /* At most two spans: the error's and its auxiliary one, each sorted in. */
    chn_ast_span spans[2];
    size_t span_count = 0;
    spans[span_count++] = err->span;
    if (err->has_aux) spans[span_count++] = err->aux;
    if (span_count == 2 && span_cmp(&spans[1], &spans[0]) < 0) {
        const chn_ast_span t = spans[0];
        spans[0] = spans[1];
        spans[1] = t;
    }

    char kind_msg[160];
    if (err->kind >= CHN_HIR_ERR_UNICODE_NOT_ALLOWED) chn_hir_error_kind_display(err, kind_msg, sizeof(kind_msg));
    else chn_ast_error_kind_display(err, kind_msg, sizeof(kind_msg));

    out_buf b = {NULL, 0, 0, false};
    const bool multi_line_pattern = memchr(pattern, '\n', pattern_len) != NULL;
    put_str(&b, "regex parse error:\n");
    if (multi_line_pattern) {
        put_repeat(&b, '~', 79);
        put_str(&b, "\n");
    }
    /* Spans::notate */
    {
        line_iter it = {pattern, pattern + pattern_len};
        const char *line;
        size_t len, i = 0;
        while (next_line(&it, &line, &len)) {
            if (line_number_width > 0) {
                put_repeat(&b, ' ', line_number_width - digit_count(i + 1));
                put_size(&b, i + 1);
                put_str(&b, ": ");
            } else {
                put_str(&b, "    ");
            }
            put(&b, line, len);
            put_str(&b, "\n");
            /* Spans::notate_line: the one-line spans on line i + 1, in order. */
            bool any = false;
            size_t pos = 0;
            for (size_t k = 0; k < span_count; ++k) {
                const chn_ast_span *s = &spans[k];
                if (s->start.line != s->end.line || s->start.line != i + 1) continue;
                if (!any) {
                    put_repeat(&b, ' ', line_number_width == 0 ? 4 : 2 + line_number_width);
                    any = true;
                }
                while (pos < s->start.column - 1) {
                    put_str(&b, " ");
                    ++pos;
                }
                const size_t note_len = s->end.column > s->start.column ? s->end.column - s->start.column : 0;
                const size_t carets = note_len > 1 ? note_len : 1;
                put_repeat(&b, '^', carets);
                pos += carets;
            }
            if (any) put_str(&b, "\n");
            ++i;
        }
    }
    if (multi_line_pattern) {
        put_repeat(&b, '~', 79);
        put_str(&b, "\n");
        bool first = true;
        for (size_t k = 0; k < span_count; ++k) {
            const chn_ast_span *s = &spans[k];
            if (s->start.line == s->end.line) continue;
            if (!first) put_str(&b, "\n");
            first = false;
            put_str(&b, "on line ");
            put_size(&b, s->start.line);
            put_str(&b, " (column ");
            put_size(&b, s->start.column);
            put_str(&b, ") through line ");
            put_size(&b, s->end.line);
            put_str(&b, " (column ");
            put_size(&b, s->end.column - 1);
            put_str(&b, ")");
        }
        if (!first) put_str(&b, "\n");
    }
    put_str(&b, "error: ");
    put_str(&b, kind_msg);
    if (b.failed) {
        free(b.ptr);
        return CHN_HIR_ERROR_NOMEM;
    }
    out->ptr = b.ptr;
    out->len = b.len;
    return CHN_HIR_OK;
}
