/* Port of regex-syntax 0.7.2 src/hir/mod.rs (with src/debug.rs's utf8_decode), MIT OR Apache-2.0. */
/* The HIR smart constructors with their simplifications, Properties, equality (for
 * lift_common_prefix), Drop, and the Display of hir::ErrorKind. */
#include "ast.h"
#include "interval.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

size_t chn_hir_error_kind_display(const chn_syntax_error *err, char *buf, size_t cap) {
    const char *msg = "";
    switch (err->kind) {
    case CHN_HIR_ERR_UNICODE_NOT_ALLOWED: msg = "Unicode not allowed here"; break;
    case CHN_HIR_ERR_INVALID_UTF8: msg = "pattern can match invalid UTF-8"; break;
    case CHN_HIR_ERR_UNICODE_PROPERTY_NOT_FOUND: msg = "Unicode property not found"; break;
    case CHN_HIR_ERR_UNICODE_PROPERTY_VALUE_NOT_FOUND: msg = "Unicode property value not found"; break;
    default: break;
    }
    const int n = snprintf(buf, cap, "%s", msg);
    return n < 0 ? 0 : (size_t)n;
}

/* Option<usize> */
static chn_hir_option_usize some(size_t value) {
    chn_hir_option_usize o = {true, value};
    return o;
}

static chn_hir_option_usize none(void) {
    chn_hir_option_usize o = {false, 0};
    return o;
}

static bool option_eq(chn_hir_option_usize a, chn_hir_option_usize b) {
    return a.some == b.some && (!a.some || a.value == b.value);
}

static size_t saturating_add(size_t a, size_t b) {
    return a > SIZE_MAX - b ? SIZE_MAX : a + b;
}

static size_t char_len_utf8(uint32_t c) {
    return c < 0x80 ? 1 : (c < 0x800 ? 2 : (c < 0x10000 ? 3 : 4));
}

static size_t encode_utf8(uint32_t c, uint8_t out[4]) {
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

/* Decodes one scalar value from a strictly valid UTF-8 sequence starting at s (core::str's
 * rules: no overlongs, no surrogates, nothing above U+10FFFF): its length, or 0. */
static size_t utf8_decode_strict(const uint8_t *s, size_t len, uint32_t *out) {
    if (len == 0) return 0;
    const uint8_t b = s[0];
    if (b < 0x80) {
        *out = b;
        return 1;
    }
    size_t n;
    uint32_t c;
    if (b >= 0xC2 && b <= 0xDF) {
        n = 2;
        c = b & 0x1Fu;
    } else if (b >= 0xE0 && b <= 0xEF) {
        n = 3;
        c = b & 0x0Fu;
    } else if (b >= 0xF0 && b <= 0xF4) {
        n = 4;
        c = b & 0x07u;
    } else {
        return 0;
    }
    if (len < n) return 0;
    for (size_t i = 1; i < n; ++i) {
        if ((s[i] & 0xC0) != 0x80) return 0;
        c = (c << 6) | (s[i] & 0x3Fu);
    }
    if ((n == 3 && c < 0x800) || (n == 4 && c < 0x10000) || c > 0x10FFFF || (c >= 0xD800 && c <= 0xDFFF)) {
        return 0;
    }
    *out = c;
    return n;
}

/* core::str::from_utf8(bytes).is_ok() */
static bool utf8_valid(const uint8_t *s, size_t len) {
    size_t i = 0;
    while (i < len) {
        uint32_t c;
        const size_t n = utf8_decode_strict(s + i, len - i, &c);
        if (n == 0) return false;
        i += n;
    }
    return true;
}

/* Properties::empty */
static chn_hir_properties props_empty(void) {
    chn_hir_properties p;
    memset(&p, 0, sizeof(p));
    p.minimum_len = some(0);
    p.maximum_len = some(0);
    p.utf8 = true;
    p.static_explicit_captures_len = some(0);
    return p;
}

/* Properties::literal */
static chn_hir_properties props_literal(const uint8_t *bytes, size_t len) {
    chn_hir_properties p = props_empty();
    p.minimum_len = some(len);
    p.maximum_len = some(len);
    p.utf8 = utf8_valid(bytes, len);
    p.literal = true;
    p.alternation_literal = true;
    return p;
}

/* Properties::class (Class::minimum_len, maximum_len, is_utf8) */
static chn_hir_properties props_class(const chn_hir_class *cls) {
    chn_hir_properties p = props_empty();
    if (cls->kind == CHN_HIR_CLASS_UNICODE) {
        const chn_hir_class_unicode *u = &cls->unicode;
        p.minimum_len = u->len ? some(char_len_utf8(u->ranges[0].start)) : none();
        p.maximum_len = u->len ? some(char_len_utf8(u->ranges[u->len - 1].end)) : none();
        p.utf8 = true;
    } else {
        p.minimum_len = cls->bytes.len ? some(1) : none();
        p.maximum_len = cls->bytes.len ? some(1) : none();
        p.utf8 = chn_class_bytes_is_ascii(&cls->bytes);
    }
    return p;
}

/* Properties::look */
static chn_hir_properties props_look(chn_hir_look look) {
    chn_hir_properties p = props_empty();
    const chn_hir_look_set set = chn_hir_look_set_singleton(look);
    p.look_set = set;
    p.look_set_prefix = set;
    p.look_set_suffix = set;
    p.look_set_prefix_any = set;
    p.look_set_suffix_any = set;
    return p;
}

/* Properties::repetition */
static chn_hir_properties props_repetition(const chn_hir_repetition *rep) {
    const chn_hir_properties *p = &rep->sub->props;
    chn_hir_properties inner = props_empty();
    if (p->minimum_len.some) {
        const size_t child_min = p->minimum_len.value, rep_min = rep->min;
        inner.minimum_len = some(rep_min != 0 && child_min > SIZE_MAX / rep_min ? SIZE_MAX : child_min * rep_min);
    } else {
        inner.minimum_len = none();
    }
    inner.maximum_len = none();
    if (rep->max.some && p->maximum_len.some) {
        const size_t child_max = p->maximum_len.value, rep_max = rep->max.value;
        if (rep_max == 0 || child_max <= SIZE_MAX / rep_max) inner.maximum_len = some(child_max * rep_max);
    }
    inner.look_set = p->look_set;
    inner.look_set_prefix = chn_hir_look_set_empty();
    inner.look_set_suffix = chn_hir_look_set_empty();
    inner.look_set_prefix_any = p->look_set_prefix_any;
    inner.look_set_suffix_any = p->look_set_suffix_any;
    inner.utf8 = p->utf8;
    inner.explicit_captures_len = p->explicit_captures_len;
    inner.static_explicit_captures_len = p->static_explicit_captures_len;
    inner.literal = false;
    inner.alternation_literal = false;
    if (rep->min > 0) {
        inner.look_set_prefix = p->look_set_prefix;
        inner.look_set_suffix = p->look_set_suffix;
    }
    if (rep->min == 0 && inner.static_explicit_captures_len.some && inner.static_explicit_captures_len.value > 0) {
        if (rep->max.some && rep->max.value == 0) inner.static_explicit_captures_len = some(0);
        else inner.static_explicit_captures_len = none();
    }
    return inner;
}

/* Properties::capture */
static chn_hir_properties props_capture(const chn_hir_capture *cap) {
    chn_hir_properties p = cap->sub->props;
    p.explicit_captures_len = saturating_add(p.explicit_captures_len, 1);
    if (p.static_explicit_captures_len.some) {
        p.static_explicit_captures_len.value = saturating_add(p.static_explicit_captures_len.value, 1);
    }
    p.literal = false;
    p.alternation_literal = false;
    return p;
}

/* Properties::concat */
static chn_hir_properties props_concat(const chn_hir *subs, size_t n) {
    chn_hir_properties props = props_empty();
    props.literal = true;
    props.alternation_literal = true;
    for (size_t i = 0; i < n; ++i) {
        const chn_hir_properties *p = &subs[i].props;
        props.look_set = chn_hir_look_set_union(props.look_set, p->look_set);
        props.utf8 = props.utf8 && p->utf8;
        props.explicit_captures_len = saturating_add(props.explicit_captures_len, p->explicit_captures_len);
        if (p->static_explicit_captures_len.some && props.static_explicit_captures_len.some) {
            props.static_explicit_captures_len = some(saturating_add(p->static_explicit_captures_len.value,
                props.static_explicit_captures_len.value));
        } else {
            props.static_explicit_captures_len = none();
        }
        props.literal = props.literal && p->literal;
        props.alternation_literal = props.alternation_literal && p->alternation_literal;
        if (props.minimum_len.some) {
            props.minimum_len = p->minimum_len.some
                ? some(saturating_add(props.minimum_len.value, p->minimum_len.value)) : none();
        }
        if (props.maximum_len.some) {
            if (!p->maximum_len.some || p->maximum_len.value > SIZE_MAX - props.maximum_len.value) {
                props.maximum_len = none();
            } else {
                props.maximum_len = some(props.maximum_len.value + p->maximum_len.value);
            }
        }
    }
    for (size_t i = 0; i < n; ++i) {
        const chn_hir_properties *p = &subs[i].props;
        props.look_set_prefix = chn_hir_look_set_union(props.look_set_prefix, p->look_set_prefix);
        props.look_set_prefix_any = chn_hir_look_set_union(props.look_set_prefix_any, p->look_set_prefix_any);
        if (!p->maximum_len.some || p->maximum_len.value > 0) break;
    }
    for (size_t i = n; i-- > 0;) {
        const chn_hir_properties *p = &subs[i].props;
        props.look_set_suffix = chn_hir_look_set_union(props.look_set_suffix, p->look_set_suffix);
        props.look_set_suffix_any = chn_hir_look_set_union(props.look_set_suffix_any, p->look_set_suffix_any);
        if (!p->maximum_len.some || p->maximum_len.value > 0) break;
    }
    return props;
}

/* Properties::alternation (Properties::union) */
static chn_hir_properties props_alternation(const chn_hir *subs, size_t n) {
    const chn_hir_look_set fix = n == 0 ? chn_hir_look_set_empty() : chn_hir_look_set_full();
    chn_hir_properties props;
    memset(&props, 0, sizeof(props));
    props.minimum_len = none();
    props.maximum_len = none();
    props.look_set_prefix = fix;
    props.look_set_suffix = fix;
    props.utf8 = true;
    props.static_explicit_captures_len = n ? subs[0].props.static_explicit_captures_len : none();
    props.literal = false;
    props.alternation_literal = true;
    bool min_poisoned = false, max_poisoned = false;
    for (size_t i = 0; i < n; ++i) {
        const chn_hir_properties *p = &subs[i].props;
        props.look_set = chn_hir_look_set_union(props.look_set, p->look_set);
        props.look_set_prefix = chn_hir_look_set_intersect(props.look_set_prefix, p->look_set_prefix);
        props.look_set_suffix = chn_hir_look_set_intersect(props.look_set_suffix, p->look_set_suffix);
        props.look_set_prefix_any = chn_hir_look_set_union(props.look_set_prefix_any, p->look_set_prefix_any);
        props.look_set_suffix_any = chn_hir_look_set_union(props.look_set_suffix_any, p->look_set_suffix_any);
        props.utf8 = props.utf8 && p->utf8;
        props.explicit_captures_len = saturating_add(props.explicit_captures_len, p->explicit_captures_len);
        if (!option_eq(props.static_explicit_captures_len, p->static_explicit_captures_len)) {
            props.static_explicit_captures_len = none();
        }
        props.alternation_literal = props.alternation_literal && p->literal;
        if (!min_poisoned) {
            if (p->minimum_len.some) {
                if (!props.minimum_len.some || p->minimum_len.value < props.minimum_len.value) {
                    props.minimum_len = p->minimum_len;
                }
            } else {
                props.minimum_len = none();
                min_poisoned = true;
            }
        }
        if (!max_poisoned) {
            if (p->maximum_len.some) {
                if (!props.maximum_len.some || p->maximum_len.value > props.maximum_len.value) {
                    props.maximum_len = p->maximum_len;
                }
            } else {
                props.maximum_len = none();
                max_poisoned = true;
            }
        }
    }
    return props;
}

static bool props_eq(const chn_hir_properties *a, const chn_hir_properties *b) {
    return option_eq(a->minimum_len, b->minimum_len) && option_eq(a->maximum_len, b->maximum_len)
        && a->look_set.bits == b->look_set.bits && a->look_set_prefix.bits == b->look_set_prefix.bits
        && a->look_set_suffix.bits == b->look_set_suffix.bits
        && a->look_set_prefix_any.bits == b->look_set_prefix_any.bits
        && a->look_set_suffix_any.bits == b->look_set_suffix_any.bits && a->utf8 == b->utf8
        && a->explicit_captures_len == b->explicit_captures_len
        && option_eq(a->static_explicit_captures_len, b->static_explicit_captures_len)
        && a->literal == b->literal && a->alternation_literal == b->alternation_literal;
}

/* PartialEq for Hir (kind, then props). */
static bool hir_eq(const chn_hir *a, const chn_hir *b) {
    if (a->kind != b->kind || !props_eq(&a->props, &b->props)) return false;
    switch (a->kind) {
    case CHN_HIR_EMPTY:
        return true;
    case CHN_HIR_LITERAL:
        return a->literal.len == b->literal.len && memcmp(a->literal.bytes, b->literal.bytes, a->literal.len) == 0;
    case CHN_HIR_CLASS:
        if (a->cls.kind != b->cls.kind) return false;
        return a->cls.kind == CHN_HIR_CLASS_UNICODE ? chn_class_unicode_eq(&a->cls.unicode, &b->cls.unicode)
                                                    : chn_class_bytes_eq(&a->cls.bytes, &b->cls.bytes);
    case CHN_HIR_LOOK:
        return a->look == b->look;
    case CHN_HIR_REPETITION:
        return a->repetition.min == b->repetition.min && a->repetition.max.some == b->repetition.max.some
            && (!a->repetition.max.some || a->repetition.max.value == b->repetition.max.value)
            && a->repetition.greedy == b->repetition.greedy && hir_eq(a->repetition.sub, b->repetition.sub);
    case CHN_HIR_CAPTURE:
        if (a->capture.index != b->capture.index) return false;
        if ((a->capture.name.ptr == NULL) != (b->capture.name.ptr == NULL)) return false;
        if (a->capture.name.ptr && (a->capture.name.len != b->capture.name.len
                || memcmp(a->capture.name.ptr, b->capture.name.ptr, a->capture.name.len) != 0)) {
            return false;
        }
        return hir_eq(a->capture.sub, b->capture.sub);
    case CHN_HIR_CONCAT:
    case CHN_HIR_ALTERNATION: {
        const chn_hir_vec *va = a->kind == CHN_HIR_CONCAT ? &a->concat : &a->alternation;
        const chn_hir_vec *vb = a->kind == CHN_HIR_CONCAT ? &b->concat : &b->alternation;
        if (va->len != vb->len) return false;
        for (size_t i = 0; i < va->len; ++i) {
            if (!hir_eq(&va->ptr[i], &vb->ptr[i])) return false;
        }
        return true;
    }
    }
    return false;
}

void chn_hir_make_empty(chn_hir *out) {
    memset(out, 0, sizeof(*out));
    out->kind = CHN_HIR_EMPTY;
    out->props = props_empty();
}

void chn_hir_make_fail(chn_hir *out) {
    memset(out, 0, sizeof(*out));
    out->kind = CHN_HIR_CLASS;
    out->cls.kind = CHN_HIR_CLASS_BYTES;
    chn_class_bytes_init(&out->cls.bytes);
    out->props = props_class(&out->cls);
}

void chn_hir_make_literal(chn_hir *out, uint8_t *bytes, size_t len) {
    if (len == 0) {
        free(bytes);
        chn_hir_make_empty(out);
        return;
    }
    memset(out, 0, sizeof(*out));
    out->kind = CHN_HIR_LITERAL;
    out->literal.bytes = bytes;
    out->literal.len = len;
    out->props = props_literal(bytes, len);
}

static void class_drop(chn_hir_class *cls) {
    if (cls->kind == CHN_HIR_CLASS_UNICODE) chn_class_unicode_drop(&cls->unicode);
    else chn_class_bytes_drop(&cls->bytes);
}

chn_hir_status chn_hir_make_class(chn_hir *out, chn_hir_class *cls) {
    const bool unicode = cls->kind == CHN_HIR_CLASS_UNICODE;
    const size_t len = unicode ? cls->unicode.len : cls->bytes.len;
    if (len == 0) {
        class_drop(cls);
        chn_hir_make_fail(out);
        return CHN_HIR_OK;
    }
    /* Class::literal */
    if (len == 1 && (unicode ? cls->unicode.ranges[0].start == cls->unicode.ranges[0].end
                             : cls->bytes.ranges[0].start == cls->bytes.ranges[0].end)) {
        uint8_t utf8[4];
        size_t n = 1;
        if (unicode) n = encode_utf8(cls->unicode.ranges[0].start, utf8);
        else utf8[0] = cls->bytes.ranges[0].start;
        class_drop(cls);
        uint8_t *bytes = (uint8_t *)malloc(n);
        if (!bytes) {
            chn_hir_make_empty(out);
            return CHN_HIR_ERROR_NOMEM;
        }
        memcpy(bytes, utf8, n);
        chn_hir_make_literal(out, bytes, n);
        return CHN_HIR_OK;
    }
    memset(out, 0, sizeof(*out));
    out->kind = CHN_HIR_CLASS;
    out->cls = *cls;
    if (unicode) chn_class_unicode_init(&cls->unicode);
    else chn_class_bytes_init(&cls->bytes);
    out->props = props_class(&out->cls);
    return CHN_HIR_OK;
}

void chn_hir_make_look(chn_hir *out, chn_hir_look look) {
    memset(out, 0, sizeof(*out));
    out->kind = CHN_HIR_LOOK;
    out->look = look;
    out->props = props_look(look);
}

void chn_hir_make_repetition(chn_hir *out, chn_hir_repetition *rep) {
    chn_hir *sub = rep->sub;
    rep->sub = NULL;
    if (rep->min == 0 && rep->max.some && rep->max.value == 0) {
        chn_hir_free(sub);
        chn_hir_make_empty(out);
        return;
    }
    if (rep->min == 1 && rep->max.some && rep->max.value == 1) {
        *out = *sub;
        free(sub);
        return;
    }
    chn_hir_repetition moved = *rep;
    moved.sub = sub;
    memset(out, 0, sizeof(*out));
    out->kind = CHN_HIR_REPETITION;
    out->repetition = moved;
    out->props = props_repetition(&out->repetition);
}

void chn_hir_make_capture(chn_hir *out, chn_hir_capture *cap) {
    chn_hir_capture moved = *cap;
    cap->name.ptr = NULL;
    cap->name.len = 0;
    cap->sub = NULL;
    memset(out, 0, sizeof(*out));
    out->kind = CHN_HIR_CAPTURE;
    out->capture = moved;
    out->props = props_capture(&out->capture);
}

/* Vec<Hir> helpers. */
static void vec_init(chn_hir_vec *v) {
    v->ptr = NULL;
    v->len = 0;
    v->cap = 0;
}

static void vec_drop(chn_hir_vec *v) {
    for (size_t i = 0; i < v->len; ++i) chn_hir_drop(&v->ptr[i]);
    free(v->ptr);
    vec_init(v);
}

static chn_hir_status vec_reserve(chn_hir_vec *v, size_t extra) {
    if (v->cap - v->len >= extra) return CHN_HIR_OK;
    if (extra > SIZE_MAX / sizeof(chn_hir) - v->len) return CHN_HIR_ERROR_NOMEM;
    size_t cap = v->cap ? v->cap : 4;
    while (cap < v->len + extra) cap *= 2;
    chn_hir *ptr = (chn_hir *)realloc(v->ptr, cap * sizeof(chn_hir));
    if (!ptr) return CHN_HIR_ERROR_NOMEM;
    v->ptr = ptr;
    v->cap = cap;
    return CHN_HIR_OK;
}

/* Vec::push of a moved Hir: *hir is left Hir::empty(); on NOMEM it is dropped. */
static chn_hir_status vec_push(chn_hir_vec *v, chn_hir *hir) {
    const chn_hir_status status = vec_reserve(v, 1);
    if (status != CHN_HIR_OK) {
        chn_hir_drop(hir);
        return status;
    }
    v->ptr[v->len++] = *hir;
    chn_hir_make_empty(hir);
    return CHN_HIR_OK;
}

/* Ends a smart constructor over a vector: 0 subs -> zero (empty or fail), 1 -> the sub,
 * otherwise the vector itself with its props. */
static void finish_vec(chn_hir *out, chn_hir_vec *v, chn_hir_kind kind) {
    if (v->len == 0) {
        free(v->ptr);
        if (kind == CHN_HIR_CONCAT) chn_hir_make_empty(out);
        else chn_hir_make_fail(out);
        return;
    }
    if (v->len == 1) {
        *out = v->ptr[0];
        free(v->ptr);
        return;
    }
    memset(out, 0, sizeof(*out));
    out->kind = kind;
    if (kind == CHN_HIR_CONCAT) {
        out->concat = *v;
        out->props = props_concat(v->ptr, v->len);
    } else {
        out->alternation = *v;
        out->props = props_alternation(v->ptr, v->len);
    }
}

/* The literal accumulator of Hir::concat (prior_lit). */
typedef struct lit_buf {
    uint8_t *ptr;
    size_t len;
    size_t cap;
} lit_buf;

/* prior_lit.extend_from_slice / Some(bytes.to_vec()): takes the literal's buffer. */
static chn_hir_status lit_take(lit_buf *buf, chn_hir *lit) {
    if (!buf->ptr) {
        buf->ptr = lit->literal.bytes;
        buf->len = lit->literal.len;
        buf->cap = lit->literal.len;
        lit->literal.bytes = NULL;
        lit->literal.len = 0;
        chn_hir_make_empty(lit);
        return CHN_HIR_OK;
    }
    const size_t n = lit->literal.len;
    if (buf->cap - buf->len < n) {
        size_t cap = buf->cap * 2;
        if (cap < buf->len + n) cap = buf->len + n;
        uint8_t *ptr = (uint8_t *)realloc(buf->ptr, cap);
        if (!ptr) {
            chn_hir_drop(lit);
            return CHN_HIR_ERROR_NOMEM;
        }
        buf->ptr = ptr;
        buf->cap = cap;
    }
    memcpy(buf->ptr + buf->len, lit->literal.bytes, n);
    buf->len += n;
    chn_hir_drop(lit);
    return CHN_HIR_OK;
}

/* new.push(Hir::literal(prior_bytes)) for a pending prior_lit. */
static chn_hir_status lit_flush(lit_buf *buf, chn_hir_vec *v) {
    if (!buf->ptr) return CHN_HIR_OK;
    chn_hir lit;
    chn_hir_make_literal(&lit, buf->ptr, buf->len);
    buf->ptr = NULL;
    buf->len = 0;
    buf->cap = 0;
    return vec_push(v, &lit);
}

chn_hir_status chn_hir_make_concat(chn_hir *out, chn_hir_vec *subs) {
    chn_hir_vec new_subs;
    vec_init(&new_subs);
    lit_buf prior = {NULL, 0, 0};
    chn_hir_status status = CHN_HIR_OK;
    for (size_t i = 0; i < subs->len && status == CHN_HIR_OK; ++i) {
        chn_hir *sub = &subs->ptr[i];
        switch (sub->kind) {
        case CHN_HIR_LITERAL:
            status = lit_take(&prior, sub);
            break;
        case CHN_HIR_CONCAT: {
            chn_hir_vec inner = sub->concat;
            vec_init(&sub->concat);
            chn_hir_drop(sub);
            for (size_t j = 0; j < inner.len; ++j) {
                if (status != CHN_HIR_OK) {
                    chn_hir_drop(&inner.ptr[j]);
                } else if (inner.ptr[j].kind == CHN_HIR_LITERAL) {
                    status = lit_take(&prior, &inner.ptr[j]);
                } else {
                    status = lit_flush(&prior, &new_subs);
                    if (status == CHN_HIR_OK) status = vec_push(&new_subs, &inner.ptr[j]);
                    else chn_hir_drop(&inner.ptr[j]);
                }
            }
            free(inner.ptr);
            break;
        }
        case CHN_HIR_EMPTY:
            break;
        default:
            status = lit_flush(&prior, &new_subs);
            if (status == CHN_HIR_OK) status = vec_push(&new_subs, sub);
            break;
        }
    }
    if (status == CHN_HIR_OK) status = lit_flush(&prior, &new_subs);
    vec_drop(subs);
    if (status != CHN_HIR_OK) {
        free(prior.ptr);
        vec_drop(&new_subs);
        chn_hir_make_empty(out);
        return status;
    }
    finish_vec(out, &new_subs, CHN_HIR_CONCAT);
    return CHN_HIR_OK;
}

/* debug::utf8_decode on a literal, then the length check of singleton_chars. */
static bool literal_single_char(const chn_hir_literal *lit, uint32_t *c) {
    const size_t n = utf8_decode_strict(lit->bytes, lit->len, c);
    return n != 0 && n == lit->len;
}

/* singleton_chars, singleton_bytes, class_chars and class_bytes: the class an
 * alternation collapses into, written to *cls; false when it does not collapse. */
static bool alternation_singleton_chars(const chn_hir_vec *v) {
    for (size_t i = 0; i < v->len; ++i) {
        uint32_t c;
        if (v->ptr[i].kind != CHN_HIR_LITERAL || !literal_single_char(&v->ptr[i].literal, &c)) return false;
    }
    return true;
}

static bool alternation_singleton_bytes(const chn_hir_vec *v) {
    for (size_t i = 0; i < v->len; ++i) {
        if (v->ptr[i].kind != CHN_HIR_LITERAL || v->ptr[i].literal.len != 1) return false;
    }
    return true;
}

/* class_chars: every sub a class, every Bytes one ASCII. */
static bool alternation_class_chars(const chn_hir_vec *v) {
    for (size_t i = 0; i < v->len; ++i) {
        if (v->ptr[i].kind != CHN_HIR_CLASS) return false;
        if (v->ptr[i].cls.kind == CHN_HIR_CLASS_BYTES && !chn_class_bytes_is_ascii(&v->ptr[i].cls.bytes)) return false;
    }
    return true;
}

/* class_bytes: every sub a class, every Unicode one ASCII. */
static bool alternation_class_bytes(const chn_hir_vec *v) {
    for (size_t i = 0; i < v->len; ++i) {
        if (v->ptr[i].kind != CHN_HIR_CLASS) return false;
        if (v->ptr[i].cls.kind == CHN_HIR_CLASS_UNICODE && !chn_class_unicode_is_ascii(&v->ptr[i].cls.unicode)) return false;
    }
    return true;
}

/* Builds the collapsed class of an alternation already accepted by one of the tests
 * above, as a Unicode (want_unicode) or Bytes class. */
static chn_hir_status alternation_class(const chn_hir_vec *v, bool want_unicode, chn_hir_class *cls) {
    chn_hir_status status = CHN_HIR_OK;
    cls->kind = want_unicode ? CHN_HIR_CLASS_UNICODE : CHN_HIR_CLASS_BYTES;
    if (want_unicode) chn_class_unicode_init(&cls->unicode);
    else chn_class_bytes_init(&cls->bytes);
    for (size_t i = 0; i < v->len && status == CHN_HIR_OK; ++i) {
        const chn_hir *h = &v->ptr[i];
        if (h->kind == CHN_HIR_LITERAL) {
            /* ClassUnicode::new(singletons) / ClassBytes::new(singletons): pushing each
             * singleton canonicalizes to the same set. */
            if (want_unicode) {
                uint32_t c = 0;
                literal_single_char(&h->literal, &c);
                status = chn_class_unicode_push(&cls->unicode, c, c);
            } else {
                status = chn_class_bytes_push(&cls->bytes, h->literal.bytes[0], h->literal.bytes[0]);
            }
            continue;
        }
        const chn_hir_class *sub = &h->cls;
        if (want_unicode && sub->kind == CHN_HIR_CLASS_UNICODE) {
            status = chn_class_unicode_union(&cls->unicode, &sub->unicode);
        } else if (!want_unicode && sub->kind == CHN_HIR_CLASS_BYTES) {
            status = chn_class_bytes_union(&cls->bytes, &sub->bytes);
        } else if (want_unicode) {
            /* ClassBytes::to_unicode_class (ASCII) */
            chn_hir_class_unicode conv;
            chn_class_unicode_init(&conv);
            for (size_t k = 0; k < sub->bytes.len && status == CHN_HIR_OK; ++k) {
                status = chn_class_unicode_append(&conv, sub->bytes.ranges[k].start, sub->bytes.ranges[k].end);
            }
            conv.folded = conv.len == 0; /* ClassUnicode::new */
            if (status == CHN_HIR_OK) status = chn_class_unicode_union(&cls->unicode, &conv);
            chn_class_unicode_drop(&conv);
        } else {
            /* ClassUnicode::to_byte_class (ASCII) */
            chn_hir_class_bytes conv;
            chn_class_bytes_init(&conv);
            for (size_t k = 0; k < sub->unicode.len && status == CHN_HIR_OK; ++k) {
                status = chn_class_bytes_push(&conv, (uint8_t)sub->unicode.ranges[k].start,
                    (uint8_t)sub->unicode.ranges[k].end);
            }
            if (status == CHN_HIR_OK) status = chn_class_bytes_union(&cls->bytes, &conv);
            chn_class_bytes_drop(&conv);
        }
    }
    if (status != CHN_HIR_OK) class_drop(cls);
    return status;
}

/* lift_common_prefix: the length of the common prefix of concats, or 0 for Err. */
static size_t common_prefix_len(const chn_hir_vec *v) {
    if (v->len <= 1 || v->ptr[0].kind != CHN_HIR_CONCAT) return 0;
    const chn_hir *prefix = v->ptr[0].concat.ptr;
    size_t prefix_len = v->ptr[0].concat.len;
    if (prefix_len == 0) return 0;
    for (size_t i = 1; i < v->len; ++i) {
        if (v->ptr[i].kind != CHN_HIR_CONCAT) return 0;
        const chn_hir_vec *concat = &v->ptr[i].concat;
        size_t common = 0;
        while (common < prefix_len && common < concat->len && hir_eq(&prefix[common], &concat->ptr[common])) {
            ++common;
        }
        prefix_len = common;
        if (prefix_len == 0) return 0;
    }
    return prefix_len;
}

/* lift_common_prefix's Ok branch: consumes *v. */
static chn_hir_status lift_common_prefix(chn_hir *out, chn_hir_vec *v, size_t len) {
    chn_hir_vec prefix_concat, suffix_alts;
    vec_init(&prefix_concat);
    vec_init(&suffix_alts);
    chn_hir_status status = vec_reserve(&suffix_alts, v->len);
    for (size_t i = 0; i < v->len; ++i) {
        chn_hir_vec concat = v->ptr[i].concat;
        vec_init(&v->ptr[i].concat);
        chn_hir_make_empty(&v->ptr[i]);
        if (status != CHN_HIR_OK) {
            vec_drop(&concat);
            continue;
        }
        /* concat.split_off(len) */
        chn_hir_vec suffix;
        vec_init(&suffix);
        status = vec_reserve(&suffix, concat.len - len);
        if (status == CHN_HIR_OK) {
            if (concat.len > len) memcpy(suffix.ptr, concat.ptr + len, (concat.len - len) * sizeof(chn_hir));
            suffix.len = concat.len - len;
            concat.len = len;
            chn_hir alt_sub;
            status = chn_hir_make_concat(&alt_sub, &suffix);
            if (status == CHN_HIR_OK) status = vec_push(&suffix_alts, &alt_sub);
        }
        if (status == CHN_HIR_OK && i == 0) {
            prefix_concat = concat;
        } else {
            vec_drop(&concat);
        }
    }
    vec_drop(v);
    chn_hir alt;
    if (status == CHN_HIR_OK) {
        status = chn_hir_make_alternation(&alt, &suffix_alts);
    } else {
        vec_drop(&suffix_alts);
    }
    if (status == CHN_HIR_OK) status = vec_push(&prefix_concat, &alt);
    if (status != CHN_HIR_OK) {
        vec_drop(&prefix_concat);
        chn_hir_make_empty(out);
        return status;
    }
    return chn_hir_make_concat(out, &prefix_concat);
}

chn_hir_status chn_hir_make_alternation(chn_hir *out, chn_hir_vec *subs) {
    chn_hir_vec new_subs;
    vec_init(&new_subs);
    chn_hir_status status = CHN_HIR_OK;
    for (size_t i = 0; i < subs->len; ++i) {
        chn_hir *sub = &subs->ptr[i];
        if (status != CHN_HIR_OK) {
            chn_hir_drop(sub);
        } else if (sub->kind == CHN_HIR_ALTERNATION) {
            chn_hir_vec inner = sub->alternation;
            vec_init(&sub->alternation);
            chn_hir_drop(sub);
            status = vec_reserve(&new_subs, inner.len);
            if (status == CHN_HIR_OK) {
                memcpy(new_subs.ptr + new_subs.len, inner.ptr, inner.len * sizeof(chn_hir));
                new_subs.len += inner.len;
                free(inner.ptr);
            } else {
                vec_drop(&inner);
            }
        } else {
            status = vec_push(&new_subs, sub);
        }
    }
    free(subs->ptr);
    vec_init(subs);
    if (status != CHN_HIR_OK) {
        vec_drop(&new_subs);
        chn_hir_make_empty(out);
        return status;
    }
    if (new_subs.len <= 1) {
        finish_vec(out, &new_subs, CHN_HIR_ALTERNATION);
        return CHN_HIR_OK;
    }
    int collapse = -1; /* 1: a Unicode class, 0: a Bytes class */
    if (alternation_singleton_chars(&new_subs)) collapse = 1;
    else if (alternation_singleton_bytes(&new_subs)) collapse = 0;
    else if (alternation_class_chars(&new_subs)) collapse = 1;
    else if (alternation_class_bytes(&new_subs)) collapse = 0;
    if (collapse >= 0) {
        chn_hir_class cls;
        status = alternation_class(&new_subs, collapse == 1, &cls);
        vec_drop(&new_subs);
        if (status != CHN_HIR_OK) {
            chn_hir_make_empty(out);
            return status;
        }
        return chn_hir_make_class(out, &cls);
    }
    const size_t prefix_len = common_prefix_len(&new_subs);
    if (prefix_len > 0) return lift_common_prefix(out, &new_subs, prefix_len);
    finish_vec(out, &new_subs, CHN_HIR_ALTERNATION);
    return CHN_HIR_OK;
}

chn_hir_status chn_hir_make_dot(chn_hir *out, chn_hir_dot dot) {
    chn_hir_class cls;
    chn_hir_status status = CHN_HIR_OK;
    const bool unicode = dot == CHN_HIR_DOT_ANY_CHAR || dot == CHN_HIR_DOT_ANY_CHAR_EXCEPT_LF
        || dot == CHN_HIR_DOT_ANY_CHAR_EXCEPT_CRLF;
    const uint32_t max = unicode ? 0x10FFFF : 0xFF;
    uint32_t ranges[3][2];
    size_t n;
    switch (dot) {
    case CHN_HIR_DOT_ANY_CHAR:
    case CHN_HIR_DOT_ANY_BYTE:
        ranges[0][0] = 0;
        ranges[0][1] = max;
        n = 1;
        break;
    case CHN_HIR_DOT_ANY_CHAR_EXCEPT_LF:
    case CHN_HIR_DOT_ANY_BYTE_EXCEPT_LF:
        ranges[0][0] = 0;
        ranges[0][1] = 0x09;
        ranges[1][0] = 0x0B;
        ranges[1][1] = max;
        n = 2;
        break;
    default:
        ranges[0][0] = 0;
        ranges[0][1] = 0x09;
        ranges[1][0] = 0x0B;
        ranges[1][1] = 0x0C;
        ranges[2][0] = 0x0E;
        ranges[2][1] = max;
        n = 3;
        break;
    }
    cls.kind = unicode ? CHN_HIR_CLASS_UNICODE : CHN_HIR_CLASS_BYTES;
    if (unicode) chn_class_unicode_init(&cls.unicode);
    else chn_class_bytes_init(&cls.bytes);
    for (size_t i = 0; i < n && status == CHN_HIR_OK; ++i) {
        if (unicode) status = chn_class_unicode_push(&cls.unicode, ranges[i][0], ranges[i][1]);
        else status = chn_class_bytes_push(&cls.bytes, (uint8_t)ranges[i][0], (uint8_t)ranges[i][1]);
    }
    if (status != CHN_HIR_OK) {
        class_drop(&cls);
        chn_hir_make_empty(out);
        return status;
    }
    return chn_hir_make_class(out, &cls);
}

void chn_hir_drop(chn_hir *hir) {
    switch (hir->kind) {
    case CHN_HIR_EMPTY:
    case CHN_HIR_LOOK:
        break;
    case CHN_HIR_LITERAL:
        free(hir->literal.bytes);
        break;
    case CHN_HIR_CLASS:
        class_drop(&hir->cls);
        break;
    case CHN_HIR_REPETITION:
        chn_hir_free(hir->repetition.sub);
        break;
    case CHN_HIR_CAPTURE:
        free(hir->capture.name.ptr);
        chn_hir_free(hir->capture.sub);
        break;
    case CHN_HIR_CONCAT:
        vec_drop(&hir->concat);
        break;
    case CHN_HIR_ALTERNATION:
        vec_drop(&hir->alternation);
        break;
    }
    chn_hir_make_empty(hir);
}

void chn_hir_free(chn_hir *hir) {
    if (!hir) return;
    chn_hir_drop(hir);
    free(hir);
}
