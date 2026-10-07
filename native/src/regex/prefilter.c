/* Port of regex 1.8.4 src/literal/imp.rs (LiteralSearcher, Matcher selection, SingleByteSet, Memmem), with regex-syntax 0.7.2 src/rank.rs's byte ranks, MIT OR Apache-2.0. */
/* The literal searchers of regex 1.8.4, with the same literal sets and the same matcher
 * selection (Empty, Bytes, Memmem, Packed/AC). Packed (Teddy) and AhoCorasick both search
 * leftmost-first, so one leftmost-first multi-literal searcher stands for both. Each
 * matcher has two implementations with identical results, chosen by the harness's prefilter
 * switch: the fast one (memchr, a rare-byte memmem, a first-byte scan with a trie) and a
 * plain scan (every position, every literal in preference order). */
#include "prog.h"
#include <stdlib.h>
#include <string.h>
#if defined(_M_X64) || defined(__SSE2__)
#include <emmintrin.h>
#define CHN_HAVE_SSE2 1
#endif

/* regex-syntax rank.rs BYTE_FREQUENCIES: a lower rank is a rarer byte. */
static const uint8_t byte_rank[256] = {
    55, 52, 51, 50, 49, 48, 47, 46, 45, 103, 242, 66, 67, 229, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35,
    34, 33, 56, 32, 31, 30, 29, 28, 255, 148, 164, 149, 136, 160, 155, 173, 221, 222, 134, 122, 232, 202, 215, 224,
    208, 220, 204, 187, 183, 179, 177, 168, 178, 200, 226, 195, 154, 184, 174, 126, 120, 191, 157, 194, 170, 189, 162, 161,
    150, 193, 142, 137, 171, 176, 185, 167, 186, 112, 175, 192, 188, 156, 140, 143, 123, 133, 128, 147, 138, 146, 114, 223,
    151, 249, 216, 238, 236, 253, 227, 218, 230, 247, 135, 180, 241, 233, 246, 244, 231, 139, 245, 243, 251, 235, 201, 196,
    240, 214, 152, 182, 205, 181, 127, 27, 212, 211, 210, 213, 228, 197, 169, 159, 131, 172, 105, 80, 98, 96, 97, 81,
    207, 145, 116, 115, 144, 130, 153, 121, 107, 132, 109, 110, 124, 111, 82, 108, 118, 141, 113, 129, 119, 125, 165, 117,
    92, 106, 83, 72, 99, 93, 65, 79, 166, 237, 163, 199, 190, 225, 209, 203, 198, 217, 219, 206, 234, 248, 158, 239,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
};

/* ---- byte scans ---- */

/* memchr2 / memchr3: the first position holding a byte of set[0..n] (n is 2 or 3). */
static const uint8_t *find_bytes(const uint8_t *p, const uint8_t *end, const uint8_t *set, size_t n) {
#ifdef CHN_HAVE_SSE2
    const __m128i b0 = _mm_set1_epi8((char)set[0]);
    const __m128i b1 = _mm_set1_epi8((char)set[1]);
    const __m128i b2 = _mm_set1_epi8((char)set[n == 3 ? 2 : 1]);
    while ((size_t)(end - p) >= 16) {
        const __m128i v = _mm_loadu_si128((const __m128i *)(const void *)p);
        const __m128i eq = _mm_or_si128(_mm_or_si128(_mm_cmpeq_epi8(v, b0), _mm_cmpeq_epi8(v, b1)),
            _mm_cmpeq_epi8(v, b2));
        const unsigned mask = (unsigned)_mm_movemask_epi8(eq);
        if (mask) {
            unsigned k = 0;
            while (!(mask & (1u << k))) ++k;
            return p + k;
        }
        p += 16;
    }
#endif
    for (; p < end; ++p) {
        const uint8_t c = *p;
        if (c == set[0] || c == set[1] || (n == 3 && c == set[2])) return p;
    }
    return NULL;
}

/* The first position whose byte is in table. */
static const uint8_t *find_table(const uint8_t *p, const uint8_t *end, const bool *table) {
    for (; p < end; ++p) {
        if (table[*p]) return p;
    }
    return NULL;
}

/* ---- Memmem ---- */

/* String::from_utf8_lossy(bytes).chars().count(): the valid chars, plus one U+FFFD per
 * invalid run as core::str::Utf8Chunks delimits them. */
static size_t char_len_lossy(const uint8_t *s, size_t n) {
    /* Each sequence attempt that fails ends an invalid run (one U+FFFD); the next chunk
     * starts where the attempt stopped, as in Utf8Chunks::next. */
    size_t count = 0, i = 0;
    while (i < n) {
        const uint8_t byte = s[i++];
        ++count;
        if (byte < 128) continue;
        const uint8_t b1 = i < n ? s[i] : 0;
        if (byte >= 0xC2 && byte <= 0xDF) {
            if ((int8_t)b1 < -64) ++i;
        } else if (byte >= 0xE0 && byte <= 0xEF) {
            const bool ok = (byte == 0xE0 && b1 >= 0xA0 && b1 <= 0xBF)
                || (byte >= 0xE1 && byte <= 0xEC && b1 >= 0x80 && b1 <= 0xBF)
                || (byte == 0xED && b1 >= 0x80 && b1 <= 0x9F)
                || (byte >= 0xEE && byte <= 0xEF && b1 >= 0x80 && b1 <= 0xBF);
            if (!ok) continue;
            ++i;
            if ((int8_t)(i < n ? s[i] : 0) < -64) ++i;
        } else if (byte >= 0xF0 && byte <= 0xF4) {
            const bool ok = (byte == 0xF0 && b1 >= 0x90 && b1 <= 0xBF)
                || (byte >= 0xF1 && byte <= 0xF3 && b1 >= 0x80 && b1 <= 0xBF)
                || (byte == 0xF4 && b1 >= 0x80 && b1 <= 0x8F);
            if (!ok) continue;
            ++i;
            if ((int8_t)(i < n ? s[i] : 0) >= -64) continue;
            ++i;
            if ((int8_t)(i < n ? s[i] : 0) < -64) ++i;
        }
    }
    return count;
}

/* Memmem::new */
static bool memmem_new(chn_memmem *m, const uint8_t *pat, size_t len) {
    m->needle = NULL;
    m->len = len;
    m->char_len = char_len_lossy(pat, len);
    m->rare = 0;
    if (len) {
        if (!(m->needle = (uint8_t *)malloc(len))) return false;
        memcpy(m->needle, pat, len);
        for (size_t i = 1; i < len; ++i) {
            if (byte_rank[pat[i]] < byte_rank[pat[m->rare]]) m->rare = i;
        }
    }
    return true;
}

static void memmem_drop(chn_memmem *m) {
    free(m->needle);
    m->needle = NULL;
    m->len = 0;
}

bool chn_memmem_find(const chn_memmem *m, bool fast, const uint8_t *haystack, size_t len, size_t *pos) {
    const size_t n = m->len;
    if (n == 0) {
        *pos = 0;
        return true;
    }
    if (n > len) return false;
    const size_t last = len - n; /* the last candidate start */
    if (!fast) {
        for (size_t i = 0; i <= last; ++i) {
            if (memcmp(haystack + i, m->needle, n) == 0) {
                *pos = i;
                return true;
            }
        }
        return false;
    }
    const uint8_t rare = m->needle[m->rare];
    const uint8_t *p = haystack + m->rare;
    const uint8_t *end = haystack + last + m->rare + 1;
    while (p < end) {
        const uint8_t *q = (const uint8_t *)memchr(p, rare, (size_t)(end - p));
        if (!q) return false;
        const uint8_t *cand = q - m->rare;
        if (memcmp(cand, m->needle, n) == 0) {
            *pos = (size_t)(cand - haystack);
            return true;
        }
        p = q + 1;
    }
    return false;
}

bool chn_memmem_is_suffix(const chn_memmem *m, const uint8_t *text, size_t len) {
    if (len < m->len) return false;
    return m->len == 0 || memcmp(text + (len - m->len), m->needle, m->len) == 0;
}

/* ---- the leftmost-first multi-literal searcher ---- */

void chn_multi_drop(chn_multi *m) {
    chn_seq_drop(&m->lits);
    free(m->trie_child);
    free(m->trie_sib);
    free(m->trie_byte);
    free(m->trie_lit);
    m->trie_child = m->trie_sib = NULL;
    m->trie_byte = NULL;
    m->trie_lit = NULL;
    m->trie_len = 0;
}

chn_lits_status chn_multi_new(chn_multi *m, chn_seq *lits) {
    memset(m, 0, sizeof(*m));
    m->lits = *lits;
    chn_seq_init_empty(lits);
    size_t total = 1;
    for (size_t i = 0; i < m->lits.len; ++i) total += m->lits.lits[i].len;
    m->trie_child = (uint32_t *)calloc(total, sizeof(uint32_t));
    m->trie_sib = (uint32_t *)calloc(total, sizeof(uint32_t));
    m->trie_byte = (uint8_t *)calloc(total, 1);
    m->trie_lit = (size_t *)malloc(total * sizeof(size_t));
    if (!m->trie_child || !m->trie_sib || !m->trie_byte || !m->trie_lit) {
        chn_multi_drop(m);
        return CHN_LITS_NOMEM;
    }
    m->trie_lit[0] = SIZE_MAX;
    m->trie_len = 1;
    for (size_t i = 0; i < m->lits.len; ++i) {
        const chn_lit *lit = &m->lits.lits[i];
        uint32_t node = 0;
        for (size_t k = 0; k < lit->len; ++k) {
            const uint8_t b = lit->bytes[k];
            uint32_t child = m->trie_child[node];
            while (child && m->trie_byte[child] != b) child = m->trie_sib[child];
            if (!child) {
                child = (uint32_t)m->trie_len++;
                m->trie_byte[child] = b;
                m->trie_lit[child] = SIZE_MAX;
                m->trie_sib[child] = m->trie_child[node];
                m->trie_child[node] = child;
            }
            node = child;
        }
        if (i < m->trie_lit[node]) m->trie_lit[node] = i;
        if (lit->len && !m->first[lit->bytes[0]]) {
            m->first[lit->bytes[0]] = true;
            if (m->first_count < 3) m->first_list[m->first_count] = lit->bytes[0];
            ++m->first_count;
        }
    }
    return CHN_LITS_OK;
}

/* The literal of smallest index matching at haystack[p..] (trie walk); SIZE_MAX: none. */
static size_t multi_at(const chn_multi *m, const uint8_t *haystack, size_t len, size_t p) {
    size_t best = m->trie_lit[0];
    uint32_t node = 0;
    for (size_t k = p; k < len; ++k) {
        uint32_t child = m->trie_child[node];
        while (child && m->trie_byte[child] != haystack[k]) child = m->trie_sib[child];
        if (!child) break;
        node = child;
        if (m->trie_lit[node] < best) best = m->trie_lit[node];
    }
    return best;
}

bool chn_multi_find(const chn_multi *m, bool fast, const uint8_t *haystack, size_t len,
    size_t *s, size_t *e) {
    if (!fast) {
        for (size_t p = 0; p <= len; ++p) {
            for (size_t i = 0; i < m->lits.len; ++i) {
                const chn_lit *lit = &m->lits.lits[i];
                if (lit->len <= len - p && (lit->len == 0 || memcmp(haystack + p, lit->bytes, lit->len) == 0)) {
                    *s = p;
                    *e = p + lit->len;
                    return true;
                }
            }
        }
        return false;
    }
    if (m->trie_lit[0] != SIZE_MAX) {
        /* An empty literal matches at 0 (not reached: the matchers exclude them). */
        *s = *e = 0;
        return true;
    }
    const uint8_t *p = haystack, *end = haystack + len;
    while (p < end) {
        const uint8_t *q;
        if (m->first_count == 1) q = (const uint8_t *)memchr(p, m->first_list[0], (size_t)(end - p));
        else if (m->first_count <= 3) q = find_bytes(p, end, m->first_list, m->first_count);
        else q = find_table(p, end, m->first);
        if (!q) return false;
        const size_t at = (size_t)(q - haystack);
        const size_t i = multi_at(m, haystack, len, at);
        if (i != SIZE_MAX) {
            *s = at;
            *e = at + m->lits.lits[i].len;
            return true;
        }
        p = q + 1;
    }
    return false;
}

/* ---- LiteralSearcher ---- */

void chn_literal_searcher_empty(chn_literal_searcher *ls) {
    memset(ls, 0, sizeof(*ls));
    chn_seq_init_empty(&ls->multi.lits);
    ls->complete = false; /* Seq::infinite().is_exact() */
    ls->kind = CHN_MATCHER_EMPTY;
    ls->fast = true;
}

void chn_literal_searcher_drop(chn_literal_searcher *ls) {
    memmem_drop(&ls->lcp);
    memmem_drop(&ls->lcs);
    memmem_drop(&ls->single);
    chn_multi_drop(&ls->multi);
    ls->kind = CHN_MATCHER_EMPTY;
}

chn_lits_status chn_literal_searcher_new(chn_literal_searcher *ls, chn_seq *lits, bool suffixes,
    bool fast) {
    chn_literal_searcher_empty(ls);
    ls->fast = fast;
    /* LiteralSearcher::new */
    ls->complete = chn_seq_is_exact(lits);
    const uint8_t *fix = NULL;
    size_t fix_len = 0;
    if (!chn_seq_longest_common_prefix(lits, &fix, &fix_len)) fix_len = 0;
    if (!memmem_new(&ls->lcp, fix, fix_len)) goto nomem;
    if (!chn_seq_longest_common_suffix(lits, &fix, &fix_len)) fix_len = 0;
    if (!memmem_new(&ls->lcs, fix, fix_len)) goto nomem;
    /* SingleByteSet::prefixes / ::suffixes (all_ascii only chooses between AC and Packed,
     * which give the same leftmost-first results, so it is not kept). */
    bool sset_complete = true;
    if (lits->finite) {
        for (size_t i = 0; i < lits->len; ++i) {
            const chn_lit *lit = &lits->lits[i];
            sset_complete = sset_complete && lit->len == 1;
            if (lit->len) {
                const uint8_t b = suffixes ? lit->bytes[lit->len - 1] : lit->bytes[0];
                if (!ls->sset_sparse[b]) {
                    ls->sset_dense[ls->sset_len++] = b;
                    ls->sset_sparse[b] = true;
                }
            }
        }
    }
    /* Matcher::new */
    size_t min_len;
    if ((lits->finite && lits->len == 0) || (chn_seq_min_literal_len(lits, &min_len) && min_len == 0)
        || !lits->finite || ls->sset_len >= 26) {
        ls->kind = CHN_MATCHER_EMPTY;
    } else if (sset_complete) {
        ls->kind = CHN_MATCHER_BYTES;
    } else if (lits->len == 1) {
        ls->kind = CHN_MATCHER_MEMMEM;
        if (!memmem_new(&ls->single, lits->lits[0].bytes, lits->lits[0].len)) goto nomem;
    } else {
        ls->kind = CHN_MATCHER_MULTI;
        if (chn_multi_new(&ls->multi, lits) != CHN_LITS_OK) goto nomem;
    }
    if (ls->kind != CHN_MATCHER_BYTES) {
        memset(ls->sset_sparse, 0, sizeof(ls->sset_sparse));
        ls->sset_len = 0;
    }
    chn_seq_drop(lits);
    return CHN_LITS_OK;
nomem:
    chn_seq_drop(lits);
    chn_literal_searcher_drop(ls);
    return CHN_LITS_NOMEM;
}

bool chn_literal_searcher_complete(const chn_literal_searcher *ls) {
    return ls->complete && !chn_literal_searcher_is_empty(ls);
}

size_t chn_literal_searcher_len(const chn_literal_searcher *ls) {
    switch (ls->kind) {
    case CHN_MATCHER_EMPTY: return 0;
    case CHN_MATCHER_BYTES: return ls->sset_len;
    case CHN_MATCHER_MEMMEM: return 1;
    case CHN_MATCHER_MULTI: return ls->multi.lits.len;
    }
    return 0;
}

bool chn_literal_searcher_find(const chn_literal_searcher *ls, const uint8_t *haystack, size_t len,
    size_t *s, size_t *e) {
    switch (ls->kind) {
    case CHN_MATCHER_EMPTY:
        *s = *e = 0;
        return true;
    case CHN_MATCHER_BYTES: {
        const uint8_t *q;
        if (!ls->fast) q = find_table(haystack, haystack + len, ls->sset_sparse);
        else if (ls->sset_len == 1) q = (const uint8_t *)memchr(haystack, ls->sset_dense[0], len);
        else if (ls->sset_len <= 3) q = find_bytes(haystack, haystack + len, ls->sset_dense, ls->sset_len);
        else q = find_table(haystack, haystack + len, ls->sset_sparse);
        if (!q) return false;
        *s = (size_t)(q - haystack);
        *e = *s + 1;
        return true;
    }
    case CHN_MATCHER_MEMMEM: {
        size_t pos;
        if (!chn_memmem_find(&ls->single, ls->fast, haystack, len, &pos)) return false;
        *s = pos;
        *e = pos + ls->single.len;
        return true;
    }
    case CHN_MATCHER_MULTI:
        return chn_multi_find(&ls->multi, ls->fast, haystack, len, s, e);
    }
    return false;
}

/* LiteralSearcher::iter's i-th literal (false when exhausted). */
static bool literal_at(const chn_literal_searcher *ls, size_t i, const uint8_t **bytes, size_t *len) {
    switch (ls->kind) {
    case CHN_MATCHER_EMPTY:
        return false;
    case CHN_MATCHER_BYTES:
        if (i >= ls->sset_len) return false;
        *bytes = &ls->sset_dense[i];
        *len = 1;
        return true;
    case CHN_MATCHER_MEMMEM:
        if (i > 0 || ls->single.len == 0) return false;
        *bytes = ls->single.needle;
        *len = ls->single.len;
        return true;
    case CHN_MATCHER_MULTI:
        if (i >= ls->multi.lits.len) return false;
        *bytes = ls->multi.lits.lits[i].bytes;
        *len = ls->multi.lits.lits[i].len;
        return true;
    }
    return false;
}

bool chn_literal_searcher_find_start(const chn_literal_searcher *ls, const uint8_t *haystack,
    size_t len, size_t *s, size_t *e) {
    const uint8_t *lit;
    size_t n;
    for (size_t i = 0; literal_at(ls, i, &lit, &n); ++i) {
        if (n > len) continue;
        if (n == 0 || memcmp(lit, haystack, n) == 0) {
            *s = 0;
            *e = n;
            return true;
        }
    }
    return false;
}

bool chn_literal_searcher_find_end(const chn_literal_searcher *ls, const uint8_t *haystack,
    size_t len, size_t *s, size_t *e) {
    const uint8_t *lit;
    size_t n;
    for (size_t i = 0; literal_at(ls, i, &lit, &n); ++i) {
        if (n > len) continue;
        if (n == 0 || memcmp(lit, haystack + (len - n), n) == 0) {
            *s = len - n;
            *e = len;
            return true;
        }
    }
    return false;
}
