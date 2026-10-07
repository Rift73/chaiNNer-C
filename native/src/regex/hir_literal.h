/* Port of regex-syntax 0.7.2 src/hir/literal.rs (Extractor, Seq, Literal, PreferenceTrie), MIT OR Apache-2.0. */
/* X2's private header: the literal sequences regex 1.8.4's exec.rs extracts from a HIR
 * (literal_analysis) and literal/imp.rs searches for. Only the parts regex 1.8.4 reaches
 * are ported. */
#ifndef CHAINNER_REGEX_HIR_LITERAL_H
#define CHAINNER_REGEX_HIR_LITERAL_H
#include "chainner_hir.h"

/* Literal { bytes: Vec<u8>, exact: bool } */
typedef struct chn_lit {
    uint8_t *bytes;
    size_t len;
    size_t cap;
    bool exact;
} chn_lit;

/* Seq { literals: Option<Vec<Literal>> }: finite false is None (the infinite sequence). */
typedef struct chn_seq {
    bool finite;
    chn_lit *lits;
    size_t len;
    size_t cap;
} chn_seq;

/* Drop for Seq; leaves *seq as Seq::empty(). */
void chn_seq_drop(chn_seq *seq);
/* Seq::empty */
void chn_seq_init_empty(chn_seq *seq);
/* Seq::is_exact */
bool chn_seq_is_exact(const chn_seq *seq);
/* Seq::make_inexact */
void chn_seq_make_inexact(chn_seq *seq);
/* Seq::min_literal_len: false is None. */
bool chn_seq_min_literal_len(const chn_seq *seq, size_t *out);
/* Seq::longest_common_prefix / longest_common_suffix: false is None; otherwise
 * [*ptr, *ptr + *len) points into the first literal. */
bool chn_seq_longest_common_prefix(const chn_seq *seq, const uint8_t **ptr, size_t *len);
bool chn_seq_longest_common_suffix(const chn_seq *seq, const uint8_t **ptr, size_t *len);
/* Seq::union (other is drained); false on allocation failure. */
bool chn_seq_union(chn_seq *seq, chn_seq *other);

/* exec.rs's literal_analysis(expr): the prefix and suffix sequences, each shrunk by the
 * (5,50)/(4,30)/(3,20) keep_*_bytes + minimize_by_preference loop. false on allocation
 * failure (both outputs are then empty). */
bool chn_literal_analysis(const chn_hir *expr, chn_seq *prefixes, chn_seq *suffixes);

#endif
