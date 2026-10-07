/* Port of regex 1.8.4 src/prog.rs, MIT OR Apache-2.0. */
#include "prog.h"
#include <stdlib.h>
#include <string.h>

void chn_prog_init(chn_prog *prog) {
    memset(prog, 0, sizeof(*prog));
    prog->only_utf8 = true;
    chn_literal_searcher_empty(&prog->prefixes);
    prog->dfa_size_limit = CHN_REGEX_DFA_SIZE_LIMIT;
}

void chn_prog_drop(chn_prog *prog) {
    free(prog->insts);
    free(prog->ranges);
    for (size_t i = 0; i < prog->captures_len; ++i) free(prog->captures[i].ptr);
    free(prog->captures);
    chn_literal_searcher_drop(&prog->prefixes);
    chn_prog_init(prog);
}

bool chn_inst_ranges_matches(const chn_prog *prog, const chn_inst *inst, uint32_t c) {
    const chn_char_range *ranges = prog->ranges + inst->ranges;
    const size_t n = inst->nranges;
    /* The first four ranges linearly, then a binary search. */
    const size_t head = n < 4 ? n : 4;
    for (size_t i = 0; i < head; ++i) {
        if (c < ranges[i].start) return false;
        if (c <= ranges[i].end) return true;
    }
    size_t lo = 0, hi = n;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (ranges[mid].end < c) lo = mid + 1;
        else if (ranges[mid].start > c) hi = mid;
        else return true;
    }
    return false;
}
