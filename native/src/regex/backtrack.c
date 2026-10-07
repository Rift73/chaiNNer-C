/* Port of regex 1.8.4 src/backtrack.rs, MIT OR Apache-2.0. */
/* The bounded backtracker: each (instruction, position) pair is visited at most once,
 * tracked in a bitset of prog.len() * (text.len() + 1) bits. */
#include "prog.h"
#include <stdlib.h>
#include <string.h>

#define BIT_SIZE ((size_t)32)
#define MAX_SIZE_BYTES ((size_t)256 * (1 << 10))

bool chn_backtrack_should_exec(size_t num_insts, size_t text_len) {
    const size_t size = ((num_insts * (text_len + 1) + BIT_SIZE - 1) / BIT_SIZE) * 4;
    return size <= MAX_SIZE_BYTES;
}

void chn_backtrack_cache_drop(chn_backtrack_cache *cache) {
    free(cache->jobs);
    free(cache->visited);
    memset(cache, 0, sizeof(*cache));
}

/* Bounded { prog, input, matches, slots, m } */
typedef struct bounded {
    const chn_prog *prog;
    const chn_input *input;
    bool *matched;
    size_t *slots;
    size_t nslots;
    chn_backtrack_cache *m;
    bool oom;
} bounded;

/* Bounded::clear */
static bool clear(bounded *b) {
    b->m->jobs_len = 0;
    const size_t visited_len = (b->prog->len * (b->input->len + 1) + BIT_SIZE - 1) / BIT_SIZE;
    if (visited_len > b->m->visited_cap) {
        uint32_t *visited = (uint32_t *)realloc(b->m->visited, visited_len * sizeof(uint32_t));
        if (!visited) return false;
        b->m->visited = visited;
        b->m->visited_cap = visited_len;
    }
    memset(b->m->visited, 0, visited_len * sizeof(uint32_t));
    return true;
}

static bool push_job(bounded *b, const chn_backtrack_job *job) {
    chn_backtrack_cache *m = b->m;
    if (m->jobs_len == m->jobs_cap) {
        const size_t cap = m->jobs_cap ? m->jobs_cap * 2 : 64;
        chn_backtrack_job *jobs = (chn_backtrack_job *)realloc(m->jobs, cap * sizeof(*jobs));
        if (!jobs) {
            b->oom = true;
            return false;
        }
        m->jobs = jobs;
        m->jobs_cap = cap;
    }
    m->jobs[m->jobs_len++] = *job;
    return true;
}

/* Bounded::has_visited */
static inline bool has_visited(bounded *b, size_t ip, chn_input_at at) {
    const size_t k = ip * (b->input->len + 1) + at.pos;
    const size_t k1 = k / BIT_SIZE;
    const uint32_t k2 = (uint32_t)1 << (k & (BIT_SIZE - 1));
    if ((b->m->visited[k1] & k2) == 0) {
        b->m->visited[k1] |= k2;
        return false;
    }
    return true;
}

/* Bounded::step */
static bool step(bounded *b, size_t ip, chn_input_at at) {
    const chn_prog *prog = b->prog;
    for (;;) {
        if (has_visited(b, ip, at)) return false;
        const chn_inst *inst = &prog->insts[ip];
        switch (inst->kind) {
        case CHN_INST_MATCH:
            *b->matched = true;
            return true;
        case CHN_INST_SAVE:
            if (inst->slot < b->nslots) {
                chn_backtrack_job job;
                job.restore = true;
                job.ip_or_slot = inst->slot;
                job.old_pos = b->slots[inst->slot];
                job.at = at;
                if (!push_job(b, &job)) return false;
                b->slots[inst->slot] = at.pos;
            }
            ip = inst->next;
            break;
        case CHN_INST_SPLIT: {
            chn_backtrack_job job;
            job.restore = false;
            job.ip_or_slot = inst->next2;
            job.at = at;
            job.old_pos = 0;
            if (!push_job(b, &job)) return false;
            ip = inst->next;
            break;
        }
        case CHN_INST_EMPTY_LOOK:
            if (!chn_input_is_empty_match(b->input, at, (chn_empty_look)inst->look)) return false;
            ip = inst->next;
            break;
        case CHN_INST_CHAR:
            if (inst->c != at.c) return false;
            ip = inst->next;
            at = chn_input_at_pos(b->input, at.pos + at.len);
            break;
        case CHN_INST_RANGES:
            if (!chn_inst_ranges_matches(prog, inst, at.c)) return false;
            ip = inst->next;
            at = chn_input_at_pos(b->input, at.pos + at.len);
            break;
        case CHN_INST_BYTES:
            if (at.byte < 0 || (uint8_t)at.byte < inst->start || (uint8_t)at.byte > inst->end) return false;
            ip = inst->next;
            at = chn_input_at_pos(b->input, at.pos + at.len);
            break;
        default:
            return false;
        }
    }
}

/* Bounded::backtrack */
static bool backtrack(bounded *b, chn_input_at start) {
    chn_backtrack_job first;
    first.restore = false;
    first.ip_or_slot = 0;
    first.at = start;
    first.old_pos = 0;
    if (!push_job(b, &first)) return false;
    while (b->m->jobs_len) {
        const chn_backtrack_job job = b->m->jobs[--b->m->jobs_len];
        if (job.restore) {
            if (job.ip_or_slot < b->nslots) b->slots[job.ip_or_slot] = job.old_pos;
            continue;
        }
        if (step(b, job.ip_or_slot, job.at)) return true; /* prog.matches.len() == 1 */
        if (b->oom) return false;
    }
    return false;
}

int chn_backtrack_exec(const chn_prog *prog, chn_backtrack_cache *cache, bool *matched,
    size_t *slots, size_t nslots, const chn_input *input, size_t start, size_t end) {
    bounded b = {prog, input, matched, slots, nslots, cache, false};
    chn_input_at at = chn_input_at_pos(input, start);
    /* Bounded::exec_ */
    if (!clear(&b)) return -1;
    if (prog->is_anchored_start) {
        if (at.pos != 0) return 0;
        const bool m = backtrack(&b, at);
        return b.oom ? -1 : (m ? 1 : 0);
    }
    const bool has_prefixes = !chn_literal_searcher_is_empty(&prog->prefixes);
    for (;;) {
        if (has_prefixes) {
            chn_input_at next;
            if (!chn_input_prefix_at(input, &prog->prefixes, at, &next)) break;
            at = next;
        }
        const bool m = backtrack(&b, at);
        if (b.oom) return -1;
        if (m) return 1; /* prog.matches.len() == 1 */
        if (at.pos >= end) break;
        at = chn_input_at_pos(input, at.pos + at.len);
    }
    return 0;
}
