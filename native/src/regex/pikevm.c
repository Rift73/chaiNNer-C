/* Port of regex 1.8.4 src/pikevm.rs (with src/sparse.rs's SparseSet), MIT OR Apache-2.0. */
/* The Pike VM: an NFA simulation in time linear in the text, recording captures. matches
 * is regex 1.8.4's one-element `&mut [false]` (exec_nfa always passes it), so
 * all_matched follows matched, as it does there. */
#include "prog.h"
#include <stdlib.h>

/* Fsm { prog, stack, input } */
typedef struct fsm {
    const chn_prog *prog;
    chn_pikevm_cache *cache;
    const chn_input *input;
    bool oom;
} fsm;

static void threads_drop(chn_pikevm_threads *t) {
    free(t->dense);
    free(t->sparse);
    free(t->caps);
    t->dense = t->sparse = t->caps = NULL;
    t->len = 0;
}

void chn_pikevm_cache_drop(chn_pikevm_cache *cache) {
    threads_drop(&cache->clist);
    threads_drop(&cache->nlist);
    free(cache->stack);
    cache->stack = NULL;
    cache->stack_cap = 0;
    cache->ready = false;
}

/* Threads::resize (done once: a cache belongs to one program). */
static bool threads_init(chn_pikevm_threads *t, size_t num_insts, size_t ncaps) {
    t->slots_per_thread = ncaps * 2;
    t->len = 0;
    t->dense = (size_t *)malloc((num_insts ? num_insts : 1) * sizeof(size_t));
    t->sparse = (size_t *)calloc(num_insts ? num_insts : 1, sizeof(size_t));
    const size_t ncells = t->slots_per_thread * num_insts;
    t->caps = (size_t *)malloc((ncells ? ncells : 1) * sizeof(size_t));
    if (!t->dense || !t->sparse || !t->caps) return false;
    for (size_t i = 0; i < ncells; ++i) t->caps[i] = CHN_REGEX_NO_SLOT;
    return true;
}

static bool cache_ready(chn_pikevm_cache *cache, const chn_prog *prog) {
    if (cache->ready) return true;
    if (!threads_init(&cache->clist, prog->len, prog->captures_len)
        || !threads_init(&cache->nlist, prog->len, prog->captures_len)) {
        chn_pikevm_cache_drop(cache);
        return false;
    }
    /* add() pushes at most one frame per instruction it inserts, plus its first. */
    cache->stack_cap = prog->len + 1;
    cache->stack = (chn_pikevm_frame *)malloc(cache->stack_cap * sizeof(chn_pikevm_frame));
    if (!cache->stack) {
        chn_pikevm_cache_drop(cache);
        return false;
    }
    cache->ready = true;
    return true;
}

/* SparseSet::contains */
static inline bool set_contains(const chn_pikevm_threads *t, size_t value) {
    const size_t i = t->sparse[value];
    return i < t->len && t->dense[i] == value;
}

/* SparseSet::insert */
static inline void set_insert(chn_pikevm_threads *t, size_t value) {
    t->dense[t->len] = value;
    t->sparse[value] = t->len;
    ++t->len;
}

/* Threads::caps */
static inline size_t *thread_caps_of(chn_pikevm_threads *t, size_t pc) {
    return t->caps + pc * t->slots_per_thread;
}

static inline bool stack_push(fsm *f, size_t *len, size_t ip_or_slot, size_t pos, bool capture) {
    chn_pikevm_cache *cache = f->cache;
    if (*len == cache->stack_cap) {
        const size_t cap = cache->stack_cap * 2;
        chn_pikevm_frame *stack = (chn_pikevm_frame *)realloc(cache->stack, cap * sizeof(*stack));
        if (!stack) {
            f->oom = true;
            return false;
        }
        cache->stack = stack;
        cache->stack_cap = cap;
    }
    chn_pikevm_frame *frame = &cache->stack[(*len)++];
    frame->ip_or_slot = ip_or_slot;
    frame->pos = pos;
    frame->capture = capture;
    return true;
}

/* Fsm::add (with add_step inlined): follows epsilon transitions from ip into nlist.
 * thread_caps has ncaps slots. */
static void add(fsm *f, chn_pikevm_threads *nlist, size_t *thread_caps, size_t ncaps, size_t ip0,
    chn_input_at at) {
    const chn_prog *prog = f->prog;
    size_t len = 0;
    if (!stack_push(f, &len, ip0, 0, false)) return;
    while (len) {
        const chn_pikevm_frame frame = f->cache->stack[--len];
        if (frame.capture) {
            thread_caps[frame.ip_or_slot] = frame.pos;
            continue;
        }
        size_t ip = frame.ip_or_slot;
        for (;;) {
            if (set_contains(nlist, ip)) break;
            set_insert(nlist, ip);
            const chn_inst *inst = &prog->insts[ip];
            switch (inst->kind) {
            case CHN_INST_EMPTY_LOOK:
                if (!chn_input_is_empty_match(f->input, at, (chn_empty_look)inst->look)) goto next_frame;
                ip = inst->next;
                continue;
            case CHN_INST_SAVE:
                if (inst->slot < ncaps) {
                    if (!stack_push(f, &len, inst->slot, thread_caps[inst->slot], true)) return;
                    thread_caps[inst->slot] = at.pos;
                }
                ip = inst->next;
                continue;
            case CHN_INST_SPLIT:
                if (!stack_push(f, &len, inst->next2, 0, false)) return;
                ip = inst->next;
                continue;
            default: {
                /* Match, Char, Ranges, Bytes: copy the captures (zip). */
                size_t *t = thread_caps_of(nlist, ip);
                const size_t n = ncaps < nlist->slots_per_thread ? ncaps : nlist->slots_per_thread;
                for (size_t k = 0; k < n; ++k) t[k] = thread_caps[k];
                goto next_frame;
            }
            }
        }
    next_frame:;
    }
}

/* Fsm::step: true on a Match instruction. */
static bool step(fsm *f, chn_pikevm_threads *nlist, bool *matched, size_t *slots, size_t nslots,
    size_t *thread_caps, size_t ncaps, size_t ip, chn_input_at at, chn_input_at at_next) {
    const chn_inst *inst = &f->prog->insts[ip];
    switch (inst->kind) {
    case CHN_INST_MATCH: {
        *matched = true; /* matches[match_slot] = true (match_slot 0 < matches.len() 1) */
        const size_t n = nslots < ncaps ? nslots : ncaps;
        for (size_t k = 0; k < n; ++k) slots[k] = thread_caps[k];
        return true;
    }
    case CHN_INST_CHAR:
        if (inst->c == at.c) add(f, nlist, thread_caps, ncaps, inst->next, at_next);
        return false;
    case CHN_INST_RANGES:
        if (chn_inst_ranges_matches(f->prog, inst, at.c)) add(f, nlist, thread_caps, ncaps, inst->next, at_next);
        return false;
    case CHN_INST_BYTES:
        if (at.byte >= 0 && inst->start <= (uint8_t)at.byte && (uint8_t)at.byte <= inst->end) {
            add(f, nlist, thread_caps, ncaps, inst->next, at_next);
        }
        return false;
    default:
        return false;
    }
}

int chn_pikevm_exec(const chn_prog *prog, chn_pikevm_cache *cache, bool *matches0,
    size_t *slots, size_t nslots, bool quit_after_match, const chn_input *input, size_t start,
    size_t end) {
    if (!cache_ready(cache, prog)) return -1;
    fsm f = {prog, cache, input, false};
    chn_pikevm_threads *clist = &cache->clist, *nlist = &cache->nlist;
    chn_input_at at = chn_input_at_pos(input, start);
    bool matched = false, all_matched = false;
    clist->len = 0;
    nlist->len = 0;
    const size_t ncaps = clist->slots_per_thread;
    const bool has_prefixes = !chn_literal_searcher_is_empty(&prog->prefixes);
    for (;;) {
        if (clist->len == 0) {
            if (matched || all_matched || (at.pos != 0 && prog->is_anchored_start)) break;
            if (has_prefixes) {
                chn_input_at next;
                if (!chn_input_prefix_at(input, &prog->prefixes, at, &next)) break;
                at = next;
            }
        }
        if (clist->len == 0 || (!prog->is_anchored_start && !all_matched)) {
            add(&f, clist, slots, nslots, 0, at);
        }
        const chn_input_at at_next = chn_input_at_pos(input, at.pos + at.len);
        for (size_t i = 0; i < clist->len; ++i) {
            const size_t ip = clist->dense[i];
            if (step(&f, nlist, matches0, slots, nslots, thread_caps_of(clist, ip), ncaps, ip, at, at_next)) {
                matched = true;
                all_matched = all_matched || *matches0;
                if (quit_after_match) goto done;
                break; /* prog.matches.len() == 1: leftmost-first */
            }
        }
        if (at.pos >= end) break;
        at = at_next;
        chn_pikevm_threads tmp = *clist;
        *clist = *nlist;
        *nlist = tmp;
        nlist->len = 0;
    }
done:
    /* The swaps may leave clist and nlist exchanged; both stay owned by the cache. */
    if (f.oom) return -1;
    return matched ? 1 : 0;
}
