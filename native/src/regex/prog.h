/* Port of regex 1.8.4 (src/prog.rs, with the engine-internal declarations of src/compile.rs, src/input.rs, src/utf8.rs, src/sparse.rs, src/pikevm.rs, src/backtrack.rs, src/dfa.rs, src/literal/imp.rs and src/exec.rs), MIT OR Apache-2.0. */
/* X2's private engine header: the compiled program (prog.rs), the inputs (input.rs), the
 * literal searcher (literal/imp.rs, in prefilter.c), the engines' caches and the compiled
 * regex itself. It is shared by X2's files only (compile, prog, input, pikevm, backtrack,
 * dfa, prefilter, exec, iter); nothing outside chainner_regex.lib includes it. */
#ifndef CHAINNER_REGEX_PROG_H
#define CHAINNER_REGEX_PROG_H
#include "chainner_hir.h"
#include "chainner_regex.h"
#include "hir_literal.h"

/* InstPtr (prog.rs). */
typedef size_t chn_inst_ptr;

/* Inst's discriminant (prog.rs). */
typedef enum chn_inst_kind {
    CHN_INST_MATCH = 0,      /* Inst::Match(usize) */
    CHN_INST_SAVE = 1,       /* Inst::Save(InstSave) */
    CHN_INST_SPLIT = 2,      /* Inst::Split(InstSplit) */
    CHN_INST_EMPTY_LOOK = 3, /* Inst::EmptyLook(InstEmptyLook) */
    CHN_INST_CHAR = 4,       /* Inst::Char(InstChar) */
    CHN_INST_RANGES = 5,     /* Inst::Ranges(InstRanges) */
    CHN_INST_BYTES = 6       /* Inst::Bytes(InstBytes) */
} chn_inst_kind;

/* EmptyLook (prog.rs). */
typedef enum chn_empty_look {
    CHN_LOOK_START_LINE = 0,
    CHN_LOOK_END_LINE = 1,
    CHN_LOOK_START_TEXT = 2,
    CHN_LOOK_END_TEXT = 3,
    CHN_LOOK_WORD_BOUNDARY = 4,
    CHN_LOOK_NOT_WORD_BOUNDARY = 5,
    CHN_LOOK_WORD_BOUNDARY_ASCII = 6,
    CHN_LOOK_NOT_WORD_BOUNDARY_ASCII = 7
} chn_empty_look;

/* (char, char), one range of InstRanges::ranges. */
typedef struct chn_char_range {
    uint32_t start;
    uint32_t end;
} chn_char_range;

/* Inst (prog.rs). The Rust variants' fields, flattened:
 *   Match(i)                      match_index
 *   Save { goto, slot }           next, slot
 *   Split { goto1, goto2 }        next, next2
 *   EmptyLook { goto, look }      next, look
 *   Char { goto, c }              next, c
 *   Ranges { goto, ranges }       next, prog->ranges[ranges .. ranges + nranges]
 *   Bytes { goto, start, end }    next, start, end */
typedef struct chn_inst {
    uint8_t kind;
    uint8_t look;
    uint8_t start;
    uint8_t end;
    uint32_t c;
    size_t next;
    union {
        size_t next2;
        size_t slot;
        size_t nranges;
        size_t match_index;
    };
    size_t ranges;
} chn_inst;

/* Rust's mem::size_of::<Inst>() on 64-bit targets, which compile.rs's check_size charges
 * per instruction (prog.rs's test_size_of_inst). */
#define CHN_RUST_INST_SIZE ((size_t)32)

/* Matcher's discriminant (literal/imp.rs). */
typedef enum chn_matcher_kind {
    CHN_MATCHER_EMPTY = 0,  /* Matcher::Empty */
    CHN_MATCHER_BYTES = 1,  /* Matcher::Bytes(SingleByteSet) */
    CHN_MATCHER_MEMMEM = 2, /* Matcher::Memmem(Memmem) */
    CHN_MATCHER_MULTI = 3   /* Matcher::AC / Matcher::Packed: leftmost-first over lits */
} chn_matcher_kind;

/* Memmem (literal/imp.rs): a needle and its lossy char length. */
typedef struct chn_memmem {
    uint8_t *needle;
    size_t len;
    size_t char_len; /* String::from_utf8_lossy(needle).chars().count() */
    size_t rare;     /* the needle offset the fast scan looks for first */
} chn_memmem;

/* A leftmost-first multi-literal searcher: the literals in preference order and a trie
 * over them (the fast scan), plus the set of first bytes. */
typedef struct chn_multi {
    chn_seq lits;     /* the literals, in preference order */
    uint32_t *trie_child; /* per node: first child (0: none) */
    uint32_t *trie_sib;   /* per node: next sibling (0: none) */
    uint8_t *trie_byte;   /* per node: its edge byte */
    size_t *trie_lit;     /* per node: smallest literal index ending here, SIZE_MAX: none */
    size_t trie_len;
    bool first[256];      /* the literals' first bytes */
    uint8_t first_list[3];
    size_t first_count;
} chn_multi;

/* LiteralSearcher (literal/imp.rs). */
typedef struct chn_literal_searcher {
    bool complete;          /* lits.is_exact() */
    chn_memmem lcp;         /* longest common prefix */
    chn_memmem lcs;         /* longest common suffix */
    chn_matcher_kind kind;  /* matcher */
    bool fast;              /* the harness's prefilter switch: false scans plainly */
    /* Matcher::Bytes: SingleByteSet { sparse, dense } */
    bool sset_sparse[256];
    uint8_t sset_dense[256];
    size_t sset_len;
    chn_memmem single;      /* Matcher::Memmem */
    chn_multi multi;        /* Matcher::AC / Packed */
} chn_literal_searcher;

/* Program (prog.rs). Every program here is a single regex (matches.len() == 1). */
typedef struct chn_prog {
    chn_inst *insts;
    size_t len;
    size_t cap;
    chn_char_range *ranges; /* the storage of every Ranges instruction */
    size_t ranges_len;
    size_t ranges_cap;
    chn_inst_ptr match_pc;    /* matches[0] */
    chn_hir_string *captures; /* captures: Vec<Option<String>> (ptr NULL: None) */
    size_t captures_len;
    size_t captures_cap;
    chn_inst_ptr start;
    uint8_t byte_classes[256];
    bool only_utf8;
    bool is_bytes;
    bool is_dfa;
    bool is_reverse;
    bool is_anchored_start;
    bool is_anchored_end;
    bool has_unicode_word_boundary;
    chn_literal_searcher prefixes;
    size_t dfa_size_limit;
} chn_prog;

/* Program::new into *prog. */
void chn_prog_init(chn_prog *prog);
/* Drop for Program. */
void chn_prog_drop(chn_prog *prog);
/* Program::uses_bytes */
static inline bool chn_prog_uses_bytes(const chn_prog *prog) {
    return prog->is_bytes || prog->is_dfa;
}
/* Program::needs_dotstar */
static inline bool chn_prog_needs_dotstar(const chn_prog *prog) {
    return prog->is_dfa && !prog->is_reverse && !prog->is_anchored_start;
}
/* InstRanges::matches */
bool chn_inst_ranges_matches(const chn_prog *prog, const chn_inst *inst, uint32_t c);

/* compile.rs's errors. */
typedef enum chn_compile_status {
    CHN_COMPILE_OK = 0,
    CHN_COMPILE_SYNTAX = 1,    /* Error::Syntax(message) */
    CHN_COMPILE_TOO_BIG = 2,   /* Error::CompiledTooBig(size_limit) */
    CHN_COMPILE_NOMEM = 3
} chn_compile_status;

/* Compiler::new().size_limit(size_limit).bytes(bytes).dfa(dfa).reverse(reverse)
 * .compile(&[expr]) (only_utf8 is always true here). On CHN_COMPILE_SYNTAX, *message is
 * the static message. */
chn_compile_status chn_compile(const chn_hir *expr, size_t size_limit, bool bytes, bool dfa,
    bool reverse, chn_prog *out, const char **message);

/* utf8.rs. */
#define CHN_CHAR_NONE UINT32_MAX /* input.rs's Char(u32::MAX): an absent char */
/* decode_utf8: the scalar value, or CHN_CHAR_NONE; *n gets the bytes read (when valid). */
uint32_t chn_decode_utf8(const uint8_t *src, size_t len, size_t *n);
/* decode_last_utf8 */
uint32_t chn_decode_last_utf8(const uint8_t *src, size_t len);
/* next_utf8 */
size_t chn_next_utf8(const uint8_t *text, size_t len, size_t i);

/* input.rs: CharInput (bytes false) and ByteInput (bytes true). */
typedef struct chn_input {
    const uint8_t *text;
    size_t len;
    bool bytes;
    bool only_utf8;
} chn_input;

/* InputAt */
typedef struct chn_input_at {
    size_t pos;
    uint32_t c;   /* Char */
    int byte;     /* Option<u8>: -1 is None */
    size_t len;
} chn_input_at;

/* Input::at */
chn_input_at chn_input_at_pos(const chn_input *input, size_t i);
/* Input::is_empty_match */
bool chn_input_is_empty_match(const chn_input *input, chn_input_at at, chn_empty_look look);
/* Input::prefix_at: false is None. */
bool chn_input_prefix_at(const chn_input *input, const chn_literal_searcher *prefixes,
    chn_input_at at, chn_input_at *out);

/* LiteralSearcher (prefilter.c). */
typedef enum chn_lits_status {
    CHN_LITS_OK = 0,
    CHN_LITS_NOMEM = 1
} chn_lits_status;
/* LiteralSearcher::empty */
void chn_literal_searcher_empty(chn_literal_searcher *ls);
/* LiteralSearcher::prefixes(lits) / ::suffixes(lits); lits is consumed. fast selects the
 * matcher implementation (the harness's prefilter switch); the selection is unchanged. */
chn_lits_status chn_literal_searcher_new(chn_literal_searcher *ls, chn_seq *lits,
    bool suffixes, bool fast);
/* Drop */
void chn_literal_searcher_drop(chn_literal_searcher *ls);
/* LiteralSearcher::complete */
bool chn_literal_searcher_complete(const chn_literal_searcher *ls);
/* LiteralSearcher::len */
size_t chn_literal_searcher_len(const chn_literal_searcher *ls);
/* LiteralSearcher::is_empty */
static inline bool chn_literal_searcher_is_empty(const chn_literal_searcher *ls) {
    return chn_literal_searcher_len(ls) == 0;
}
/* LiteralSearcher::find: true with [*s, *e) relative to haystack. */
bool chn_literal_searcher_find(const chn_literal_searcher *ls, const uint8_t *haystack,
    size_t len, size_t *s, size_t *e);
/* LiteralSearcher::find_start */
bool chn_literal_searcher_find_start(const chn_literal_searcher *ls, const uint8_t *haystack,
    size_t len, size_t *s, size_t *e);
/* LiteralSearcher::find_end */
bool chn_literal_searcher_find_end(const chn_literal_searcher *ls, const uint8_t *haystack,
    size_t len, size_t *s, size_t *e);
/* Memmem::find (the lcp / lcs searchers) */
bool chn_memmem_find(const chn_memmem *m, bool fast, const uint8_t *haystack, size_t len,
    size_t *pos);
/* Memmem::is_suffix */
bool chn_memmem_is_suffix(const chn_memmem *m, const uint8_t *text, size_t len);
/* A leftmost-first multi-literal searcher over lits (consumed), for exec.rs's AhoCorasick
 * match type. */
chn_lits_status chn_multi_new(chn_multi *m, chn_seq *lits);
void chn_multi_drop(chn_multi *m);
bool chn_multi_find(const chn_multi *m, bool fast, const uint8_t *haystack, size_t len,
    size_t *s, size_t *e);

/* Engine results: 1 match, 0 no match, -1 out of memory. */

/* pikevm.rs Cache */
typedef struct chn_pikevm_threads {
    size_t *dense;   /* SparseSet::dense */
    size_t *sparse;  /* SparseSet::sparse */
    size_t len;      /* SparseSet::dense.len() */
    size_t *caps;    /* Threads::caps (CHN_REGEX_NO_SLOT: None) */
    size_t slots_per_thread;
} chn_pikevm_threads;

typedef struct chn_pikevm_frame {
    size_t ip_or_slot; /* FollowEpsilon::IP(ip) / Capture { slot } */
    size_t pos;        /* Capture { pos } */
    bool capture;
} chn_pikevm_frame;

typedef struct chn_pikevm_cache {
    bool ready;
    chn_pikevm_threads clist;
    chn_pikevm_threads nlist;
    chn_pikevm_frame *stack;
    size_t stack_cap;
} chn_pikevm_cache;

void chn_pikevm_cache_drop(chn_pikevm_cache *cache);
/* pikevm::Fsm::exec (matches is always one bool here). */
int chn_pikevm_exec(const chn_prog *prog, chn_pikevm_cache *cache, bool *matched,
    size_t *slots, size_t nslots, bool quit_after_match, const chn_input *input, size_t start,
    size_t end);

/* backtrack.rs */
bool chn_backtrack_should_exec(size_t num_insts, size_t text_len);

typedef struct chn_backtrack_job {
    size_t ip_or_slot; /* Job::Inst { ip } / SaveRestore { slot } */
    chn_input_at at;   /* Job::Inst { at } */
    size_t old_pos;    /* SaveRestore { old_pos } */
    bool restore;
} chn_backtrack_job;

typedef struct chn_backtrack_cache {
    chn_backtrack_job *jobs;
    size_t jobs_len;
    size_t jobs_cap;
    uint32_t *visited;
    size_t visited_cap;
} chn_backtrack_cache;

void chn_backtrack_cache_drop(chn_backtrack_cache *cache);
/* backtrack::Bounded::exec */
int chn_backtrack_exec(const chn_prog *prog, chn_backtrack_cache *cache, bool *matched,
    size_t *slots, size_t nslots, const chn_input *input, size_t start, size_t end);

/* dfa.rs */
typedef enum chn_dfa_status {
    CHN_DFA_MATCH = 0,    /* Result::Match(value) */
    CHN_DFA_NO_MATCH = 1, /* Result::NoMatch(value) */
    CHN_DFA_QUIT = 2,     /* Result::Quit */
    CHN_DFA_NOMEM = 3     /* allocation failure (Rust: abort) */
} chn_dfa_status;

typedef struct chn_dfa_result {
    chn_dfa_status status;
    size_t value;
} chn_dfa_result;

/* dfa.rs Cache (dfa.c), created on first use. */
typedef struct chn_dfa_cache chn_dfa_cache;
void chn_dfa_cache_free(chn_dfa_cache *cache);
/* dfa::can_exec */
bool chn_dfa_can_exec(const chn_prog *prog);
/* Fsm::forward(prog, cache, quit_after_match = false, text, at), at <= len (the caller
 * reproduces start_flags's text[at - 1] panic). prefixes is the program's prefix searcher
 * (the forward DFA program shares the NFA's). */
chn_dfa_result chn_dfa_forward(const chn_prog *prog, const chn_literal_searcher *prefixes,
    chn_dfa_cache **cache, const uint8_t *text, size_t len, size_t at);
/* Fsm::reverse(prog, cache, quit_after_match = false, text, at) */
chn_dfa_result chn_dfa_reverse(const chn_prog *prog, chn_dfa_cache **cache, const uint8_t *text,
    size_t len, size_t at);

/* ProgramCacheInner (exec.rs). */
typedef struct chn_cache {
    chn_pikevm_cache pikevm;
    chn_backtrack_cache backtrack;
    chn_dfa_cache *dfa;
    chn_dfa_cache *dfa_reverse;
} chn_cache;

/* MatchType / MatchLiteralType / MatchNfaType (exec.rs). */
typedef enum chn_match_type {
    CHN_MATCH_LITERAL_UNANCHORED = 0,
    CHN_MATCH_LITERAL_ANCHORED_START = 1,
    CHN_MATCH_LITERAL_ANCHORED_END = 2,
    CHN_MATCH_LITERAL_AHO_CORASICK = 3,
    CHN_MATCH_DFA = 4,
    CHN_MATCH_DFA_ANCHORED_REVERSE = 5,
    CHN_MATCH_DFA_SUFFIX = 6,
    CHN_MATCH_NFA_AUTO = 7,
    CHN_MATCH_NFA_BACKTRACK = 8,
    CHN_MATCH_NFA_PIKEVM = 9,
    CHN_MATCH_NOTHING = 10
} chn_match_type;

#define CHN_POOL_SLOTS 8

/* regex_py::Regex over regex::Regex (Exec { ro: ExecReadOnly, pool }). */
struct chn_regex {
    char *pattern;              /* ExecReadOnly::res[0] */
    size_t pattern_len;
    chn_prog nfa;               /* ExecReadOnly::nfa */
    chn_prog dfa;               /* ExecReadOnly::dfa */
    chn_prog dfa_reverse;       /* ExecReadOnly::dfa_reverse */
    chn_literal_searcher suffixes; /* ExecReadOnly::suffixes */
    bool has_ac;                /* ExecReadOnly::ac.is_some() */
    chn_multi ac;               /* ExecReadOnly::ac */
    chn_match_type match_type;  /* ExecReadOnly::match_type */
    bool fast;                  /* the prefilter switch */
    void *volatile pool[CHN_POOL_SLOTS]; /* Pool<ProgramCache>: free chn_cache values */
};

/* exec.c, for iter.c: ExecNoSyncStr::find_at and ::captures_read_at with a pooled cache. */
chn_regex_status chn_exec_find_at(const chn_regex *re, const uint8_t *text, size_t len,
    size_t start, size_t *s, size_t *e, chn_regex_string *panic);
chn_regex_status chn_exec_captures_read_at(const chn_regex *re, size_t *slots, size_t nslots,
    const uint8_t *text, size_t len, size_t start, chn_regex_string *panic);

#endif
