/* Port of regex 1.8.4 (src/exec.rs, src/re_trait.rs, src/re_unicode.rs, src/re_builder.rs) and chainner_ext 0.3.10 (crates/regex-py/src/lib.rs, position.rs), MIT OR Apache-2.0. */
/* The regex engine's C API: chainner_regex.lib, linked into chainner_ext.pyd (RustRegex,
 * RegexMatch, MatchGroup) and regex_probe.exe; _chainner_graph.pyd reaches the same
 * functions through the chn_regex_api capsule below. regex-py composes them as follows,
 * and so must every caller:
 *   Regex::search(text, pos)  = chn_regex_to_byte_pos, chn_regex_search_at, then one
 *                               PosTranslator over the groups in index order, start then end;
 *   Regex::findall(text)      = chn_regex_iter_next_captures until NO_MATCH, one
 *                               PosTranslator shared by every match, same order;
 *   Regex::split, split_without_captures = chn_regex_split, chn_regex_split_without_captures.
 *
 * Conventions:
 * - Text and patterns are UTF-8 (Rust &str) with explicit byte lengths; they may contain
 *   U+0000. Python callers pass PyUnicode_AsUTF8AndSize's bytes, which, like pyo3's &str
 *   extraction, refuses lone surrogates before the engine is reached.
 * - All positions here are byte offsets, except where "char" is said: regex-py turns them
 *   into character positions with PosTranslator.
 * - size_t arithmetic wraps exactly as 0.3.10's release build does (no overflow checks).
 * - A chn_regex is immutable once compiled and may be searched from several threads at
 *   once; per-search scratch (Rust's Pool<ProgramCache>) is the engine's internal concern.
 * - Outputs are written on every return. *_drop releases a value in caller storage and
 *   leaves it empty; chn_regex_free releases a handle. Strings and span vectors are
 *   allocated inside chainner_regex.lib and must be released through these functions
 *   (through the capsule from another module: each module has its own CRT heap).
 * - CHN_REGEX_PANIC reproduces a pyo3 PanicException of 0.3.10 (for example a search
 *   position past the end of the text on a literal-scan path, or a PosTranslator slice
 *   inside a character). *panic then holds Rust's panic message, byte for byte.
 *
 * regex_probe.exe (native/src/regex/probe.c): the fixed I/O format, shared with X4's
 * harness. Do not change it.
 *   CLI: regex_probe.exe --mode default|nfa|backtrack|dfa --prefilter on|off < in.jsonl > out.jsonl
 *        (--mode sets chn_regex_options.engine, --prefilter its prefilter flag; both
 *        default to default and on.)
 *   Input: one JSON object per line, {"id": str, "pattern": str, "ops": [op, ...]}.
 *   Ops and their results:
 *     {"op":"compile"}  -> {"ok":true,"groups":n,"groupindex":{name:idx,...}}
 *                          or {"ok":false,"error":"<the full message, 'Invalid regex: ' prefix included>"}
 *     {"op":"search","text":s,"pos":p}       -> null or a match
 *     {"op":"findall","text":s}              -> a list of matches
 *     {"op":"split","text":s}                -> a list of strings
 *     {"op":"split_without_captures","text":s} -> a list of strings
 *   A match: {"start":a,"end":b,"len":l,"groups":[[a,b] | null, ...],"names":{name:[a,b] | null}}.
 *   Every position is a CHARACTER position, as regex-py returns them through
 *   PosTranslator. "groups" covers index 0..groups.
 *   When compile fails, the other ops are skipped and their results are null.
 *   An op on which 0.3.10 raises pyo3's PanicException yields {"panic":"<message>"} as its
 *   result. The message must be byte-identical to Rust's panic text (spec section 2's
 *   identical-message rule); native/tests/chainner_ext/regex_expected.jsonl holds the real
 *   module's. The forms seen there, from searches at positions past the text end:
 *   - "index out of bounds: the len is N but the index is M" (an engine indexing the text);
 *   - "range start index N out of range for slice of length M" (a path slicing the text,
 *     e.g. search("abc", pos=10) on a literal scan).
 *   A third is possible: PosTranslator::get_char_pos slicing `&self.text[start..]` inside a
 *   character gives core::str's "byte index N is not a char boundary; it is inside 'X'
 *   (bytes A..B) of `T`" (T shortened as Rust does).
 *   Output: one line per input line, {"id": same, "results": [one per op, in order]}.
 *   "groupindex" and "names" list names in group-index order.
 *   Exit status: 0; 3 when a scaffold stub was hit (CHN_REGEX_UNIMPLEMENTED: compile yields
 *   {"ok":false,"error":"unimplemented: ..."}); 2 on malformed input (bad JSON, a lone
 *   surrogate escape, a negative or non-integer pos, an unknown op or option), which
 *   stops the probe with a message on stderr. */
#ifndef CHAINNER_REGEX_H
#define CHAINNER_REGEX_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* RegexOptions::default (re_builder.rs), the values regex-py's Regex::new compiles with. */
#define CHN_REGEX_SIZE_LIMIT ((size_t)10 << 20)    /* RegexOptions::size_limit */
#define CHN_REGEX_DFA_SIZE_LIMIT ((size_t)2 << 20) /* RegexOptions::dfa_size_limit */
#define CHN_REGEX_NEST_LIMIT 250u                  /* RegexOptions::nest_limit */

/* Slot = Option<usize> (re_trait.rs): this value is None. */
#define CHN_REGEX_NO_SLOT SIZE_MAX

typedef enum chn_regex_status {
    CHN_REGEX_OK = 0,            /* Ok(..) / Some(..) */
    CHN_REGEX_NO_MATCH = 1,      /* None: no match, or the iterator is exhausted */
    CHN_REGEX_ERROR_INVALID = 2, /* regex_py::RegexError: *error holds its message */
    CHN_REGEX_ERROR_NOMEM = 3,   /* allocation failure (Rust: abort); outputs are empty */
    CHN_REGEX_PANIC = 4,         /* 0.3.10 panics here: *panic holds the panic message */
    CHN_REGEX_UNIMPLEMENTED = 5  /* X0 scaffold stub; never returned once X2 lands */
} chn_regex_status;

/* String: UTF-8, len bytes, with one trailing NUL that len does not count; may contain
 * U+0000. {NULL, 0} is the empty value. */
typedef struct chn_regex_string {
    char *ptr;
    size_t len;
} chn_regex_string;

/* Releases s->ptr and leaves *s empty. */
void chn_regex_string_drop(chn_regex_string *s);

/* The harness-only engine switch. Production always uses DEFAULT with the prefilter on;
 * every setting must give identical results (spec section 4). */
typedef enum chn_regex_engine {
    CHN_REGEX_ENGINE_DEFAULT = 0,   /* ExecReadOnly::choose_match_type(None) */
    CHN_REGEX_ENGINE_NFA = 1,       /* ExecBuilder::nfa(): MatchType::Nfa(MatchNfaType::PikeVM) */
    CHN_REGEX_ENGINE_BACKTRACK = 2, /* ExecBuilder::bounded_backtracking(): MatchType::Nfa(MatchNfaType::Backtrack) */
    CHN_REGEX_ENGINE_DFA = 3        /* the lazy DFA wherever dfa::can_exec, with exec.rs's Quit fallbacks
                                     * (dfa.c gives it meaning; until then it behaves as DEFAULT) */
} chn_regex_engine;

typedef struct chn_regex_options {
    chn_regex_engine engine; /* harness only */
    bool prefilter;          /* harness only: false disables every candidate finder (prefilter.c) */
} chn_regex_options;

/* regex_py::Regex { inner: regex::Regex, names: Arc<Vec<Option<String>>> }. Opaque. */
typedef struct chn_regex chn_regex;

/* regex_py::Regex::new: regex::Regex::new(pattern) (RegexBuilder defaults, the limits
 * above) plus the capture names. options NULL means production (DEFAULT, prefilter on).
 * - CHN_REGEX_OK: *out is the handle; *error is empty.
 * - CHN_REGEX_ERROR_INVALID: *error is format!("Invalid regex: {}", e) for e: regex::Error:
 *   Syntax(s) displays s (regex_syntax's caret-formatted text, or compile.rs's own
 *   messages), CompiledTooBig(limit) displays "Compiled regex exceeds size limit of
 *   {limit} bytes."; *out is NULL.
 * - CHN_REGEX_ERROR_NOMEM, CHN_REGEX_UNIMPLEMENTED: *out is NULL; *error may describe it. */
chn_regex_status chn_regex_compile(const char *pattern, size_t pattern_len,
    const chn_regex_options *options, chn_regex **out, chn_regex_string *error);

/* Drop for Regex. NULL is allowed. */
void chn_regex_free(chn_regex *re);

/* Regex::pattern (regex::Regex::as_str): the pattern as compiled; *len gets its length. */
const char *chn_regex_pattern(const chn_regex *re, size_t *len);

/* Regex::groups: names.len() - 1. */
size_t chn_regex_groups(const chn_regex *re);

/* names.len() (regex::Regex::captures_len): groups + 1. Slot arrays hold twice this. */
size_t chn_regex_captures_len(const chn_regex *re);

/* names[index] (Regex::capture_names), index < chn_regex_captures_len: NULL for an
 * unnamed group (always for 0), else the name with *len its byte length. Regex::groupindex
 * is every named index; RegexMatch::get_by_name takes the first index with the name. */
const char *chn_regex_capture_name(const chn_regex *re, size_t index, size_t *len);

/* regex::Regex::captures_at(text, byte_pos) (captures_read_at). slots_len must be
 * 2 * chn_regex_captures_len(re). byte_pos is regex-py's to_byte_pos result: it may lie
 * inside a character or past text_len, and the engine behaves as 1.8.4 does there.
 * - CHN_REGEX_OK: slots[2i], slots[2i+1] are group i's byte span, or CHN_REGEX_NO_SLOT.
 * - CHN_REGEX_NO_MATCH: every slot is CHN_REGEX_NO_SLOT.
 * - CHN_REGEX_PANIC: *panic holds the message (otherwise *panic is left empty). */
chn_regex_status chn_regex_search_at(const chn_regex *re, const char *text, size_t text_len,
    size_t byte_pos, size_t *slots, size_t slots_len, chn_regex_string *panic);

/* Matches<'t, ExecNoSyncStr> (re_trait.rs), also the state of CaptureMatches. Plain
 * state: it owns nothing, and re and text must outlive it. */
typedef struct chn_regex_iter {
    const chn_regex *re; /* Matches::re */
    const char *text;    /* Matches::text */
    size_t text_len;     /* Matches::text (length) */
    size_t last_end;     /* Matches::last_end */
    size_t last_match;   /* Matches::last_match (CHN_REGEX_NO_SLOT: None) */
} chn_regex_iter;

/* RegularExpression::find_iter / captures_iter: last_end 0, last_match None. */
void chn_regex_iter_init(chn_regex_iter *it, const chn_regex *re, const char *text, size_t text_len);

/* Matches::next (find_at; empty matches advance by next_utf8 and never directly follow a
 * match). CHN_REGEX_OK with *start and *end, or CHN_REGEX_NO_MATCH when done. */
chn_regex_status chn_regex_iter_next(chn_regex_iter *it, size_t *start, size_t *end);

/* CaptureMatches::next (captures_read_at), slots as for chn_regex_search_at. */
chn_regex_status chn_regex_iter_next_captures(chn_regex_iter *it, size_t *slots, size_t slots_len);

/* One piece of text as a byte range (a &str slice, or a String's source). */
typedef struct chn_regex_span {
    size_t start;
    size_t end;
} chn_regex_span;

/* Vec<String> as spans into the text. */
typedef struct chn_regex_spans {
    chn_regex_span *ptr; /* Vec pointer */
    size_t len;          /* Vec length */
    size_t cap;          /* Vec capacity */
} chn_regex_spans;

/* Releases spans->ptr and leaves *spans empty. */
void chn_regex_spans_drop(chn_regex_spans *spans);

/* regex_py::Regex::split over captures_iter: for each match, text[last..start] when
 * non-empty, then every participating group 1.. (empty ones included); finally
 * text[last..] when non-empty. */
chn_regex_status chn_regex_split(const chn_regex *re, const char *text, size_t text_len,
    chn_regex_spans *out);

/* regex_py::Regex::split_without_captures: regex::Regex::split (re_unicode.rs Split),
 * empty pieces included. */
chn_regex_status chn_regex_split_without_captures(const chn_regex *re, const char *text,
    size_t text_len, chn_regex_spans *out);

/* position::to_byte_pos(text, char_pos), out-of-range rules and wrapping included. */
size_t chn_regex_to_byte_pos(const char *text, size_t text_len, size_t char_pos);

/* An element of PosTranslator::known: (usize, usize). */
typedef struct chn_regex_pos_known {
    size_t byte_pos; /* .0 */
    size_t char_pos; /* .1 */
} chn_regex_pos_known;

/* position::PosTranslator { text: &str, known: Vec<(usize, usize)> }. text must outlive it. */
typedef struct chn_regex_pos_translator {
    const char *text;           /* PosTranslator::text */
    size_t text_len;            /* PosTranslator::text (length) */
    chn_regex_pos_known *known; /* PosTranslator::known (Vec pointer) */
    size_t known_len;           /* PosTranslator::known (Vec length) */
    size_t known_cap;           /* PosTranslator::known (Vec capacity) */
} chn_regex_pos_translator;

/* PosTranslator::new */
void chn_regex_pos_translator_init(chn_regex_pos_translator *t, const char *text, size_t text_len);

/* PosTranslator::get_char_pos. CHN_REGEX_OK with *char_pos; CHN_REGEX_PANIC when Rust's
 * `&self.text[start..]` would slice inside a character (*panic holds the message). */
chn_regex_status chn_regex_pos_translator_get_char_pos(chn_regex_pos_translator *t,
    size_t byte_pos, size_t *char_pos, chn_regex_string *panic);

/* Drop: releases known and leaves *t empty. */
void chn_regex_pos_translator_drop(chn_regex_pos_translator *t);

/* The capsule chainner_ext exports for _chainner_graph: the extension module
 * chainner_ext.chainner_ext holds it as _C_API (the package re-exports only __all__), and
 * PyCapsule_GetPointer(capsule, CHN_REGEX_CAPSULE_NAME) gives a const chn_regex_api *.
 * The harness copy chainner_ext_c holds one too, which nothing imports; its regex_of
 * recognizes only its own RustRegex. */
#define CHN_REGEX_CAPSULE_NAME "chainner_ext._C_API"
#define CHN_REGEX_API_VERSION 2u

typedef struct chn_regex_api {
    uint32_t version; /* CHN_REGEX_API_VERSION */
    void (*string_drop)(chn_regex_string *s);
    chn_regex_status (*compile)(const char *pattern, size_t pattern_len,
        const chn_regex_options *options, chn_regex **out, chn_regex_string *error);
    void (*regex_free)(chn_regex *re);
    const char *(*pattern)(const chn_regex *re, size_t *len);
    size_t (*groups)(const chn_regex *re);
    size_t (*captures_len)(const chn_regex *re);
    const char *(*capture_name)(const chn_regex *re, size_t index, size_t *len);
    chn_regex_status (*search_at)(const chn_regex *re, const char *text, size_t text_len,
        size_t byte_pos, size_t *slots, size_t slots_len, chn_regex_string *panic);
    void (*iter_init)(chn_regex_iter *it, const chn_regex *re, const char *text, size_t text_len);
    chn_regex_status (*iter_next)(chn_regex_iter *it, size_t *start, size_t *end);
    chn_regex_status (*iter_next_captures)(chn_regex_iter *it, size_t *slots, size_t slots_len);
    void (*spans_drop)(chn_regex_spans *spans);
    chn_regex_status (*split)(const chn_regex *re, const char *text, size_t text_len,
        chn_regex_spans *out);
    chn_regex_status (*split_without_captures)(const chn_regex *re, const char *text,
        size_t text_len, chn_regex_spans *out);
    size_t (*to_byte_pos)(const char *text, size_t text_len, size_t char_pos);
    void (*pos_translator_init)(chn_regex_pos_translator *t, const char *text, size_t text_len);
    chn_regex_status (*pos_translator_get_char_pos)(chn_regex_pos_translator *t,
        size_t byte_pos, size_t *char_pos, chn_regex_string *panic);
    void (*pos_translator_drop)(chn_regex_pos_translator *t);
    /* Version 2. The engine of a RustRegex: object is a PyObject *; for an instance of
     * exactly chainner_ext's RustRegex it gives that instance's handle, valid while the
     * caller holds a reference to object; for any other object NULL, with no Python
     * error set. The GIL must be held. */
    const chn_regex *(*regex_of)(void *object);
} chn_regex_api;

#ifdef __cplusplus
}
#endif
#endif
