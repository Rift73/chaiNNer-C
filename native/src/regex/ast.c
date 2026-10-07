/* Port of regex-syntax 0.7.2 src/ast/mod.rs, MIT OR Apache-2.0. */
/* Display for ast::ErrorKind, and the arena that holds an AST (Rust's Box/Vec nodes and
 * its heap-based Drop become one bump allocator freed at once). */
#include "ast.h"
#include <stdio.h>
#include <stdlib.h>

size_t chn_ast_error_kind_display(const chn_syntax_error *err, char *buf, size_t cap) {
    const char *msg = "";
    int n;
    switch (err->kind) {
    case CHN_AST_ERR_CAPTURE_LIMIT_EXCEEDED:
        n = snprintf(buf, cap, "exceeded the maximum number of capturing groups (%lu)", (unsigned long)UINT32_MAX);
        return n < 0 ? 0 : (size_t)n;
    case CHN_AST_ERR_NEST_LIMIT_EXCEEDED:
        n = snprintf(buf, cap, "exceed the maximum number of nested parentheses/brackets (%lu)",
            (unsigned long)err->limit);
        return n < 0 ? 0 : (size_t)n;
    case CHN_AST_ERR_CLASS_ESCAPE_INVALID: msg = "invalid escape sequence found in character class"; break;
    case CHN_AST_ERR_CLASS_RANGE_INVALID: msg = "invalid character class range, the start must be <= the end"; break;
    case CHN_AST_ERR_CLASS_RANGE_LITERAL: msg = "invalid range boundary, must be a literal"; break;
    case CHN_AST_ERR_CLASS_UNCLOSED: msg = "unclosed character class"; break;
    case CHN_AST_ERR_DECIMAL_EMPTY: msg = "decimal literal empty"; break;
    case CHN_AST_ERR_DECIMAL_INVALID: msg = "decimal literal invalid"; break;
    case CHN_AST_ERR_ESCAPE_HEX_EMPTY: msg = "hexadecimal literal empty"; break;
    case CHN_AST_ERR_ESCAPE_HEX_INVALID: msg = "hexadecimal literal is not a Unicode scalar value"; break;
    case CHN_AST_ERR_ESCAPE_HEX_INVALID_DIGIT: msg = "invalid hexadecimal digit"; break;
    case CHN_AST_ERR_ESCAPE_UNEXPECTED_EOF: msg = "incomplete escape sequence, reached end of pattern prematurely"; break;
    case CHN_AST_ERR_ESCAPE_UNRECOGNIZED: msg = "unrecognized escape sequence"; break;
    case CHN_AST_ERR_FLAG_DANGLING_NEGATION: msg = "dangling flag negation operator"; break;
    case CHN_AST_ERR_FLAG_DUPLICATE: msg = "duplicate flag"; break;
    case CHN_AST_ERR_FLAG_REPEATED_NEGATION: msg = "flag negation operator repeated"; break;
    case CHN_AST_ERR_FLAG_UNEXPECTED_EOF: msg = "expected flag but got end of regex"; break;
    case CHN_AST_ERR_FLAG_UNRECOGNIZED: msg = "unrecognized flag"; break;
    case CHN_AST_ERR_GROUP_NAME_DUPLICATE: msg = "duplicate capture group name"; break;
    case CHN_AST_ERR_GROUP_NAME_EMPTY: msg = "empty capture group name"; break;
    case CHN_AST_ERR_GROUP_NAME_INVALID: msg = "invalid capture group character"; break;
    case CHN_AST_ERR_GROUP_NAME_UNEXPECTED_EOF: msg = "unclosed capture group name"; break;
    case CHN_AST_ERR_GROUP_UNCLOSED: msg = "unclosed group"; break;
    case CHN_AST_ERR_GROUP_UNOPENED: msg = "unopened group"; break;
    case CHN_AST_ERR_REPETITION_COUNT_INVALID: msg = "invalid repetition count range, the start must be <= the end"; break;
    case CHN_AST_ERR_REPETITION_COUNT_DECIMAL_EMPTY: msg = "repetition quantifier expects a valid decimal"; break;
    case CHN_AST_ERR_REPETITION_COUNT_UNCLOSED: msg = "unclosed counted repetition"; break;
    case CHN_AST_ERR_REPETITION_MISSING: msg = "repetition operator missing expression"; break;
    case CHN_AST_ERR_UNICODE_CLASS_INVALID: msg = "invalid Unicode character class"; break;
    case CHN_AST_ERR_UNSUPPORTED_BACKREFERENCE: msg = "backreferences are not supported"; break;
    case CHN_AST_ERR_UNSUPPORTED_LOOK_AROUND: msg = "look-around, including look-ahead and look-behind, is not supported"; break;
    default: break;
    }
    n = snprintf(buf, cap, "%s", msg);
    return n < 0 ? 0 : (size_t)n;
}

struct chn_arena_chunk {
    chn_arena_chunk *next;
};

enum { ARENA_ALIGN = 16, ARENA_CHUNK = 64 * 1024 };

void chn_arena_init(chn_arena *arena) {
    arena->head = NULL;
    arena->next = NULL;
    arena->left = 0;
}

void *chn_arena_alloc(chn_arena *arena, size_t size) {
    size = (size + (ARENA_ALIGN - 1)) & ~(size_t)(ARENA_ALIGN - 1);
    if (size > arena->left) {
        const size_t header = (sizeof(chn_arena_chunk) + (ARENA_ALIGN - 1)) & ~(size_t)(ARENA_ALIGN - 1);
        const size_t body = size > ARENA_CHUNK ? size : ARENA_CHUNK;
        if (body > SIZE_MAX - header) return NULL;
        chn_arena_chunk *chunk = (chn_arena_chunk *)malloc(header + body);
        if (!chunk) return NULL;
        chunk->next = arena->head;
        arena->head = chunk;
        arena->next = (char *)chunk + header;
        arena->left = body;
    }
    void *ptr = arena->next;
    arena->next += size;
    arena->left -= size;
    return ptr;
}

void chn_arena_free(chn_arena *arena) {
    chn_arena_chunk *chunk = arena->head;
    while (chunk) {
        chn_arena_chunk *next = chunk->next;
        free(chunk);
        chunk = next;
    }
    chn_arena_init(arena);
}
