#ifndef CHAINNER_PARALLEL_H
#define CHAINNER_PARALLEL_H

#include "chainner.h"

typedef void (*cn_range_fn)(void *context, size_t begin, size_t end);

#ifdef __cplusplus
extern "C" {
#endif

/* Synchronous completion; callbacks write disjoint ranges and cannot fail.
 * count <= grain runs as one call on the caller. A larger call runs
 * min(1 + (count - 1) / grain, 128) contiguous chunks: the partition depends
 * on (count, grain) alone, never on the CPU count or on how busy the pool is.
 * The caller works on its own call; pool helpers join the active call with the
 * fewest helpers. A helper is admitted only while registered callers plus
 * helpers fit the CPU count, and leaves at its next chunk boundary when a new
 * caller needs the room. Every chunk runs under the caller's MXCSR. No callback
 * remains alive when this function returns. The library must stay loaded for
 * the life of the process: a helper woken for a call may still be returning to
 * the pool after that call returned (it no longer touches that call's job).
 * Non-Windows builds run the same chunks serially.
 */
CN_EXPORT cn_status cn_parallel_for(size_t count, size_t grain, cn_range_fn fn, void *context);
CN_EXPORT unsigned int cn_parallel_capacity(void);
/* Calls that ran a chunk list (count > grain). */
CN_EXPORT uint64_t cn_parallel_dispatches(void);
/* Helper wake-ups submitted (including the re-submit after a call leaves while
 * other calls remain), and helper callbacks that claimed no chunk. Non-Windows
 * builds have no helpers: both are 0. */
CN_EXPORT uint64_t cn_parallel_wakes(void);
CN_EXPORT uint64_t cn_parallel_empty_wakes(void);
/* Test-only: cn_parallel_for(count, 1, ...) with a range function that does
 * nothing, for the dispatch floor. count 0-128, else CN_INVALID_ARGUMENT. */
CN_EXPORT cn_status cn_parallel_noop(size_t count);

#ifdef __cplusplus
}
#endif

#endif
