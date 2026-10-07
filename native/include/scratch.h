/* The shared scratch pool (SP4b D10, spec 4.5), built from resample_filters.c's
 * workspace lease: CN_SCRATCH_SLOTS slots, each leased exclusively by one call and
 * retaining at most CN_SCRATCH_RETAIN_BYTES when idle. A lease takes, under a short
 * lock, the smallest idle slot that holds the request, else the smallest idle slot,
 * which it grows outside the lock; when every slot is busy it takes an ordinary
 * temporary and never waits. Only kernels that write every byte of their scratch
 * before reading it lease from the pool: a lease holds a previous call's bytes. */
#ifndef CHAINNER_SCRATCH_H
#define CHAINNER_SCRATCH_H
#include "chainner.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A lease: data holds bytes (at least the request); slot is the pool slot, or -1 for
 * a temporary. */
typedef struct cn_scratch {
    void *data;
    size_t bytes;
    int slot;
} cn_scratch;

/* Callers take one lease per call and carve it into their buffers at 64-byte offsets:
 * *offset receives the next buffer's offset and *total grows by bytes rounded up to a
 * multiple of 64. Returns 0, leaving both unchanged, when the sum overflows size_t;
 * the caller then returns CN_ALLOCATION_FAILED, as its separate mallocs did. */
int cn_scratch_carve(size_t *total, size_t bytes, size_t *offset);
/* Leases bytes (> 0). Unless allocated is NULL, *allocated is 1 when the lease
 * allocated (a slot grown, or a temporary), else 0 (resample counts it). A failed
 * allocation returns CN_ALLOCATION_FAILED with nothing leased (lease->data NULL,
 * lease->slot -1). */
cn_status cn_scratch_lease(size_t bytes, cn_scratch *lease, int *allocated);
/* Returns the lease to the pool; a slot grown above the retention cap, and a
 * temporary, are freed. A lease that holds nothing (failed or never taken, data NULL
 * and slot -1) is ignored. */
void cn_scratch_release(cn_scratch *lease);
/* values[0] the leases that allocated, [1] the leases served by a temporary, [2] the
 * idle slots' retained bytes, [3] the slots leased now. count must be 4, else
 * CN_INVALID_ARGUMENT (as is a NULL values), and nothing is written. */
CN_EXPORT cn_status cn_scratch_info(uint64_t *values, size_t count);
/* Test-only: grows every idle slot to min(bytes, CN_SCRATCH_RETAIN_BYTES) and fills
 * its whole capacity with the 32-bit word bits (little-endian, then its low bytes for
 * a remainder); returns the slots filled, or -1 when a growth fails. Counts no
 * allocation. */
CN_EXPORT int cn_scratch_fill(uint32_t bits, size_t bytes);

#ifdef __cplusplus
}
#endif

#endif
