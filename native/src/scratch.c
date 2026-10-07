/* The shared scratch pool (scratch.h; SP4b D10). The lease logic is resample's
 * workspace lease (resample_filters.c): exclusive slots taken under a short lock, a
 * retention cap per slot, and a busy caller takes an ordinary temporary and never
 * waits. No source or output pointer is kept between calls. */
#include "scratch.h"
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <stdatomic.h>
#endif

#define CN_SCRATCH_SLOTS 4
/* An idle slot retains at most 16 MiB, so the pool retains at most 64 MiB. */
#define CN_SCRATCH_RETAIN_BYTES ((size_t)16 * 1024 * 1024)

/* data and capacity change only while the slot is leased (by its leaseholder, outside
 * the lock); leased and the counters change only under the lock. */
typedef struct scratch_slot {
    void *data;
    size_t capacity;
    int leased;
} scratch_slot;

static scratch_slot slots[CN_SCRATCH_SLOTS];
static uint64_t allocations, temporaries;
#ifdef _WIN32
static SRWLOCK scratch_lock = SRWLOCK_INIT;
static void lock_scratch(void) { AcquireSRWLockExclusive(&scratch_lock); }
static void unlock_scratch(void) { ReleaseSRWLockExclusive(&scratch_lock); }
#else
static atomic_flag scratch_lock = ATOMIC_FLAG_INIT;
static void lock_scratch(void)
{
    while (atomic_flag_test_and_set_explicit(&scratch_lock, memory_order_acquire)) {}
}
static void unlock_scratch(void)
{
    atomic_flag_clear_explicit(&scratch_lock, memory_order_release);
}
#endif

int cn_scratch_carve(size_t *total, size_t bytes, size_t *offset)
{
    size_t rounded;
    if (bytes > SIZE_MAX - 63) return 0;
    rounded = (bytes + 63) & ~(size_t)63;
    if (rounded > SIZE_MAX - *total) return 0;
    *offset = *total;
    *total += rounded;
    return 1;
}

/* Marks and returns an idle slot under the lock: the smallest that holds bytes, else
 * the smallest (an empty one first), whose buffer the caller replaces; -1 when every
 * slot is leased. */
static int take_slot(size_t bytes)
{
    int fit = -1, smallest = -1;
    lock_scratch();
    for (int i = 0; i < CN_SCRATCH_SLOTS; ++i) {
        if (slots[i].leased) continue;
        if (slots[i].capacity >= bytes && (fit < 0 || slots[i].capacity < slots[fit].capacity))
            fit = i;
        if (smallest < 0 || slots[i].capacity < slots[smallest].capacity) smallest = i;
    }
    if (fit < 0) fit = smallest;
    if (fit >= 0) slots[fit].leased = 1;
    unlock_scratch();
    return fit;
}

static void return_slot(int index)
{
    lock_scratch();
    slots[index].leased = 0;
    unlock_scratch();
}

/* Grows a leased slot to at least bytes, keeping its old buffer on failure. */
static int grow_slot(int index, size_t bytes)
{
    void *data;
    if (slots[index].capacity >= bytes) return 1;
    data = malloc(bytes);
    if (!data) return 0;
    free(slots[index].data);
    slots[index].data = data;
    slots[index].capacity = bytes;
    return 1;
}

cn_status cn_scratch_lease(size_t bytes, cn_scratch *lease, int *allocated)
{
    int index = take_slot(bytes), grew = 0;
    lease->data = NULL;
    lease->bytes = 0;
    lease->slot = -1;
    if (allocated) *allocated = 0;
    if (index < 0) {
        void *data = malloc(bytes);
        if (!data) return CN_ALLOCATION_FAILED;
        lease->data = data;
        lease->bytes = bytes;
        lock_scratch();
        ++allocations;
        ++temporaries;
        unlock_scratch();
        if (allocated) *allocated = 1;
        return CN_OK;
    }
    if (slots[index].capacity < bytes) {
        if (!grow_slot(index, bytes)) {
            return_slot(index);
            return CN_ALLOCATION_FAILED;
        }
        grew = 1;
        lock_scratch();
        ++allocations;
        unlock_scratch();
    }
    lease->data = slots[index].data;
    lease->bytes = slots[index].capacity;
    lease->slot = index;
    if (allocated) *allocated = grew;
    return CN_OK;
}

void cn_scratch_release(cn_scratch *lease)
{
    if (lease->slot < 0) {
        free(lease->data);
    } else {
        scratch_slot *slot = &slots[lease->slot];
        void *drop = NULL;
        if (slot->capacity > CN_SCRATCH_RETAIN_BYTES) {
            drop = slot->data;
            slot->data = NULL;
            slot->capacity = 0;
        }
        return_slot(lease->slot);
        free(drop);
    }
    lease->data = NULL;
    lease->bytes = 0;
    lease->slot = -1;
}

CN_EXPORT cn_status cn_scratch_info(uint64_t *values, size_t count)
{
    if (!values || count != 4) return CN_INVALID_ARGUMENT;
    lock_scratch();
    values[0] = allocations;
    values[1] = temporaries;
    values[2] = values[3] = 0;
    for (int i = 0; i < CN_SCRATCH_SLOTS; ++i) {
        if (slots[i].leased) ++values[3];
        else values[2] += slots[i].capacity;
    }
    unlock_scratch();
    return CN_OK;
}

CN_EXPORT int cn_scratch_fill(uint32_t bits, size_t bytes)
{
    size_t target = bytes < CN_SCRATCH_RETAIN_BYTES ? bytes : CN_SCRATCH_RETAIN_BYTES;
    int filled = 0;
    for (int i = 0; i < CN_SCRATCH_SLOTS; ++i) {
        int idle;
        lock_scratch();
        idle = !slots[i].leased;
        if (idle) slots[i].leased = 1;
        unlock_scratch();
        if (!idle) continue;
        if (!grow_slot(i, target)) {
            return_slot(i);
            return -1;
        }
        if (slots[i].capacity) {
            /* One word, then the filled prefix doubled: every 4-byte offset holds bits. */
            unsigned char *data = (unsigned char *)slots[i].data;
            size_t capacity = slots[i].capacity;
            size_t done = capacity < sizeof(bits) ? capacity : sizeof(bits);
            memcpy(data, &bits, done);
            for (; done < capacity; done *= 2)
                memcpy(data + done, data, capacity - done < done ? capacity - done : done);
        }
        return_slot(i);
        ++filled;
    }
    return filled;
}
