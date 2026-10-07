#include "parallel.h"

#define CN_PARALLEL_MAX_CHUNKS 128

static size_t chunk_count(size_t count, size_t grain)
{
    size_t chunks = 1 + (count - 1) / grain;
    return chunks < CN_PARALLEL_MAX_CHUNKS ? chunks : CN_PARALLEL_MAX_CHUNKS;
}

static void run_chunk(cn_range_fn fn, void *context, size_t count, size_t chunks, size_t index)
{
    size_t block = count / chunks;
    size_t remainder = count % chunks;
    size_t begin = block * index + (index < remainder ? index : remainder);
    size_t end = begin + block + (index < remainder ? 1 : 0);
    fn(context, begin, end);
}

static void noop_range(void *context, size_t begin, size_t end)
{
    (void)context; (void)begin; (void)end;
}

cn_status cn_parallel_noop(size_t count)
{
    if (count > CN_PARALLEL_MAX_CHUNKS) return CN_INVALID_ARGUMENT;
    return cn_parallel_for(count, 1, noop_range, NULL);
}

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#define CN_X86_FP 1
#endif

#define CN_PARALLEL_SLOTS 32

typedef struct range_job {
    cn_range_fn fn;
    void *context;
    size_t count;
    size_t chunks;
    volatile LONG next; /* next unclaimed chunk; claimed without the lock */
    LONG helpers;       /* helpers pinned to this job; guarded by registry */
#ifdef CN_X86_FP
    unsigned int mxcsr;
#endif
} range_job;

static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
static PTP_POOL pool;
static TP_CALLBACK_ENVIRON environment;
static PTP_WORK helper_work;
static unsigned int capacity = 1;
static volatile LONG64 dispatches = 0;
/* Helper wake-ups submitted, and helper callbacks that claimed no chunk: one
 * interlocked add per dispatch and at most one per helper callback. */
static volatile LONG64 wakes = 0;
static volatile LONG64 empty_wakes = 0;

/* registry guards jobs, callers, helpers_running and every job's helpers.
 * generation changes with every registration and removal and is written only
 * under the lock; helpers read it without the lock to notice that the set of
 * jobs changed, so it is bumped with an interlocked increment (which wraps)
 * and compared for equality only. */
static SRWLOCK registry = SRWLOCK_INIT;
static CONDITION_VARIABLE released = CONDITION_VARIABLE_INIT;
static range_job *jobs[CN_PARALLEL_SLOTS];
static LONG callers = 0;
static LONG helpers_running = 0;
static volatile LONG generation = 0;

static int claim(range_job *job)
{
    LONG index = InterlockedIncrement(&job->next) - 1;
    if ((size_t)index >= job->chunks) return 0;
    run_chunk(job->fn, job->context, job->count, job->chunks, (size_t)index);
    return 1;
}

/* Release the helper's previous job, then pin the job with unclaimed chunks
 * and the fewest helpers. NULL means the helper leaves: nothing needs help, or
 * callers plus helpers would exceed the CPU count. */
static range_job *switch_job(range_job *previous, LONG *seen)
{
    range_job *best = NULL;
    int last = 0;
    AcquireSRWLockExclusive(&registry);
    if (previous) {
        last = --previous->helpers == 0;
    } else {
        ++helpers_running;
    }
    if (helpers_running + callers <= (LONG)capacity) {
        for (int slot = 0; slot < CN_PARALLEL_SLOTS; ++slot) {
            range_job *job = jobs[slot];
            if (job && job->next < (LONG)job->chunks && (!best || job->helpers < best->helpers)) best = job;
        }
    }
    if (best) ++best->helpers;
    else --helpers_running;
    *seen = ReadAcquire(&generation);
    ReleaseSRWLockExclusive(&registry);
    /* Wake after the release: a caller woken under the lock would block on it
     * and need a second wake. */
    if (last) WakeAllConditionVariable(&released);
    return best;
}

static VOID CALLBACK helper(PTP_CALLBACK_INSTANCE instance, PVOID parameter, PTP_WORK work)
{
    range_job *job = NULL;
    LONG seen = 0;
    int claimed = 0;
#ifdef CN_X86_FP
    /* A helper runs every chunk under its job's rounding/denormal controls and
     * returns to the pool with the controls it arrived with. */
    unsigned int arrived = _mm_getcsr();
#endif
    (void)instance; (void)parameter; (void)work;
    while ((job = switch_job(job, &seen)) != NULL) {
#ifdef CN_X86_FP
        _mm_setcsr(job->mxcsr);
#endif
        while (seen == ReadAcquire(&generation) && claim(job)) claimed = 1;
#ifdef CN_X86_FP
        _mm_setcsr(arrived);
#endif
    }
    if (!claimed) InterlockedIncrement64(&empty_wakes);
}

static BOOL CALLBACK initialize(PINIT_ONCE init, PVOID parameter, PVOID *context)
{
    (void)init; (void)parameter; (void)context;
    DWORD_PTR process_mask = 0, system_mask = 0;
    unsigned int cpus = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)) {
        while (process_mask) {
            cpus += (unsigned int)(process_mask & 1);
            process_mask >>= 1;
        }
    }
    if (!cpus) cpus = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    capacity = cpus ? cpus : 1;
    if (capacity == 1) return TRUE;
    pool = CreateThreadpool(NULL);
    if (!pool) return FALSE;
    SetThreadpoolThreadMaximum(pool, capacity - 1);
    InitializeThreadpoolEnvironment(&environment);
    SetThreadpoolCallbackPool(&environment, pool);
    helper_work = CreateThreadpoolWork(helper, NULL, &environment);
    if (!helper_work) {
        CloseThreadpool(pool);
        pool = NULL;
        return FALSE;
    }
    return TRUE;
}

/* Register the job and return its slot (-1: the registry is full and the
 * caller works alone). *wake is how many more helpers the budget admits. */
static int enter(range_job *job, LONG *wake)
{
    int slot = -1;
    *wake = 0;
    AcquireSRWLockExclusive(&registry);
    for (int candidate = 0; candidate < CN_PARALLEL_SLOTS; ++candidate) {
        if (!jobs[candidate]) {
            slot = candidate;
            break;
        }
    }
    if (slot >= 0) {
        jobs[slot] = job;
        ++callers;
        InterlockedIncrement(&generation);
        LONG room = (LONG)capacity - callers - helpers_running;
        LONG wanted = (LONG)job->chunks - 1;
        *wake = room < wanted ? room : wanted;
        /* Another call's helpers can fill the pool (they leave at their next
         * chunk boundary), so room may be negative: no wake-up, and none counted. */
        if (*wake < 0) *wake = 0;
    }
    ReleaseSRWLockExclusive(&registry);
    return slot;
}

/* Unregister, then wait until no helper is still inside one of the job's
 * chunks. Returns nonzero when other calls remain: the budget grew by one. */
static int leave(range_job *job, int slot)
{
    int others;
    AcquireSRWLockExclusive(&registry);
    jobs[slot] = NULL;
    --callers;
    InterlockedIncrement(&generation);
    while (job->helpers) SleepConditionVariableSRW(&released, &registry, INFINITE, 0);
    others = callers > 0;
    ReleaseSRWLockExclusive(&registry);
    return others;
}

cn_status cn_parallel_for(size_t count, size_t grain, cn_range_fn fn, void *context)
{
    if (!fn || !grain) return CN_INVALID_ARGUMENT;
    if (!count) return CN_OK;
    if (count <= grain) {
        fn(context, 0, count);
        return CN_OK;
    }
    if (!InitOnceExecuteOnce(&once, initialize, NULL, NULL)) return CN_ALLOCATION_FAILED;
    range_job job = {0};
    job.fn = fn; job.context = context; job.count = count;
    job.chunks = chunk_count(count, grain);
#ifdef CN_X86_FP
    job.mxcsr = _mm_getcsr();
#endif
    InterlockedIncrement64(&dispatches);
    LONG wake = 0;
    int slot = capacity > 1 ? enter(&job, &wake) : -1;
    for (LONG i = 0; i < wake; ++i) SubmitThreadpoolWork(helper_work);
    while (claim(&job)) {}
    if (slot >= 0 && leave(&job, slot)) {
        SubmitThreadpoolWork(helper_work);
        ++wake;
    }
    if (wake) InterlockedExchangeAdd64(&wakes, wake);
    return CN_OK;
}

unsigned int cn_parallel_capacity(void)
{
    return InitOnceExecuteOnce(&once, initialize, NULL, NULL) ? capacity : 0;
}

uint64_t cn_parallel_dispatches(void)
{
    return (uint64_t)InterlockedCompareExchange64(&dispatches, 0, 0);
}

uint64_t cn_parallel_wakes(void)
{
    return (uint64_t)InterlockedCompareExchange64(&wakes, 0, 0);
}

uint64_t cn_parallel_empty_wakes(void)
{
    return (uint64_t)InterlockedCompareExchange64(&empty_wakes, 0, 0);
}
#else
cn_status cn_parallel_for(size_t count, size_t grain, cn_range_fn fn, void *context)
{
    if (!fn || !grain) return CN_INVALID_ARGUMENT;
    if (!count) return CN_OK;
    if (count <= grain) {
        fn(context, 0, count);
        return CN_OK;
    }
    size_t chunks = chunk_count(count, grain);
    for (size_t index = 0; index < chunks; ++index) run_chunk(fn, context, count, chunks, index);
    return CN_OK;
}
unsigned int cn_parallel_capacity(void) { return 1; }
uint64_t cn_parallel_dispatches(void) { return 0; }
uint64_t cn_parallel_wakes(void) { return 0; }
uint64_t cn_parallel_empty_wakes(void) { return 0; }
#endif
