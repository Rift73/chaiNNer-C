#ifdef __linux__
#define _GNU_SOURCE /* sched_getaffinity (cn_platform.h) under -std=c17 */
#endif
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
#elif defined(__linux__)
/* The Windows branch's pool on POSIX threads: the same registry, claims, helper budget,
 * MXCSR hand-over and counters. The thread pool's work object becomes a queue of
 * submitted helper calls, each run once by one of capacity - 1 helper threads, which
 * start with the pool and live as long as the process. Interlocked operations are
 * sequentially consistent atomics, ReadAcquire an acquire load, the SRW lock a mutex
 * and its condition variable a pthread one. */
#include "cn_platform.h"
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <string.h>
#include <unistd.h>
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#define CN_X86_FP 1
#endif

#define CN_PARALLEL_SLOTS 32

typedef struct range_job {
    cn_range_fn fn;
    void *context;
    size_t count;
    size_t chunks;
    atomic_long next; /* next unclaimed chunk; claimed without the lock */
    long helpers;     /* helpers pinned to this job; guarded by registry */
#ifdef CN_X86_FP
    unsigned int mxcsr;
#endif
} range_job;

/* InitOnceExecuteOnce: initialized is set (release) once initialize succeeded, under
 * init_lock; a failed initialization runs again at the next call. */
static pthread_mutex_t init_lock = PTHREAD_MUTEX_INITIALIZER;
static atomic_int initialized;
static unsigned int capacity = 1;
static unsigned int threads = 0; /* helper threads started; guarded by init_lock */
static _Atomic uint64_t dispatches;
/* Helper wake-ups submitted, and helper calls that claimed no chunk: one atomic add
 * per dispatch and at most one per helper call. */
static _Atomic uint64_t wakes;
static _Atomic uint64_t empty_wakes;

/* registry guards jobs, callers, helpers_running and every job's helpers, as in the
 * Windows branch; generation is compared for equality only. */
static pthread_mutex_t registry = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t released = PTHREAD_COND_INITIALIZER;
static range_job *jobs[CN_PARALLEL_SLOTS];
static long callers = 0;
static long helpers_running = 0;
static atomic_uint generation;

/* Submitted helper calls not yet started; guarded by queue_lock. */
static pthread_mutex_t queue_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t queue_filled = PTHREAD_COND_INITIALIZER;
static size_t queued = 0;

static int claim(range_job *job)
{
    long index = atomic_fetch_add(&job->next, 1);
    if ((size_t)index >= job->chunks) return 0;
    run_chunk(job->fn, job->context, job->count, job->chunks, (size_t)index);
    return 1;
}

/* Release the helper's previous job, then pin the job with unclaimed chunks
 * and the fewest helpers. NULL means the helper leaves: nothing needs help, or
 * callers plus helpers would exceed the CPU count. */
static range_job *switch_job(range_job *previous, unsigned int *seen)
{
    range_job *best = NULL;
    int last = 0;
    pthread_mutex_lock(&registry);
    if (previous) {
        last = --previous->helpers == 0;
    } else {
        ++helpers_running;
    }
    if (helpers_running + callers <= (long)capacity) {
        for (int slot = 0; slot < CN_PARALLEL_SLOTS; ++slot) {
            range_job *job = jobs[slot];
            if (job && atomic_load_explicit(&job->next, memory_order_relaxed) < (long)job->chunks
                && (!best || job->helpers < best->helpers)) best = job;
        }
    }
    if (best) ++best->helpers;
    else --helpers_running;
    *seen = atomic_load_explicit(&generation, memory_order_acquire);
    pthread_mutex_unlock(&registry);
    /* Wake after the release: a caller woken under the lock would block on it
     * and need a second wake. */
    if (last) pthread_cond_broadcast(&released);
    return best;
}

/* One submitted helper call. */
static void helper(void)
{
    range_job *job = NULL;
    unsigned int seen = 0;
    int claimed = 0;
#ifdef CN_X86_FP
    /* A helper runs every chunk under its job's rounding/denormal controls and
     * returns to the pool with the controls it arrived with. */
    unsigned int arrived = _mm_getcsr();
#endif
    while ((job = switch_job(job, &seen)) != NULL) {
#ifdef CN_X86_FP
        _mm_setcsr(job->mxcsr);
#endif
        while (seen == atomic_load_explicit(&generation, memory_order_acquire) && claim(job)) claimed = 1;
#ifdef CN_X86_FP
        _mm_setcsr(arrived);
#endif
    }
    if (!claimed) atomic_fetch_add(&empty_wakes, 1);
}

/* A helper thread: runs one submitted call at a time, for the life of the process. */
static void *serve(void *parameter)
{
    (void)parameter;
    pthread_mutex_lock(&queue_lock);
    for (;;) {
        while (!queued) pthread_cond_wait(&queue_filled, &queue_lock);
        --queued;
        pthread_mutex_unlock(&queue_lock);
        helper();
        pthread_mutex_lock(&queue_lock);
    }
}

/* SubmitThreadpoolWork: one more helper call, run by the first idle helper thread. */
static void submit(void)
{
    pthread_mutex_lock(&queue_lock);
    ++queued;
    pthread_mutex_unlock(&queue_lock);
    pthread_cond_signal(&queue_filled);
}

/* A fork copies only the forking thread, which is never inside a call here: the
 * child's pool has no helper thread and no other caller, and starts afresh at its
 * first call. The locks are held across the fork so the child's copies are
 * consistent; its condition variables lose the parent's waiters. */
static void fork_prepare(void)
{
    pthread_mutex_lock(&init_lock);
    pthread_mutex_lock(&registry);
    pthread_mutex_lock(&queue_lock);
}

static void fork_parent(void)
{
    pthread_mutex_unlock(&queue_lock);
    pthread_mutex_unlock(&registry);
    pthread_mutex_unlock(&init_lock);
}

static void fork_child(void)
{
    memset(jobs, 0, sizeof jobs);
    callers = 0;
    helpers_running = 0;
    queued = 0;
    threads = 0;
    atomic_store(&initialized, 0);
    pthread_cond_init(&released, NULL);
    pthread_cond_init(&queue_filled, NULL);
    fork_parent();
}

/* Under init_lock: the CPU count (the affinity mask's, as GetProcessAffinityMask's)
 * and capacity - 1 helper threads, which block every signal so that the threads that
 * call in keep receiving them. */
static int initialize(void)
{
    static int fork_handlers = 0;
    unsigned int cpus = cn_affinity_cpus();
    if (!cpus) {
        long online = sysconf(_SC_NPROCESSORS_ONLN);
        cpus = online > 0 ? (unsigned int)online : 0;
    }
    capacity = cpus ? cpus : 1;
    if (capacity == 1) return 1;
    if (!fork_handlers) {
        if (pthread_atfork(fork_prepare, fork_parent, fork_child) != 0) return 0;
        fork_handlers = 1;
    }
    sigset_t all, previous;
    sigfillset(&all);
    pthread_sigmask(SIG_SETMASK, &all, &previous);
    while (threads + 1 < capacity) {
        pthread_t thread;
        if (pthread_create(&thread, NULL, serve, NULL) != 0) break;
        pthread_detach(thread);
        ++threads;
    }
    pthread_sigmask(SIG_SETMASK, &previous, NULL);
    return threads + 1 >= capacity;
}

static int initialize_once(void)
{
    if (atomic_load_explicit(&initialized, memory_order_acquire)) return 1;
    pthread_mutex_lock(&init_lock);
    int done = atomic_load_explicit(&initialized, memory_order_relaxed) || initialize();
    if (done) atomic_store_explicit(&initialized, 1, memory_order_release);
    pthread_mutex_unlock(&init_lock);
    return done;
}

/* Register the job and return its slot (-1: the registry is full and the
 * caller works alone). *wake is how many more helpers the budget admits. */
static int enter(range_job *job, long *wake)
{
    int slot = -1;
    *wake = 0;
    pthread_mutex_lock(&registry);
    for (int candidate = 0; candidate < CN_PARALLEL_SLOTS; ++candidate) {
        if (!jobs[candidate]) {
            slot = candidate;
            break;
        }
    }
    if (slot >= 0) {
        jobs[slot] = job;
        ++callers;
        atomic_fetch_add(&generation, 1);
        long room = (long)capacity - callers - helpers_running;
        long wanted = (long)job->chunks - 1;
        *wake = room < wanted ? room : wanted;
        /* Another call's helpers can fill the pool (they leave at their next
         * chunk boundary), so room may be negative: no wake-up, and none counted. */
        if (*wake < 0) *wake = 0;
    }
    pthread_mutex_unlock(&registry);
    return slot;
}

/* Unregister, then wait until no helper is still inside one of the job's
 * chunks. Returns nonzero when other calls remain: the budget grew by one. */
static int leave(range_job *job, int slot)
{
    int others;
    pthread_mutex_lock(&registry);
    jobs[slot] = NULL;
    --callers;
    atomic_fetch_add(&generation, 1);
    while (job->helpers) pthread_cond_wait(&released, &registry);
    others = callers > 0;
    pthread_mutex_unlock(&registry);
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
    if (!initialize_once()) return CN_ALLOCATION_FAILED;
    range_job job = {0};
    job.fn = fn; job.context = context; job.count = count;
    job.chunks = chunk_count(count, grain);
#ifdef CN_X86_FP
    job.mxcsr = _mm_getcsr();
#endif
    atomic_fetch_add(&dispatches, 1);
    long wake = 0;
    int slot = capacity > 1 ? enter(&job, &wake) : -1;
    for (long i = 0; i < wake; ++i) submit();
    while (claim(&job)) {}
    if (slot >= 0 && leave(&job, slot)) {
        submit();
        ++wake;
    }
    if (wake) atomic_fetch_add(&wakes, (uint64_t)wake);
    return CN_OK;
}

unsigned int cn_parallel_capacity(void)
{
    return initialize_once() ? capacity : 0;
}

uint64_t cn_parallel_dispatches(void)
{
    return atomic_load(&dispatches);
}

uint64_t cn_parallel_wakes(void)
{
    return atomic_load(&wakes);
}

uint64_t cn_parallel_empty_wakes(void)
{
    return atomic_load(&empty_wakes);
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
