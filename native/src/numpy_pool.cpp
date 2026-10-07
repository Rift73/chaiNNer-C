/* SP4c: a NumPy data-memory handler that keeps freed blocks of 512 KiB or more warm
 * (docs/superpowers/specs/2026-10-05-sp4c-output-allocation-design.md, D1-D9, 4.1-4.5).
 *
 * Switch. numpy_pool_install reads CHAINNER_C_NUMPY_POOL and CHAINNER_C_PROFILE once per
 * process, at its first call. Unset or empty = installed with the default cap,
 * min(512 MiB, 10 % of physical memory); "0" = off: never installed, so NumPy's default
 * handler serves every request exactly as without this unit; other ASCII digits =
 * installed with that cap in MiB; anything else = invalid: not installed, explained in
 * the returned message. Only then is NumPy's C API imported and its feature version
 * checked; a NumPy without the handler API (or of another ABI) is unsupported: not
 * installed, explained. None of these raises.
 *
 * Routing. A request below the threshold (512 KiB) goes to NumPy's default handler
 * unchanged, which keeps its small-block cache. A request at or above it is served by
 * the smallest idle pool block of at least its size and at most twice it (the newest
 * among equals), else by a fresh VirtualAlloc block: page-aligned, so 64-byte-aligned,
 * and zero-filled. A pointer is the pool's iff the table of pool blocks holds it, and
 * every free and realloc is routed by that lookup, never by size (NumPy passes realloc
 * no old size). Pool blocks start at the allocation granularity (64 KiB), so a pointer
 * that does not cannot be one and is passed on without the lock. realloc keeps a
 * default block on the default handler; it keeps a pool block in place while the new
 * size stays at or above the threshold and within [capacity / 2, capacity], else moves
 * it (allocate, copy, free). calloc zeroes a reused block with memset; a fresh one is
 * already zero. An idle block's address passed to free or realloc is a double free: one
 * line on stderr, then abort().
 *
 * Retention. A freed pool block becomes the newest idle block. Idle bytes never exceed
 * the cap: the oldest idle blocks are released first, and a block larger than the cap
 * is released at once. numpy_pool_release releases every idle block (the worker calls
 * it at the end of each run). Blocks go back with VirtualFree outside the lock; a
 * failure is counted for the process, never ignored.
 *
 * Install. NumPy keeps the current handler in a context variable, which a new thread
 * does not inherit, so every thread that allocates node outputs calls
 * numpy_pool_install: the worker's main thread before its event loop copies its
 * context into tasks, and each executor thread through its initializer. An array keeps
 * the handler that allocated it and is freed through it on any thread.
 *
 * Counting, with CHAINNER_C_PROFILE=1 only (as native_profile reads it), so that timed
 * runs never pay for it. Per size class of 64 KiB or more, on both sides of the
 * threshold, and per malloc / calloc / realloc: requests and bytes (atomics), and the
 * residency of up to 8 evenly spaced pages of each block a malloc or calloc returns
 * (QueryWorkingSetEx: a page outside the working set faults on first touch). At or
 * above the threshold, under the lock: the requests, pool hits, misses (cold: no idle
 * block at all; fit: none within [size, 2 size]), in-place and default-block reallocs,
 * frees, evictions, held and live bytes with their peaks, and the threads that made
 * such requests (at most 64). The request entries are compiled twice and the first
 * install picks one table: without profiling, below the threshold they only forward,
 * with no lock and no atomic. Without profiling every counter stays 0 but the idle
 * bytes, which the cap needs, and the VirtualFree failures.
 *
 * Windows only (VirtualAlloc, SRW lock, QueryWorkingSetEx), as parallel.c and scratch.c
 * are. NumPy's headers are included in this unit only, without PY_ARRAY_UNIQUE_SYMBOL,
 * so the C API table stays private to it. */
#include <pybind11/pybind11.h>
#include <pybind11/gil_safe_call_once.h>
#define NPY_NO_DEPRECATED_API NPY_1_22_API_VERSION
#include <numpy/arrayobject.h>
#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <psapi.h>

namespace py = pybind11;

namespace {

constexpr const char *capsule_name = "mem_handler";
constexpr unsigned handler_api = NPY_1_22_API_VERSION;  // PyDataMem_SetHandler and its kin
constexpr size_t threshold = size_t{1} << 19;
constexpr size_t default_cap_limit = size_t{512} << 20;
constexpr size_t class_minimum = size_t{64} << 10;  // counted and sampled from here
constexpr int sample_pages = 8;
constexpr int max_threads = 64;

/* Size classes of the counted requests: one per power of two [2^k, 2^(k+1)) for
 * k = 16..26, then ">=128M". The threshold, 2^19, is a class boundary. */
constexpr int class_count = 12;
constexpr const char *class_labels[class_count] = {
    "64K", "128K", "256K", "512K", "1M", "2M", "4M", "8M", "16M", "32M", "64M", ">=128M"};
enum Op { op_malloc, op_calloc, op_realloc, op_count };
constexpr const char *op_labels[op_count] = {"malloc", "calloc", "realloc"};

int size_class(size_t size)  // size >= class_minimum
{
    return std::min(std::bit_width(size) - 17, class_count - 1);
}

/* Written once by the first install, before the handler can be installed; read-only
 * afterwards. */
struct Config {
    size_t cap = 0;          // idle bytes retained at most
    bool profile = false;    // CHAINNER_C_PROFILE=1: counting and sampling
    size_t page = 4096;
    size_t granularity = 65536;  // VirtualAlloc's, a power of two
};
Config config;
const PyDataMemAllocator *base = nullptr;  // NumPy's default handler

struct Block {
    char *data;
    size_t capacity;  // committed bytes, whole pages
    size_t size;      // the request it serves while live
    bool live;
    Block *older;     // idle list (and a release chain), newest at the end
    Block *newer;
};

/* Profile counters at or above the threshold; under the lock. */
struct Counts {
    uint64_t requests, hits, hit_bytes, misses_cold, misses_fit, fresh, fresh_bytes;
    uint64_t realloc_inplace, realloc_passthrough, frees, evictions, evicted_bytes;
    uint64_t dropped, dropped_bytes, releases, released_bytes;
    uint64_t class_hits[class_count][2];
};

struct Snapshot {
    Counts counts;
    size_t held_bytes, held_blocks, held_peak, live_bytes, live_blocks, live_peak;
    int threads;
};

/* Profile counters per size class and op: [count, bytes]; and per size class, the
 * sampled pages and those outside the working set. */
std::atomic<uint64_t> class_requests[class_count][op_count][2];
std::atomic<uint64_t> class_sampled[class_count][2];
std::atomic<uint64_t> sample_failures{0};
std::atomic<uint64_t> release_failures{0};  // the process's, whatever the profile

SRWLOCK lock = SRWLOCK_INIT;
// Under the lock: the pool's blocks, live and idle, by data address (open addressing,
// linear probing, backward-shift deletion), the idle list and its bytes; with
// profiling, the other totals and the counts.
Block **slots = nullptr;
int slot_bits = 0;
size_t table_blocks = 0;
Block *oldest = nullptr;
Block *newest = nullptr;
size_t held_bytes = 0;
size_t held_blocks = 0, held_peak = 0, live_bytes = 0, live_blocks = 0, live_peak = 0;
Counts counts = {};
DWORD thread_ids[max_threads];
int thread_count = 0;

struct Locked {
    Locked() { AcquireSRWLockExclusive(&lock); }
    ~Locked() { ReleaseSRWLockExclusive(&lock); }
    Locked(const Locked &) = delete;
    Locked &operator=(const Locked &) = delete;
};

size_t slot_of(const void *data)
{
    uint64_t key = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(data) / config.granularity);
    return static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> (64 - slot_bits));
}

Block *table_find(const void *data)
{
    if (!slots) return nullptr;
    size_t mask = (size_t{1} << slot_bits) - 1;
    for (size_t i = slot_of(data);; i = (i + 1) & mask) {
        if (!slots[i] || slots[i]->data == data) return slots[i];
    }
}

void table_place(Block *block)
{
    size_t mask = (size_t{1} << slot_bits) - 1;
    size_t i = slot_of(block->data);
    while (slots[i]) i = (i + 1) & mask;
    slots[i] = block;
}

bool table_insert(Block *block)
{
    if (!slots || (table_blocks + 1) * 2 > (size_t{1} << slot_bits)) {
        int bits = slots ? slot_bits + 1 : 6;
        Block **grown = static_cast<Block **>(std::calloc(size_t{1} << bits, sizeof(Block *)));
        if (!grown) return false;
        Block **old = slots;
        size_t old_size = old ? size_t{1} << slot_bits : 0;
        slots = grown;
        slot_bits = bits;
        for (size_t j = 0; j < old_size; ++j) {
            if (old[j]) table_place(old[j]);
        }
        std::free(old);
    }
    table_place(block);
    ++table_blocks;
    return true;
}

void table_erase(const Block *block)
{
    size_t mask = (size_t{1} << slot_bits) - 1;
    size_t hole = slot_of(block->data);
    while (slots[hole] != block) hole = (hole + 1) & mask;
    slots[hole] = nullptr;
    --table_blocks;
    for (size_t j = (hole + 1) & mask; slots[j]; j = (j + 1) & mask) {
        // An entry may fill the hole unless its home slot lies cyclically in (hole, j].
        size_t home = slot_of(slots[j]->data);
        if (((j - home) & mask) >= ((j - hole) & mask)) {
            slots[hole] = slots[j];
            slots[j] = nullptr;
            hole = j;
        }
    }
}

void idle_push(Block *block)
{
    block->newer = nullptr;
    block->older = newest;
    (newest ? newest->newer : oldest) = block;
    newest = block;
    held_bytes += block->capacity;
    if (config.profile) {
        ++held_blocks;
        held_peak = std::max(held_peak, held_bytes);
    }
}

void idle_unlink(Block *block)
{
    (block->older ? block->older->newer : oldest) = block->newer;
    (block->newer ? block->newer->older : newest) = block->older;
    held_bytes -= block->capacity;
    if (config.profile) --held_blocks;
}

/* A request at or above the threshold, counted with its thread; profile only. */
void note_request()
{
    ++counts.requests;
    DWORD id = GetCurrentThreadId();
    for (int i = 0; i < thread_count; ++i) {
        if (thread_ids[i] == id) return;
    }
    if (thread_count < max_threads) thread_ids[thread_count++] = id;
}

void note_live(Block *block, size_t size)
{
    block->live = true;
    block->size = size;
    if (config.profile) {
        live_bytes += block->capacity;
        ++live_blocks;
        live_peak = std::max(live_peak, live_bytes);
    }
}

/* The smallest idle block holding size bytes and at most twice size, the newest among
 * equals, unlinked from the idle list; null when none fits. */
Block *take(size_t size)
{
    size_t limit = size <= SIZE_MAX / 2 ? size * 2 : SIZE_MAX;
    Block *best = nullptr;
    for (Block *block = newest; block; block = block->older) {
        if (block->capacity >= size && block->capacity <= limit &&
            (!best || block->capacity < best->capacity))
            best = block;
    }
    if (best) idle_unlink(best);
    return best;
}

/* A live block freed by NumPy: kept as the newest idle block, after releasing the oldest
 * ones while the idle bytes would exceed the cap, or released itself when larger than
 * the cap. Returns the blocks to release, chained through older. */
Block *retire(Block *block)
{
    block->live = false;
    if (config.profile) {
        live_bytes -= block->capacity;
        --live_blocks;
        ++counts.frees;
    }
    if (block->capacity > config.cap) {
        table_erase(block);
        if (config.profile) {
            ++counts.dropped;
            counts.dropped_bytes += block->capacity;
        }
        block->older = nullptr;
        return block;
    }
    Block *released = nullptr;
    while (held_bytes + block->capacity > config.cap) {
        Block *victim = oldest;
        idle_unlink(victim);
        table_erase(victim);
        if (config.profile) {
            ++counts.evictions;
            counts.evicted_bytes += victim->capacity;
        }
        victim->older = released;
        released = victim;
    }
    idle_push(block);
    return released;
}

void release(Block *chain)
{
    while (chain) {
        Block *next = chain->older;
        if (!VirtualFree(chain->data, 0, MEM_RELEASE)) release_failures.fetch_add(1, std::memory_order_relaxed);
        std::free(chain);
        chain = next;
    }
}

/* The live pool block at ptr, under the lock; null for a pointer the pool does not own.
 * An idle block's address cannot come back from NumPy: no allocator hands it out while
 * the pool holds it, so it is a double free, and it stops the process (D9). */
Block *live_find(void *ptr)
{
    Block *block = table_find(ptr);
    if (block && !block->live) {
        std::fprintf(stderr, "chaiNNer-C numpy pool: double free of pool block %p; aborting\n", ptr);
        std::fflush(stderr);
        std::abort();
    }
    return block;
}

bool pool_aligned(const void *ptr)
{
    return ptr && (reinterpret_cast<uintptr_t>(ptr) & (config.granularity - 1)) == 0;
}

/* Profile only: a request of size bytes by op, counted from class_minimum. */
void count(size_t size, Op op)
{
    if (size < class_minimum) return;
    std::atomic<uint64_t> *entry = class_requests[size_class(size)][op];
    entry[0].fetch_add(1, std::memory_order_relaxed);
    entry[1].fetch_add(size, std::memory_order_relaxed);
}

/* Profile only: the residency of up to 8 evenly spaced pages of a block of size bytes
 * handed out at data, from class_minimum. */
void sample(const void *data, size_t size)
{
    if (!data || size < class_minimum) return;
    uintptr_t first = reinterpret_cast<uintptr_t>(data) / config.page * config.page;
    uintptr_t last = (reinterpret_cast<uintptr_t>(data) + size - 1) / config.page * config.page;
    size_t pages = (last - first) / config.page + 1;
    int taken = pages < static_cast<size_t>(sample_pages) ? static_cast<int>(pages) : sample_pages;
    PSAPI_WORKING_SET_EX_INFORMATION info[sample_pages] = {};
    for (int i = 0; i < taken; ++i) {
        size_t page = taken == 1 ? 0 : (pages - 1) * static_cast<size_t>(i) / static_cast<size_t>(taken - 1);
        info[i].VirtualAddress = reinterpret_cast<void *>(first + page * config.page);
    }
    if (!QueryWorkingSetEx(GetCurrentProcess(), info, static_cast<DWORD>(sizeof(info[0]) * taken))) {
        sample_failures.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    uint64_t absent = 0;
    for (int i = 0; i < taken; ++i) absent += info[i].VirtualAttributes.Valid ? 0 : 1;
    std::atomic<uint64_t> *entry = class_sampled[size_class(size)];
    entry[0].fetch_add(static_cast<uint64_t>(taken), std::memory_order_relaxed);
    entry[1].fetch_add(absent, std::memory_order_relaxed);
}

/* A fresh pool block of size bytes, live; null when memory runs out (NumPy then raises
 * MemoryError). */
void *pool_fresh(size_t size)
{
    if (size > SIZE_MAX - (config.page - 1)) return nullptr;
    size_t capacity = (size + config.page - 1) / config.page * config.page;
    void *data = VirtualAlloc(nullptr, capacity, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!data) return nullptr;
    auto *block = static_cast<Block *>(std::malloc(sizeof(Block)));
    bool kept = false;
    if (block) {
        *block = Block{static_cast<char *>(data), capacity, size, false, nullptr, nullptr};
        Locked held;
        kept = table_insert(block);
        if (kept) {
            note_live(block, size);
            if (config.profile) {
                ++counts.fresh;
                counts.fresh_bytes += capacity;
            }
        }
    }
    if (!kept) {
        std::free(block);
        if (!VirtualFree(data, 0, MEM_RELEASE)) release_failures.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    return data;
}

/* A request at or above the threshold: an idle block, else a fresh one; null when memory
 * runs out. With profiling, a malloc or calloc samples the block before a calloc zeroes
 * a reused one (realloc copies, so it touches every page it keeps: never sampled). */
void *pool_allocate(size_t size, Op op)
{
    Block *block;
    {
        Locked held;
        block = take(size);
        if (block) note_live(block, size);
        if (config.profile) {
            note_request();
            if (block) {
                ++counts.hits;
                counts.hit_bytes += size;
                uint64_t *hits = counts.class_hits[size_class(size)];
                ++hits[0];
                hits[1] += size;
            } else {
                ++(oldest ? counts.misses_fit : counts.misses_cold);
            }
        }
    }
    void *data = block ? block->data : pool_fresh(size);
    if (config.profile && op != op_realloc) sample(data, size);
    if (block && op == op_calloc) std::memset(data, 0, size);
    return data;
}

template <bool profiled>
void *handler_malloc(void *, size_t size)
{
    if constexpr (profiled) count(size, op_malloc);
    if (size >= threshold) return pool_allocate(size, op_malloc);
    void *data = base->malloc(base->ctx, size);
    if constexpr (profiled) sample(data, size);
    return data;
}

template <bool profiled>
void *handler_calloc(void *, size_t nelem, size_t elsize)
{
    // An overflowing product has no size to route by; the default handler fails it.
    // NumPy asks for (bytes, 1), which skips the division.
    if (elsize > 1 && nelem > SIZE_MAX / elsize) return base->calloc(base->ctx, nelem, elsize);
    size_t size = nelem * elsize;
    if constexpr (profiled) count(size, op_calloc);
    if (size >= threshold) return pool_allocate(size, op_calloc);
    void *data = base->calloc(base->ctx, nelem, elsize);
    if constexpr (profiled) sample(data, size);
    return data;
}

template <bool profiled>
void *handler_realloc(void *, void *ptr, size_t new_size)
{
    if constexpr (profiled) count(new_size, op_realloc);
    Block *block = nullptr;
    size_t old_size = 0;
    if (pool_aligned(ptr)) {
        Locked held;
        block = live_find(ptr);
        if (block) {
            if (new_size >= threshold && new_size <= block->capacity && block->capacity / 2 <= new_size) {
                block->size = new_size;
                if constexpr (profiled) {
                    note_request();
                    ++counts.realloc_inplace;
                }
                return ptr;
            }
            old_size = block->size;
        }
    }
    if (!block) {
        // A default block (or none) stays with the default handler at any size.
        if constexpr (profiled) {
            if (new_size >= threshold) {
                Locked held;
                note_request();
                ++counts.realloc_passthrough;
            }
        }
        return base->realloc(base->ctx, ptr, new_size);
    }
    void *data = new_size < threshold ? base->malloc(base->ctx, new_size) : pool_allocate(new_size, op_realloc);
    if (!data) return nullptr;  // The old block stays the caller's, as with realloc.
    std::memcpy(data, ptr, std::min(old_size, new_size));
    Block *released;
    {
        Locked held;
        released = retire(block);
    }
    release(released);
    return data;
}

void handler_free(void *, void *ptr, size_t size)
{
    if (pool_aligned(ptr)) {
        Block *released = nullptr;
        bool pooled;
        {
            Locked held;
            Block *block = live_find(ptr);
            pooled = block != nullptr;
            if (pooled) released = retire(block);
        }
        if (pooled) {
            release(released);
            return;
        }
    }
    base->free(base->ctx, ptr, size);
}

template <bool profiled>
PyDataMem_Handler handler = {
    "chainner_c_numpy_pool", 1,
    {nullptr, handler_malloc<profiled>, handler_calloc<profiled>, handler_realloc<profiled>, handler_free}};

/* The variable's value; empty when it is unset or empty, which the switch treats alike. */
std::wstring environment(const wchar_t *name)
{
    std::wstring value;
    DWORD size = GetEnvironmentVariableW(name, nullptr, 0);  // with the terminator; 0 when unset
    while (size) {
        value.resize(size);
        DWORD length = GetEnvironmentVariableW(name, value.data(), size);
        if (length < size) {  // copied, without the terminator
            value.resize(length);
            return value;
        }
        size = length;  // the variable grew meanwhile
    }
    return {};
}

/* The pending Python error as "Type: message"; the error indicator is cleared. */
py::str pending_error()
{
    py::error_already_set error;
    return py::str("{}: {}").format(error.type().attr("__name__"), error.value());
}

py::str unsupported(const py::str &reason)
{
    return py::str("NumPy's data-memory handler API is unavailable ({}); the NumPy pool is not installed")
        .format(reason);
}

/* What the first install found; kept until interpreter shutdown, when pybind11 3.1
 * releases its call-once storage. The handler's capsule lives as long, and longer while
 * an array allocated through it still holds it. */
struct Setup {
    const char *status;  // "installed", "off", "invalid" or "unsupported"
    py::str message;     // empty unless invalid or unsupported
    py::object capsule;  // the handler's, when installed
};

Setup configure()
{
    std::wstring value = environment(L"CHAINNER_C_NUMPY_POOL");
    size_t cap = 0;
    if (!value.empty()) {
        size_t mib = 0;
        for (wchar_t digit : value) {
            if (digit < L'0' || digit > L'9' || mib > ((SIZE_MAX >> 20) - 9) / 10) {
                PyObject *text = PyUnicode_FromWideChar(value.data(), static_cast<Py_ssize_t>(value.size()));
                if (!text) throw py::error_already_set();
                py::str message = py::str("CHAINNER_C_NUMPY_POOL='{}' is not empty, 0 or a cap in MiB (ASCII digits); "
                                          "the NumPy pool is not installed")
                                      .format(py::reinterpret_steal<py::str>(text));
                return {"invalid", message, py::object()};
            }
            mib = mib * 10 + static_cast<size_t>(digit - L'0');
        }
        if (mib == 0) return {"off", py::str(), py::object()};
        cap = mib << 20;
    }
    bool profile = environment(L"CHAINNER_C_PROFILE") == L"1";
    if (!cap) {
        MEMORYSTATUSEX memory = {};
        memory.dwLength = sizeof(memory);
        if (!GlobalMemoryStatusEx(&memory)) {
            PyErr_SetFromWindowsErr(0);
            throw py::error_already_set();
        }
        cap = static_cast<size_t>(std::min<uint64_t>(default_cap_limit, memory.ullTotalPhys / 10));
    }

    if (_import_array() < 0) return {"unsupported", unsupported(pending_error()), py::object()};
    unsigned feature = PyArray_GetNDArrayCFeatureVersion();
    if (feature < handler_api) {
        py::str reason = py::str("C API feature version {:#x} is below {:#x}, NumPy 1.22's").format(feature, handler_api);
        return {"unsupported", unsupported(reason), py::object()};
    }
    auto *defaults = static_cast<PyDataMem_Handler *>(PyCapsule_GetPointer(PyDataMem_DefaultHandler, capsule_name));
    if (!defaults) return {"unsupported", unsupported(pending_error()), py::object()};

    PyDataMem_Handler *own = profile ? &handler<true> : &handler<false>;
    py::object capsule = py::reinterpret_steal<py::object>(PyCapsule_New(own, capsule_name, nullptr));
    if (!capsule) throw py::error_already_set();
    Py_INCREF(PyDataMem_DefaultHandler);  // kept with base, never released
    SYSTEM_INFO system;
    GetSystemInfo(&system);
    // No Python call from here on: a thread that reads the settings holds the GIL.
    config = Config{cap, profile, system.dwPageSize, system.dwAllocationGranularity};
    base = &defaults->allocator;
    return {"installed", py::str(), std::move(capsule)};
}

/* Installs the handler in the calling thread's context when the first call found it
 * installable; returns (status, message). */
py::tuple install()
{
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<Setup> storage;
    const Setup &setup = storage.call_once_and_store_result(configure).get_stored();
    if (setup.capsule) {
        PyObject *previous = PyDataMem_SetHandler(setup.capsule.ptr());
        if (!previous) throw py::error_already_set();
        Py_DECREF(previous);
    }
    return py::make_tuple(setup.status, setup.message);
}

py::tuple pair(uint64_t first, uint64_t second) { return py::make_tuple(first, second); }

py::dict stats()
{
    Snapshot s;
    {
        Locked held;
        s = Snapshot{counts, held_bytes, held_blocks, held_peak, live_bytes, live_blocks, live_peak, thread_count};
    }
    py::dict result;
    result["requests"] = s.counts.requests;
    result["hits"] = s.counts.hits;
    result["hit_bytes"] = s.counts.hit_bytes;
    result["misses_cold"] = s.counts.misses_cold;
    result["misses_fit"] = s.counts.misses_fit;
    result["fresh"] = s.counts.fresh;
    result["fresh_bytes"] = s.counts.fresh_bytes;
    result["realloc_inplace"] = s.counts.realloc_inplace;
    result["realloc_passthrough"] = s.counts.realloc_passthrough;
    result["frees"] = s.counts.frees;
    result["evictions"] = s.counts.evictions;
    result["evicted_bytes"] = s.counts.evicted_bytes;
    result["dropped"] = s.counts.dropped;
    result["dropped_bytes"] = s.counts.dropped_bytes;
    result["releases"] = s.counts.releases;
    result["released_bytes"] = s.counts.released_bytes;
    result["release_failures"] = release_failures.load(std::memory_order_relaxed);
    result["sample_failures"] = sample_failures.load(std::memory_order_relaxed);
    result["held_bytes"] = s.held_bytes;
    result["held_blocks"] = s.held_blocks;
    result["held_peak"] = s.held_peak;
    result["live_bytes"] = s.live_bytes;
    result["live_blocks"] = s.live_blocks;
    result["live_peak"] = s.live_peak;
    result["threads"] = s.threads;
    result["cap_bytes"] = config.cap;
    result["threshold_bytes"] = threshold;
    py::dict classes;
    for (int c = 0; c < class_count; ++c) {
        py::dict entry;
        for (int op = 0; op < op_count; ++op) {
            uint64_t requests = class_requests[c][op][0].load(std::memory_order_relaxed);
            if (requests) entry[op_labels[op]] = pair(requests, class_requests[c][op][1].load(std::memory_order_relaxed));
        }
        if (s.counts.class_hits[c][0]) entry["hits"] = pair(s.counts.class_hits[c][0], s.counts.class_hits[c][1]);
        uint64_t sampled = class_sampled[c][0].load(std::memory_order_relaxed);
        if (sampled) entry["sampled"] = pair(sampled, class_sampled[c][1].load(std::memory_order_relaxed));
        if (!entry.empty()) classes[class_labels[c]] = entry;
    }
    result["classes"] = classes;
    return result;
}

/* Zeroes the profile counters; the peaks restart from the current bytes. The idle blocks
 * and the process's VirtualFree failures stay. */
void reset()
{
    {
        Locked held;
        counts = {};
        if (config.profile) {
            held_peak = held_bytes;
            live_peak = live_bytes;
        }
        thread_count = 0;
    }
    for (auto &by_op : class_requests) {
        for (auto &entry : by_op) {
            entry[0].store(0, std::memory_order_relaxed);
            entry[1].store(0, std::memory_order_relaxed);
        }
    }
    for (auto &entry : class_sampled) {
        entry[0].store(0, std::memory_order_relaxed);
        entry[1].store(0, std::memory_order_relaxed);
    }
    sample_failures.store(0, std::memory_order_relaxed);
}

/* Releases every idle block; returns the bytes released and the process's VirtualFree
 * failures so far. */
py::tuple release_idle()
{
    Block *chain = nullptr;
    size_t bytes;
    {
        Locked held;
        bytes = held_bytes;
        while (oldest) {
            Block *block = oldest;
            idle_unlink(block);
            table_erase(block);
            block->older = chain;
            chain = block;
        }
        if (config.profile) {
            ++counts.releases;
            counts.released_bytes += bytes;
        }
    }
    {
        py::gil_scoped_release unlocked;
        release(chain);
    }
    return py::make_tuple(bytes, release_failures.load(std::memory_order_relaxed));
}

/* The idle blocks, oldest first, as (address, capacity). */
py::list idle_blocks()
{
    std::vector<std::pair<uintptr_t, size_t>> blocks;
    {
        Locked held;
        for (Block *block = oldest; block; block = block->newer)
            blocks.emplace_back(reinterpret_cast<uintptr_t>(block->data), block->capacity);
    }
    py::list result;
    for (const auto &[address, capacity] : blocks) result.append(py::make_tuple(address, capacity));
    return result;
}

}  // namespace

void cn_bind_numpy_pool(py::module_ &module)
{
    module.def("numpy_pool_install", &install,
               "Install the NumPy memory handler (CHAINNER_C_NUMPY_POOL) on the calling thread; "
               "returns (status, message).");
    module.def("numpy_pool_release", &release_idle,
               "Release every idle pool block; returns (bytes released, the process's VirtualFree failures).");
    module.def("numpy_pool_stats", &stats, "The handler's counters since the last reset, as a dict.");
    module.def("numpy_pool_reset", &reset, "Zero the handler's counters.");
    module.def("numpy_pool_idle_blocks", &idle_blocks, "The idle pool blocks, oldest first, as (address, capacity).");
}
