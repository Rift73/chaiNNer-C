/* The native run engine (native/local/instant-exec/DESIGN.md section 2, amendments A2 and A6), behind
 * backend/src/native_executor.py.
 *
 * A run has one driver thread, which follows process.Executor's control flow: the generator groups one after
 * another (each generator run once, its collectors started, then the items in order: the generators' advance, the
 * item's side-effect nodes, its collectors, its progress; a context item before a generator's range runs its loop's
 * temporal nodes alone), then the non-iterated side-effect nodes. A node that may
 * not run beside others (not pure_cpu_node: side effects, a context, a GPU, a lazy input) runs on the driver, in the
 * order process.Executor's demand-driven recursion runs it (inputs in position order); a lazy input is evaluated on
 * the thread that reads it, nested in its reader (its upstream's events between the reader's start and finish).
 * Pure nodes run on worker threads as soon as they are wanted and their inputs are done: every phase first wants
 * the pure nodes its sinks need, so independent branches and sinks overlap.
 *
 * Values: a cell (a node's run: one per node, or one per node and item for a generator group's iterated nodes) holds
 * its outputs; a Python node's outputs stay the Python objects run_node returned (PyRef), a native node's are native
 * Values; each crossing converts zero-copy (python.cpp). Counted cache entries are released after their last read
 * (chain/cache.py's strategies), iterated ones after their item.
 *
 * Python: node bodies go through process.run_node (via the hooks of native_executor.py); at most
 * CHAINNER_C_PYTHON_THREADS (default 8) engine threads are in Python at once; a thread waiting for a cell releases
 * the GIL and its Python slot first; references dropped without the GIL are released in batches. Events are built
 * here and posted to the server's queue in batches by one pump thread; broadcasts are computed latest-first, one at
 * a time, by one broadcaster thread (process.Executor's LatestBroadcasts). */
#include "engine.hpp"
#include "image_decode.hpp"
#include "image_preview.hpp"

#include <pybind11/gil_safe_call_once.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <cstring>
#include <exception>
#include <functional>
#include <list>
#include <future>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <semaphore>
#include <set>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#else
#include "cn_platform.h"
#endif

namespace py = pybind11;

namespace cn::run {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

int environment_count(const char *name, int fallback)
{
#ifdef _WIN32
    char *value = nullptr;
    size_t size = 0;
    if (_dupenv_s(&value, &size, name) != 0 || !value) return fallback;
    std::string text(value);
    free(value);
#else
    const char *value = std::getenv(name);
    if (!value) return fallback;
    std::string text(value);
#endif
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos || text.size() > 4) return fallback;
    int count = std::stoi(text);
    return count >= 1 ? count : fallback;
}

/* A Python exception that ends a step of the run (a failed cell's error, Aborted, an iteration error). */
struct PythonError {
    PyRef error;
};
/* Stop: raised at the driver's suspend points; becomes progress_controller.Aborted. */
struct StopRun {};
/* A generator's supplier is exhausted (process.Executor's StopIteration). */
struct ItemsEnd {};

constexpr int python_slot_limit = 1024;  // python_slots' hard limit

/* The engine threads allowed in Python at once: CHAINNER_C_PYTHON_THREADS (default 8), a larger value than
 * python_slot_limit counting as that limit. */
int python_threads()
{
    static const int count = std::min(environment_count("CHAINNER_C_PYTHON_THREADS", 8), python_slot_limit);
    return count;
}

/* At most python_threads() engine threads in Python (A2): a slot, then the GIL; per thread, nested sections share
 * the outer slot. */
std::counting_semaphore<python_slot_limit> &python_slots()
{
    static std::counting_semaphore<python_slot_limit> slots(python_threads());
    return slots;
}
thread_local int slot_depth = 0;

class PythonSection {
public:
    PythonSection()
    {
        if (slot_depth++ == 0) python_slots().acquire();
        acquire_.emplace();
        release_deferred();
    }
    ~PythonSection()
    {
        release_deferred();
        acquire_.reset();
        if (--slot_depth == 0) python_slots().release();
    }
    PythonSection(const PythonSection &) = delete;
    PythonSection &operator=(const PythonSection &) = delete;

private:
    std::optional<py::gil_scoped_acquire> acquire_;
};

/* Inside a PythonSection: the GIL and the thread's slot released while the thread waits for the engine. */
class PythonPause {
public:
    PythonPause() : depth_(slot_depth)
    {
        release_.emplace();
        if (depth_ > 0) {
            slot_depth = 0;
            python_slots().release();
        }
    }
    ~PythonPause()
    {
        if (depth_ > 0) {
            python_slots().acquire();
            slot_depth = depth_;
        }
        release_.reset();
    }
    PythonPause(const PythonPause &) = delete;
    PythonPause &operator=(const PythonPause &) = delete;

private:
    int depth_;
    std::optional<py::gil_scoped_release> release_;
};

PyRef current_error(py::error_already_set &error)
{
    return py_ref(error.value());
}

/* An error no caller can receive (an event's or a broadcast's), reported as Python's unraisable hook reports one
 * (GIL held). */
void report_unraisable(const char *where)
{
    try {
        throw;
    } catch (py::error_already_set &error) {
        error.discard_as_unraisable(where);
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        PyErr_WriteUnraisable(py::str(where).ptr());
    }
}

/* A C++ exception escaping native code (an allocation, a status check) as the node's error: a NodeError as itself,
 * any other as pybind11 raises it when the same code runs under a Python body. */
NodeError escaped(const std::exception &error)
{
    if (const auto *node = dynamic_cast<const NodeError *>(&error)) return *node;
    const char *type = "RuntimeError";
    if (dynamic_cast<const std::bad_alloc *>(&error))
        type = "MemoryError";
    else if (dynamic_cast<const std::out_of_range *>(&error))
        type = "IndexError";
    else if (dynamic_cast<const std::overflow_error *>(&error))
        type = "OverflowError";
    else if (dynamic_cast<const std::domain_error *>(&error) || dynamic_cast<const std::invalid_argument *>(&error) ||
             dynamic_cast<const std::length_error *>(&error) || dynamic_cast<const std::range_error *>(&error))
        type = "ValueError";
    return NodeError(type, error.what());
}

/* A memo key: 128 bits of two independent 64-bit mixes over the key's parts. */
struct Key {
    uint64_t a = 0x9E3779B97F4A7C15ull, b = 0xC2B2AE3D27D4EB4Full;
    bool operator==(const Key &other) const noexcept { return a == other.a && b == other.b; }
    bool operator<(const Key &other) const noexcept { return a != other.a ? a < other.a : b < other.b; }

    static uint64_t mix(uint64_t h)
    {
        h ^= h >> 33;
        h *= 0xFF51AFD7ED558CCDull;
        h ^= h >> 33;
        h *= 0xC4CEB9FE1A85EC53ull;
        return h ^ (h >> 33);
    }
    Key &add(uint64_t value)
    {
        a = mix(a ^ (value + 0x9E3779B97F4A7C15ull + (a << 6) + (a >> 2)));
        b = mix(b + (value ^ 0xD6E8FEB86659FD93ull) * 0x2545F4914F6CDD1Dull);
        return *this;
    }
    Key &add(std::string_view text)
    {
        add(uint64_t(text.size()));
        for (size_t i = 0; i < text.size(); i += 8) {
            uint64_t word = 0;
            std::memcpy(&word, text.data() + i, std::min<size_t>(8, text.size() - i));
            add(word);
        }
        return *this;
    }
    Key &add(const Key &other) { return add(other.a).add(other.b); }
};

/* The memo cache (DESIGN.md section 2, amendment A3): in this process only, the outputs of deterministic,
 * side-effect-free nodes by key (schema, enforced constant inputs, package settings, upstream keys; a file's
 * identity for Load Image and Load Images' items), least recently used first out, within a budget apart from the item
 * window's: the Result Cache setting's, which each run sets at its start (memo_set_budget), min(2 GiB, 10 % of the RAM)
 * until then; CHAINNER_C_MEMO_BUDGET_MIB (0 = off), a diagnostic override, wins over both. An entry above a budget's
 * eighth is not kept. A dependency change restarts the worker process (server_host.change_dependencies), which drops
 * it. */
class Memo {
public:
    static Memo &get()
    {
        static Memo *memo = new Memo();
        return *memo;
    }
    struct Entry {
        std::vector<Value> outputs;
        PyRef object;  // a regular Python node's RegularOutput (an individual run returns it)
        int64_t bytes = 0;
    };
    bool enabled() const noexcept { return budget_ > 0; }
    std::optional<Entry> find(const Key &key)
    {
        std::lock_guard<std::mutex> locked(mutex_);
        auto found = entries_.find(key);
        if (found == entries_.end()) {
            ++misses_;
            return std::nullopt;
        }
        order_.splice(order_.end(), order_, found->second.position);
        ++hits_;
        return found->second.entry;
    }
    void insert(const Key &key, Entry entry)
    {
        // Even scalar/text entries retain a key, outputs and LRU bookkeeping.
        entry.bytes += int64_t(sizeof(Entry) + sizeof(Key) * 2 + sizeof(void *) * 6);
        std::vector<Entry> evicted;  // released outside the lock
        std::lock_guard<std::mutex> locked(mutex_);
        if (!enabled() || entry.bytes > budget_ / 8 || entries_.count(key)) return;
        while (bytes_ + entry.bytes > budget_ && !order_.empty()) evict(order_.begin(), evicted);
        bytes_ += entry.bytes;
        order_.push_back(key);
        entries_.emplace(key, Slot{std::move(entry), std::prev(order_.end())});
    }
    void clear()
    {
        std::map<Key, Slot> released;
        std::lock_guard<std::mutex> locked(mutex_);
        released.swap(entries_);
        order_.clear();
        bytes_ = 0;
    }
    /* The Result Cache setting's budget (0: off), unless CHAINNER_C_MEMO_BUDGET_MIB set one, which stays: the
     * entries above its eighth go, then the least recently used until the rest fits; off, every entry goes. Returns
     * the budget in effect. */
    int64_t set_budget(int64_t budget)
    {
        std::vector<Entry> evicted;  // released outside the lock
        std::lock_guard<std::mutex> locked(mutex_);
        if (overridden_) return budget_;
        budget_ = std::max<int64_t>(budget, 0);
        for (auto position = order_.begin(); position != order_.end();) {
            auto next = std::next(position);
            if (!enabled() || entries_.find(*position)->second.entry.bytes > budget_ / 8) evict(position, evicted);
            position = next;
        }
        while (bytes_ > budget_ && !order_.empty()) evict(order_.begin(), evicted);
        return budget_;
    }
    py::dict stats()
    {
        std::lock_guard<std::mutex> locked(mutex_);
        py::dict result;
        result["entries"] = entries_.size();
        result["bytes"] = bytes_;
        result["budget"] = budget_;
        result["hits"] = hits_;
        result["misses"] = misses_;
        return result;
    }

private:
    Memo()
    {
#ifdef _WIN32
        MEMORYSTATUSEX memory = {};
        memory.dwLength = sizeof(memory);
        int64_t total = GlobalMemoryStatusEx(&memory) ? int64_t(memory.ullTotalPhys) : 0;
#else
        int64_t total = int64_t(cn_physical_memory());
#endif
        budget_ = std::min<int64_t>(int64_t(2) << 30, total / 10);
#ifdef _WIN32
        char *value = nullptr;
        size_t size = 0;
        if (_dupenv_s(&value, &size, "CHAINNER_C_MEMO_BUDGET_MIB") == 0 && value) {
            std::string text(value);
            free(value);
#else
        if (const char *value = std::getenv("CHAINNER_C_MEMO_BUDGET_MIB")) {
            std::string text(value);
#endif
            if (!text.empty() && text.size() <= 9 && text.find_first_not_of("0123456789") == std::string::npos) {
                budget_ = int64_t(std::stoll(text)) << 20;
                overridden_ = true;
            }
        }
    }
    struct Slot {
        Entry entry;
        std::list<Key>::iterator position;
    };
    /* mutex_ held; the entry released by the caller, outside it. */
    void evict(std::list<Key>::iterator position, std::vector<Entry> &evicted)
    {
        auto victim = entries_.find(*position);
        bytes_ -= victim->second.entry.bytes;
        evicted.push_back(std::move(victim->second.entry));
        entries_.erase(victim);
        order_.erase(position);
    }
    std::mutex mutex_;
    std::map<Key, Slot> entries_;
    std::list<Key> order_;  // least recently used first
    int64_t budget_ = 0, bytes_ = 0;
    bool overridden_ = false;  // CHAINNER_C_MEMO_BUDGET_MIB set the budget
    uint64_t hits_ = 0, misses_ = 0;
};

struct InputPlan {
    int source = -1;  // the node feeding it, -1 for a constant
    int output = 0;
    PyRef constant;   // a constant's value, as the Python body receives it before enforce
    Value enforced;   // a native node's constant, enforced
    bool lazy = false;
    bool ignored = false;  // a collector's iterated input at construction
    bool image = false;    // ImageInput: the engine enforces images
    std::vector<int> channels;  // ImageInput.channels (empty: any)
    bool colors = false;        // ImageInput.allow_colors
    std::optional<Key> constant_key;  // a constant's memo key part (its JSON); none: unkeyable
};

struct OutputPlan {
    int64_t id = 0;
    bool image = false;  // ImageOutput or LargeImageOutput: the engine enforces and broadcasts images
    bool large = false;  // LargeImageOutput: its broadcast carries the UI's previews
    int channels = 0;    // 0: any
    bool assume_normalized = false;
};

enum class Kind { regular, generator, collector };

struct NodePlan {
    std::string id, schema, name;
    Kind kind = Kind::regular;
    bool side_effects = false, pure = false, node_context = false;
    std::vector<InputPlan> inputs;
    std::vector<OutputPlan> outputs;
    std::vector<int> iterable_outputs;
    int64_t hits = 0;  // -1: static
    const Entry *native = nullptr;  // a regular node's or a generator's native entry point
    std::map<std::string, Value, std::less<>> settings;
    bool precached = false;
    std::vector<Value> precached_outputs;
    std::string precached_identity;  // a precached Load Image's file identity as its individual run loaded it
    bool phased = false;  // registered prepare and commit phases (item_window.register_phases)
    bool not_speculated = false;  // Into Directory: never computed ahead of its item (process._NOT_SPECULATED)
    bool memo = false;    // its outputs may be memoized (amendment A3's allow-list)
    bool run_keyed = false;  // a generator whose per-run outputs follow from its inputs alone (MEMO_GENERATORS)
    int identity_input = -1;  // Load Image: its file input, whose file identity keys it
    Key settings_key;     // its package settings
};

struct GeneratorPlan {
    int node;
    std::vector<int> collectors;
};

struct GroupPlan {
    std::vector<GeneratorPlan> generators;
    std::vector<int> outputs;
    std::vector<int> iterated;
    // A group of one generator: its temporal nodes (NodeData.temporal_context), in topological order, and the context
    // items (before, after) they want, which a generator that reads a range of its source reads around it.
    std::vector<int> temporal;
    int64_t before = 0, after = 0;
};

// released: done, its value dropped after its last counted read (chain/cache.py's hits_to_live)
enum class State { idle, queued, running, done, failed, released };

struct Cell {
    State state = State::idle;
    bool wanted = false;
    std::vector<Value> outputs;
    PyRef object;  // run_node's result (a generator's GeneratorOutput), a collector's Collector
    PyRef inputs;  // a collector's inputs at its construction (the window's writer directories)
    PyRef error;
    int64_t hits = 0;
    int64_t bytes = 0;  // its outputs' array bytes (the window's per-item estimate, a memo entry's size)
    enum class KeyState : uint8_t { unknown, none, known } key_state = KeyState::unknown;
    Key key;
    // A native generator's item source, with the context and inputs it was created with.
    std::shared_ptr<ItemSource> items;
    std::shared_ptr<NodeContext> items_context;
    std::vector<Value> items_inputs;
    bool demanded = false;  // the commit path needs it (its events are sent as it runs)
    std::string identity;   // Load Image: its file's identity as it ran (memo keys)
    // A deferred job's events (see Run::defers), sent when the commit path takes it.
    bool deferred = false, deferred_ok = false;
    double deferred_seconds = 0;
    std::vector<Value> deferred_outputs;
};

/* A done cell's output; reading a released one (more reads than its counted hits) is an engine bug. */
const Value &output_of(const Cell &cell, int output)
{
    if (cell.state != State::done || size_t(output) >= cell.outputs.size())
        throw std::logic_error("The engine read an output that is not there");
    return cell.outputs[size_t(output)];
}

struct CellRef {
    int node;
    int64_t slot;  // -1: the node's run outside an item
};

/* The hooks' bound methods (native_executor._Hooks), looked up once. */
struct Hooks {
    py::object post, finish, warn, info, run_node, enforce_input, enforce_output, native_failure,
        regular_outputs, generator_outputs, collector, node_cleanups, chain_cleanups, output_bytes,
        broadcast, items, describe, materialize, item_failure, collector_iterate, collector_complete, prepare,
        begin_loop, iteration, end_loop, window, warn_look_ahead_off, native_generator, file_identity, item_identity,
        regular_output, memo_bytes, loaded, preview_url, aborted_type, waited;
};

struct Event {
    enum Type { chain_start, start, finish, progress, raw } type;
    explicit Event(Type kind, int id = -1, double seconds = 0) : type(kind), node(id), time(seconds) {}
    int node = -1;
    double time = 0, progress_value = 0, eta = 0;
    int64_t index = 0, total = 0;
    PyRef payload;  // raw: the event dict (broadcasts, warnings)
    // node-progress's Python types: int 1 for an empty sequence and when done, an int 0 ETA without item times
    bool progress_int = false, eta_int = false;
};

class Run;

/* A native node's context. */
class EngineContext final : public NodeContext {
public:
    EngineContext(const Run &run, const NodePlan &node, std::filesystem::path storage)
        : run_(run), node_(node), storage_(std::move(storage))
    {
    }
    bool aborted() const noexcept override;
    void check_aborted() const override;
    std::string_view node_id() const noexcept override { return node_.id; }
    const std::filesystem::path &storage_dir() const noexcept override { return storage_; }
    const Value *setting(std::string_view key) const noexcept override
    {
        auto found = node_.settings.find(key);
        return found == node_.settings.end() ? nullptr : &found->second;
    }

private:
    const Run &run_;
    const NodePlan &node_;
    std::filesystem::path storage_;
};

/* An upstream failure met by a native node through a lazy input: the upstream's error, unchanged. */
struct UpstreamFailed {
    PyRef error;
};

class EngineLazy;

/* The run whose job (a service, a worker's job) the thread runs: close() never waits for its own thread. */
thread_local const Run *serving_run = nullptr;

class Serving {
public:
    explicit Serving(const Run *run) : previous_(serving_run) { serving_run = run; }
    ~Serving() { serving_run = previous_; }
    Serving(const Serving &) = delete;
    Serving &operator=(const Serving &) = delete;

private:
    const Run *previous_;
};

/* Python state for an engine thread, made once for its life: a thread state and NumPy's data-memory handler
 * (nodes/impl/numpy_pool.py), which a thread does not inherit; released when a service thread ends (not once the
 * interpreter finalizes, when no thread may take the GIL). */
class PythonThread {
public:
    PythonThread() : state_(PyGILState_Ensure())
    {
        try {
            py::module_::import("nodes.impl.numpy_pool").attr("install")();
        } catch (py::error_already_set &error) {
            error.discard_as_unraisable("chaiNNer-C engine thread setup");
        }
        thread_ = PyEval_SaveThread();
    }
    ~PythonThread()
    {
        if (Py_IsFinalizing()) return;
        PyEval_RestoreThread(thread_);
        PyGILState_Release(state_);
    }
    PythonThread(const PythonThread &) = delete;
    PythonThread &operator=(const PythonThread &) = delete;

private:
    PyGILState_STATE state_;
    PyThreadState *thread_ = nullptr;
};

/* The process's engine threads, shared by every run (and preview): workers, as many as the CPUs of the affinity
 * mask (cn_parallel_for's capacity: CHAINNER_C_ENGINE_THREADS overrides), that run the queued jobs of every run;
 * and service threads, reused for each run's driver, event pump, broadcaster and window producer, which end after
 * a while idle (a burst of previews leaves none behind).
 * A run offers one token per job it queues; a worker takes a token, then a job from that run (none when it was
 * taken, dropped or paused meanwhile). The engine's lock is never held while taking a run's. */
class Engine {
public:
    static Engine &get()
    {
        static Engine *engine = new Engine();  // never destroyed: its threads live as long as the process
        return *engine;
    }
    void attach(Run *run);
    void offer(Run *run, size_t jobs = 1);
    /* Drops the run's tokens and waits until no worker is inside it. */
    void detach(Run *run);
    std::future<void> service(std::function<void()> task);
    int workers() const noexcept { return workers_; }

private:
    Engine();
    void work();
    void serve();

    int workers_ = 1;
    std::chrono::seconds idle_time_{30};  // a service thread's idle time before it ends (CHAINNER_C_ENGINE_IDLE_SECONDS)
    std::mutex mutex_;
    std::condition_variable work_, idle_, services_changed_;
    std::deque<Run *> tokens_;
    std::unordered_map<Run *, int> active_;
    std::deque<std::packaged_task<void()>> services_;
    int idle_services_ = 0;
};

struct GeneratorRun {
    int node = -1;
    PyRef output, iterator;
    std::shared_ptr<ItemSource> native;  // a native generator's items (describe serial, materialize anywhere)
    std::vector<Value> partial;  // its GeneratorOutput's partial_output (its static outputs)
    bool split = false, fail_fast = true;
    int64_t expected = 0;
    Clock::time_point loop_start, last;
    double paused_start = 0, last_paused = 0;
    std::vector<double> times;
};

struct CollectorRun {
    int node;
    PyRef collector, inputs;
    double seconds = 0;
};

/* A generator's advance at a slot of the item window. */
struct Outcome {
    enum Kind { pending, ready, ended, failed } kind = pending;
    PyRef error;
};

/* A sink's prepare phase at a slot (process.ItemWindow's prepares). */
struct Prepare {
    enum Kind { waiting, running, done } kind = waiting;
    PyRef value;  // empty: nothing prepared (the node runs whole)
};

/* Window work for the workers besides cells: a single generator's materialize, a sink's prepare. */
struct Task {
    enum Kind { materialize, prepare } kind;
    int64_t slot;
    int index;  // materialize: the generator's position in its group; prepare: the sink
    PyRef token;      // a Python generator's item token
    int64_t item = 0;  // a native generator's item token
    std::optional<Key> key;  // the item's memo key (a split iterator's file identity)
};

/* One generator group's loop and its item window (process.Executor's _LoopRun and ItemWindow). */
struct Loop {
    const GroupPlan *group = nullptr;
    std::vector<GeneratorRun> *generators = nullptr;
    std::vector<CollectorRun> *collectors = nullptr;
    std::vector<int> sinks;  // the group's side-effect nodes, then its collectors
    bool decided = false, active = false, halted = false, stop = false;
    bool fail_fast = true;  // every generator's errors end the loop (serial nodes may then run ahead)
    // Context items (process._LoopRun): the slots before `first` and, with items after the range, from `end` on (-1:
    // none; the commit path never begins them). There only the group's temporal nodes and the nodes they need run.
    int64_t first = 0, end = -1;
    int k = 1;
    int64_t budget = 0, estimate = 0, committed = 0;
    std::optional<std::string> blocker;
    std::set<int> warned;
    int64_t produced = 0;  // the next slot the producer starts
    int64_t demand = -1;   // the furthest slot a look-ahead waits for (admitted beyond the budget)
    int materializing = 0;
    std::map<int64_t, std::vector<Outcome>> outcomes;
    std::map<std::pair<int, int64_t>, Prepare> prepares;
    std::map<int, bool> ahead;
    std::set<int> serial;  // nodes computed ahead that are serial or depend on a serial one
    std::future<void> producer;

    int generator_index(int node) const
    {
        for (size_t g = 0; g < generators->size(); ++g)
            if ((*generators)[g].node == node) return int(g);
        return -1;
    }
    bool context(int64_t slot) const { return slot < first || (end >= 0 && slot >= end); }
    PyRef collector_of(int node) const
    {
        for (const CollectorRun &collector : *collectors)
            if (collector.node == node) return collector.collector;
        return {};
    }
};

/* The outcome of a native entry point's call. python: the node's Python body runs instead (a decline, or an input
 * whose enforce fails, which the body reports as run_node does). */
enum class NativeOutcome { done, failed, declined, python };

class Run {
public:
    Run(const py::dict &plan, py::object hooks_object);
    ~Run();

    void start(const std::string &mode, int target);
    void kill();
    void pause();
    void resume();
    bool aborted() const noexcept { return aborted_.load(); }
    bool paused() const noexcept { return paused_.load(); }
    int64_t commit_slot() const noexcept { return commit_slot_.load(); }
    void flush_broadcasts();
    void close();
    int lookahead(int node, int input);
    py::object peek(int node, int input, int64_t offset, int64_t origin);

    /* A lazy input's value (on the reading thread): its upstream evaluated (nested), the read consumed. */
    void read_lazy(EngineLazy &lazy);

private:
    friend class EngineContext;
    friend class EngineLazy;

    // Plan
    std::vector<NodePlan> nodes_;
    std::vector<GroupPlan> groups_;
    std::vector<int> tail_;
    std::vector<std::string> chain_start_;
    bool send_broadcast_data_ = false;
    std::filesystem::path storage_;
    Hooks hooks_;
    PyRef hooks_object_;

    // Cells, under mutex_
    std::mutex mutex_;
    std::condition_variable changed_;
    std::vector<Cell> static_cells_;
    std::map<int64_t, std::vector<Cell>> slot_cells_;
    std::vector<char> iterated_;
    std::vector<CellRef> wanted_;
    std::deque<CellRef> ready_;
    std::deque<Task> tasks_;
    std::unique_ptr<Loop> loop_;  // the running group's loop (set and reset by the driver under mutex_)
    int running_ = 0, running_slots_ = 0;
    bool dropping_ = false;  // the run failed: queued pure cells are not started
    bool slots_closing_ = false;  // the loop's end drains its slots: no slot work is queued any more

    // Run state
    std::atomic<bool> aborted_{false}, paused_{false};
    std::atomic<int64_t> commit_slot_{0};
    double time_paused_ = 0;
    std::string mode_;
    int target_ = -1;
    std::future<void> driver_;
    bool started_ = false, closed_ = false;
    // Cleared at close: a lazy input a Python node kept and reads later no longer reaches the run.
    std::shared_ptr<std::atomic<bool>> open_ = std::make_shared<std::atomic<bool>>(true);
    std::string fault_;  // an engine thread's fault (under mutex_): the run's error

    // Events
    std::mutex events_mutex_;
    std::condition_variable events_changed_;
    std::vector<Event> events_;
    bool events_stop_ = false, pumping_ = false, events_failed_ = false;
    std::future<void> pump_;

    // Broadcasts (LatestBroadcasts): one pending job per node, sent one at a time in submission order; a loop's
    // LargeImage previews (limited) at most one start per preview_interval_ and node, the newest kept meanwhile
    struct BroadcastJob {
        int node;
        std::vector<Value> outputs;
        bool limited = false;
    };
    std::mutex broadcasts_mutex_;
    std::condition_variable broadcasts_changed_;
    std::deque<int> broadcast_order_;
    std::unordered_map<int, BroadcastJob> broadcast_jobs_;
    std::unordered_map<int, Clock::time_point> preview_started_;
    Clock::duration preview_interval_{};  // 0: no limit
    int flushing_ = 0;  // a flush sends the limited previews at once
    bool broadcasting_ = false, broadcasts_stop_ = false, broadcasts_failed_ = false;
    std::future<void> broadcaster_;

    Cell &cell(const CellRef &ref);
    CellRef source_of(const CellRef &ref, int input) const;
    bool settled(const Cell &cell) const
    {
        return cell.state == State::done || cell.state == State::failed || cell.state == State::released;
    }

    // Driver
    void drive();
    void suspend();
    void ensure(const CellRef &ref, bool commit = true);
    bool parallel(const CellRef &ref, std::optional<std::map<std::pair<int, int64_t>, bool>> &known);
    void demand(const CellRef &ref);
    void want(const CellRef &ref);
    void prefetch(const CellRef &ref, std::set<std::pair<int, int64_t>> &seen);
    void dispatch();
    void drain_slots();
    PyRef process_regular(int node, int64_t slot);
    PyRef process_generator(int node);
    void iterate_group(const GroupPlan &group);
    void open_slot(int64_t slot);
    bool admits(const Loop &loop) const;
    void produce();
    void set_item(int node, int64_t slot, std::vector<Value> outputs, std::optional<Key> key = std::nullopt);
    void want_window(int64_t slot);
    void run_task(const Task &task);
    PyRef take_prepared(int node, int64_t slot);
    void decide(int64_t start);
    int64_t item_bytes(const GeneratorRun &generator, const std::vector<Value> &outputs);
    bool measured(const CellRef &ref) const;
    bool discarded(const CellRef &ref) const;
    Outcome take_advance(int64_t slot, size_t generator);
    bool ahead(int node);
    bool servable(int node, int input);
    void decide_early();
    void take(const CellRef &ref);
    void release_slot(int64_t slot);
    void drain_workers();
    void fault(std::exception_ptr error) noexcept;

    // Execution
    friend class Engine;
    void work_one();
    void queue(const CellRef &ref);
    void queue(Task task);
    void execute(const CellRef &ref, bool on_driver);
    void execute_cell(const CellRef &ref, bool on_driver);
    int64_t memo_bytes(const std::vector<Value> &outputs);
    struct Gathered {
        std::vector<Value> values;               // inputs as the node receives them (lazy: an EngineLazy)
        std::vector<std::shared_ptr<EngineLazy>> lazies;
        bool defer = false;                      // a window job: its events wait for the commit path
    };
    bool defers(const CellRef &ref, bool on_driver);
    Gathered gather(const CellRef &ref, bool on_driver);
    bool run_python(const CellRef &ref, const Gathered &inputs, Cell &result, double &other_seconds,
                    const PyRef &prepared, bool ahead);
    NativeOutcome run_native(const CellRef &ref, const Gathered &inputs, Cell &result, double &other_seconds);
    std::optional<NodeError> enforce_outputs(int node, std::vector<Value> &outputs, const std::vector<size_t> &which);
    std::optional<Key> key_of(const CellRef &ref);
    void identify_file(const CellRef &ref);
    std::optional<Key> file_key(const NodePlan &node, const std::string &identity) const;
    std::optional<Key> item_key(const GeneratorRun &generator, const py::object &token);
    std::vector<Value> fill_item(const GeneratorRun &generator, std::vector<Value> item);
    std::vector<Value> native_item(const GeneratorRun &generator, int64_t token);
    PythonError item_failure(const GeneratorRun &generator, const PyRef &error);
    py::list python_inputs(const CellRef &ref, const Gathered &inputs);
    void store(const CellRef &ref, Cell &&result);
    void consume(const CellRef &source);
    /* A native node's edge input enforced as the input's enforce does: images (ImageInput) here, anything else by
     * the hooks (GIL); an enforce error is the node's NodeError. */
    Value enforce_input(int node, int input, const Value &value);

    // Events and broadcasts
    void emit(Event event);
    void emit_start(int node) { emit(Event(Event::start, node)); }
    void emit_finish(int node, double time) { emit(Event(Event::finish, node, time)); }
    void emit_progress(int node, const std::vector<double> &times, int64_t index, int64_t total);
    void pump();
    void flush_events();
    py::object event_dict(const Event &event);
    void submit_broadcast(int node, const std::vector<Value> &outputs);
    void broadcast_now(int node, const std::vector<Value> &outputs, const PyRef &generator);
    void broadcaster();
};

std::unordered_map<std::string, int64_t> &decline_counts()
{
    static std::unordered_map<std::string, int64_t> counts;
    return counts;
}
std::mutex decline_mutex;

/* A lazy input of a node the engine runs, read on the node's thread (the driver: a node with a lazy input is never
 * pure), its upstream evaluated nested in the read. A Python reader receives the upstream value (run_node enforces
 * it, as process.enforce_inputs wraps lazy inputs); a native reader receives it enforced. */
class EngineLazy final : public LazyValue {
public:
    EngineLazy(Run &run, CellRef reader, int input, CellRef source, int output)
        : run_(run), open_(run.open_), reader_(reader), input_(input), source_(source), output_(output)
    {
    }

    const Value &value() override
    {
        const Value &upstream = raw();
        if (!enforced_) {
            try {
                value_ = run_.enforce_input(reader_.node, input_, upstream);
            } catch (const NodeError &) {
                // As an edge input whose enforce fails (run_native): the body enforces it itself, and runs or
                // raises as it does today.
                throw RunPython{};
            }
            enforced_ = true;
        }
        return value_;
    }

    /* The upstream value for a Python reader; Stop raises progress_controller's Aborted, an upstream failure its
     * error. */
    py::object python_value()
    {
        if (!*open_ && (!read_ || error_)) {
            PyErr_SetString(PyExc_RuntimeError, "A lazy input was read after its run ended");
            throw py::error_already_set();
        }
        const Value *upstream = nullptr;
        bool stopped = false;
        PyRef failed;
        {
            PythonPause unlocked;
            try {
                upstream = &raw();
            } catch (const UpstreamFailed &failure) {
                failed = failure.error;
            } catch (const StopRun &) {
                stopped = true;
            }
        }
        if (stopped) {
            PyErr_SetNone(run_.hooks_.aborted_type.ptr());
            throw py::error_already_set();
        }
        if (failed.handle) {
            PyObject *error = static_cast<PyObject *>(failed.handle.get());
            PyErr_SetObject(reinterpret_cast<PyObject *>(Py_TYPE(error)), error);
            throw py::error_already_set();
        }
        return to_python(*upstream);
    }

    double seconds() const noexcept { return seconds_; }

private:
    friend class Run;

    /* The upstream value; throws the upstream's failure (UpstreamFailed) or StopRun. */
    const Value &raw()
    {
        if (!read_) {
            read_ = true;
            run_.read_lazy(*this);
        }
        if (upstream_.handle) throw UpstreamFailed{upstream_};
        if (error_) std::rethrow_exception(error_);
        return raw_;
    }

    Run &run_;
    std::shared_ptr<std::atomic<bool>> open_;
    CellRef reader_;
    int input_;
    CellRef source_;
    int output_;
    bool read_ = false, enforced_ = false;
    Value raw_, value_;
    PyRef upstream_;
    std::exception_ptr error_;
    double seconds_ = 0;
};

/* A constant on a native node's lazy input: never hops (process.enforce_inputs' Lazy.ready). */
class ReadyLazy final : public LazyValue {
public:
    explicit ReadyLazy(Value value) : value_(std::move(value)) {}
    const Value &value() override { return value_; }

private:
    Value value_;
};

bool EngineContext::aborted() const noexcept { return run_.aborted(); }
void EngineContext::check_aborted() const
{
    if (run_.aborted()) throw Aborted{};
}

Value value_of(py::handle object)
{
    return to_value(object);
}

Run::Run(const py::dict &plan, py::object hooks_object) : hooks_object_(py_ref(hooks_object))
{
    auto bind = [&](const char *name) { return hooks_object.attr(name); };
    hooks_ = Hooks{bind("post"), bind("finish"), bind("warn"), bind("info"), bind("run_node"),
                   bind("enforce_input"), bind("enforce_output"), bind("native_failure"),
                   bind("regular_outputs"), bind("generator_outputs"), bind("collector"), bind("node_cleanups"),
                   bind("chain_cleanups"), bind("output_bytes"), bind("broadcast"), bind("items"), bind("describe"),
                   bind("materialize"), bind("item_failure"), bind("collector_iterate"), bind("collector_complete"),
                   bind("prepare"), bind("begin_loop"), bind("iteration"), bind("end_loop"), bind("window"),
                   bind("warn_look_ahead_off"), bind("native_generator"), bind("file_identity"),
                   bind("item_identity"), bind("regular_output"), bind("memo_bytes"), bind("loaded"), bind("preview_url"),
                   bind("aborted_type"), bind("waited")};
    send_broadcast_data_ = plan["send_broadcast_data"].cast<bool>();
    preview_interval_ = std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(plan["preview_interval"].cast<double>()));
    for (py::handle id : plan["chain_start"]) chain_start_.push_back(id.cast<std::string>());
    for (py::handle node_handle : plan["nodes"]) {
        py::dict node = py::reinterpret_borrow<py::dict>(node_handle);
        NodePlan plan_node;
        plan_node.id = node["id"].cast<std::string>();
        plan_node.schema = node["schema"].cast<std::string>();
        plan_node.name = node["name"].cast<std::string>();
        plan_node.kind = Kind(node["kind"].cast<int>());
        plan_node.side_effects = node["side_effects"].cast<bool>();
        plan_node.pure = node["pure"].cast<bool>();
        plan_node.node_context = node["node_context"].cast<bool>();
        plan_node.hits = node["hits"].cast<int64_t>();
        if (node["native"].cast<bool>() && plan_node.kind != Kind::collector) {
            const Entry *entry = find_entry(plan_node.schema);
            if (entry && (plan_node.kind == Kind::regular ? entry->run != nullptr : entry->generator != nullptr))
                plan_node.native = entry;
        }
        for (py::handle input_handle : node["inputs"]) {
            py::tuple input = py::reinterpret_borrow<py::tuple>(input_handle);
            InputPlan plan_input;
            plan_input.source = input[0].cast<int>();
            plan_input.output = input[1].cast<int>();
            plan_input.constant = py_ref(input[2]);
            plan_input.lazy = input[3].cast<bool>();
            plan_input.ignored = input[4].cast<bool>();
            py::object image = input[5];
            if (!image.is_none()) {
                py::tuple description = py::reinterpret_borrow<py::tuple>(image);
                plan_input.image = true;
                if (!description[0].is_none())
                    for (py::handle channel : description[0]) plan_input.channels.push_back(channel.cast<int>());
                plan_input.colors = description[1].cast<bool>();
            }
            if (plan_node.native && plan_input.source < 0) plan_input.enforced = value_of(input[6]);
            if (plan_input.source < 0 && !input[7].is_none()) plan_input.constant_key = Key().add(input[7].cast<std::string>());
            plan_node.inputs.push_back(std::move(plan_input));
        }
        py::list output_ids = node["output_ids"];
        size_t position = 0;
        for (py::handle output_handle : node["image_outputs"]) {
            OutputPlan output;
            output.id = output_ids[position++].cast<int64_t>();
            if (!output_handle.is_none()) {
                py::tuple description = py::reinterpret_borrow<py::tuple>(output_handle);
                output.image = true;
                output.channels = description[0].cast<int>();
                output.assume_normalized = description[1].cast<bool>();
                output.large = description[2].cast<bool>();
            }
            plan_node.outputs.push_back(output);
        }
        for (py::handle index : node["iterable_outputs"]) plan_node.iterable_outputs.push_back(index.cast<int>());
        if (plan_node.native) {
            for (auto [key, value] : py::reinterpret_borrow<py::dict>(node["settings"]))
                plan_node.settings.emplace(py::str(key), value_of(value));
        }
        plan_node.phased = node["phased"].cast<bool>();
        plan_node.not_speculated = plan_node.schema == "chainner:utility:into_directory";
        plan_node.memo = node["memo"].cast<bool>() && Memo::get().enabled();
        plan_node.run_keyed = node["run_keyed"].cast<bool>();
        plan_node.identity_input = node["identity_input"].cast<int>();
        plan_node.settings_key = Key().add(node["settings_key"].cast<std::string>());
        py::object precached = node["precached"];
        if (!precached.is_none()) {
            plan_node.precached = true;
            for (py::handle value : precached) plan_node.precached_outputs.push_back(py_ref(value));
            py::object identity = node["precached_identity"];
            if (!identity.is_none()) plan_node.precached_identity = identity.cast<std::string>();
        }
        nodes_.push_back(std::move(plan_node));
    }
    for (py::handle group_handle : plan["groups"]) {
        py::dict group = py::reinterpret_borrow<py::dict>(group_handle);
        GroupPlan plan_group;
        for (py::handle generator_handle : group["generators"]) {
            py::dict generator = py::reinterpret_borrow<py::dict>(generator_handle);
            GeneratorPlan plan_generator{generator["node"].cast<int>(), {}};
            for (py::handle collector : generator["collectors"]) plan_generator.collectors.push_back(collector.cast<int>());
            plan_group.generators.push_back(std::move(plan_generator));
        }
        for (py::handle output : group["outputs"]) plan_group.outputs.push_back(output.cast<int>());
        for (py::handle node : group["iterated"]) plan_group.iterated.push_back(node.cast<int>());
        for (py::handle node : group["temporal"]) plan_group.temporal.push_back(node.cast<int>());
        py::tuple context = group["context"];
        plan_group.before = context[0].cast<int64_t>();
        plan_group.after = context[1].cast<int64_t>();
        groups_.push_back(std::move(plan_group));
    }
    for (py::handle node : plan["tail"]) tail_.push_back(node.cast<int>());
    storage_ = std::get<Path>(value_of(plan["storage_dir"])).value;
    static_cells_.resize(nodes_.size());
    iterated_.assign(nodes_.size(), 0);
    for (size_t i = 0; i < nodes_.size(); ++i) {
        Cell &c = static_cells_[i];
        c.hits = nodes_[i].hits;
        if (nodes_[i].precached) {
            c.state = State::done;
            c.outputs = nodes_[i].precached_outputs;
            c.hits = -1;
            if (nodes_[i].memo && !nodes_[i].precached_identity.empty()) {
                std::optional<Key> key = file_key(nodes_[i], nodes_[i].precached_identity);
                c.key_state = key ? Cell::KeyState::known : Cell::KeyState::none;
                if (key) c.key = *key;
            }
        }
    }
}

Run::~Run()
{
    try {
        if (PyGILState_Check()) {
            py::gil_scoped_release unlocked;
            close();
        } else {
            close();
        }
    } catch (...) {
    }
}

/* A cell (mutex_ held); an item's cell only while its slot is open (open_slot makes it). */
Cell &Run::cell(const CellRef &ref)
{
    if (ref.slot >= 0 && iterated_[size_t(ref.node)]) {
        auto found = slot_cells_.find(ref.slot);
        if (found == slot_cells_.end()) throw std::logic_error("The engine reached a cell of a slot it closed");
        return found->second[size_t(ref.node)];
    }
    return static_cells_[size_t(ref.node)];
}

CellRef Run::source_of(const CellRef &ref, int input) const
{
    int source = nodes_[size_t(ref.node)].inputs[size_t(input)].source;
    return {source, ref.slot >= 0 && iterated_[size_t(source)] ? ref.slot : -1};
}

// Threads

void Run::start(const std::string &mode, int target)
{
    if (started_ || closed_) throw std::logic_error("A RunEngine starts once, before it closes");
    started_ = true;
    mode_ = mode;
    target_ = target;
    py::gil_scoped_release unlocked;
    Engine &engine = Engine::get();
    engine.attach(this);
    // A failed pump or broadcaster stops the run with its fault; their flushes no longer wait for them.
    pump_ = engine.service([this] {
        Serving serving(this);
        try {
            pump();
        } catch (...) {
            fault(std::current_exception());
            std::lock_guard<std::mutex> locked(events_mutex_);
            events_failed_ = true;
            events_changed_.notify_all();
        }
    });
    broadcaster_ = engine.service([this] {
        Serving serving(this);
        try {
            broadcaster();
        } catch (...) {
            fault(std::current_exception());
            std::lock_guard<std::mutex> locked(broadcasts_mutex_);
            broadcasts_failed_ = true;
            broadcasts_changed_.notify_all();
        }
    });
    driver_ = engine.service([this] {
        Serving serving(this);
        drive();
    });
}

void Run::kill()
{
    aborted_ = true;
    {
        // Workers drop a stopped run's queued jobs, also those a pause left without a worker.
        std::lock_guard<std::mutex> locked(mutex_);
        if (started_ && !closed_) Engine::get().offer(this, ready_.size() + tasks_.size());
    }
    changed_.notify_all();
}

void Run::pause() { paused_ = true; }

void Run::resume()
{
    paused_ = false;
    {
        // Workers drop a paused run's tokens: its queued jobs are offered again.
        std::lock_guard<std::mutex> locked(mutex_);
        if (started_ && !closed_) Engine::get().offer(this, ready_.size() + tasks_.size());
    }
    changed_.notify_all();
}

/* Joins the run's threads, then drops every Python reference it holds (the hooks, cells, plan constants, broadcasts
 * never sent): references the garbage collector cannot see, which would keep the executor and its outputs alive. */
void Run::close()
{
    if (closed_) return;
    // A run that never started (stopped while it waited for a stopped run's node) has no thread to join, but holds
    // its plan's and hooks' references all the same: the hooks reach back to it, a cycle the collector cannot see.
    if (started_ && serving_run == this) throw std::logic_error("A run cannot close on one of its own threads");
    closed_ = true;
    *open_ = false;
    if (started_) {
        if (driver_.valid()) driver_.wait();
        flush_broadcasts();
        Engine::get().detach(this);
        {
            std::lock_guard<std::mutex> locked(broadcasts_mutex_);
            broadcasts_stop_ = true;
        }
        broadcasts_changed_.notify_all();
        if (broadcaster_.valid()) broadcaster_.wait();
        {
            std::lock_guard<std::mutex> locked(events_mutex_);
            events_stop_ = true;
        }
        events_changed_.notify_all();
        if (pump_.valid()) pump_.wait();
    }
    py::gil_scoped_acquire python;
    hooks_ = Hooks{};
    hooks_object_ = PyRef{};
    for (NodePlan &node : nodes_) {
        for (InputPlan &input : node.inputs) {
            input.constant = PyRef{};
            input.enforced = Value{};
        }
        node.settings.clear();
        node.precached_outputs.clear();
    }
    static_cells_.clear();
    slot_cells_.clear();
    broadcast_jobs_.clear();
    broadcast_order_.clear();
    events_.clear();
    release_deferred();
}

/* An exception escaping an engine thread's job (an engine fault, an allocation outside any node): the run stops
 * and reports it as its error, never a crash or a hang. */
void Run::fault(std::exception_ptr error) noexcept
{
    try {
        std::string text = "The native run engine failed";
        try {
            std::rethrow_exception(error);
        } catch (const std::exception &raised) {
            text += std::string(": ") + raised.what();
        } catch (...) {
            text += " (an exception of an unknown type)";
        }
        {
            std::lock_guard<std::mutex> locked(mutex_);
            if (fault_.empty()) fault_ = std::move(text);
        }
        kill();
    } catch (...) {
        // Out of memory even for the report: the run still stops (its driver then reports a Stop).
        aborted_ = true;
        changed_.notify_all();
    }
}

/* One job of the run for a worker of the engine: a window task first, else the oldest queued cell. A paused run's
 * jobs wait for resume(), which offers them again; a stopped run's are dropped. */
void Run::work_one()
{
    Serving serving(this);
    std::optional<Task> task;
    CellRef ref{};
    {
        std::unique_lock<std::mutex> locked(mutex_);
        if (paused_ && !aborted_) return;
        if (!tasks_.empty()) {
            task = std::move(tasks_.front());
            tasks_.pop_front();
            if (aborted_) {
                if (task->kind == Task::materialize) {
                    --loop_->materializing;
                } else if (auto found = loop_->prepares.find({task->index, task->slot}); found != loop_->prepares.end()) {
                    found->second.kind = Prepare::done;  // nothing prepared
                }
                locked.unlock();
                changed_.notify_all();
                return;
            }
            ++running_slots_;
        } else {
            if (ready_.empty()) return;
            ref = ready_.front();
            ready_.pop_front();
            Cell &c = cell(ref);
            if (aborted_ || dropping_) {
                c.state = State::idle;
                c.wanted = false;
                locked.unlock();
                changed_.notify_all();
                return;
            }
            c.state = State::running;
            if (ref.slot >= 0) ++running_slots_;
        }
        ++running_;
    }
    bool faulted = false;
    try {
        if (task)
            run_task(*task);
        else
            execute_cell(ref, false);
    } catch (...) {
        faulted = true;
        fault(std::current_exception());
    }
    {
        std::lock_guard<std::mutex> locked(mutex_);
        --running_;
        if (task || ref.slot >= 0) --running_slots_;
        // A faulted cell is left to the stopped driver, which no longer waits for it.
        if (faulted && !task && cell(ref).state == State::running) cell(ref).state = State::idle;
        dispatch();
    }
    changed_.notify_all();
}

/* A ready cell or a window task queued for the engine's workers (mutex_ held). */
void Run::queue(const CellRef &ref)
{
    ready_.push_back(ref);
    Engine::get().offer(this);
}

void Run::queue(Task task)
{
    tasks_.push_back(std::move(task));
    Engine::get().offer(this);
}

// Driver

void Run::suspend()
{
    if (aborted_) throw StopRun{};
    if (!paused_) return;
    Clock::time_point start = Clock::now();
    while (paused_ && !aborted_) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    time_paused_ += seconds_since(start);
    if (aborted_) throw StopRun{};
}

void Run::want(const CellRef &ref)
{
    Cell &c = cell(ref);
    if (c.state != State::idle || c.wanted) return;
    const NodePlan &node = nodes_[size_t(ref.node)];
    if (!node.pure || (node.not_speculated && ref.slot > commit_slot_)) return;
    c.wanted = true;
    for (size_t i = 0; i < node.inputs.size(); ++i) {
        const InputPlan &input = node.inputs[i];
        if (input.source >= 0 && !input.lazy && !input.ignored) want(source_of(ref, int(i)));
    }
    wanted_.push_back(ref);
}

/* Wants every pure node a sink needs (through the driver's nodes too), so that it starts as soon as its inputs are
 * done, beside the driver's work and other sinks' (A6). */
void Run::prefetch(const CellRef &ref, std::set<std::pair<int, int64_t>> &seen)
{
    if (!seen.insert({ref.node, ref.slot}).second) return;
    const Cell &c = cell(ref);
    if (c.state != State::idle) return;
    const NodePlan &node = nodes_[size_t(ref.node)];
    if (node.pure) {
        want(ref);
        return;
    }
    for (size_t i = 0; i < node.inputs.size(); ++i) {
        const InputPlan &input = node.inputs[i];
        if (input.source >= 0 && !input.lazy && !input.ignored) prefetch(source_of(ref, int(i)), seen);
    }
}

void Run::dispatch()
{
    bool queued = false;
    for (size_t w = 0; w < wanted_.size();) {
        const CellRef ref = wanted_[w];
        Cell &c = cell(ref);
        if (c.state != State::idle || !c.wanted) {
            wanted_[w] = wanted_.back();
            wanted_.pop_back();
            continue;
        }
        const NodePlan &node = nodes_[size_t(ref.node)];
        bool ready = true;
        for (size_t i = 0; i < node.inputs.size() && ready; ++i) {
            const InputPlan &input = node.inputs[i];
            if (input.source < 0 || input.lazy || input.ignored) continue;
            ready = cell(source_of(ref, int(i))).state == State::done;
        }
        if (ready && !aborted_ && !dropping_ && !(ref.slot >= 0 && slots_closing_)) {
            c.state = State::queued;
            queue(ref);
            queued = true;
            wanted_[w] = wanted_.back();
            wanted_.pop_back();
            continue;
        }
        ++w;
    }
    if (loop_ && loop_->active && !aborted_ && !dropping_ && !slots_closing_) {
        for (auto &[key, prepare] : loop_->prepares) {
            if (prepare.kind != Prepare::waiting) continue;
            const auto [sink, slot] = key;
            if (slot_cells_.find(slot) == slot_cells_.end()) continue;
            const NodePlan &node = nodes_[size_t(sink)];
            bool ready = true;
            for (size_t i = 0; i < node.inputs.size() && ready; ++i) {
                const InputPlan &input = node.inputs[i];
                if (input.source < 0 || (node.kind == Kind::collector && !input.ignored)) continue;
                ready = cell(source_of({sink, slot}, int(i))).state == State::done;
            }
            if (!ready) continue;
            prepare.kind = Prepare::running;
            queue(Task{Task::prepare, slot, sink, {}, 0, std::nullopt});
            queued = true;
        }
    }
    if (queued) changed_.notify_all();
}

/* The cell computed (or failed), its inputs first. commit: on the commit path, which takes what ran before it
 * (replaying deferred events); a look-ahead's computation ahead takes nothing. */
void Run::ensure(const CellRef &ref, bool commit)
{
    bool done;
    {
        std::lock_guard<std::mutex> locked(mutex_);
        Cell &c = cell(ref);
        if (commit) c.demanded = true;
        done = settled(c);
    }
    if (done) {
        if (commit) take(ref);
        return;
    }
    const NodePlan &node = nodes_[size_t(ref.node)];
    // A pure node runs on the workers, beside the driver's work; an individual run's node on the driver (no hop).
    bool workers = node.pure && mode_ == "run";
    if (workers) {
        std::lock_guard<std::mutex> locked(mutex_);
        want(ref);
        dispatch();
    }
    // process.Executor.__gather_inputs: up to four consecutive inputs whose upstream is pure CPU (or computed) are
    // gathered together: each started and drained, the first error in input order reported; any other input (a
    // constant, a lazy one included) ends the batch and is read alone, in order.
    std::vector<CellRef> batch;
    PyRef failed;
    auto read = [&](const CellRef &source) {
        ensure(source, commit);
        std::lock_guard<std::mutex> locked(mutex_);
        const Cell &from = cell(source);
        if (from.state == State::failed && !failed.handle) failed = from.error;
    };
    auto gather = [&] {
        if (commit) {
            std::lock_guard<std::mutex> locked(mutex_);
            for (const CellRef &source : batch) demand(source);
        }
        for (const CellRef &source : batch) read(source);
        batch.clear();
        return !failed.handle;
    };
    for (size_t i = 0; i < node.inputs.size() && !failed.handle; ++i) {
        const InputPlan &input = node.inputs[i];
        bool edge = input.source >= 0 && !input.lazy && !input.ignored;
        CellRef source = edge ? source_of(ref, int(i)) : CellRef{-1, -1};
        bool together;
        {
            std::lock_guard<std::mutex> locked(mutex_);
            std::optional<std::map<std::pair<int, int64_t>, bool>> known;
            together = edge && parallel(source, known);
        }
        if (together) {
            batch.push_back(source);
            if (batch.size() < 4) continue;
        }
        if (gather() && edge && !together) read(source);
    }
    if (!failed.handle) gather();
    if (failed.handle) {
        {
            std::lock_guard<std::mutex> locked(mutex_);
            Cell &c = cell(ref);
            if (c.state == State::idle) {
                c.state = State::failed;
                c.error = failed;
                c.wanted = false;
            }
        }
        changed_.notify_all();
        return;
    }
    if (workers) {
        std::unique_lock<std::mutex> locked(mutex_);
        dispatch();
        changed_.wait(locked, [&] {
            const Cell &c = cell(ref);
            return settled(c) || (aborted_ && c.state != State::running);
        });
        Cell &c = cell(ref);
        if (!settled(c)) throw StopRun{};
        locked.unlock();
        if (commit) take(ref);
        return;
    }
    {
        std::lock_guard<std::mutex> locked(mutex_);
        cell(ref).state = State::running;
    }
    execute_cell(ref, true);
}

/* process's pure_cpu_dependency: a cell computed, or pure with every input such a cell (mutex_ held). */
bool Run::parallel(const CellRef &ref, std::optional<std::map<std::pair<int, int64_t>, bool>> &known)
{
    const Cell &c = cell(ref);
    if (c.state == State::done || c.state == State::released) return true;
    const NodePlan &node = nodes_[size_t(ref.node)];
    if (!node.pure) return false;
    if (!known) known.emplace();
    auto [entry, inserted] = known->try_emplace(std::pair{ref.node, ref.slot}, false);
    if (!inserted) return entry->second;
    for (size_t i = 0; i < node.inputs.size(); ++i)
        if (node.inputs[i].source >= 0 && !parallel(source_of(ref, int(i)), known)) return false;
    entry->second = true;
    return true;
}

/* A gathered input's upstream, started together as Python's gather starts each branch: its cells not yet running
 * send their events as they run (mutex_ held). */
void Run::demand(const CellRef &ref)
{
    Cell &c = cell(ref);
    if (c.demanded || settled(c)) return;
    c.demanded = true;
    const NodePlan &node = nodes_[size_t(ref.node)];
    for (size_t i = 0; i < node.inputs.size(); ++i) {
        const InputPlan &input = node.inputs[i];
        if (input.source >= 0 && !input.lazy && !input.ignored) demand(source_of(ref, int(i)));
    }
}

void Run::consume(const CellRef &source)
{
    Cell &c = cell(source);
    if (c.hits > 0 && --c.hits == 0 && c.state == State::done) {
        c.outputs.clear();
        c.object = PyRef{};
        c.state = State::released;
        c.wanted = false;
    }
}

PyRef Run::process_regular(int node, int64_t slot)
{
    CellRef ref{node, slot};
    ensure(ref);
    std::lock_guard<std::mutex> locked(mutex_);
    Cell &c = cell(ref);
    if (c.state == State::failed) throw PythonError{c.error};
    PyRef object = c.object;
    if (nodes_[size_t(node)].side_effects) consume(ref);
    return object;
}

PyRef Run::process_generator(int node)
{
    CellRef ref{node, -1};
    ensure(ref);
    std::lock_guard<std::mutex> locked(mutex_);
    Cell &c = cell(ref);
    if (c.state == State::failed) throw PythonError{c.error};
    return c.object;
}

/* After a slot's commit: every slot up to it whose cells and window work are idle is dropped (one still in flight
 * goes at a later release, or at the loop's end). */
void Run::release_slot(int64_t slot)
{
    {
        std::lock_guard<std::mutex> locked(mutex_);
        for (auto it = slot_cells_.begin(); it != slot_cells_.end() && it->first <= slot;) {
            int64_t current = it->first;
            bool busy = std::any_of(it->second.begin(), it->second.end(), [](const Cell &c) {
                return c.state == State::running || c.state == State::queued;
            }) || std::any_of(tasks_.begin(), tasks_.end(), [&](const Task &t) { return t.slot == current; });
            if (loop_)
                for (const auto &[key, prepare] : loop_->prepares)
                    busy = busy || (key.second == current && prepare.kind == Prepare::running);
            if (busy || (loop_ && loop_->materializing > 0)) {
                ++it;
                continue;
            }
            std::erase_if(wanted_, [&](const CellRef &r) { return r.slot == current; });
            if (loop_) {
                std::erase_if(loop_->prepares, [&](const auto &entry) { return entry.first.second == current; });
                loop_->outcomes.erase(current);
            }
            it = slot_cells_.erase(it);
        }
    }
    changed_.notify_all();
}

/* ETA as process.Executor.__send_node_progress computes it. */
double eta_of(const std::vector<double> &times, int64_t index, int64_t total)
{
    double average = 0;
    if (!times.empty()) {
        size_t count = std::min<size_t>(times.size(), 100);
        double weighted = 0, weights = 0;
        for (size_t k = 0; k < count; ++k) {
            double i = double(count - k);  // weights for the last `count` times, oldest first: i = count .. 1
            double weight = std::max(1 / i, std::pow(0.9, i));
            weighted += times[times.size() - count + k] * weight;
            weights += weight;
        }
        average = weighted / weights;
    }
    return average * double(std::max<int64_t>(0, total - index));
}

void Run::emit_progress(int node, const std::vector<double> &times, int64_t index, int64_t total)
{
    Event event(Event::progress, node);
    event.index = index;
    event.total = total;
    event.progress_value = total == 0 ? 1 : double(index) / double(total);
    event.progress_int = total == 0;
    event.eta = eta_of(times, index, total);
    event.eta_int = times.empty();
    emit(std::move(event));
}

void Run::open_slot(int64_t slot)
{
    std::vector<Cell> &cells = slot_cells_[slot];
    if (!cells.empty()) return;
    cells.resize(nodes_.size());
    for (int node : loop_->group->iterated) cells[size_t(node)].hits = nodes_[size_t(node)].hits;
}

/* Whether the producer may start its next slot: one a look-ahead waits for (process.ItemWindow.peek: up to its
 * origin + K - 1, beyond the committing slot's window when a node computed ahead looks ahead itself), else within
 * the window (K slots from the committing one) and the memory budget; not paused or stopped (_may_admit). */
bool Run::admits(const Loop &loop) const
{
    if (loop.halted || aborted_ || paused_) return false;
    if (loop.produced <= loop.demand) return true;
    int64_t commit = commit_slot_;
    if (loop.produced >= commit + loop.k) return false;
    return (loop.produced - commit) * loop.estimate <= loop.budget;
}

/* The item window's producer: each slot's generators described in order (Python's iterator is serial), a single
 * generator's item materialized by a worker (its load and enforce), zipped generators advanced whole here; a slot
 * stops at its first end or error, as the loop's advance does. */
void Run::produce()
{
    Loop &loop = *loop_;
    std::vector<GeneratorRun> &generators = *loop.generators;
    for (;;) {
        int64_t slot;
        {
            std::unique_lock<std::mutex> locked(mutex_);
            changed_.wait(locked, [&] { return loop.stop || admits(loop); });
            if (loop.stop) return;
            slot = loop.produced++;
            open_slot(slot);
            loop.outcomes[slot];
        }
        for (size_t g = 0; g < generators.size(); ++g) {
            GeneratorRun &generator = generators[g];
            Outcome outcome;
            PyRef token;
            int64_t item = 0;
            bool native_token = false;
            std::optional<Key> key;
            std::vector<Value> outputs;
            if (generator.native) {
                try {
                    if (!generator.native->describe(item)) {
                        outcome.kind = Outcome::ended;
                    } else if (generators.size() > 1) {
                        outputs = native_item(generator, item);
                        outcome.kind = Outcome::ready;
                    } else {
                        native_token = true;
                    }
                } catch (const std::exception &error) {
                    PythonSection python;
                    outcome.kind = Outcome::failed;
                    outcome.error = py_ref(python_exception(escaped(error)));
                } catch (const PythonError &failure) {
                    outcome.kind = Outcome::failed;
                    outcome.error = failure.error;
                } catch (const StopRun &) {
                    return;
                }
            } else {
                PythonSection section;
                try {
                    py::handle iterator(static_cast<PyObject *>(generator.iterator.handle.get()));
                    py::object described = hooks_.describe(iterator, generator.split);
                    if (described.is_none()) {
                        outcome.kind = Outcome::ended;
                    } else if (generators.size() > 1) {
                        py::list values = hooks_.materialize(
                            generator.node, iterator, py::handle(static_cast<PyObject *>(generator.output.handle.get())),
                            described, generator.split);
                        for (py::handle value : values) outputs.push_back(py_ref(value));
                        outcome.kind = Outcome::ready;
                    } else {
                        key = generator.split ? item_key(generator, described) : std::nullopt;
                        std::optional<Memo::Entry> memoized;
                        if (key) memoized = Memo::get().find(*key);
                        if (memoized) {
                            outputs = memoized->outputs;  // the item's load, from the memo
                            outcome.kind = Outcome::ready;
                        } else {
                            token = py_ref(described);
                        }
                    }
                } catch (py::error_already_set &error) {
                    outcome.kind = Outcome::failed;
                    outcome.error = current_error(error);
                }
            }
            Outcome::Kind kind = outcome.kind;
            {
                std::lock_guard<std::mutex> locked(mutex_);
                if (kind == Outcome::ready) set_item(generator.node, slot, std::move(outputs), key);
                if (kind == Outcome::ended && generators.size() == 1) loop.halted = true;
                if (kind == Outcome::failed && generator.fail_fast) loop.halted = true;
                if (token.handle || native_token) {
                    ++loop.materializing;
                    queue(Task{Task::materialize, slot, int(g), token, item, key});
                }
                loop.outcomes[slot].push_back(std::move(outcome));
            }
            changed_.notify_all();
            if (kind == Outcome::ended || kind == Outcome::failed) break;
        }
    }
}

/* A generator's item for a slot (mutex_ held): its cell done (keyed when its file identity keys it), the slot's
 * window work wanted. */
void Run::set_item(int node, int64_t slot, std::vector<Value> outputs, std::optional<Key> key)
{
    Cell &c = cell({node, slot});
    c.outputs = std::move(outputs);
    c.state = State::done;
    c.hits = -1;
    c.key_state = key ? Cell::KeyState::known : Cell::KeyState::none;
    if (key) c.key = *key;
    if (loop_ && loop_->active && !slots_closing_) want_window(slot);
    dispatch();
}

/* The pure nodes the slot's sinks need, through lazy inputs too (process.Executor's _ItemPlanner candidates), and
 * the prepare phases of its sinks (mutex_ held). A context item's: the pure nodes its temporal nodes need, and no
 * prepare (process.ItemWindow's context slots: nothing else runs for it). */
void Run::want_window(int64_t slot)
{
    std::set<std::pair<int, int64_t>> seen;
    std::function<void(const CellRef &)> walk = [&](const CellRef &ref) {
        if (!seen.insert({ref.node, ref.slot}).second) return;
        const NodePlan &node = nodes_[size_t(ref.node)];
        if (cell(ref).state != State::idle) return;
        if (node.pure) {
            want(ref);
            return;
        }
        for (size_t i = 0; i < node.inputs.size(); ++i) {
            const InputPlan &input = node.inputs[i];
            if (input.source >= 0 && !(input.ignored && node.kind != Kind::collector)) walk(source_of(ref, int(i)));
        }
    };
    if (loop_->context(slot)) {
        for (int node : loop_->group->temporal) walk({node, slot});
        return;
    }
    for (int sink : loop_->sinks) {
        walk({sink, slot});
        if (nodes_[size_t(sink)].phased) loop_->prepares.try_emplace({sink, slot});
    }
}

void Run::run_task(const Task &task)
{
    Loop &loop = *loop_;
    if (task.kind == Task::materialize) {
        GeneratorRun &generator = (*loop.generators)[size_t(task.index)];
        Outcome outcome;
        std::vector<Value> outputs;
        if (generator.native) {
            try {
                outputs = native_item(generator, task.item);
                outcome.kind = Outcome::ready;
            } catch (const PythonError &failure) {
                outcome.kind = Outcome::failed;
                outcome.error = failure.error;
            } catch (const StopRun &) {
                std::lock_guard<std::mutex> locked(mutex_);
                --loop.materializing;
                return;
            }
        } else {
            PythonSection section;
            try {
                py::list values = hooks_.materialize(
                    generator.node, py::handle(static_cast<PyObject *>(generator.iterator.handle.get())),
                    py::handle(static_cast<PyObject *>(generator.output.handle.get())),
                    py::handle(static_cast<PyObject *>(task.token.handle.get())), generator.split);
                for (py::handle value : values) outputs.push_back(py_ref(value));
                outcome.kind = Outcome::ready;
                if (task.key)
                    Memo::get().insert(*task.key,
                                       Memo::Entry{outputs, PyRef{}, hooks_.memo_bytes(values).cast<int64_t>()});
            } catch (py::error_already_set &error) {
                outcome.kind = Outcome::failed;
                outcome.error = current_error(error);
            }
        }
        std::lock_guard<std::mutex> locked(mutex_);
        --loop.materializing;
        if (outcome.kind == Outcome::ready) set_item(generator.node, task.slot, std::move(outputs), task.key);
        if (outcome.kind == Outcome::failed && generator.fail_fast) loop.halted = true;
        for (Outcome &pending : loop.outcomes[task.slot])
            if (pending.kind == Outcome::pending) pending = std::move(outcome);
        return;
    }
    // A sink's prepare phase: its phase inputs (a lazy input as its upstream value), as process.prepare_node takes
    // them; an error prepares nothing (the node then runs whole and fails or succeeds as it does today).
    const NodePlan &node = nodes_[size_t(task.index)];
    std::vector<Value> values;
    PyRef collector;
    {
        std::lock_guard<std::mutex> locked(mutex_);
        for (size_t i = 0; i < node.inputs.size(); ++i) {
            const InputPlan &input = node.inputs[i];
            if (node.kind == Kind::collector && !input.ignored) continue;
            if (input.source < 0) {
                values.push_back(input.constant);
                continue;
            }
            const Cell &from = cell(source_of({task.index, task.slot}, int(i)));
            values.push_back(from.state == State::done ? from.outputs[size_t(input.output)] : Value{});
        }
        if (node.kind == Kind::collector) collector = loop.collector_of(task.index);
    }
    PyRef value;
    {
        PythonSection section;
        try {
            py::list inputs;
            for (const Value &v : values) inputs.append(to_python(v));
            py::object prepared = hooks_.prepare(
                task.index, inputs,
                collector.handle ? py::handle(static_cast<PyObject *>(collector.handle.get())) : py::none());
            if (!prepared.is_none()) value = py_ref(prepared);
        } catch (py::error_already_set &error) {
            error.restore();
            PyErr_Clear();
        }
    }
    std::lock_guard<std::mutex> locked(mutex_);
    Prepare &prepare = loop.prepares[{task.index, task.slot}];
    prepare.kind = Prepare::done;
    prepare.value = value;
}

/* The slot's prepared value of a sink, for its commit (process.ItemWindow.take_prepared): waited for when in flight;
 * none when its phase never started (the node runs whole). Taken once. */
PyRef Run::take_prepared(int node, int64_t slot)
{
    std::unique_lock<std::mutex> locked(mutex_);
    if (aborted_) throw StopRun{};  // a stopped run runs no sink (whole or committed)
    if (!loop_ || !loop_->active) return {};
    auto found = loop_->prepares.find({node, slot});
    if (found == loop_->prepares.end()) return {};
    if (found->second.kind == Prepare::waiting) {
        loop_->prepares.erase(found);
        return {};
    }
    // Stop ends the wait: a paused run's workers take no task (they drop it once stopped).
    changed_.wait(locked, [&] { return found->second.kind == Prepare::done || aborted_; });
    if (aborted_) throw StopRun{};  // also a prepare its Stop dropped (nothing prepared)
    PyRef value = std::move(found->second.value);
    loop_->prepares.erase(found);
    return value;
}

/* The window's decision (process.Executor.__open_window): K, the read-ahead blocker and the budget, from the
 * hooks; a window of K > 1 starts its producer at `start`: the range's second slot (slot 1 without context items) at
 * the regular decision, the slot after the committing one at a look-ahead's (the commit path advanced that one). */
void Run::decide(int64_t start)
{
    Loop &loop = *loop_;
    if (loop.decided) return;
    loop.decided = true;
    std::vector<GeneratorRun> &generators = *loop.generators;
    {
        PythonSection section;
        py::list generator_nodes, generator_outputs, writers, iterated;
        for (const GeneratorRun &generator : generators) {
            generator_nodes.append(generator.node);
            generator_outputs.append(py::handle(static_cast<PyObject *>(generator.output.handle.get())));
        }
        for (int sink : loop.group->outputs) writers.append(sink);
        py::dict collector_inputs;
        for (const CollectorRun &collector : *loop.collectors) {
            writers.append(collector.node);
            collector_inputs[py::int_(collector.node)] = py::handle(static_cast<PyObject *>(collector.inputs.handle.get()));
        }
        for (int node : loop.group->iterated) iterated.append(node);
        py::cpp_function cached([this](int node, int output) -> py::tuple {
            Value value;
            bool found;
            {
                std::lock_guard<std::mutex> locked(mutex_);
                const Cell &c = static_cells_[size_t(node)];
                found = c.state == State::done && size_t(output) < c.outputs.size();
                if (found) value = c.outputs[size_t(output)];
            }
            return py::make_tuple(found, found ? to_python(value) : py::object(py::none()));
        });
        py::tuple decision = hooks_.window(generator_nodes, generator_outputs, writers, iterated, cached,
                                           collector_inputs);
        loop.k = decision[0].cast<int>();
        if (!decision[2].is_none()) loop.blocker = decision[2].cast<std::string>();
        loop.budget = decision[3].cast<int64_t>();
        for (py::handle warning : decision[4]) {
            Event event(Event::raw);
            event.payload = py_ref(warning);
            emit(std::move(event));
        }
    }
    if (loop.k <= 1) return;
    std::lock_guard<std::mutex> locked(mutex_);
    loop.active = true;
    loop.produced = start;
    loop.producer = Engine::get().service([this] {
        Serving serving(this);
        try {
            produce();
        } catch (...) {
            fault(std::current_exception());
        }
    });
}

/* The outcome of a generator's advance at a slot of the window, waited for (process.ItemWindow.take_advance). */
Outcome Run::take_advance(int64_t slot, size_t generator)
{
    std::unique_lock<std::mutex> locked(mutex_);
    Loop &loop = *loop_;
    changed_.wait(locked, [&] {
        auto found = loop.outcomes.find(slot);
        return aborted_ || (found != loop.outcomes.end() && found->second.size() > generator &&
                            found->second[generator].kind != Outcome::pending);
    });
    auto found = loop.outcomes.find(slot);
    if (found == loop.outcomes.end() || found->second.size() <= generator ||
        found->second[generator].kind == Outcome::pending)
        throw StopRun{};
    return found->second[generator];
}

/* A generator's item's array bytes, for the window's per-item estimate: a native generator's images, a Python
 * generator's values as hooks.output_bytes counts them. */
int64_t Run::item_bytes(const GeneratorRun &generator, const std::vector<Value> &outputs)
{
    if (generator.native) {
        int64_t bytes = 0;
        for (const Value &value : outputs)
            if (const Image *image = std::get_if<Image>(&value)) bytes += int64_t(image->values() * sizeof(float));
        return bytes;
    }
    PythonSection python;
    try {
        py::list values;
        for (const Value &value : outputs) values.append(to_python(value));
        return hooks_.output_bytes(values).cast<int64_t>();
    } catch (py::error_already_set &error) {
        throw PythonError{current_error(error)};
    }
}

/* Whether the cell's outputs count to the window's per-item estimate: a cell of the range's first item. A slot's
 * cell runs only while its loop is set (its end drains them first). */
bool Run::measured(const CellRef &ref) const { return ref.slot >= 0 && ref.slot == loop_->first; }

/* Whether the cell's outputs are discarded: an iterated node's at a context item before the range, which is never
 * broadcast (process.Executor.__process); a static node first needed there broadcasts as always. */
bool Run::discarded(const CellRef &ref) const
{
    return ref.slot >= 0 && iterated_[size_t(ref.node)] && ref.slot < loop_->first;
}

/* process.Executor's _ItemPlanner.ahead (memoized per loop): a node a look-ahead may wait for at a later item: an
 * iterated regular node without side effects, pure CPU and not Into Directory, or serial (any other, without lazy
 * inputs: computed on the driver one item at a time, in order) when every error ends the loop; whose inputs are
 * constants, generator outputs, computed static values or such nodes. */
bool Run::ahead(int node)
{
    Loop &loop = *loop_;
    auto memo = loop.ahead.find(node);
    if (memo != loop.ahead.end()) return memo->second;
    loop.ahead[node] = false;  // a cycle never qualifies
    const NodePlan &plan = nodes_[size_t(node)];
    bool serial = !plan.pure && std::none_of(plan.inputs.begin(), plan.inputs.end(),
                                             [](const InputPlan &input) { return input.lazy; });
    bool result = iterated_[size_t(node)] && plan.kind == Kind::regular && !plan.side_effects &&
                  ((plan.pure && !plan.not_speculated) || (serial && loop.fail_fast));
    for (size_t i = 0; result && i < plan.inputs.size(); ++i) {
        int source = plan.inputs[i].source;
        if (source < 0 || loop.generator_index(source) >= 0) continue;
        if (!iterated_[size_t(source)]) {
            std::lock_guard<std::mutex> locked(mutex_);
            result = static_cells_[size_t(source)].state == State::done;
        } else {
            result = ahead(source);
            serial = serial || loop.serial.count(source) > 0;
        }
    }
    loop.ahead[node] = result;
    if (result && serial) loop.serial.insert(node);
    return result;
}

/* Whether a look-ahead on the node's input can be served: a group generator's iterated output, or a node computed
 * ahead (process.Executor.__look_ahead_source). */
bool Run::servable(int node, int input)
{
    const InputPlan &plan = nodes_[size_t(node)].inputs[size_t(input)];
    if (plan.source < 0) return false;
    int generator = loop_->generator_index(plan.source);
    if (generator >= 0) {
        const std::vector<int> &iterable = nodes_[size_t(plan.source)].iterable_outputs;
        return std::find(iterable.begin(), iterable.end(), plan.output) != iterable.end();
    }
    return ahead(plan.source);
}

/* A look-ahead before the range's second slot (during slot 0 without context items) decides the window there
 * (process.Executor.__decide_early); the producer continues from the next slot. */
void Run::decide_early()
{
    if (!loop_->decided) decide(commit_slot_ + 1);
}

int Run::lookahead(int node, int input)
{
    if (!loop_) return 0;
    decide_early();
    Loop &loop = *loop_;
    if (!loop.active) {
        if (loop.blocker && loop.warned.insert(node).second) {
            Event event(Event::raw);
            event.payload = py_ref(hooks_.warn_look_ahead_off(node, *loop.blocker));
            emit(std::move(event));
        }
        return 0;
    }
    return servable(node, input) ? loop.k - 1 : 0;
}

/* process.ItemWindow.peek: the node's input at a later item (offset items after origin, the committing item when
 * -1, as a node computed ahead looks from its own), within the window; the source computed ahead now if needed: a
 * serial one (or one depending on a serial one) at every item in between first, in order. None when the item never
 * comes, its source fails there, or it is not computed ahead. */
py::object Run::peek(int node, int input, int64_t offset, int64_t origin)
{
    if (!loop_) return py::none();
    decide_early();
    Loop &loop = *loop_;
    int64_t commit = commit_slot_;
    int64_t start = origin < 0 ? commit : origin;
    int64_t target = start + offset;
    if (!loop.active || target <= commit || target >= start + loop.k || !servable(node, input)) return py::none();
    if (aborted_) {
        PyErr_SetNone(hooks_.aborted_type.ptr());
        throw py::error_already_set();
    }
    const InputPlan &plan = nodes_[size_t(node)].inputs[size_t(input)];
    bool generator = loop.generator_index(plan.source) >= 0;
    int64_t first = loop.serial.count(plan.source) ? commit + 1 : target;
    Value value;
    bool found = false, stopped = false;
    {
        PythonPause unlocked;
        auto resolved = [&](int64_t slot) {  // the slot's generators all yielded, or it never comes (mutex_ held)
            auto outcome = loop.outcomes.find(slot);
            if (outcome != loop.outcomes.end()) {
                for (const Outcome &o : outcome->second)
                    if (o.kind != Outcome::ready) return o.kind != Outcome::pending;
                return outcome->second.size() == loop.generators->size();
            }
            return loop.halted && loop.produced <= slot;
        };
        try {
            {
                std::lock_guard<std::mutex> locked(mutex_);
                loop.demand = std::max(loop.demand, target);
            }
            changed_.notify_all();
            for (int64_t slot = first; slot <= target; ++slot) {
                {
                    std::unique_lock<std::mutex> locked(mutex_);
                    changed_.wait(locked, [&] { return aborted_ || resolved(slot); });
                    if (aborted_) throw StopRun{};
                    auto outcome = loop.outcomes.find(slot);
                    if (outcome == loop.outcomes.end() || outcome->second.size() != loop.generators->size() ||
                        !std::all_of(outcome->second.begin(), outcome->second.end(),
                                     [](const Outcome &o) { return o.kind == Outcome::ready; }))
                        break;
                }
                CellRef source{plan.source, slot};
                if (!generator) ensure(source, false);
                std::lock_guard<std::mutex> locked(mutex_);
                const Cell &c = cell(source);
                if (aborted_) throw StopRun{};
                if (c.state != State::done) break;
                if (slot == target) {
                    value = c.outputs[size_t(plan.output)];
                    found = true;
                }
            }
        } catch (const StopRun &) {
            stopped = true;
        }
    }
    if (stopped) {
        PyErr_SetNone(hooks_.aborted_type.ptr());
        throw py::error_already_set();
    }
    return found ? to_python(value) : py::none();
}

/* A window job's deferred events, at its place on the commit path (process.ItemWindow.take_node's replay): its
 * inputs' first, then its start, broadcast and finish (only its start when it failed). */
void Run::take(const CellRef &ref)
{
    std::vector<CellRef> inputs;
    bool ok;
    double seconds;
    std::vector<Value> outputs;
    {
        std::lock_guard<std::mutex> locked(mutex_);
        Cell &c = cell(ref);
        if (!c.deferred) return;
        c.deferred = false;
        ok = c.deferred_ok;
        seconds = c.deferred_seconds;
        outputs = std::move(c.deferred_outputs);
        const NodePlan &node = nodes_[size_t(ref.node)];
        for (size_t i = 0; i < node.inputs.size(); ++i) {
            const InputPlan &input = node.inputs[i];
            if (input.source >= 0 && !input.lazy && !input.ignored) inputs.push_back(source_of(ref, int(i)));
        }
    }
    for (const CellRef &input : inputs) take(input);
    emit_start(ref.node);
    if (ok) {
        if (!discarded(ref)) submit_broadcast(ref.node, outputs);
        emit_finish(ref.node, seconds);
    }
}

void Run::iterate_group(const GroupPlan &group)
{
    suspend();
    std::vector<GeneratorRun> generators;
    std::vector<CollectorRun> collectors;
    std::set<int> collector_ids;
    std::set<int64_t> lengths, bases;
    // Context items (process.Executor.__iterate_generator_nodes): a group's only generator reads, for its temporal
    // nodes, `before` items before its range and `after` items after it.
    int64_t before = 0, after = 0;
    for (const GeneratorPlan &plan : group.generators) {
        GeneratorRun run;
        run.node = plan.node;
        run.output = process_generator(plan.node);
        {
            std::lock_guard<std::mutex> locked(mutex_);
            run.partial = static_cells_[size_t(plan.node)].outputs;
            run.native = static_cells_[size_t(plan.node)].items;
        }
        int64_t base = 0;  // a native generator reads its source whole
        bool exact_length = true;
        if (run.native) {
            run.expected = run.native->expected_length();
            run.fail_fast = run.native->fail_fast();
        } else {
            PythonSection python;
            try {
                py::object wanted = group.generators.size() == 1
                                        ? py::object(py::make_tuple(group.before, group.after))
                                        : py::object(py::none());
                py::list iterated;
                for (int node : group.iterated) iterated.append(node);
                py::tuple items = hooks_.items(plan.node, py::handle(static_cast<PyObject *>(run.output.handle.get())),
                                               wanted, iterated);
                run.iterator = py_ref(items[0]);
                run.split = items[1].cast<bool>();
                run.expected = items[2].cast<int64_t>();
                run.fail_fast = items[3].cast<bool>();
                before = items[4].cast<int64_t>();
                after = items[5].cast<int64_t>();
                base = items[6].cast<int64_t>();
                exact_length = items[7].cast<bool>();
            } catch (py::error_already_set &error) {
                throw PythonError{current_error(error)};
            }
        }
        bases.insert(base);
        for (int collector : plan.collectors) {
            if (!collector_ids.insert(collector).second) continue;
            suspend();
            Clock::time_point start = Clock::now();
            CellRef ref{collector, -1};
            ensure(ref);
            std::lock_guard<std::mutex> locked(mutex_);
            Cell &c = cell(ref);
            if (c.state == State::failed) throw PythonError{c.error};
            collectors.push_back({collector, c.object, c.inputs, seconds_since(start)});
        }
        if (exact_length) lengths.insert(run.expected);
        run.loop_start = run.last = Clock::now();
        run.paused_start = run.last_paused = time_paused_;
        emit_progress(plan.node, {}, 0, run.expected);
        generators.push_back(std::move(run));
    }
    if (lengths.size() > 1) {
        PythonSection python;
        throw PythonError{py_ref(py::reinterpret_borrow<py::object>(PyExc_AssertionError)(
            "Expected all connected iterators to have the same length"))};
    }
    {
        std::lock_guard<std::mutex> locked(mutex_);
        for (int node : group.iterated) iterated_[size_t(node)] = 1;
        slots_closing_ = false;
        loop_ = std::make_unique<Loop>();
        loop_->group = &group;
        loop_->generators = &generators;
        loop_->collectors = &collectors;
        loop_->sinks = group.outputs;
        for (const CollectorRun &collector : collectors) loop_->sinks.push_back(collector.node);
        for (const GeneratorRun &generator : generators) loop_->fail_fast = loop_->fail_fast && generator.fail_fast;
        // The range's slots: `before` context items come first; with items after it, the loop ends after its last
        // one (they only serve look-aheads), else as always, when the generators are exhausted.
        loop_->first = before;
        loop_->end = after > 0 ? before + generators[0].expected : -1;
    }
    {
        PythonSection python;
        py::list generator_nodes, iterated;
        for (const GeneratorRun &generator : generators) generator_nodes.append(generator.node);
        for (int node : group.iterated) iterated.append(node);
        // The loop's index base: its generators' shared one, else 0.
        hooks_.begin_loop(generator_nodes, iterated, before, bases.size() == 1 ? *bases.begin() : 0);
    }
    size_t stops = 0;
    std::vector<std::string> deferred;
    // process.Executor.__end_loop and __close_window, on every exit of the loop: the producer stopped, views
    // detached, no item's cell left running or kept.
    auto end_loop = [&] {
        {
            std::lock_guard<std::mutex> locked(mutex_);
            loop_->stop = true;
        }
        changed_.notify_all();
        if (loop_->producer.valid()) loop_->producer.wait();
        {
            PythonSection python;
            try {
                hooks_.end_loop();
                if (loop_->active) hooks_.info(py::str("item window K={}: {} items").format(loop_->k, loop_->committed));
            } catch (py::error_already_set &error) {
                error.discard_as_unraisable("chaiNNer-C engine loop end");
            }
        }
        drain_slots();
        std::lock_guard<std::mutex> locked(mutex_);
        for (int node : group.iterated) iterated_[size_t(node)] = 0;
        loop_.reset();
    };
    try {
        for (int64_t slot = 0; slot != loop_->end; ++slot) {
            {
                std::lock_guard<std::mutex> locked(mutex_);
                commit_slot_ = slot;
                if (slot == 0 || !loop_->active) open_slot(slot);
            }
            changed_.notify_all();
            // A context item before the range: it runs the temporal nodes, with the nodes they need, and nothing else.
            bool context = slot < loop_->first;
            if (slot == loop_->first + 1) decide(slot);
            const GeneratorRun *current = nullptr;
            // The window's per-item estimate measures the range's first item (the items before it meanwhile, which
            // run their generators' part of it).
            int64_t estimate = 0;
            try {
                for (size_t g = 0; g < generators.size(); ++g) {
                    GeneratorRun &generator = generators[g];
                    suspend();
                    current = &generator;
                    std::vector<Value> outputs;
                    if (loop_->active && slot >= 1) {
                        Outcome outcome = take_advance(slot, g);
                        if (outcome.kind == Outcome::ended) throw ItemsEnd{};
                        if (outcome.kind == Outcome::failed) throw item_failure(generator, outcome.error);
                        std::lock_guard<std::mutex> locked(mutex_);
                        outputs = cell({generator.node, slot}).outputs;
                    } else if (generator.native) {
                        int64_t token = 0;
                        bool more;
                        try {
                            more = generator.native->describe(token);
                        } catch (const std::exception &error) {
                            PythonSection python;
                            throw item_failure(generator, py_ref(python_exception(escaped(error))));
                        }
                        if (!more) throw ItemsEnd{};
                        try {
                            outputs = native_item(generator, token);
                        } catch (const PythonError &failure) {
                            throw item_failure(generator, failure.error);
                        }
                        suspend();
                        {
                            std::lock_guard<std::mutex> locked(mutex_);
                            set_item(generator.node, slot, outputs);
                        }
                        changed_.notify_all();
                    } else {
                        bool more;
                        std::optional<Key> item;
                        {
                            PythonSection python;
                            try {
                                py::handle iterator(static_cast<PyObject *>(generator.iterator.handle.get()));
                                py::object token = hooks_.describe(iterator, generator.split);
                                more = !token.is_none();
                                if (more) {
                                    std::optional<Key> key = generator.split ? item_key(generator, token) : std::nullopt;
                                    std::optional<Memo::Entry> memoized;
                                    if (key) memoized = Memo::get().find(*key);
                                    if (memoized) {
                                        outputs = memoized->outputs;
                                    } else {
                                        py::list values = hooks_.materialize(
                                            generator.node, iterator,
                                            py::handle(static_cast<PyObject *>(generator.output.handle.get())), token,
                                            generator.split);
                                        for (py::handle value : values) outputs.push_back(py_ref(value));
                                        if (key)
                                            Memo::get().insert(*key, Memo::Entry{outputs, PyRef{},
                                                                                 hooks_.memo_bytes(values).cast<int64_t>()});
                                    }
                                    item = key;
                                }
                            } catch (py::error_already_set &error) {
                                throw item_failure(generator, current_error(error));
                            }
                        }
                        if (!more) throw ItemsEnd{};
                        suspend();
                        {
                            std::lock_guard<std::mutex> locked(mutex_);
                            set_item(generator.node, slot, outputs, item);
                        }
                        changed_.notify_all();
                    }
                    if (slot <= loop_->first) {
                        estimate += item_bytes(generator, outputs);
                        std::lock_guard<std::mutex> locked(mutex_);  // the producer reads it (an early decision's)
                        loop_->estimate = estimate;
                    }
                    if (!context) submit_broadcast(generator.node, outputs);
                }
                {
                    std::lock_guard<std::mutex> locked(mutex_);
                    std::set<std::pair<int, int64_t>> seen;
                    if (context) {
                        for (int node : group.temporal) prefetch({node, slot}, seen);
                    } else {
                        for (int output : group.outputs) prefetch({output, slot}, seen);
                        for (const CollectorRun &collector : collectors) {
                            const NodePlan &node = nodes_[size_t(collector.node)];
                            for (size_t i = 0; i < node.inputs.size(); ++i)
                                if (node.inputs[i].ignored && node.inputs[i].source >= 0)
                                    prefetch(source_of({collector.node, slot}, int(i)), seen);
                        }
                    }
                    dispatch();
                }
                if (context) {
                    // A context item updates the temporal nodes' state: they run, in topological order, with the
                    // nodes they need, and nothing else (process.Executor.__iterate_generator_nodes).
                    for (int node : group.temporal) process_regular(node, slot);
                } else {
                    for (int output : group.outputs) process_regular(output, slot);
                    for (CollectorRun &collector : collectors) {
                        suspend();
                        const NodePlan &node = nodes_[size_t(collector.node)];
                        // process.Executor.__gather_collector_inputs: the iterated inputs in order.
                        std::vector<Value> values;
                        for (size_t i = 0; i < node.inputs.size(); ++i) {
                            const InputPlan &input = node.inputs[i];
                            if (!input.ignored) continue;
                            if (input.source < 0) {
                                values.push_back(input.constant);
                                continue;
                            }
                            CellRef source = source_of({collector.node, slot}, int(i));
                            ensure(source);
                            std::lock_guard<std::mutex> locked(mutex_);
                            Cell &from = cell(source);
                            if (from.state == State::failed) throw PythonError{from.error};
                            values.push_back(output_of(from, input.output));
                            consume(source);
                        }
                        suspend();
                        Clock::time_point start = Clock::now();
                        PyRef prepared = take_prepared(collector.node, slot);
                        {
                            PythonSection python;
                            try {
                                py::list inputs;
                                for (const Value &value : values) inputs.append(to_python(value));
                                hooks_.collector_iterate(
                                    collector.node, inputs,
                                    py::handle(static_cast<PyObject *>(collector.collector.handle.get())),
                                    prepared.handle ? py::handle(static_cast<PyObject *>(prepared.handle.get()))
                                                    : py::none());
                            } catch (py::error_already_set &error) {
                                collector.seconds += seconds_since(start);
                                throw PythonError{current_error(error)};
                            }
                        }
                        collector.seconds += seconds_since(start);
                    }
                }
                if (slot == loop_->first) {
                    std::lock_guard<std::mutex> locked(mutex_);
                    for (int node : group.iterated) loop_->estimate += slot_cells_[slot][size_t(node)].bytes;
                }
                ++loop_->committed;
                release_slot(slot);
                suspend();
                for (GeneratorRun &generator : generators) {
                    if (context) break;  // no progress for a context item: its time counts to the range's first item
                    Clock::time_point now = Clock::now();
                    double paused = std::max(0.0, time_paused_ - generator.last_paused);
                    generator.times.push_back(std::chrono::duration<double>(now - generator.last).count() - paused);
                    generator.last = now;
                    generator.last_paused = time_paused_;
                    int64_t iterations = int64_t(generator.times.size());
                    emit_progress(generator.node, generator.times, iterations,
                                  std::max(generator.expected, iterations));
                }
                suspend();
            } catch (const ItemsEnd &) {
                release_slot(slot);
                if (++stops >= generators.size()) break;
            } catch (const PythonError &failure) {
                if (current && current->fail_fast) throw;
                {
                    PythonSection python;
                    deferred.push_back(py::str(py::handle(static_cast<PyObject *>(failure.error.handle.get()))));
                }
                release_slot(slot);
                suspend();
            }
        }
    } catch (...) {
        end_loop();
        throw;
    }
    end_loop();
    flush_broadcasts();
    for (GeneratorRun &generator : generators) {
        suspend();
        {
            // process.Executor restores the GeneratorOutput in the cache, with its strategy, for later readers.
            std::lock_guard<std::mutex> locked(mutex_);
            Cell &c = static_cells_[size_t(generator.node)];
            c.state = State::done;
            c.outputs = generator.partial;
            c.hits = nodes_[size_t(generator.node)].hits;
        }
        submit_broadcast(generator.node, generator.partial);
        int64_t iterations = int64_t(generator.times.size());
        Event done(Event::progress, generator.node);
        done.index = done.total = iterations;
        done.progress_value = 1;
        done.progress_int = done.eta_int = true;
        emit(std::move(done));
        double paused = std::max(0.0, time_paused_ - generator.paused_start);
        emit_finish(generator.node, seconds_since(generator.loop_start) - paused);
    }
    for (CollectorRun &collector : collectors) {
        suspend();
        Clock::time_point start = Clock::now();
        std::vector<Value> outputs;
        {
            PythonSection python;
            try {
                py::list values = hooks_.collector_complete(
                    collector.node, py::handle(static_cast<PyObject *>(collector.collector.handle.get())));
                for (py::handle value : values) outputs.push_back(py_ref(value));
            } catch (py::error_already_set &error) {
                throw PythonError{current_error(error)};
            }
        }
        collector.seconds += seconds_since(start);
        submit_broadcast(collector.node, outputs);
        emit_finish(collector.node, collector.seconds);
        std::lock_guard<std::mutex> locked(mutex_);
        Cell &c = static_cells_[size_t(collector.node)];
        c.outputs = std::move(outputs);
        c.hits = nodes_[size_t(collector.node)].hits;
    }
    if (!deferred.empty()) {
        std::string message = "Errors occurred during iteration:\n";
        for (size_t i = 0; i < deferred.size(); ++i) message += (i ? "\n- " : "- ") + deferred[i];
        PythonSection python;
        throw PythonError{py_ref(py::reinterpret_borrow<py::object>(PyExc_Exception)(message))};
    }
}

/* The end of a group's loop: no item's cell is queued or running afterwards, and none is kept. Slot work a job
 * finishing meanwhile would queue (a materialize's item wanting its window) is refused from the start, and purged
 * again with the slots under one hold of the lock. */
void Run::drain_slots()
{
    std::unique_lock<std::mutex> locked(mutex_);
    slots_closing_ = true;
    auto purge = [&] {
        std::erase_if(ready_, [&](const CellRef &ref) {
            if (ref.slot < 0) return false;
            Cell &c = cell(ref);
            c.state = State::idle;
            c.wanted = false;
            return true;
        });
        std::erase_if(wanted_, [](const CellRef &ref) { return ref.slot >= 0; });
        tasks_.clear();
    };
    purge();
    changed_.wait(locked, [&] { return running_slots_ == 0; });
    purge();
    slot_cells_.clear();
}

void Run::drain_workers()
{
    std::unique_lock<std::mutex> locked(mutex_);
    dropping_ = true;
    for (const CellRef &ref : ready_) {
        Cell &c = cell(ref);
        c.state = State::idle;
        c.wanted = false;
    }
    ready_.clear();
    changed_.wait(locked, [&] { return running_ == 0; });
}

void Run::drive()
{
    PyRef error, result;
    bool stopped = false;
    try {
        if (mode_ == "run") {
            emit(Event(Event::chain_start));
            for (const GroupPlan &group : groups_) iterate_group(group);
            for (size_t i = 0; i < tail_.size(); ++i) {
                {
                    // The sink's pure upstream, beside the driver's work; a later sink's waits for it (Python runs
                    // the sinks one after another: a later sink may read a file an earlier one writes, and every
                    // branch at once could hold every branch's outputs).
                    std::lock_guard<std::mutex> locked(mutex_);
                    std::set<std::pair<int, int64_t>> seen;
                    prefetch({tail_[i], -1}, seen);
                    dispatch();
                }
                suspend();
                process_regular(tail_[i], -1);
            }
        } else if (mode_ == "regular") {
            // An individual run's result, as the server caches it: the node's RegularOutput, a native node's made of
            // its outputs. Nothing consumes the cell (a side-effect node's run would release them): the run ends.
            CellRef ref{target_, -1};
            ensure(ref);
            std::vector<Value> outputs;
            std::string identity;
            {
                std::lock_guard<std::mutex> locked(mutex_);
                Cell &c = cell(ref);
                if (c.state == State::failed) throw PythonError{c.error};
                result = c.object;
                outputs = c.outputs;
                identity = c.identity;
            }
            PythonSection python;
            if (!result.handle) {
                py::list values;
                for (const Value &value : outputs) values.append(to_python(value));
                result = py_ref(hooks_.regular_output(values));
            }
            if (!identity.empty()) hooks_.loaded(py::handle(static_cast<PyObject *>(result.handle.get())), identity);
        } else {
            result = process_generator(target_);
        }
    } catch (const PythonError &failure) {
        error = failure.error;
    } catch (const StopRun &) {
        stopped = true;
    } catch (const std::exception &failure) {
        PythonSection python;
        error = py_ref(py::reinterpret_borrow<py::object>(PyExc_RuntimeError)(failure.what()));
    } catch (...) {
        // Whatever ends the run, the request gets its result (and the run its cleanups).
        PythonSection python;
        error = py_ref(py::reinterpret_borrow<py::object>(PyExc_RuntimeError)("The native run engine failed"));
    }
    drain_workers();
    {
        std::string fault;
        {
            std::lock_guard<std::mutex> locked(mutex_);
            fault = fault_;
        }
        PythonSection section;
        if (stopped)
            error = fault.empty() ? py_ref(hooks_.aborted_type())
                                  : py_ref(py::reinterpret_borrow<py::object>(PyExc_RuntimeError)(fault));
    }
    if (mode_ == "run") {
        try {
            flush_broadcasts();
        } catch (...) {
        }
        {
            std::lock_guard<std::mutex> locked(mutex_);
            for (Cell &c : static_cells_) {
                c.outputs.clear();
                c.object = PyRef{};
            }
            slot_cells_.clear();
        }
        PythonSection section;
        try {
            hooks_.chain_cleanups(bool(error.handle));
        } catch (py::error_already_set &failure) {
            if (!error.handle) error = current_error(failure);
        }
    }
    flush_events();
    PythonSection section;
    py::handle error_object = error.handle ? py::handle(static_cast<PyObject *>(error.handle.get())) : py::none();
    py::handle result_object = result.handle ? py::handle(static_cast<PyObject *>(result.handle.get())) : py::none();
    try {
        hooks_.finish(error_object, result_object);
    } catch (py::error_already_set &failure) {
        failure.discard_as_unraisable("chaiNNer-C engine finish");
    }
    error = PyRef{};
    result = PyRef{};
}

// Execution

/* Whether a cell's events wait for the commit path to take it (mutex_ held): a window job (process.ItemWindow's
 * take_node replay), a serial node computed ahead on the driver for a look-ahead, or a cell a worker runs before the
 * commit path needs it (a prefetch), whose events Python never sends when the run fails first. */
bool Run::defers(const CellRef &ref, bool on_driver)
{
    bool window = ref.slot >= 0 && loop_ && loop_->active;
    if (on_driver) return window && ref.slot > commit_slot_;
    return window || !cell(ref).demanded;
}

Run::Gathered Run::gather(const CellRef &ref, bool on_driver)
{
    const NodePlan &node = nodes_[size_t(ref.node)];
    Gathered gathered;
    std::lock_guard<std::mutex> locked(mutex_);
    gathered.defer = defers(ref, on_driver);
    for (size_t i = 0; i < node.inputs.size(); ++i) {
        const InputPlan &input = node.inputs[i];
        if (input.ignored) {
            gathered.values.emplace_back(std::monostate{});
        } else if (input.source < 0) {
            if (!node.native)
                gathered.values.push_back(input.constant);
            else if (input.lazy)
                gathered.values.push_back(std::shared_ptr<LazyValue>(std::make_shared<ReadyLazy>(input.enforced)));
            else
                gathered.values.push_back(input.enforced);
        } else if (input.lazy) {
            auto lazy = std::make_shared<EngineLazy>(*this, ref, int(i), source_of(ref, int(i)), input.output);
            gathered.lazies.push_back(lazy);
            gathered.values.push_back(std::shared_ptr<LazyValue>(lazy));
        } else {
            CellRef source = source_of(ref, int(i));
            Cell &from = cell(source);
            gathered.values.push_back(output_of(from, input.output));
            consume(source);
        }
    }
    return gathered;
}

void Run::read_lazy(EngineLazy &lazy)
{
    Clock::time_point start = Clock::now();
    try {
        ensure(lazy.source_);
        std::lock_guard<std::mutex> locked(mutex_);
        // Stopped while the upstream ran (a worker finishes it): the reader, already running, gets no value, as
        // process.Executor's lazy wait ends with Aborted; else Save Image would still write its file after Stop.
        if (aborted_) throw StopRun{};
        Cell &from = cell(lazy.source_);
        if (from.state == State::failed) {
            lazy.upstream_ = from.error;
        } else {
            lazy.raw_ = output_of(from, lazy.output_);
            consume(lazy.source_);
        }
    } catch (...) {
        lazy.error_ = std::current_exception();
    }
    lazy.seconds_ = seconds_since(start);
}

py::list Run::python_inputs(const CellRef &ref, const Gathered &inputs)
{
    const NodePlan &node = nodes_[size_t(ref.node)];
    py::list values;
    size_t lazy = 0;
    for (size_t i = 0; i < node.inputs.size(); ++i) {
        const InputPlan &input = node.inputs[i];
        if (input.ignored) {
            values.append(py::none());
        } else if (input.source < 0) {
            values.append(py::handle(static_cast<PyObject *>(input.constant.handle.get())));
        } else if (input.lazy) {
            std::shared_ptr<EngineLazy> reader = inputs.lazies[lazy++];
            values.append(py::module_::import("api").attr("Lazy")(
                py::cpp_function([reader]() { return reader->python_value(); })));
        } else {
            values.append(to_python(inputs.values[i]));
        }
    }
    return values;
}

bool Run::run_python(const CellRef &ref, const Gathered &inputs, Cell &result, double &other_seconds,
                     const PyRef &prepared, bool ahead)
{
    const NodePlan &node = nodes_[size_t(ref.node)];
    PythonSection python;
    // The body's waits for other work (util.waiting: a session an item ahead holds, a look-ahead) are not its time,
    // as process.Executor's timed_supplier leaves them out.
    double waited = hooks_.waited().cast<double>();
    try {
        py::list values = python_inputs(ref, inputs);
        py::object output = hooks_.run_node(
            ref.node, values,
            prepared.handle ? py::handle(static_cast<PyObject *>(prepared.handle.get())) : py::none(),
            ahead ? ref.slot : int64_t(-1));
        result.object = py_ref(output);
        if (node.kind == Kind::regular) {
            py::list outputs = hooks_.regular_outputs(output);
            for (py::handle value : outputs) result.outputs.push_back(py_ref(value));
            if (measured(ref)) result.bytes = hooks_.output_bytes(outputs).cast<int64_t>();
        } else if (node.kind == Kind::generator) {
            for (py::handle value : hooks_.generator_outputs(output)) result.outputs.push_back(py_ref(value));
        } else {
            result.object = py_ref(hooks_.collector(output));
            result.inputs = py_ref(values);
        }
    } catch (py::error_already_set &error) {
        result.error = current_error(error);
    }
    other_seconds += hooks_.waited().cast<double>() - waited;
    for (const auto &lazy : inputs.lazies) other_seconds += lazy->seconds();
    return !result.error.handle;
}

/* An image the engine enforces itself: an allowed channel count. */
bool allowed_channels(const std::vector<int> &channels, int count)
{
    return channels.empty() || std::find(channels.begin(), channels.end(), count) != channels.end();
}

Value Run::enforce_input(int node, int input, const Value &value)
{
    const InputPlan &plan = nodes_[size_t(node)].inputs[size_t(input)];
    if (plan.image) {
        if (const Image *image = std::get_if<Image>(&value); image && allowed_channels(plan.channels, image->channels))
            return value;
        if (const Color *color = std::get_if<Color>(&value);
            color && plan.colors && allowed_channels(plan.channels, int(color->values.size())))
            return value;
    }
    PythonSection python;
    try {
        return value_of(hooks_.enforce_input(node, input, to_python(value)));
    } catch (py::error_already_set &error) {
        throw node_error(error);
    } catch (const std::exception &error) {
        throw escaped(error);  // a value an entry point cannot take (e.g. a 2-channel image)
    }
}

std::optional<NodeError> Run::enforce_outputs(int node_index, std::vector<Value> &outputs,
                                              const std::vector<size_t> &which)
{
    const NodePlan &node = nodes_[size_t(node_index)];
    std::vector<size_t> slow;
    try {
        for (size_t i : which) {
            const OutputPlan &plan = node.outputs[i];
            const Image *image = std::get_if<Image>(&outputs[i]);
            if (plan.image && image && image->height > 0 && image->width > 0 &&
                (plan.channels == 0 || plan.channels == image->channels)) {
                if (!plan.assume_normalized) outputs[i] = normalized_output(*image);
                continue;
            }
            slow.push_back(i);
        }
    } catch (const std::exception &error) {
        return escaped(error);
    }
    if (slow.empty()) return std::nullopt;
    PythonSection python;
    try {
        for (size_t i : slow) outputs[i] = py_ref(hooks_.enforce_output(node_index, int(i), to_python(outputs[i])));
    } catch (py::error_already_set &raised) {
        return node_error(raised);
    }
    return std::nullopt;
}

NativeOutcome Run::run_native(const CellRef &ref, const Gathered &inputs, Cell &result, double &other_seconds)
{
    const NodePlan &node = nodes_[size_t(ref.node)];
    std::vector<Value> values = inputs.values;
    try {
        for (size_t i = 0; i < node.inputs.size(); ++i) {
            const InputPlan &input = node.inputs[i];
            if (input.source >= 0 && !input.ignored && !input.lazy) values[i] = enforce_input(ref.node, int(i), values[i]);
        }
    } catch (const NodeError &) {
        return NativeOutcome::python;
    }
    auto failure = [&](const NodeError &error) {
        PythonSection python;
        py::list python_values;
        for (const Value &value : values)
            python_values.append(std::holds_alternative<std::shared_ptr<LazyValue>>(value) ? py::object(py::none())
                                                                                       : to_python(value));
        try {
            result.error = py_ref(hooks_.native_failure(ref.node, python_exception(error), python_values));
        } catch (py::error_already_set &again) {
            result.error = current_error(again);
        }
        return NativeOutcome::failed;
    };
    std::vector<Value> outputs;
    std::unique_ptr<ItemSource> source;
    auto context = std::make_shared<EngineContext>(*this, node, storage_);
    try {
        if (node.kind == Kind::generator)
            source = node.native->generator(*context, values, outputs);
        else
            node.native->run(*context, values, outputs);
    } catch (const Aborted &) {
        PythonSection python;
        result.error = py_ref(py::module_::import("api").attr("Aborted")());
        return NativeOutcome::failed;
    } catch (const RunPython &) {
        return NativeOutcome::declined;
    } catch (const std::bad_variant_access &) {
        // An input of a type the entry point does not take (e.g. an integer beyond int64): the body runs.
        return NativeOutcome::declined;
    } catch (const UpstreamFailed &failed) {
        result.error = failed.error;
        return NativeOutcome::failed;
    } catch (const std::exception &error) {
        return failure(escaped(error));
    }
    for (const auto &lazy : inputs.lazies) other_seconds += lazy->seconds();
    if (node.kind == Kind::generator) {
        // process.enforce_generator_output: the static outputs, enforced, beside the iterated ones (None).
        size_t statics = node.outputs.size() - node.iterable_outputs.size();
        if (!source) return failure(NodeError("AssertionError", "Expected the output to be a generator"));
        if (outputs.size() != statics) return failure(NodeError("AssertionError", ""));
        std::vector<Value> partial(node.outputs.size());
        std::vector<size_t> which;
        for (size_t i = 0, k = 0; i < node.outputs.size(); ++i) {
            if (std::find(node.iterable_outputs.begin(), node.iterable_outputs.end(), int(i)) != node.iterable_outputs.end())
                continue;
            partial[i] = outputs[k++];
            which.push_back(i);
        }
        if (auto error = enforce_outputs(ref.node, partial, which)) return failure(*error);
        std::vector<std::filesystem::path> paths;
        bool known = false;
        try {
            known = source->source_paths(paths);
        } catch (const std::exception &error) {
            return failure(escaped(error));
        }
        {
            PythonSection python;
            try {
                py::list python_partial;
                for (const Value &value : partial) python_partial.append(to_python(value));
                py::object source_paths = py::none();
                if (known) {
                    py::list listed;
                    for (const auto &p : paths) listed.append(to_python(Path{p}));
                    source_paths = py::tuple(listed);
                }
                result.object = py_ref(hooks_.native_generator(ref.node, source->expected_length(),
                                                               source->fail_fast(), source_paths, python_partial));
            } catch (py::error_already_set &error) {
                result.error = current_error(error);
                return NativeOutcome::failed;
            }
        }
        result.outputs = std::move(partial);
        result.items = std::move(source);
        result.items_context = context;
        result.items_inputs = std::move(values);
        return NativeOutcome::done;
    }
    if (outputs.size() != node.outputs.size())
        return failure(NodeError("AssertionError", "Expected all " + node.name + " nodes to have " +
                                                       std::to_string(node.outputs.size()) + " output(s) but found " +
                                                       std::to_string(outputs.size()) + "."));
    std::vector<size_t> all(outputs.size());
    for (size_t i = 0; i < all.size(); ++i) all[i] = i;
    if (auto error = enforce_outputs(ref.node, outputs, all)) return failure(*error);
    if (measured(ref))
        for (const Value &value : outputs)
            if (const Image *image = std::get_if<Image>(&value)) result.bytes += int64_t(image->values() * sizeof(float));
    result.outputs = std::move(outputs);
    return NativeOutcome::done;
}

/* A native generator's item as process.Executor.__generator_fill_partial_output fills it: the partial outputs with
 * the item's iterated values in their places, enforced; an error is the item's (its Python exception, raw). */
std::vector<Value> Run::fill_item(const GeneratorRun &generator, std::vector<Value> item)
{
    const NodePlan &node = nodes_[size_t(generator.node)];
    if (item.size() != node.iterable_outputs.size()) {
        PythonSection python;
        throw PythonError{py_ref(py::reinterpret_borrow<py::object>(PyExc_AssertionError)())};
    }
    std::vector<Value> outputs = generator.partial;
    std::vector<size_t> which;
    for (size_t k = 0; k < item.size(); ++k) {
        outputs[size_t(node.iterable_outputs[k])] = std::move(item[k]);
        which.push_back(size_t(node.iterable_outputs[k]));
    }
    if (auto error = enforce_outputs(generator.node, outputs, which)) {
        PythonSection python;
        throw PythonError{py_ref(python_exception(*error))};
    }
    return outputs;
}

/* A described item of a native generator, materialized (GIL-free) and filled. */
std::vector<Value> Run::native_item(const GeneratorRun &generator, int64_t token)
{
    std::vector<Value> item;
    try {
        generator.native->materialize(token, item);
    } catch (const RunPython &) {
        PythonSection python;
        throw PythonError{py_ref(python_exception(materialize_handed_back()))};
    } catch (const std::bad_variant_access &) {
        PythonSection python;
        throw PythonError{py_ref(python_exception(materialize_handed_back()))};
    } catch (const std::exception &error) {
        PythonSection python;
        throw PythonError{py_ref(python_exception(escaped(error)))};
    } catch (const Aborted &) {
        throw StopRun{};
    }
    return fill_item(generator, std::move(item));
}

/* A generator's item error as the run reports it, naming the generator (process.loop_node_error); an error the
 * hook itself raises is reported instead. */
PythonError Run::item_failure(const GeneratorRun &generator, const PyRef &error)
{
    PythonSection python;
    try {
        return PythonError{
            py_ref(hooks_.item_failure(generator.node, py::handle(static_cast<PyObject *>(error.handle.get()))))};
    } catch (py::error_already_set &again) {
        return PythonError{current_error(again)};
    }
}

/* A cell's memo key (mutex_ held): from its schema, package settings and inputs (a constant's JSON, a source's key
 * and output index); a generator's run only when its per-run outputs follow from its inputs alone (not Load Video's
 * FPS, read from its file); a generator's item has the key its file identity gave it; a node outside the allow-list
 * (or fed by one) has none. Structural: nothing runs to compute it. */
std::optional<Key> Run::key_of(const CellRef &ref)
{
    Cell &c = cell(ref);
    if (c.key_state == Cell::KeyState::known) return c.key;
    if (c.key_state == Cell::KeyState::none) return std::nullopt;
    const NodePlan &node = nodes_[size_t(ref.node)];
    c.key_state = Cell::KeyState::none;
    bool keyed = node.kind == Kind::generator ? node.run_keyed && (ref.slot < 0 || !iterated_[size_t(ref.node)])
                                              : node.memo;
    if (!keyed || node.identity_input >= 0) return std::nullopt;  // Load Image: identify_file keys it
    Key key = Key().add(node.schema).add(node.settings_key);
    for (size_t i = 0; i < node.inputs.size(); ++i) {
        const InputPlan &input = node.inputs[i];
        key.add(uint64_t(i));
        if (input.ignored) continue;
        if (input.source < 0) {
            if (!input.constant_key) return std::nullopt;
            key.add(*input.constant_key);
            continue;
        }
        std::optional<Key> source = key_of(source_of(ref, int(i)));
        if (!source) return std::nullopt;
        key.add(*source).add(uint64_t(input.output));
    }
    c.key = key;
    c.key_state = Cell::KeyState::known;
    return key;
}

/* Load Image's key: its schema, settings and constant inputs with its file's identity (path, size, mtime, volume and
 * file id); none when an input is not a keyable constant. */
std::optional<Key> Run::file_key(const NodePlan &node, const std::string &identity) const
{
    Key key = Key().add(node.schema).add(node.settings_key).add(identity);
    for (const InputPlan &input : node.inputs) {
        if (input.source >= 0 || !input.constant_key) return std::nullopt;
        key.add(*input.constant_key);
    }
    return key;
}

/* Load Image's key from its file's identity read as it runs, so a file an earlier sink of the run wrote is seen as
 * it is now. */
void Run::identify_file(const CellRef &ref)
{
    const NodePlan &node = nodes_[size_t(ref.node)];
    const InputPlan &input = node.inputs[size_t(node.identity_input)];
    std::string identity;
    if (input.source < 0) {
        PythonSection python;
        py::object found = hooks_.file_identity(py::handle(static_cast<PyObject *>(input.constant.handle.get())));
        if (!found.is_none()) identity = found.cast<std::string>();
    }
    std::optional<Key> key = identity.empty() ? std::nullopt : file_key(node, identity);
    std::lock_guard<std::mutex> locked(mutex_);
    Cell &c = cell(ref);
    c.identity = key ? identity : std::string();
    c.key_state = key ? Cell::KeyState::known : Cell::KeyState::none;
    if (key) c.key = *key;
}

/* A Python split iterator's item key (GIL held): the generator's key with the identity of the file its token
 * names (Load Images' (path, index)); none for other generators. */
std::optional<Key> Run::item_key(const GeneratorRun &generator, const py::object &token)
{
    std::optional<Key> base;
    {
        std::lock_guard<std::mutex> locked(mutex_);
        if (!Memo::get().enabled()) return std::nullopt;
        base = key_of({generator.node, -1});
    }
    if (!base) return std::nullopt;
    py::object identity = hooks_.item_identity(token);
    if (identity.is_none()) return std::nullopt;
    return Key(*base).add(std::string_view("item")).add(identity.cast<std::string>());
}

void Run::execute(const CellRef &ref, bool on_driver)
{
    const NodePlan &node = nodes_[size_t(ref.node)];
    Gathered inputs = gather(ref, on_driver);
    bool defer = inputs.defer;
    bool ahead = on_driver && defer;  // a serial node computed ahead for a look-ahead, with its view at that item
    if (on_driver) suspend();
    if (!defer) emit_start(ref.node);
    if (on_driver) suspend();
    PyRef prepared = on_driver && node.phased && ref.slot >= 0 ? take_prepared(ref.node, ref.slot) : PyRef{};
    if (on_driver && !ahead && node.node_context && node.kind == Kind::regular && ref.slot >= 0) {
        // process.Executor.__get_node_context: the loop's node gets its iteration view on the commit path.
        PythonSection python;
        hooks_.iteration(ref.node);
    }
    Cell result;
    double other_seconds = 0;
    Clock::time_point start = Clock::now();
    bool ok;
    std::optional<Key> key;
    std::optional<Memo::Entry> memoized;
    if (node.memo) {
        if (node.identity_input >= 0) identify_file(ref);
        {
            std::lock_guard<std::mutex> locked(mutex_);
            key = key_of(ref);
        }
        if (key) memoized = Memo::get().find(*key);
    }
    // A sink whose prepare phase ran ahead commits with it through its Python body (the window's encode ahead);
    // without one it runs natively.
    NativeOutcome outcome = memoized                         ? NativeOutcome::done
                            : node.native && !prepared.handle ? run_native(ref, inputs, result, other_seconds)
                                                              : NativeOutcome::python;
    if (memoized) {
        // A hit: the node's outputs as computed before, with the events a computed node sends.
        result.outputs = memoized->outputs;
        result.object = memoized->object;
        result.bytes = memoized->bytes;
    }
    if (outcome == NativeOutcome::declined) {
        std::lock_guard<std::mutex> locked(decline_mutex);
        ++decline_counts()[node.schema];
    }
    if (outcome == NativeOutcome::declined || outcome == NativeOutcome::python) {
        result = Cell{};
        other_seconds = 0;
        ok = run_python(ref, inputs, result, other_seconds, prepared, ahead);
    } else {
        ok = outcome == NativeOutcome::done;
    }
    double seconds = seconds_since(start) - other_seconds;
    // A stopped run's node that still finished (its request ended without it: NativeExecutor's Stop) memoizes nothing.
    if (ok && key && !memoized && !aborted_) {
        result.key = *key;
        result.key_state = Cell::KeyState::known;
        Memo::get().insert(*key, Memo::Entry{result.outputs, result.object, memo_bytes(result.outputs)});
    }
    if (node.node_context) {
        PythonSection python;
        try {
            hooks_.node_cleanups(ref.node, !ok);
        } catch (py::error_already_set &error) {
            if (ok) {
                ok = false;
                result.error = current_error(error);
            }
        }
    }
    if (ok && on_driver) {
        try {
            suspend();
        } catch (const StopRun &) {
            std::lock_guard<std::mutex> locked(mutex_);
            cell(ref).state = State::idle;
            throw;
        }
    }
    if (defer) {
        result.deferred = true;
        result.deferred_ok = ok;
        result.deferred_seconds = seconds;
        if (ok) result.deferred_outputs = result.outputs;
    } else if (ok) {
        if (node.kind == Kind::generator) {
            flush_broadcasts();
            broadcast_now(ref.node, result.outputs, result.object);
        } else if (node.kind == Kind::regular) {
            if (!discarded(ref)) submit_broadcast(ref.node, result.outputs);
            emit_finish(ref.node, seconds);
        }
    }
    store(ref, std::move(result));
}

/* execute(), with a C++ exception escaping it (a conversion, an allocation) as the node's error. */
void Run::execute_cell(const CellRef &ref, bool on_driver)
{
    std::optional<NodeError> failure;
    bool deferred = false;
    try {
        execute(ref, on_driver);
    } catch (py::error_already_set &error) {
        PythonSection python;
        failure = node_error(error);
    } catch (const std::exception &error) {
        failure = escaped(error);
    }
    if (!failure) return;
    {
        std::lock_guard<std::mutex> locked(mutex_);
        deferred = defers(ref, on_driver);
    }
    Cell result;
    {
        PythonSection python;
        try {
            result.error =
                py_ref(hooks_.native_failure(ref.node, python_exception(*failure), py::list()));
        } catch (py::error_already_set &again) {
            result.error = current_error(again);
        }
    }
    result.deferred = deferred;  // its start replayed by the commit path's take
    store(ref, std::move(result));
}

/* The memory a memo entry keeps alive: each storage its images use once, whole (a view, e.g. a full-width crop,
 * keeps its parent's block), and its Python values' buffers, a view's whole base (hooks.memo_bytes). */
int64_t Run::memo_bytes(const std::vector<Value> &outputs)
{
    std::set<const ImageStorage *> storages;
    int64_t bytes = 0;
    std::vector<PyRef> python;
    for (const Value &value : outputs) {
        if (const Image *image = std::get_if<Image>(&value)) {
            if (!storages.insert(image->storage.get()).second) continue;
            if (image->storage->owner().handle)
                python.push_back(image->storage->owner());
            else
                bytes += int64_t(image->storage->bytes());
        } else if (const PyRef *object = std::get_if<PyRef>(&value)) {
            python.push_back(*object);
        } else if (const auto *text = std::get_if<std::string>(&value)) {
            bytes += int64_t(text->capacity()) + 1;
        }
    }
    if (python.empty()) return bytes;
    PythonSection section;
    py::list values;
    for (const PyRef &object : python) values.append(py::handle(static_cast<PyObject *>(object.handle.get())));
    return bytes + hooks_.memo_bytes(values).cast<int64_t>();
}

void Run::store(const CellRef &ref, Cell &&result)
{
    {
        std::lock_guard<std::mutex> locked(mutex_);
        Cell &c = cell(ref);
        if (result.error.handle) {
            c.state = State::failed;
            c.error = std::move(result.error);
        } else {
            c.state = State::done;
            c.outputs = std::move(result.outputs);
            c.object = std::move(result.object);
            c.inputs = std::move(result.inputs);
            c.bytes = result.bytes;
            if (result.key_state == Cell::KeyState::known) {
                c.key = result.key;
                c.key_state = result.key_state;
            }
            c.items = std::move(result.items);
            c.items_context = std::move(result.items_context);
            c.items_inputs = std::move(result.items_inputs);
            c.hits = nodes_[size_t(ref.node)].kind == Kind::collector ? 0 : c.hits;
        }
        c.deferred = result.deferred;
        c.deferred_ok = result.deferred_ok;
        c.deferred_seconds = result.deferred_seconds;
        c.deferred_outputs = std::move(result.deferred_outputs);
        c.wanted = false;
        dispatch();
    }
    changed_.notify_all();
}

// Events

void Run::emit(Event event)
{
    {
        std::lock_guard<std::mutex> locked(events_mutex_);
        events_.push_back(std::move(event));
    }
    events_changed_.notify_all();
}

py::object Run::event_dict(const Event &event)
{
    py::dict data;
    const char *name = "";
    switch (event.type) {
    case Event::chain_start: {
        name = "chain-start";
        py::list ids;
        for (const std::string &id : chain_start_) ids.append(id);
        data["nodes"] = ids;
        break;
    }
    case Event::start:
        name = "node-start";
        data["nodeId"] = nodes_[size_t(event.node)].id;
        break;
    case Event::finish:
        name = "node-finish";
        data["nodeId"] = nodes_[size_t(event.node)].id;
        data["executionTime"] = event.time;
        break;
    case Event::progress:
        name = "node-progress";
        data["nodeId"] = nodes_[size_t(event.node)].id;
        data["progress"] = event.progress_int ? py::object(py::int_(1)) : py::object(py::float_(event.progress_value));
        data["index"] = event.index;
        data["total"] = event.total;
        data["eta"] = event.eta_int ? py::object(py::int_(0)) : py::object(py::float_(event.eta));
        break;
    case Event::raw:
        return py::reinterpret_borrow<py::object>(static_cast<PyObject *>(event.payload.handle.get()));
    }
    py::dict result;
    result["event"] = name;
    result["data"] = data;
    return result;
}

void Run::pump()
{
    for (;;) {
        std::vector<Event> batch;
        bool stop;
        {
            std::unique_lock<std::mutex> locked(events_mutex_);
            pumping_ = false;
            events_changed_.notify_all();
            events_changed_.wait(locked, [&] { return events_stop_ || !events_.empty(); });
            batch.swap(events_);
            stop = events_stop_;
            pumping_ = !batch.empty();
        }
        if (!batch.empty()) {
            PythonSection section;
            try {
                py::list list;
                for (const Event &event : batch) list.append(event_dict(event));
                batch.clear();
                hooks_.post(list);
            } catch (const std::exception &) {
                report_unraisable("chaiNNer-C engine events");
            }
        }
        if (stop) return;
    }
}

/* Returns once every event emitted so far is on the loop's queue of callbacks (before the run's result). */
void Run::flush_events()
{
    std::unique_lock<std::mutex> locked(events_mutex_);
    events_changed_.wait(locked, [&] { return (events_.empty() && !pumping_) || events_failed_; });
}

void Run::submit_broadcast(int node, const std::vector<Value> &outputs)
{
    const NodePlan &plan = nodes_[size_t(node)];
    if (!send_broadcast_data_ || plan.outputs.empty()) return;
    // A /run's LargeImage previews of an item (an iterated node's): rate-limited (hunt-misc 6, ruling R4).
    bool limited = mode_ == "run" && preview_interval_ > Clock::duration::zero() && iterated_[size_t(node)] &&
                   std::any_of(plan.outputs.begin(), plan.outputs.end(), [](const OutputPlan &o) { return o.large; });
    {
        std::lock_guard<std::mutex> locked(broadcasts_mutex_);
        auto [job, inserted] = broadcast_jobs_.insert_or_assign(node, BroadcastJob{node, outputs, limited});
        if (inserted) broadcast_order_.push_back(node);
    }
    broadcasts_changed_.notify_all();
}

/* A native image's broadcast (ImageOutput.get_broadcast_data and get_broadcast_type), computed without the GIL;
 * LargeImageOutput's previews through image_preview.cpp, the same bytes as today's Python path. */
struct ImageBroadcast {
    size_t position;
    Image image;
    std::vector<cn::image_preview::Preview> previews;
    bool large;
    PyRef cached;            // its previews' Python list, already encoded (PreviewCache)
    bool cacheable = false;  // pooled storage: its new previews are kept
};

/* LargeImage previews already encoded, by image (hunt-misc 8): an unchanged engine image (a memo hit, a generator's
 * static outputs again, a node passing its input through) is not encoded again. Keyed by the image's view of its
 * storage, pooled storage only (immutable once published), held weakly: an entry never keeps an image alive, and a
 * freed storage's address reused by another never matches. The previews' Python list is kept, so its URLs are not
 * copied again either. An entry whose image is freed is dropped at the next lookup of it, insert or stats(); else
 * least recently used first out within 64 MiB of URLs and 64 entries; memo_clear() drops every entry too. */
class PreviewCache {
public:
    static PreviewCache &get()
    {
        static PreviewCache *cache = new PreviewCache();
        return *cache;
    }
    PyRef find(const Image &image)
    {
        std::vector<PyRef> dropped;  // released outside the lock
        std::lock_guard<std::mutex> locked(mutex_);
        auto found = entries_.find(key(image));
        if (found == entries_.end() || found->second.storage.lock() != image.storage) {
            if (found != entries_.end()) drop(found, dropped);  // its image is gone (another at its address)
            ++misses_;
            return {};
        }
        order_.splice(order_.end(), order_, found->second.position);
        ++hits_;
        return found->second.previews;
    }
    void insert(const Image &image, PyRef previews, size_t bytes)
    {
        std::vector<PyRef> dropped;  // released outside the lock
        std::lock_guard<std::mutex> locked(mutex_);
        if (bytes > budget_ || entries_.count(key(image))) return;
        drop_expired(dropped);
        while ((bytes_ + bytes > budget_ || entries_.size() >= capacity_) && !order_.empty())
            drop(entries_.find(order_.front()), dropped);
        bytes_ += bytes;
        order_.push_back(key(image));
        entries_.emplace(key(image), Entry{image.storage, std::move(previews), bytes, std::prev(order_.end())});
    }
    py::dict stats()
    {
        std::vector<PyRef> dropped;  // released outside the lock
        std::lock_guard<std::mutex> locked(mutex_);
        drop_expired(dropped);
        py::dict result;
        result["entries"] = entries_.size();
        result["bytes"] = bytes_;
        result["hits"] = hits_;
        result["misses"] = misses_;
        return result;
    }
    void clear()
    {
        std::map<Id, Entry> released;  // released outside the lock
        std::lock_guard<std::mutex> locked(mutex_);
        released.swap(entries_);
        order_.clear();
        bytes_ = 0;
    }

private:
    using Id = std::tuple<const ImageStorage *, const float *, int32_t, int32_t, int32_t>;
    static Id key(const Image &image)
    {
        return {image.storage.get(), image.data, image.height, image.width, image.channels};
    }
    struct Entry {
        std::weak_ptr<const ImageStorage> storage;
        PyRef previews;
        size_t bytes;
        std::list<Id>::iterator position;
    };
    /* mutex_ held; the previews released by the caller, outside it. */
    void drop(std::map<Id, Entry>::iterator victim, std::vector<PyRef> &dropped)
    {
        bytes_ -= victim->second.bytes;
        dropped.push_back(std::move(victim->second.previews));
        order_.erase(victim->second.position);
        entries_.erase(victim);
    }
    void drop_expired(std::vector<PyRef> &dropped)
    {
        for (auto it = entries_.begin(); it != entries_.end();) {
            auto current = it++;
            if (current->second.storage.expired()) drop(current, dropped);
        }
    }

    std::mutex mutex_;
    std::map<Id, Entry> entries_;
    std::list<Id> order_;  // least recently used first
    size_t budget_ = size_t(64) << 20, capacity_ = 64, bytes_ = 0;
    uint64_t hits_ = 0, misses_ = 0;
};

void Run::broadcast_now(int node, const std::vector<Value> &outputs, const PyRef &generator)
{
    const NodePlan &plan = nodes_[size_t(node)];
    if (!send_broadcast_data_ || plan.outputs.empty() || aborted_) return;
    std::vector<ImageBroadcast> images;
    try {
        for (size_t i = 0; i < outputs.size() && i < plan.outputs.size(); ++i) {
            const Image *image = std::get_if<Image>(&outputs[i]);
            if (!plan.outputs[i].image || !image || image->height <= 0 || image->width <= 0) continue;
            ImageBroadcast broadcast{i, *image, {}, plan.outputs[i].large, PyRef{}, false};
            if (broadcast.large) {
                // Previews the native encoder made, while it is on (CHAINNER_C_NATIVE_CODECS, read per call).
                bool pooled = !image->storage->owner().handle && cn::image_decode::enabled();
                if (pooled) broadcast.cached = PreviewCache::get().find(*image);
                // A preview the native encoder declines stays Python's (numpy_outputs._previews).
                if (!broadcast.cached.handle &&
                    !(cn::image_decode::enabled() &&
                      cn::image_preview::previews(image->data, uint32_t(image->height), uint32_t(image->width),
                                                  uint32_t(image->channels), broadcast.previews)))
                    continue;
                broadcast.cacheable = pooled && !broadcast.cached.handle;
            }
            images.push_back(std::move(broadcast));
        }
    } catch (const std::exception &) {
        images.clear();  // Python computes every broadcast (the same data)
    }
    if (aborted_) return;
    PythonSection python;
    try {
        py::list values;
        py::dict native;
        for (const Value &value : outputs) values.append(to_python(value));
        for (const ImageBroadcast &image : images) {
            values[image.position] = py::none();
            py::dict data;
            if (image.cached.handle) {
                data["previews"] = py::handle(static_cast<PyObject *>(image.cached.handle.get()));
            } else if (image.large) {
                py::list previews;
                size_t bytes = 0;
                for (const auto &preview : image.previews) {
                    py::dict entry;
                    entry["width"] = preview.width;
                    entry["height"] = preview.height;
                    entry["url"] = hooks_.preview_url(py::str(preview.url));  // events.PreviewUrl: spliced, not escaped
                    previews.append(entry);
                    bytes += preview.url.size();
                }
                data["previews"] = previews;
                if (image.cacheable) PreviewCache::get().insert(image.image, py_ref(previews), bytes);
            }
            data["height"] = image.image.height;
            data["width"] = image.image.width;
            data["channels"] = image.image.channels;
            py::dict fields;
            fields["width"] = image.image.width;
            fields["height"] = image.image.height;
            fields["channels"] = image.image.channels;
            py::dict type;
            type["type"] = "named";
            type["name"] = "Image";
            type["fields"] = fields;
            native[py::int_(image.position)] = py::make_tuple(data, type);
        }
        py::object generator_object =
            generator.handle ? py::reinterpret_borrow<py::object>(static_cast<PyObject *>(generator.handle.get())) : py::none();
        py::object event = hooks_.broadcast(node, values, generator_object, native);
        if (!aborted_) {
            Event raw(Event::raw, node);
            raw.payload = py_ref(event);
            emit(std::move(raw));
        }
    } catch (const std::exception &) {
        report_unraisable("chaiNNer-C engine broadcast");
    }
}

/* Sends the pending broadcasts in submission order, one at a time. A limited one (a loop's LargeImage previews)
 * waits until preview_interval_ after its node's previous start, the newest kept meanwhile; the first of a run, a
 * flush (the loop's or run's end) and the run's close send at once. */
void Run::broadcaster()
{
    for (;;) {
        BroadcastJob job;
        {
            std::unique_lock<std::mutex> locked(broadcasts_mutex_);
            for (;;) {
                if (broadcasts_stop_ && broadcast_order_.empty()) return;
                Clock::time_point now = Clock::now();
                std::optional<Clock::time_point> wake;
                auto due = std::find_if(broadcast_order_.begin(), broadcast_order_.end(), [&](int node) {
                    if (!broadcast_jobs_.at(node).limited || flushing_ > 0 || broadcasts_stop_) return true;
                    auto started = preview_started_.find(node);
                    if (started == preview_started_.end() || started->second + preview_interval_ <= now) return true;
                    Clock::time_point ready = started->second + preview_interval_;
                    wake = wake ? std::min(*wake, ready) : ready;
                    return false;
                });
                if (due != broadcast_order_.end()) {
                    int node = *due;
                    broadcast_order_.erase(due);
                    job = std::move(broadcast_jobs_.at(node));
                    broadcast_jobs_.erase(node);
                    if (job.limited) preview_started_[node] = now;
                    break;
                }
                if (wake)
                    broadcasts_changed_.wait_until(locked, *wake);
                else
                    broadcasts_changed_.wait(locked);
            }
            broadcasting_ = true;
        }
        broadcast_now(job.node, job.outputs, PyRef{});
        job.outputs.clear();
        {
            std::lock_guard<std::mutex> locked(broadcasts_mutex_);
            broadcasting_ = false;
        }
        broadcasts_changed_.notify_all();
    }
}

void Run::flush_broadcasts()
{
    std::unique_lock<std::mutex> locked(broadcasts_mutex_);
    ++flushing_;
    broadcasts_changed_.notify_all();
    broadcasts_changed_.wait(locked, [&] { return (broadcast_order_.empty() && !broadcasting_) || broadcasts_failed_; });
    --flushing_;
}


Engine::Engine()
{
    unsigned cpus = std::max(1u, std::thread::hardware_concurrency());
#ifdef _WIN32
    DWORD_PTR process_mask = 0, system_mask = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) && process_mask)
        cpus = unsigned(std::popcount(uint64_t(process_mask)));
#else
    if (unsigned affinity = cn_affinity_cpus()) cpus = affinity;
#endif
    workers_ = environment_count("CHAINNER_C_ENGINE_THREADS", int(std::max(2u, cpus)));
    idle_time_ = std::chrono::seconds(environment_count("CHAINNER_C_ENGINE_IDLE_SECONDS", 30));
    for (int i = 0; i < workers_; ++i) std::thread([this] { work(); }).detach();
}

void Engine::attach(Run *run)
{
    std::lock_guard<std::mutex> locked(mutex_);
    active_[run] = 0;
}

void Engine::offer(Run *run, size_t jobs)
{
    if (jobs == 0) return;
    {
        std::lock_guard<std::mutex> locked(mutex_);
        if (active_.find(run) == active_.end()) return;
        tokens_.insert(tokens_.end(), jobs, run);
    }
    work_.notify_all();
}

void Engine::detach(Run *run)
{
    std::unique_lock<std::mutex> locked(mutex_);
    std::erase(tokens_, run);
    idle_.wait(locked, [&] { return active_[run] == 0; });
    // A worker still inside the run may have offered tokens meanwhile: none may outlive the detach.
    std::erase(tokens_, run);
    active_.erase(run);
}

std::future<void> Engine::service(std::function<void()> task)
{
    std::packaged_task<void()> job(std::move(task));
    std::future<void> done = job.get_future();
    bool spawn;
    {
        std::lock_guard<std::mutex> locked(mutex_);
        services_.push_back(std::move(job));
        spawn = idle_services_ < int(services_.size());
        if (spawn) ++idle_services_;  // counted idle until it takes its first service
    }
    if (spawn) std::thread([this] { serve(); }).detach();
    services_changed_.notify_one();
    return done;
}

void Engine::work()
{
    PythonThread python;
    for (;;) {
        Run *run;
        {
            std::unique_lock<std::mutex> locked(mutex_);
            work_.wait(locked, [&] { return !tokens_.empty(); });
            run = tokens_.front();
            tokens_.pop_front();
            ++active_[run];
        }
        try {
            run->work_one();
        } catch (...) {
            run->fault(std::current_exception());  // the job's bookkeeping failed (an engine fault, an allocation)
        }
        {
            std::lock_guard<std::mutex> locked(mutex_);
            if (--active_[run] == 0) idle_.notify_all();
        }
    }
}

void Engine::serve()
{
    PythonThread python;
    for (;;) {
        std::packaged_task<void()> job;
        {
            std::unique_lock<std::mutex> locked(mutex_);
            if (!services_changed_.wait_for(locked, idle_time_, [&] { return !services_.empty(); })) {
                --idle_services_;
                return;
            }
            job = std::move(services_.front());
            services_.pop_front();
            --idle_services_;
        }
        job();
        std::lock_guard<std::mutex> locked(mutex_);
        ++idle_services_;
    }
}

}  // namespace
}  // namespace cn::run

void cn_bind_run_executor(py::module_ &module)
{
    using cn::run::Run;
    py::class_<Run>(module, "RunEngine")
        .def(py::init<const py::dict &, py::object>(), py::arg("plan"), py::arg("hooks"))
        .def("start", &Run::start, py::arg("mode"), py::arg("target"))
        .def("kill", &Run::kill, py::call_guard<py::gil_scoped_release>())
        .def("pause", &Run::pause, py::call_guard<py::gil_scoped_release>())
        .def("resume", &Run::resume, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("aborted", &Run::aborted)
        .def_property_readonly("paused", &Run::paused)
        .def_property_readonly("commit_slot", &Run::commit_slot)
        .def("flush_broadcasts", &Run::flush_broadcasts, py::call_guard<py::gil_scoped_release>())
        .def("close", &Run::close, py::call_guard<py::gil_scoped_release>())
        .def("lookahead", &Run::lookahead)
        .def("peek", &Run::peek, py::arg("node"), py::arg("input"), py::arg("offset"), py::arg("origin"));
    module.def("memo_stats", [] { return cn::run::Memo::get().stats(); },
               "The memo cache's entries, bytes, budget, hits and misses (amendment A3).");
    module.def("memo_set_budget", [](int64_t budget) { return cn::run::Memo::get().set_budget(budget); },
               py::arg("budget"),
               "Sets the memo cache's budget in bytes (0: off), trimming it, unless CHAINNER_C_MEMO_BUDGET_MIB set "
               "one; returns the budget in effect.");
    module.def("memo_clear", [] {
        cn::run::Memo::get().clear();
        cn::run::PreviewCache::get().clear();
    }, "Drops every memo cache entry, and every preview the preview cache keeps.");
    module.def("preview_cache_stats", [] { return cn::run::PreviewCache::get().stats(); },
               "The LargeImage preview cache's entries, bytes (of URLs), hits and misses (hunt-misc 8).");
    module.def("python_threads", [] { return cn::run::python_threads(); },
               "The engine threads allowed in Python at once (CHAINNER_C_PYTHON_THREADS, at most 1024).");
    module.def("native_declines", [] {
        py::dict counts;
        std::lock_guard<std::mutex> locked(cn::run::decline_mutex);
        for (const auto &[schema, count] : cn::run::decline_counts()) counts[py::str(schema)] = count;
        return counts;
    }, "The native entry points' declines to their Python bodies (RunPython), per schema id, since the process started.");
}
