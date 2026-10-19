/* The native run engine (native/local/instant-exec/DESIGN.md section 2, amendments A2 and A6), behind
 * backend/src/native_executor.py.
 *
 * A run has one driver thread, which follows process.Executor's control flow: the generator groups one after
 * another (each generator run once, its collectors started, then the items in order: the generators' advance, the
 * item's side-effect nodes, its collectors, its progress), then the non-iterated side-effect nodes. A node that may
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
#include "kernel_status.hpp"

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
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <set>
#include <thread>
#include <unordered_map>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

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
    char *value = nullptr;
    size_t size = 0;
    if (_dupenv_s(&value, &size, name) != 0 || !value) return fallback;
    std::string text(value);
    free(value);
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

/* At most CHAINNER_C_PYTHON_THREADS engine threads in Python (A2): a slot, then the GIL; per thread, nested sections
 * share the outer slot. */
std::counting_semaphore<1024> &python_slots()
{
    static std::counting_semaphore<1024> slots(environment_count("CHAINNER_C_PYTHON_THREADS", 8));
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
};

struct OutputPlan {
    bool image = false;  // ImageOutput or LargeImageOutput: the engine enforces images
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
    const Entry *native = nullptr;
    std::map<std::string, Value, std::less<>> settings;
    bool precached = false;
    std::vector<Value> precached_outputs;
    bool phased = false;  // registered prepare and commit phases (item_window.register_phases)
};

struct GeneratorPlan {
    int node;
    std::vector<int> collectors;
};

struct GroupPlan {
    std::vector<GeneratorPlan> generators;
    std::vector<int> outputs;
    std::vector<int> iterated;
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
    int64_t bytes = 0;  // its outputs' array bytes, measured at slot 0 (the window's per-item estimate)
    // A window job's events, sent when the commit path takes it (process.ItemWindow.take_node's replay).
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
    py::object install_thread, post, finish, warn, info, run_node, enforce_input, enforce_output, native_failure,
        regular_outputs, generator_outputs, collector, node_cleanups, chain_cleanups, output_bytes,
        broadcast, items, describe, materialize, collector_iterate, collector_complete, prepare, begin_loop, iteration,
        end_loop, window, warn_look_ahead_off, aborted_type;
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

struct GeneratorRun {
    int node = -1;
    PyRef output, iterator;
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
    PyRef token;
};

/* One generator group's loop and its item window (process.Executor's _LoopRun and ItemWindow). */
struct Loop {
    const GroupPlan *group = nullptr;
    std::vector<GeneratorRun> *generators = nullptr;
    std::vector<CollectorRun> *collectors = nullptr;
    std::vector<int> sinks;  // the group's side-effect nodes, then its collectors
    bool decided = false, active = false, halted = false, stop = false;
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
    std::thread producer;

    int generator_index(int node) const
    {
        for (size_t g = 0; g < generators->size(); ++g)
            if ((*generators)[g].node == node) return int(g);
        return -1;
    }
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
    py::object peek(int node, int input, int64_t offset);

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
    bool workers_stop_ = false;
    bool dropping_ = false;  // the run failed: queued pure cells are not started
    std::vector<std::thread> workers_;

    // Run state
    std::atomic<bool> aborted_{false}, paused_{false};
    std::atomic<int64_t> commit_slot_{0};
    double time_paused_ = 0;
    std::string mode_;
    int target_ = -1;
    std::thread driver_;
    bool started_ = false, closed_ = false;

    // Events
    std::mutex events_mutex_;
    std::condition_variable events_changed_;
    std::vector<Event> events_;
    bool events_stop_ = false, pumping_ = false;
    std::thread pump_;

    // Broadcasts (LatestBroadcasts): one pending job per node, sent one at a time in submission order
    struct BroadcastJob {
        int node;
        std::vector<Value> outputs;
    };
    std::mutex broadcasts_mutex_;
    std::condition_variable broadcasts_changed_;
    std::deque<int> broadcast_order_;
    std::unordered_map<int, BroadcastJob> broadcast_jobs_;
    bool broadcasting_ = false, broadcasts_stop_ = false;
    std::thread broadcaster_;

    Cell &cell(const CellRef &ref);
    CellRef source_of(const CellRef &ref, int input) const;
    bool settled(const Cell &cell) const
    {
        return cell.state == State::done || cell.state == State::failed || cell.state == State::released;
    }

    // Driver
    void drive();
    void suspend();
    void ensure(const CellRef &ref);
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
    void set_item(int node, int64_t slot, std::vector<Value> outputs);
    void want_window(int64_t slot);
    void run_task(const Task &task);
    PyRef take_prepared(int node, int64_t slot);
    void decide();
    Outcome take_advance(int64_t slot, size_t generator);
    bool ahead(int node);
    bool servable(int node, int input);
    void decide_early();
    void take(const CellRef &ref);
    void release_slot(int64_t slot);
    void finish_run(PyRef error, PyRef result);
    void drain_workers();

    // Execution
    void worker();
    void execute(const CellRef &ref, bool on_driver);
    struct Gathered {
        std::vector<Value> values;               // inputs as the node receives them (lazy: an EngineLazy)
        std::vector<std::shared_ptr<EngineLazy>> lazies;
        bool defer = false;                      // a window job: its events wait for the commit path
    };
    Gathered gather(const CellRef &ref);
    bool run_python(const CellRef &ref, const Gathered &inputs, Cell &result, double &lazy_seconds,
                    const PyRef &prepared);
    NativeOutcome run_native(const CellRef &ref, const Gathered &inputs, Cell &result, double &lazy_seconds);
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
        : run_(run), reader_(reader), input_(input), source_(source), output_(output)
    {
    }

    const Value &value() override
    {
        const Value &upstream = raw();
        if (!enforced_) {
            value_ = run_.enforce_input(reader_.node, input_, upstream);
            enforced_ = true;
        }
        return value_;
    }

    /* The upstream value for a Python reader; Stop raises progress_controller's Aborted, an upstream failure its
     * error. */
    py::object python_value()
    {
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
    return to_value(object, ImageMode::enforced);
}

Run::Run(const py::dict &plan, py::object hooks_object) : hooks_object_(py_ref(hooks_object))
{
    auto bind = [&](const char *name) { return hooks_object.attr(name); };
    hooks_ = Hooks{bind("install_thread"), bind("post"), bind("finish"), bind("warn"), bind("info"), bind("run_node"),
                   bind("enforce_input"), bind("enforce_output"), bind("native_failure"),
                   bind("regular_outputs"), bind("generator_outputs"), bind("collector"), bind("node_cleanups"),
                   bind("chain_cleanups"), bind("output_bytes"), bind("broadcast"), bind("items"), bind("describe"),
                   bind("materialize"), bind("collector_iterate"), bind("collector_complete"), bind("prepare"),
                   bind("begin_loop"), bind("iteration"), bind("end_loop"), bind("window"),
                   bind("warn_look_ahead_off"), bind("aborted_type")};
    send_broadcast_data_ = plan["send_broadcast_data"].cast<bool>();
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
        bool native = node["native"].cast<bool>() && plan_node.kind == Kind::regular;
        if (native) {
            const Entry *entry = find_entry(plan_node.schema);
            if (entry && entry->run) plan_node.native = entry;
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
            plan_node.inputs.push_back(std::move(plan_input));
        }
        for (py::handle output_handle : node["image_outputs"]) {
            OutputPlan output;
            if (!output_handle.is_none()) {
                py::tuple description = py::reinterpret_borrow<py::tuple>(output_handle);
                output.image = true;
                output.channels = description[0].cast<int>();
                output.assume_normalized = description[1].cast<bool>();
            }
            plan_node.outputs.push_back(output);
        }
        for (py::handle index : node["iterable_outputs"]) plan_node.iterable_outputs.push_back(index.cast<int>());
        if (plan_node.native) {
            for (auto [key, value] : py::reinterpret_borrow<py::dict>(node["settings"]))
                plan_node.settings.emplace(py::str(key), value_of(value));
        }
        plan_node.phased = node["phased"].cast<bool>();
        py::object precached = node["precached"];
        if (!precached.is_none()) {
            plan_node.precached = true;
            for (py::handle value : precached) plan_node.precached_outputs.push_back(py_ref(value));
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

Cell &Run::cell(const CellRef &ref)
{
    if (ref.slot >= 0 && iterated_[size_t(ref.node)]) return slot_cells_[ref.slot][size_t(ref.node)];
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
    if (started_) throw std::logic_error("A RunEngine starts once");
    started_ = true;
    mode_ = mode;
    target_ = target;
    py::gil_scoped_release unlocked;
    unsigned cpus = std::max(1u, std::thread::hardware_concurrency());
    DWORD_PTR process_mask = 0, system_mask = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) && process_mask)
        cpus = unsigned(std::popcount(uint64_t(process_mask)));
    unsigned count = unsigned(environment_count("CHAINNER_C_ENGINE_THREADS", int(std::max(2u, cpus))));
    for (unsigned i = 0; i < count; ++i) workers_.emplace_back([this] { worker(); });
    pump_ = std::thread([this] { pump(); });
    broadcaster_ = std::thread([this] { broadcaster(); });
    driver_ = std::thread([this] { drive(); });
}

void Run::kill()
{
    aborted_ = true;
    {
        std::lock_guard<std::mutex> locked(mutex_);
    }
    changed_.notify_all();
}

void Run::pause() { paused_ = true; }

void Run::resume()
{
    paused_ = false;
    {
        std::lock_guard<std::mutex> locked(mutex_);
    }
    changed_.notify_all();
}

void Run::close()
{
    if (closed_ || !started_) return;
    closed_ = true;
    if (driver_.joinable()) driver_.join();
    flush_broadcasts();
    {
        std::lock_guard<std::mutex> locked(mutex_);
        workers_stop_ = true;
    }
    changed_.notify_all();
    for (std::thread &worker : workers_) worker.join();
    {
        std::lock_guard<std::mutex> locked(broadcasts_mutex_);
        broadcasts_stop_ = true;
    }
    broadcasts_changed_.notify_all();
    broadcaster_.join();
    {
        std::lock_guard<std::mutex> locked(events_mutex_);
        events_stop_ = true;
    }
    events_changed_.notify_all();
    pump_.join();
}

/* Python state for an engine thread: a thread state kept for its life, its NumPy handler installed. */
class ThreadPython {
public:
    explicit ThreadPython(const py::object &install)
    {
        state_ = PyGILState_Ensure();
        try {
            install();
        } catch (py::error_already_set &error) {
            error.discard_as_unraisable("chaiNNer-C engine thread setup");
        }
        saved_ = PyEval_SaveThread();
    }
    ~ThreadPython()
    {
        PyEval_RestoreThread(saved_);
        release_deferred();
        PyGILState_Release(state_);
    }
    ThreadPython(const ThreadPython &) = delete;
    ThreadPython &operator=(const ThreadPython &) = delete;

private:
    PyGILState_STATE state_;
    PyThreadState *saved_;
};

void Run::worker()
{
    ThreadPython python(hooks_.install_thread);
    for (;;) {
        CellRef ref{};
        {
            std::unique_lock<std::mutex> locked(mutex_);
            changed_.wait(locked, [&] { return workers_stop_ || ((!ready_.empty() || !tasks_.empty()) && !paused_); });
            if (ready_.empty() && tasks_.empty()) return;
            if (!tasks_.empty()) {
                Task task = std::move(tasks_.front());
                tasks_.pop_front();
                ++running_;
                ++running_slots_;
                locked.unlock();
                run_task(task);
                locked.lock();
                --running_;
                --running_slots_;
                dispatch();
                changed_.notify_all();
                continue;
            }
            ref = ready_.front();
            ready_.pop_front();
            Cell &c = cell(ref);
            if (aborted_ || dropping_) {
                c.state = State::idle;
                c.wanted = false;
                continue;
            }
            c.state = State::running;
            ++running_;
            if (ref.slot >= 0) ++running_slots_;
        }
        execute(ref, false);
        {
            std::lock_guard<std::mutex> locked(mutex_);
            --running_;
            if (ref.slot >= 0) --running_slots_;
            dispatch();
        }
        changed_.notify_all();
    }
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
    if (!node.pure) return;
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
        if (ready && !aborted_ && !dropping_) {
            c.state = State::queued;
            ready_.push_back(ref);
            queued = true;
            wanted_[w] = wanted_.back();
            wanted_.pop_back();
            continue;
        }
        ++w;
    }
    if (loop_ && loop_->active && !aborted_ && !dropping_) {
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
            tasks_.push_back(Task{Task::prepare, slot, sink, {}});
            queued = true;
        }
    }
    if (queued) changed_.notify_all();
}

void Run::ensure(const CellRef &ref)
{
    bool done;
    {
        std::lock_guard<std::mutex> locked(mutex_);
        done = settled(cell(ref));
    }
    if (done) {
        take(ref);
        return;
    }
    const NodePlan &node = nodes_[size_t(ref.node)];
    if (node.pure) {
        std::lock_guard<std::mutex> locked(mutex_);
        want(ref);
        dispatch();
    }
    for (size_t i = 0; i < node.inputs.size(); ++i) {
        const InputPlan &input = node.inputs[i];
        if (input.source < 0 || input.lazy || input.ignored) continue;
        CellRef source = source_of(ref, int(i));
        ensure(source);
        std::lock_guard<std::mutex> locked(mutex_);
        Cell &from = cell(source);
        if (from.state == State::failed) {
            Cell &c = cell(ref);
            if (c.state == State::idle) {
                c.state = State::failed;
                c.error = from.error;
                c.wanted = false;
            }
            changed_.notify_all();
            return;
        }
    }
    if (node.pure) {
        std::unique_lock<std::mutex> locked(mutex_);
        dispatch();
        changed_.wait(locked, [&] {
            const Cell &c = cell(ref);
            return settled(c) || (aborted_ && c.state != State::running);
        });
        Cell &c = cell(ref);
        if (!settled(c)) throw StopRun{};
        locked.unlock();
        take(ref);
        return;
    }
    {
        std::lock_guard<std::mutex> locked(mutex_);
        cell(ref).state = State::running;
    }
    execute(ref, true);
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

/* Whether the producer may start its next slot: within the window (K slots from the committing one), within the
 * memory budget unless a look-ahead waits for it, not paused or stopped (process.Executor's _may_admit). */
bool Run::admits(const Loop &loop) const
{
    if (loop.halted || aborted_ || paused_) return false;
    int64_t commit = commit_slot_;
    if (loop.produced >= commit + loop.k) return false;
    if (loop.produced <= loop.demand) return true;
    return (loop.produced - commit) * loop.estimate <= loop.budget;
}

/* The item window's producer: each slot's generators described in order (Python's iterator is serial), a single
 * generator's item materialized by a worker (its load and enforce), zipped generators advanced whole here; a slot
 * stops at its first end or error, as the loop's advance does. */
void Run::produce()
{
    ThreadPython python(hooks_.install_thread);
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
            std::vector<Value> outputs;
            {
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
                        token = py_ref(described);
                    }
                } catch (py::error_already_set &error) {
                    outcome.kind = Outcome::failed;
                    outcome.error = current_error(error);
                }
            }
            Outcome::Kind kind = outcome.kind;
            {
                std::lock_guard<std::mutex> locked(mutex_);
                if (kind == Outcome::ready) set_item(generator.node, slot, std::move(outputs));
                if (kind == Outcome::ended && generators.size() == 1) loop.halted = true;
                if (kind == Outcome::failed && generator.fail_fast) loop.halted = true;
                if (token.handle) {
                    ++loop.materializing;
                    tasks_.push_back(Task{Task::materialize, slot, int(g), token});
                }
                loop.outcomes[slot].push_back(std::move(outcome));
            }
            changed_.notify_all();
            if (kind == Outcome::ended || kind == Outcome::failed) break;
        }
    }
}

/* A generator's item for a slot (mutex_ held): its cell done, the slot's window work wanted. */
void Run::set_item(int node, int64_t slot, std::vector<Value> outputs)
{
    Cell &c = cell({node, slot});
    c.outputs = std::move(outputs);
    c.state = State::done;
    c.hits = -1;
    if (loop_ && loop_->active) want_window(slot);
    dispatch();
}

/* The pure nodes the slot's sinks need, through lazy inputs too (process.Executor's _ItemPlanner candidates), and
 * the prepare phases of its sinks (mutex_ held). */
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
        {
            PythonSection section;
            try {
                py::list values = hooks_.materialize(
                    generator.node, py::handle(static_cast<PyObject *>(generator.iterator.handle.get())),
                    py::handle(static_cast<PyObject *>(generator.output.handle.get())),
                    py::handle(static_cast<PyObject *>(task.token.handle.get())), generator.split);
                for (py::handle value : values) outputs.push_back(py_ref(value));
                outcome.kind = Outcome::ready;
            } catch (py::error_already_set &error) {
                outcome.kind = Outcome::failed;
                outcome.error = current_error(error);
            }
        }
        std::lock_guard<std::mutex> locked(mutex_);
        --loop.materializing;
        if (outcome.kind == Outcome::ready) set_item(generator.node, task.slot, std::move(outputs));
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
    if (!loop_ || !loop_->active) return {};
    auto found = loop_->prepares.find({node, slot});
    if (found == loop_->prepares.end()) return {};
    if (found->second.kind == Prepare::waiting) {
        loop_->prepares.erase(found);
        return {};
    }
    changed_.wait(locked, [&] { return found->second.kind == Prepare::done; });
    PyRef value = std::move(found->second.value);
    loop_->prepares.erase(found);
    return value;
}

/* The window's decision (process.Executor.__open_window): K, the read-ahead blocker and the budget, from the
 * hooks; a window of K > 1 starts its producer at slot 1 (slot 0 is the loop's own). */
void Run::decide()
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
    loop.produced = 1;
    loop.producer = std::thread([this] { produce(); });
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

/* process.Executor's _ItemPlanner.ahead: a node a look-ahead may wait for at a later item (memoized per loop). */
bool Run::ahead(int node)
{
    Loop &loop = *loop_;
    auto memo = loop.ahead.find(node);
    if (memo != loop.ahead.end()) return memo->second;
    loop.ahead[node] = false;  // a cycle never qualifies
    const NodePlan &plan = nodes_[size_t(node)];
    bool result = iterated_[size_t(node)] && plan.kind == Kind::regular && !plan.side_effects && plan.pure &&
                  plan.schema != "chainner:utility:into_directory";
    for (size_t i = 0; result && i < plan.inputs.size(); ++i) {
        int source = plan.inputs[i].source;
        if (source < 0 || loop.generator_index(source) >= 0) continue;
        if (!iterated_[size_t(source)]) {
            std::lock_guard<std::mutex> locked(mutex_);
            result = static_cells_[size_t(source)].state == State::done;
        } else {
            result = ahead(source);
        }
    }
    loop.ahead[node] = result;
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

/* A look-ahead during slot 0 decides the window there (process.Executor.__decide_early). */
void Run::decide_early()
{
    if (!loop_->decided) decide();
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

py::object Run::peek(int node, int input, int64_t offset)
{
    if (!loop_) return py::none();
    decide_early();
    Loop &loop = *loop_;
    if (!loop.active || offset >= loop.k || !servable(node, input)) return py::none();
    if (aborted_) {
        PyErr_SetNone(hooks_.aborted_type.ptr());
        throw py::error_already_set();
    }
    const InputPlan &plan = nodes_[size_t(node)].inputs[size_t(input)];
    int64_t target = commit_slot_ + offset;
    Value value;
    bool found = false, stopped = false;
    {
        PythonPause unlocked;
        std::unique_lock<std::mutex> locked(mutex_);
        loop.demand = std::max(loop.demand, target);
        changed_.notify_all();
        auto resolved = [&] {
            auto outcome = loop.outcomes.find(target);
            if (outcome != loop.outcomes.end()) {
                for (const Outcome &o : outcome->second)
                    if (o.kind != Outcome::ready) return o.kind != Outcome::pending;
                return outcome->second.size() == loop.generators->size();
            }
            return loop.halted && loop.produced <= target;
        };
        changed_.wait(locked, [&] { return aborted_ || resolved(); });
        stopped = aborted_;
        auto outcome = loop.outcomes.find(target);
        bool items = !stopped && outcome != loop.outcomes.end() && outcome->second.size() == loop.generators->size() &&
                     std::all_of(outcome->second.begin(), outcome->second.end(),
                                 [](const Outcome &o) { return o.kind == Outcome::ready; });
        if (items) {
            CellRef source{plan.source, target};
            if (loop.generator_index(plan.source) < 0) {
                want(source);
                dispatch();
                changed_.wait(locked, [&] { return aborted_ || settled(cell(source)); });
                stopped = aborted_;
            }
            const Cell &c = cell(source);
            if (!stopped && c.state == State::done) {
                value = c.outputs[size_t(plan.output)];
                found = true;
            }
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
        submit_broadcast(ref.node, outputs);
        emit_finish(ref.node, seconds);
    }
}

void Run::iterate_group(const GroupPlan &group)
{
    suspend();
    std::vector<GeneratorRun> generators;
    std::vector<CollectorRun> collectors;
    std::set<int64_t> lengths;
    for (const GeneratorPlan &plan : group.generators) {
        GeneratorRun run;
        run.node = plan.node;
        run.output = process_generator(plan.node);
        {
            std::lock_guard<std::mutex> locked(mutex_);
            run.partial = static_cells_[size_t(plan.node)].outputs;
        }
        {
            PythonSection python;
            try {
                py::tuple items = hooks_.items(py::handle(static_cast<PyObject *>(run.output.handle.get())));
                run.iterator = py_ref(items[0]);
                run.split = items[1].cast<bool>();
                run.expected = items[2].cast<int64_t>();
                run.fail_fast = items[3].cast<bool>();
            } catch (py::error_already_set &error) {
                throw PythonError{current_error(error)};
            }
        }
        for (int collector : plan.collectors) {
            suspend();
            Clock::time_point start = Clock::now();
            CellRef ref{collector, -1};
            ensure(ref);
            std::lock_guard<std::mutex> locked(mutex_);
            Cell &c = cell(ref);
            if (c.state == State::failed) throw PythonError{c.error};
            collectors.push_back({collector, c.object, c.inputs, seconds_since(start)});
        }
        lengths.insert(run.expected);
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
        loop_ = std::make_unique<Loop>();
        loop_->group = &group;
        loop_->generators = &generators;
        loop_->collectors = &collectors;
        loop_->sinks = group.outputs;
        for (const CollectorRun &collector : collectors) loop_->sinks.push_back(collector.node);
    }
    {
        PythonSection python;
        py::list generator_nodes, iterated;
        for (const GeneratorRun &generator : generators) generator_nodes.append(generator.node);
        for (int node : group.iterated) iterated.append(node);
        hooks_.begin_loop(generator_nodes, iterated);
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
        if (loop_->producer.joinable()) loop_->producer.join();
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
        for (int64_t slot = 0;; ++slot) {
            {
                std::lock_guard<std::mutex> locked(mutex_);
                commit_slot_ = slot;
                if (slot == 0 || !loop_->active) open_slot(slot);
            }
            changed_.notify_all();
            if (slot == 1) decide();
            const GeneratorRun *current = nullptr;
            try {
                for (size_t g = 0; g < generators.size(); ++g) {
                    GeneratorRun &generator = generators[g];
                    suspend();
                    current = &generator;
                    std::vector<Value> outputs;
                    if (loop_->active && slot >= 1) {
                        Outcome outcome = take_advance(slot, g);
                        if (outcome.kind == Outcome::ended) throw ItemsEnd{};
                        if (outcome.kind == Outcome::failed) throw PythonError{outcome.error};
                        std::lock_guard<std::mutex> locked(mutex_);
                        outputs = cell({generator.node, slot}).outputs;
                    } else {
                        bool more;
                        {
                            PythonSection python;
                            try {
                                py::handle iterator(static_cast<PyObject *>(generator.iterator.handle.get()));
                                py::object token = hooks_.describe(iterator, generator.split);
                                more = !token.is_none();
                                if (more) {
                                    py::list values = hooks_.materialize(
                                        generator.node, iterator,
                                        py::handle(static_cast<PyObject *>(generator.output.handle.get())), token,
                                        generator.split);
                                    for (py::handle value : values) outputs.push_back(py_ref(value));
                                    if (slot == 0) loop_->estimate += hooks_.output_bytes(values).cast<int64_t>();
                                }
                            } catch (py::error_already_set &error) {
                                throw PythonError{current_error(error)};
                            }
                        }
                        if (!more) throw ItemsEnd{};
                        suspend();
                        {
                            std::lock_guard<std::mutex> locked(mutex_);
                            set_item(generator.node, slot, outputs);
                        }
                        changed_.notify_all();
                    }
                    submit_broadcast(generator.node, outputs);
                }
                {
                    std::lock_guard<std::mutex> locked(mutex_);
                    std::set<std::pair<int, int64_t>> seen;
                    for (int output : group.outputs) prefetch({output, slot}, seen);
                    for (const CollectorRun &collector : collectors) {
                        const NodePlan &node = nodes_[size_t(collector.node)];
                        for (size_t i = 0; i < node.inputs.size(); ++i)
                            if (node.inputs[i].ignored && node.inputs[i].source >= 0)
                                prefetch(source_of({collector.node, slot}, int(i)), seen);
                    }
                    dispatch();
                }
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
                                prepared.handle ? py::handle(static_cast<PyObject *>(prepared.handle.get())) : py::none());
                        } catch (py::error_already_set &error) {
                            collector.seconds += seconds_since(start);
                            throw PythonError{current_error(error)};
                        }
                    }
                    collector.seconds += seconds_since(start);
                }
                if (slot == 0) {
                    std::lock_guard<std::mutex> locked(mutex_);
                    for (int node : group.iterated) loop_->estimate += slot_cells_[0][size_t(node)].bytes;
                }
                ++loop_->committed;
                release_slot(slot);
                suspend();
                for (GeneratorRun &generator : generators) {
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

/* The end of a group's loop: no item's cell is queued or running afterwards, and none is kept. */
void Run::drain_slots()
{
    std::unique_lock<std::mutex> locked(mutex_);
    std::erase_if(ready_, [&](const CellRef &ref) {
        if (ref.slot < 0) return false;
        Cell &c = cell(ref);
        c.state = State::idle;
        c.wanted = false;
        return true;
    });
    std::erase_if(wanted_, [](const CellRef &ref) { return ref.slot >= 0; });
    tasks_.clear();
    changed_.wait(locked, [&] { return running_slots_ == 0; });
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
    ThreadPython python(hooks_.install_thread);
    PyRef error, result;
    bool stopped = false;
    try {
        if (mode_ == "run") {
            emit(Event(Event::chain_start));
            for (const GroupPlan &group : groups_) iterate_group(group);
            {
                std::lock_guard<std::mutex> locked(mutex_);
                std::set<std::pair<int, int64_t>> seen;
                for (int node : tail_) prefetch({node, -1}, seen);
                dispatch();
            }
            for (int node : tail_) {
                suspend();
                process_regular(node, -1);
            }
        } else if (mode_ == "regular") {
            result = process_regular(target_, -1);
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
    }
    drain_workers();
    {
        PythonSection section;
        if (stopped) error = py_ref(hooks_.aborted_type());
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

Run::Gathered Run::gather(const CellRef &ref)
{
    const NodePlan &node = nodes_[size_t(ref.node)];
    Gathered gathered;
    std::lock_guard<std::mutex> locked(mutex_);
    gathered.defer = ref.slot >= 0 && loop_ && loop_->active;
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

bool Run::run_python(const CellRef &ref, const Gathered &inputs, Cell &result, double &lazy_seconds,
                     const PyRef &prepared)
{
    const NodePlan &node = nodes_[size_t(ref.node)];
    PythonSection python;
    try {
        py::list values = python_inputs(ref, inputs);
        py::object output = hooks_.run_node(
            ref.node, values,
            prepared.handle ? py::handle(static_cast<PyObject *>(prepared.handle.get())) : py::none());
        result.object = py_ref(output);
        if (node.kind == Kind::regular) {
            py::list outputs = hooks_.regular_outputs(output);
            for (py::handle value : outputs) result.outputs.push_back(py_ref(value));
            if (ref.slot == 0) result.bytes = hooks_.output_bytes(outputs).cast<int64_t>();
        } else if (node.kind == Kind::generator) {
            for (py::handle value : hooks_.generator_outputs(output)) result.outputs.push_back(py_ref(value));
        } else {
            result.object = py_ref(hooks_.collector(output));
            result.inputs = py_ref(values);
        }
    } catch (py::error_already_set &error) {
        result.error = current_error(error);
    }
    for (const auto &lazy : inputs.lazies) lazy_seconds += lazy->seconds();
    return !result.error.handle;
}

/* An image the engine enforces itself: an allowed channel count. */
bool allowed_channels(const std::vector<int> &channels, int count)
{
    return channels.empty() || std::find(channels.begin(), channels.end(), count) != channels.end();
}

/* A Python exception as a native node's NodeError (its type's name, qualified unless a builtin, and its text). */
NodeError node_error(py::error_already_set &error)
{
    py::object type = error.type();
    std::string module = py::str(type.attr("__module__"));
    std::string name = py::str(type.attr("__qualname__"));
    return NodeError(module == "builtins" ? name : module + "." + name, py::str(error.value()));
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
    }
}

NativeOutcome Run::run_native(const CellRef &ref, const Gathered &inputs, Cell &result, double &lazy_seconds)
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
            result.error = py_ref(hooks_.native_failure(ref.node, error.python_type, error.what(), python_values));
        } catch (py::error_already_set &again) {
            result.error = current_error(again);
        }
        return NativeOutcome::failed;
    };
    std::vector<Value> outputs;
    EngineContext context(*this, node, storage_);
    try {
        node.native->run(context, values, outputs);
    } catch (const NodeError &error) {
        return failure(error);
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
    }
    for (const auto &lazy : inputs.lazies) lazy_seconds += lazy->seconds();
    if (outputs.size() != node.outputs.size())
        return failure(NodeError("AssertionError", "Expected all " + node.name + " nodes to have " +
                                                       std::to_string(node.outputs.size()) + " output(s) but found " +
                                                       std::to_string(outputs.size()) + "."));
    std::vector<size_t> enforce;
    for (size_t i = 0; i < outputs.size(); ++i) {
        const OutputPlan &plan = node.outputs[i];
        const Image *image = std::get_if<Image>(&outputs[i]);
        if (plan.image && image && image->height > 0 && image->width > 0 &&
            (plan.channels == 0 || plan.channels == image->channels)) {
            if (!plan.assume_normalized) {
                // ImageOutput.enforce: borrowed when its clip changes no bit, else normalize's copy.
                int same = 0;
                check(cn_pixels_normalized_f32(image->data, image->values(), &same));
                if (!same) {
                    MutableImage copy = MutableImage::allocate(image->height, image->width, image->channels);
                    check(cn_pixels_convert_checked(image->data, copy.data(), image->values(), 0, 0, 1, nullptr));
                    outputs[i] = std::move(copy).publish();
                }
            }
            continue;
        }
        enforce.push_back(i);
    }
    if (!enforce.empty()) {
        std::optional<NodeError> error;
        {
            PythonSection python;
            try {
                for (size_t i : enforce)
                    outputs[i] = py_ref(hooks_.enforce_output(ref.node, int(i), to_python(outputs[i])));
            } catch (py::error_already_set &raised) {
                error = node_error(raised);
            }
        }
        if (error) return failure(*error);
    }
    if (ref.slot == 0)
        for (const Value &value : outputs)
            if (const Image *image = std::get_if<Image>(&value)) result.bytes += int64_t(image->values() * sizeof(float));
    result.outputs = std::move(outputs);
    return NativeOutcome::done;
}

void Run::execute(const CellRef &ref, bool on_driver)
{
    const NodePlan &node = nodes_[size_t(ref.node)];
    Gathered inputs = gather(ref);
    bool defer = inputs.defer && !on_driver;
    if (on_driver) suspend();
    if (!defer) emit_start(ref.node);
    if (on_driver) suspend();
    PyRef prepared = on_driver && node.phased && ref.slot >= 0 ? take_prepared(ref.node, ref.slot) : PyRef{};
    if (on_driver && node.node_context && node.kind == Kind::regular && ref.slot >= 0) {
        // process.Executor.__get_node_context: the loop's node gets its iteration view on the commit path.
        PythonSection python;
        hooks_.iteration(ref.node);
    }
    Cell result;
    double lazy_seconds = 0;
    Clock::time_point start = Clock::now();
    bool ok;
    NativeOutcome outcome = node.native ? run_native(ref, inputs, result, lazy_seconds) : NativeOutcome::python;
    if (outcome == NativeOutcome::declined) {
        std::lock_guard<std::mutex> locked(decline_mutex);
        ++decline_counts()[node.schema];
    }
    if (outcome == NativeOutcome::declined || outcome == NativeOutcome::python) {
        result = Cell{};
        lazy_seconds = 0;
        ok = run_python(ref, inputs, result, lazy_seconds, prepared);
    } else {
        ok = outcome == NativeOutcome::done;
    }
    double seconds = seconds_since(start) - lazy_seconds;
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
            submit_broadcast(ref.node, result.outputs);
            emit_finish(ref.node, seconds);
        }
    }
    store(ref, std::move(result));
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
    ThreadPython python(hooks_.install_thread);
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
            } catch (py::error_already_set &error) {
                error.discard_as_unraisable("chaiNNer-C engine events");
            }
        }
        if (stop) return;
    }
}

/* Returns once every event emitted so far is on the loop's queue of callbacks (before the run's result). */
void Run::flush_events()
{
    std::unique_lock<std::mutex> locked(events_mutex_);
    events_changed_.wait(locked, [&] { return events_.empty() && !pumping_; });
}

void Run::submit_broadcast(int node, const std::vector<Value> &outputs)
{
    if (!send_broadcast_data_ || nodes_[size_t(node)].outputs.empty()) return;
    {
        std::lock_guard<std::mutex> locked(broadcasts_mutex_);
        auto [job, inserted] = broadcast_jobs_.insert_or_assign(node, BroadcastJob{node, outputs});
        if (inserted) broadcast_order_.push_back(node);
    }
    broadcasts_changed_.notify_all();
}

void Run::broadcast_now(int node, const std::vector<Value> &outputs, const PyRef &generator)
{
    if (!send_broadcast_data_ || nodes_[size_t(node)].outputs.empty() || aborted_) return;
    PythonSection python;
    try {
        py::list values;
        for (const Value &value : outputs) values.append(to_python(value));
        py::object generator_object =
            generator.handle ? py::reinterpret_borrow<py::object>(static_cast<PyObject *>(generator.handle.get())) : py::none();
        py::object event = hooks_.broadcast(node, values, generator_object);
        if (!aborted_) {
            Event raw(Event::raw, node);
            raw.payload = py_ref(event);
            emit(std::move(raw));
        }
    } catch (py::error_already_set &error) {
        error.discard_as_unraisable("chaiNNer-C engine broadcast");
    }
}

void Run::broadcaster()
{
    ThreadPython python(hooks_.install_thread);
    for (;;) {
        BroadcastJob job;
        {
            std::unique_lock<std::mutex> locked(broadcasts_mutex_);
            broadcasts_changed_.wait(locked, [&] { return broadcasts_stop_ || !broadcast_order_.empty(); });
            if (broadcast_order_.empty()) return;
            int node = broadcast_order_.front();
            broadcast_order_.pop_front();
            job = std::move(broadcast_jobs_.at(node));
            broadcast_jobs_.erase(node);
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
    broadcasts_changed_.wait(locked, [&] { return broadcast_order_.empty() && !broadcasting_; });
}


}  // namespace
}  // namespace cn::run

void cn_bind_run_executor(py::module_ &module)
{
    using cn::run::Run;
    py::class_<Run>(module, "RunEngine")
        .def(py::init<const py::dict &, py::object>(), py::arg("plan"), py::arg("hooks"))
        .def("start", &Run::start, py::arg("mode"), py::arg("target"))
        .def("kill", &Run::kill)
        .def("pause", &Run::pause)
        .def("resume", &Run::resume)
        .def_property_readonly("aborted", &Run::aborted)
        .def_property_readonly("paused", &Run::paused)
        .def_property_readonly("commit_slot", &Run::commit_slot)
        .def("flush_broadcasts", &Run::flush_broadcasts, py::call_guard<py::gil_scoped_release>())
        .def("close", &Run::close, py::call_guard<py::gil_scoped_release>())
        .def("lookahead", &Run::lookahead)
        .def("peek", &Run::peek);
    module.def("native_declines", [] {
        py::dict counts;
        std::lock_guard<std::mutex> locked(cn::run::decline_mutex);
        for (const auto &[schema, count] : cn::run::decline_counts()) counts[py::str(schema)] = count;
        return counts;
    }, "The native entry points' declines to their Python bodies (RunPython), per schema id, since the process started.");
}
