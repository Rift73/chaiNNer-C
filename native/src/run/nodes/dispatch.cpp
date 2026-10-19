/* Process facts for native node entry points (dispatch.hpp). The one file under nodes/ that includes Python: a fact's
 * first read takes the GIL (no lock held) to call the bridge's own Python function; the result is published in a
 * lock-free list that later reads walk without the GIL. Two threads may compute the same fact at once; both get the
 * same value and the list keeps both, which is harmless. */
#include "dispatch.hpp"

#include "../engine.hpp"

#include <atomic>
#include <sstream>
#include <variant>

namespace py = pybind11;

namespace cn::run::dispatch {
namespace {

using Fact = std::variant<int64_t, std::string>;

/* A computed fact; entries live for the process (a few dozen at most) and are never freed. */
struct Entry {
    std::string key;
    Fact value;
    const Entry *next;
};

std::atomic<const Entry *> facts{nullptr};

const Fact *find(const std::string &key)
{
    for (const Entry *entry = facts.load(std::memory_order_acquire); entry; entry = entry->next)
        if (entry->key == key) return &entry->value;
    return nullptr;
}

const Fact &publish(std::string key, Fact value)
{
    auto *entry = new Entry{std::move(key), std::move(value), facts.load(std::memory_order_relaxed)};
    while (!facts.compare_exchange_weak(entry->next, entry, std::memory_order_release, std::memory_order_relaxed)) {
    }
    return entry->value;
}

/* module.function(*arguments) under the GIL; a Python error becomes the NodeError naming it. */
Fact call(std::string_view module, std::string_view function, std::initializer_list<std::string_view> arguments,
          bool as_text)
{
    py::gil_scoped_acquire gil;
    try {
        py::object target = py::module_::import(std::string(module).c_str());
        std::string path(function);
        for (size_t start = 0;;) {
            size_t dot = path.find('.', start);
            target = target.attr(path.substr(start, dot - start).c_str());
            if (dot == std::string::npos) break;
            start = dot + 1;
        }
        py::tuple values(arguments.size());
        size_t index = 0;
        for (std::string_view argument : arguments) values[index++] = py::str(argument.data(), argument.size());
        py::object result = target(*values);
        if (as_text) return Fact(result.cast<std::string>());
        return Fact(result.cast<int64_t>());
    } catch (py::error_already_set &error) {
        rethrow_python_error(error);
    }
}

const Fact &fact(std::string_view module, std::string_view function,
                 std::initializer_list<std::string_view> arguments, bool as_text)
{
    std::string key = std::string(as_text ? "text:" : "number:") + std::string(module) + ":" + std::string(function);
    for (std::string_view argument : arguments) key += ":" + std::string(argument);
    if (const Fact *found = find(key)) return *found;
    return publish(std::move(key), call(module, function, arguments, as_text));
}

constexpr std::string_view simd = "nodes.impl.native_opencv_simd";

}  // namespace

int64_t number(std::string_view module, std::string_view function, std::initializer_list<std::string_view> arguments)
{
    return std::get<int64_t>(fact(module, function, arguments, false));
}

std::string text(std::string_view module, std::string_view function, std::initializer_list<std::string_view> arguments)
{
    return std::get<std::string>(fact(module, function, arguments, true));
}

size_t opencv_lanes(std::string_view family)
{
    return size_t(number(simd, "float32_lanes", {opencv_target(family)}));
}

std::string opencv_target(std::string_view family)
{
    return text(simd, "target", {family});
}

bool cv2_tested(std::string_view gate)
{
    return number("nodes.impl.native_versions", "cv2_is_tested", {gate}) != 0;
}

bool cv2_use_ipp()
{
    return number("cv2", "ipp.useIPP", {}) != 0;
}

bool cv2_use_ipp_not_exact()
{
    return number("cv2", "ipp.useIPP_NotExact", {}) != 0;
}

bool cv2_sse3()
{
    std::istringstream features(text("cv2", "getCPUFeaturesLine", {}));
    for (std::string feature; features >> feature;)
        if (feature == "SSE3" || feature == "*SSE3") return true;
    return false;
}

}  // namespace cn::run::dispatch
