/* chaiNNer text application algorithms, adapted from chaiNNer's GPL-3.0
 * nodes/utils/replacement.py, nodes/impl/rust_regex.py and utility/text nodes.
 * A chainner_ext RustRegex runs on its engine through chainner_ext's _C_API
 * capsule (chainner_regex.h), with no Python call per match; other regex
 * objects keep their Python protocol. CPython owns Unicode storage, scalar
 * arithmetic and the existing str padding/replacement primitives.
 * No runtime source evaluation and no shared mutable state. */
#include <pybind11/gil_safe_call_once.h>
#include <pybind11/pybind11.h>
#include <optional>
#include <vector>
#include "chainner_regex.h"
#include "graph_python.hpp"
#include "graph_unpack.hpp"

namespace {
namespace py = pybind11;
using O = py::object;
using graphpy::compare;
using graphpy::contains;

O owned(PyObject* result) {
    if (!result) throw py::error_already_set();
    return py::reinterpret_steal<O>(result);
}
O add(const O& a, const O& b) { return owned(PyNumber_Add(a.ptr(), b.ptr())); }
O append(const O& a, const O& b) { return owned(PyNumber_InPlaceAdd(a.ptr(), b.ptr())); }
O slice(const O& text, const O& start, const O& end) {
    O key = owned(PySlice_New(start.ptr(), end.ptr(), Py_None));
    return owned(PyObject_GetItem(text.ptr(), key.ptr()));
}
O substring(const O& text, Py_ssize_t start, Py_ssize_t end) {
    return owned(PyUnicode_Substring(text.ptr(), start, end));
}
O formatted(const char* pattern, const O& value) {
    return py::str(pattern).attr("format")(value);
}
[[noreturn]] void value_error(const O& message) {
    PyErr_SetObject(PyExc_ValueError, message.ptr());
    throw py::error_already_set();
}

// PyNumber_InPlaceAdd cannot use bytecode's unique-reference Unicode +=
// optimization. Buffer exact strings, then copy each fragment once in join.
// Materialize before a custom operand so its __radd__/__iadd__ sees precisely
// the original accumulated value. Empty/single-fragment identity is preserved.
class TextBuilder {
    py::list fragments_;
    O object_;
    bool plain_ = true;
public:
    O finish() const {
        if (!plain_) return object_;
        const size_t count = fragments_.size();
        if (count == 0) return py::str("");
        if (count == 1) return py::reinterpret_borrow<O>(fragments_[0]);
        py::str separator("");
        return owned(PyUnicode_Join(separator.ptr(), fragments_.ptr()));
    }
    void push(const O& value) {
        if (plain_ && PyUnicode_CheckExact(value.ptr())) {
            if (PyUnicode_GET_LENGTH(value.ptr()) != 0) fragments_.append(value);
            return;
        }
        O previous = finish();
        O combined = append(previous, value);
        fragments_ = py::list();
        if (PyUnicode_CheckExact(combined.ptr())) {
            plain_ = true;
            object_ = O();
            if (PyUnicode_GET_LENGTH(combined.ptr()) != 0) fragments_.append(combined);
        } else {
            plain_ = false;
            object_ = combined;
        }
    }
};

void replacement_init(O self, O pattern, O interpolation_type) {
    self.attr("tokens") = py::list();
    self.attr("names") = py::set();
    // The fixed regular expression scans the underlying Unicode value even
    // for a str subclass; slicing literals still uses that object's protocol.
    if (!PyUnicode_Check(pattern.ptr())) {
        Py_buffer buffer{};
        if (PyObject_GetBuffer(pattern.ptr(), &buffer, PyBUF_SIMPLE) == 0) {
            PyBuffer_Release(&buffer);
            PyErr_SetString(PyExc_TypeError, "cannot use a string pattern on a bytes-like object");
        } else {
            // re's string acquisition replaces buffer errors (including a
            // non-contiguous memoryview) with this input-type diagnostic.
            PyErr_Clear();
            PyErr_Format(PyExc_TypeError, "expected string or bytes-like object, got '%.200s'", Py_TYPE(pattern.ptr())->tp_name);
        }
        throw py::error_already_set();
    }
    if (PyUnicode_READY(pattern.ptr()) < 0) throw py::error_already_set();
    const Py_ssize_t size = PyUnicode_GET_LENGTH(pattern.ptr());
    const int kind = PyUnicode_KIND(pattern.ptr());
    const void* data = PyUnicode_DATA(pattern.ptr());
    Py_ssize_t cursor = 0, last_index = 0;
    TextBuilder literal;
    while (cursor < size) {
        if (PyUnicode_READ(kind, data, cursor) != '{') { ++cursor; continue; }
        Py_ssize_t end = cursor + 1;
        const bool escaped = end < size && PyUnicode_READ(kind, data, end) == '{';
        if (escaped) ++end;
        else {
            while (end < size && PyUnicode_READ(kind, data, end) != '{' && PyUnicode_READ(kind, data, end) != '}') ++end;
            if (end == size || PyUnicode_READ(kind, data, end) != '}') { ++cursor; continue; }
            ++end;
        }
        literal.push(slice(pattern, py::int_(last_index), py::int_(cursor)));
        last_index = end;
        if (escaped) literal.push(py::str("{"));
        else {
            O name = substring(pattern, cursor + 1, end - 1);
            if (end == cursor + 2)
                value_error(formatted("Invalid replacement pattern. {{}} is not a valid replacement. Either specify a name or id number, or escape a single \"{{\" as \"{{{{\". Full pattern: {}", pattern));
            for (Py_ssize_t i = cursor + 1; i < end - 1; ++i) {
                const Py_UCS4 c = PyUnicode_READ(kind, data, i);
                if (c != '_' && !Py_UNICODE_ISALNUM(c)) {
                    O message = py::str("Invalid replacement pattern. {{{}}} is not a valid replacement. Names and ids only allow letters and digits. Full pattern: {}").attr("format")(name, pattern);
                    value_error(message);
                }
            }
            self.attr("tokens").attr("append")(literal.finish());
            literal = TextBuilder();
            self.attr("tokens").attr("append")(interpolation_type(name));
            self.attr("names").attr("add")(name);
        }
        cursor = end;
    }
    literal.push(slice(pattern, py::int_(last_index), py::none()));
    self.attr("tokens").attr("append")(literal.finish());
}

O replacement_replace(O self, O replacements) {
    TextBuilder result;
    for (py::handle token_handle : self.attr("tokens")) {
        O token = py::reinterpret_borrow<O>(token_handle);
        const int is_string = PyObject_IsInstance(token.ptr(), reinterpret_cast<PyObject*>(&PyUnicode_Type));
        if (is_string < 0) throw py::error_already_set();
        if (is_string) result.push(token);
        else if (contains(replacements, token.attr("name"))) {
            O name = token.attr("name");
            O replacement = owned(PyObject_GetItem(replacements.ptr(), name.ptr()));
            result.push(replacement);
        } else {
            // Keep attribute and keys/join evaluation in Python's original order.
            O name = token.attr("name");
            O prefix = formatted("Unknown replacement. There is no replacement with the name or id {}. Available replacements: ", name);
            O available = py::str(", ").attr("join")(replacements.attr("keys")());
            value_error(add(add(prefix, available), py::str(".")));
        }
    }
    return result.finish();
}

O range_text(O text, O range) {
    O start = range.attr("start");
    O end = range.attr("end");
    return slice(text, start, end);
}
O group_text(const O& text, const O& group) {
    return group.is_none() ? O(py::str("")) : range_text(text, group);
}

// chainner_ext's engine, bound once from its capsule (chainner_regex.h).
struct RegexEngine {
    const chn_regex_api* api;
    O panic_type; // pyo3_runtime.PanicException, as RustRegex raises it
};
const RegexEngine& regex_engine() {
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<RegexEngine> storage;
    return storage.call_once_and_store_result([] {
        O module = py::module_::import("chainner_ext.chainner_ext");
        O capsule = module.attr("_C_API");
        auto* api = static_cast<const chn_regex_api*>(PyCapsule_GetPointer(capsule.ptr(), CHN_REGEX_CAPSULE_NAME));
        if (!api) throw py::error_already_set();
        if (api->version != CHN_REGEX_API_VERSION) {
            PyErr_Format(PyExc_ImportError, "%s is version %u; _chainner_graph needs version %u",
                CHN_REGEX_CAPSULE_NAME, api->version, CHN_REGEX_API_VERSION);
            throw py::error_already_set();
        }
        return RegexEngine{api, module.attr("PanicException")};
    }).get_stored();
}
// chainner_ext's raise_status (regex.c): an engine failure as RustRegex raises it.
[[noreturn]] void raise_engine(const RegexEngine& engine, chn_regex_status status, chn_regex_string* message) {
    if (status == CHN_REGEX_PANIC) {
        PyObject* text = PyUnicode_DecodeUTF8(message->ptr ? message->ptr : "", static_cast<Py_ssize_t>(message->len), "replace");
        engine.api->string_drop(message);
        if (text) {
            PyErr_SetObject(engine.panic_type.ptr(), text);
            Py_DECREF(text);
        }
    } else {
        engine.api->string_drop(message);
        if (status == CHN_REGEX_ERROR_NOMEM) PyErr_NoMemory();
        else PyErr_Format(PyExc_RuntimeError, "unexpected regex engine status %d", static_cast<int>(status));
    }
    throw py::error_already_set();
}
// PosTranslator (position.rs): byte offsets to char positions, one per search or findall.
class Translator {
    const RegexEngine& engine_;
    chn_regex_pos_translator translator_{};
public:
    Translator(const RegexEngine& engine, const char* text, size_t size) : engine_(engine) {
        engine_.api->pos_translator_init(&translator_, text, size);
    }
    ~Translator() { engine_.api->pos_translator_drop(&translator_); }
    Translator(const Translator&) = delete;
    Translator& operator=(const Translator&) = delete;
    // RegexMatch::from_captures: each participating group's start, then end, in index order.
    void translate(std::vector<size_t>& slots) {
        for (size_t i = 0; i < slots.size(); i += 2) {
            if (slots[i] == CHN_REGEX_NO_SLOT || slots[i + 1] == CHN_REGEX_NO_SLOT) {
                slots[i] = slots[i + 1] = CHN_REGEX_NO_SLOT;
                continue;
            }
            for (size_t k = i; k < i + 2; ++k) {
                chn_regex_string panic{nullptr, 0};
                const chn_regex_status status = engine_.api->pos_translator_get_char_pos(&translator_, slots[k], &slots[k], &panic);
                if (status != CHN_REGEX_OK) raise_engine(engine_, status, &panic);
            }
        }
    }
};
// The capture names in index order (None for an unnamed group): RustRegex.groupindex's order.
std::vector<O> engine_names(const RegexEngine& engine, const chn_regex* re) {
    const size_t count = engine.api->captures_len(re);
    std::vector<O> names;
    names.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        size_t length = 0;
        const char* name = engine.api->capture_name(re, i, &length);
        names.push_back(name ? owned(PyUnicode_DecodeUTF8(name, static_cast<Py_ssize_t>(length), "strict")) : O(py::none()));
    }
    return names;
}
// match_to_replacements_dict with the regex's groups and groupindex read from the engine:
// str(i) for every group, then every name; group_at(i) gives group i's text.
template <class GroupText>
py::dict engine_capture_map(const std::vector<O>& names, GroupText group_at) {
    py::dict replacements;
    for (size_t i = 0; i < names.size(); ++i) {
        // Assignment evaluates its RHS before the dictionary key.
        O value = group_at(i);
        replacements[py::str(py::int_(i))] = value;
    }
    for (size_t i = 0; i < names.size(); ++i) {
        if (names[i].is_none()) continue;
        O value = group_at(i);
        replacements[names[i]] = value;
    }
    return replacements;
}
// text[start:end] (end None: to the end), through a str subclass's own protocol.
O char_slice(const O& text, size_t start, std::optional<size_t> end) {
    if (!PyUnicode_CheckExact(text.ptr()))
        return slice(text, py::int_(start), end ? O(py::int_(*end)) : O(py::none()));
    const Py_ssize_t stop = end ? static_cast<Py_ssize_t>(*end) : PyUnicode_GET_LENGTH(text.ptr());
    return substring(text, static_cast<Py_ssize_t>(start), stop);
}
// Group index's text from char spans (2 per group); "" for a group that did not participate.
O span_text(const O& text, const size_t* spans, size_t index) {
    if (spans[2 * index] == CHN_REGEX_NO_SLOT) return py::str("");
    return char_slice(text, spans[2 * index], spans[2 * index + 1]);
}

// A chainner_ext RustRegex searched over a str on the engine itself.
struct EngineSearch {
    const RegexEngine* engine;
    const chn_regex* re;
    const char* data;
    size_t size;
    std::vector<O> names;
    size_t width() const { return 2 * names.size(); }
};
// The engine path when regex is exactly chainner_ext's RustRegex and text a str it can
// read; otherwise the regex keeps its Python protocol, which also raises RustRegex's own
// conversion errors for a text the engine cannot take.
std::optional<EngineSearch> engine_search(const O& regex, const O& text) {
    const RegexEngine& engine = regex_engine();
    const chn_regex* re = engine.api->regex_of(regex.ptr());
    if (!re || !PyUnicode_Check(text.ptr())) return std::nullopt;
    Py_ssize_t size = 0;
    const char* data = PyUnicode_AsUTF8AndSize(text.ptr(), &size);
    if (!data) {
        PyErr_Clear();
        return std::nullopt;
    }
    return EngineSearch{&engine, re, data, static_cast<size_t>(size), engine_names(engine, re)};
}
// RustRegex.search(text): the char spans of the match, or empty for None.
std::vector<size_t> engine_first(const EngineSearch& search) {
    const chn_regex_api* api = search.engine->api;
    std::vector<size_t> slots(search.width());
    const size_t start = api->to_byte_pos(search.data, search.size, 0);
    chn_regex_string panic{nullptr, 0};
    const chn_regex_status status = api->search_at(search.re, search.data, search.size, start, slots.data(), slots.size(), &panic);
    if (status == CHN_REGEX_NO_MATCH) return {};
    if (status != CHN_REGEX_OK) raise_engine(*search.engine, status, &panic);
    Translator(*search.engine, search.data, search.size).translate(slots);
    return slots;
}
// RustRegex.findall(text): every match's char spans, width() per match.
std::vector<size_t> engine_all(const EngineSearch& search) {
    const chn_regex_api* api = search.engine->api;
    std::vector<size_t> spans, slots(search.width());
    Translator translator(*search.engine, search.data, search.size);
    chn_regex_iter iter;
    api->iter_init(&iter, search.re, search.data, search.size);
    for (;;) {
        const chn_regex_status status = api->iter_next_captures(&iter, slots.data(), slots.size());
        if (status == CHN_REGEX_NO_MATCH) return spans;
        if (status != CHN_REGEX_OK) {
            chn_regex_string none{nullptr, 0};
            raise_engine(*search.engine, status, &none);
        }
        translator.translate(slots);
        spans.insert(spans.end(), slots.begin(), slots.end());
    }
}

py::dict capture_map(O regex, O match, O text) {
    const RegexEngine& engine = regex_engine();
    if (const chn_regex* re = engine.api->regex_of(regex.ptr())) {
        // A RustRegex's groups and groupindex come from the engine; the match keeps its protocol.
        return engine_capture_map(engine_names(engine, re), [&](size_t i) {
            return group_text(text, match.attr("get")(py::int_(i)));
        });
    }
    py::dict replacements;
    O count = add(regex.attr("groups"), py::int_(1));
    for (py::handle ih : graphpy::builtin("range")(count)) {
        O i = py::reinterpret_borrow<O>(ih);
        // Assignment evaluates its RHS before the dictionary key.
        O value = group_text(text, match.attr("get")(i));
        replacements[py::str(i)] = value;
    }
    // The original loop is `for name, index in regex.groupindex.items()`; CPython
    // unpacks each item (graphpy::unpack), so the errors are the interpreter's own.
    O items = regex.attr("groupindex").attr("items")();
    for (py::handle pair_handle : items) {
        py::tuple pair = graphpy::unpack(py::reinterpret_borrow<O>(pair_handle), 2);
        O name = pair[0];
        O index = pair[1];
        O value = group_text(text, match.attr("get")(index));
        replacements[name] = value;
    }
    return replacements;
}

O regex_find(O text, O pattern, O mode, O output_pattern,
    O mode_type, O regex_type, O replacement_type, bool installed) {
    O cache = py::module_::import("nodes.impl.text_cache");
    O regex = cache.attr("compile_regex")(pattern, regex_type);
    const std::optional<EngineSearch> search = engine_search(regex, text);
    std::vector<size_t> spans;
    O match;
    if (search) spans = engine_first(*search);
    else match = regex.attr("search")(text);
    if (search ? spans.empty() : match.is_none()) {
        if (installed) {
            O message = formatted("No match found. Unable to find the pattern '{}' in the text.", pattern);
            PyErr_SetObject(PyExc_RuntimeError, message.ptr());
            throw py::error_already_set();
        }
        return py::make_tuple("", false);
    }
    O result;
    if (compare(mode, mode_type.attr("FULL_MATCH"), Py_EQ))
        result = search ? span_text(text, spans.data(), 0) : range_text(text, match);
    else if (compare(mode, mode_type.attr("PATTERN"), Py_EQ)) {
        O replacements = search
            ? O(engine_capture_map(search->names, [&](size_t i) { return span_text(text, spans.data(), i); }))
            : O(capture_map(regex, match, text));
        O replacement = cache.attr("compile_replacement")(output_pattern, replacement_type);
        result = replacement.attr("replace")(replacements);
    } else {
        if (installed) return py::none();
        value_error(formatted("Unknown OutputMode: {}", mode));
    }
    return installed ? result : O(py::make_tuple(result, true));
}
O regex_replace(O text, O pattern, O replacement_pattern, O mode,
    O mode_type, O regex_type, O replacement_type) {
    O cache = py::module_::import("nodes.impl.text_cache");
    O regex = cache.attr("compile_regex")(pattern, regex_type);
    O replacement = cache.attr("compile_replacement")(replacement_pattern, replacement_type);
    if (const std::optional<EngineSearch> search = engine_search(regex, text)) {
        // The engine's search gives findall's first match (iteration starts with it).
        const bool first = mode.is(mode_type.attr("REPLACE_FIRST"));
        const std::vector<size_t> spans = first ? engine_first(*search) : engine_all(*search);
        if (spans.empty()) return text;
        size_t count = spans.size() / search->width();
        if (compare(mode, mode_type.attr("REPLACE_FIRST"), Py_EQ)) count = 1;
        TextBuilder result;
        size_t last_end = 0;
        for (size_t m = 0; m < count; ++m) {
            const size_t* found = spans.data() + m * search->width();
            result.push(char_slice(text, last_end, found[0]));
            // Retrieve replace before computing the call argument, as Python does.
            O replace = replacement.attr("replace");
            O replacements = engine_capture_map(search->names, [&](size_t i) { return span_text(text, found, i); });
            result.push(replace(replacements));
            last_end = found[1];
        }
        result.push(char_slice(text, last_end, std::nullopt));
        return result.finish();
    }
    O matches;
    // The built-in engine's search returns the same first capture as findall.
    // Keep custom constructors/protocols on their original observable path.
    if (regex_type.is(cache.attr("RustRegex")) && mode.is(mode_type.attr("REPLACE_FIRST"))) {
        O match = regex.attr("search")(text);
        if (match.is_none()) return text;
        matches = py::make_tuple(match);
    } else {
        matches = regex.attr("findall")(text);
    }
    if (py::len(matches) == 0) return text;
    if (compare(mode, mode_type.attr("REPLACE_FIRST"), Py_EQ))
        matches = slice(matches, py::none(), py::int_(1));
    TextBuilder result;
    O last_end = py::int_(0);
    for (py::handle match_handle : matches) {
        O match = py::reinterpret_borrow<O>(match_handle);
        O start = match.attr("start");
        result.push(slice(text, last_end, start));
        // Retrieve replace before computing the call argument, as Python does.
        O replace = replacement.attr("replace");
        O replacements = capture_map(regex, match, text);
        result.push(replace(replacements));
        last_end = match.attr("end");
    }
    result.push(slice(text, last_end, py::none()));
    return result.finish();
}
O text_pattern(O pattern, py::tuple args, O replacement_type) {
    py::dict replacements;
    for (size_t i = 0; i < args.size(); ++i) {
        O value = args[i];
        if (!value.is_none()) replacements[py::str(py::int_(i + 1))] = value;
    }
    O cache = py::module_::import("nodes.impl.text_cache");
    return cache.attr("compile_replacement")(pattern, replacement_type).attr("replace")(replacements);
}
O text_padding(O text, O width, O padding, O alignment, O alignment_type) {
    if (compare(alignment, alignment_type.attr("START"), Py_EQ)) return text.attr("rjust")(width, padding);
    if (compare(alignment, alignment_type.attr("END"), Py_EQ)) return text.attr("ljust")(width, padding);
    if (compare(alignment, alignment_type.attr("CENTER"), Py_EQ)) return text.attr("center")(width, padding);
    value_error(formatted("Invalid alignment '{}'.", alignment));
}
O text_replace(O text, O old, O replacement, O mode, O mode_type) {
    if (compare(mode, mode_type.attr("REPLACE_ALL"), Py_EQ)) return text.attr("replace")(old, replacement);
    return text.attr("replace")(old, replacement, 1);
}
O text_slice(O text, O operation, O start, O length, O maximum,
    O alignment, O operation_type, O alignment_type) {
    if (compare(operation, operation_type.attr("START"), Py_EQ)) return slice(text, start, py::none());
    if (compare(operation, operation_type.attr("START_AND_LENGTH"), Py_EQ)) {
        O lower = py::int_(-static_cast<Py_ssize_t>(py::len(text)));
        if (!compare(start, lower, Py_GT)) start = lower;
        O end = add(start, length);
        return slice(text, start, end);
    }
    if (compare(operation, operation_type.attr("MAX_LENGTH"), Py_EQ)) {
        if (compare(maximum, py::int_(0), Py_EQ)) return py::str("");
        if (compare(alignment, alignment_type.attr("START"), Py_EQ)) return slice(text, py::none(), maximum);
        if (compare(alignment, alignment_type.attr("END"), Py_EQ)) return slice(text, graphpy::negative(maximum), py::none());
    }
    return py::none();
}
} // namespace

void cn_bind_utility_text(pybind11::module_& module) {
    module.def("utility_replacement_init", &replacement_init);
    module.def("utility_replacement_replace", &replacement_replace);
    module.def("utility_range_text", &range_text);
    module.def("utility_capture_map", &capture_map);
    module.def("utility_regex_find", &regex_find);
    module.def("utility_regex_replace", &regex_replace);
    module.def("utility_text_pattern", &text_pattern);
    module.def("utility_text_padding", &text_padding);
    module.def("utility_text_replace", &text_replace);
    module.def("utility_text_slice", &text_slice);
}
