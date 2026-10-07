/* chaiNNer file-sequence application control (GPL-3.0).
 * Traversal adapted from WCMatch 11.0.1 glob.py, MIT license,
 * Copyright (c) 2018 - 2026 Isaac Muse. Its complete notice ships with the bundled
 * wcmatch distribution (wcmatch-11.0.1.dist-info/licenses/LICENSE.md).
 * WCMatch/bracex pattern parsing remains an explicit library dependency.
 * Directory walking, pattern application, deduplication, natural-key assembly,
 * snapshot/limit policy and lazy per-file iterator state execute in C++.
 */
#include "graph_python.hpp"
#include "graph_unpack.hpp"
#include <functional>
#include <memory>
#include <vector>

namespace py = pybind11;
namespace {
using graphpy::O;
O checked(PyObject *value) {
    if (!value) throw py::error_already_set();
    return py::reinterpret_steal<O>(value);
}
[[noreturn]] void exhausted() {
    PyErr_SetNone(PyExc_StopIteration);
    throw py::error_already_set();
}
O own(py::handle value) { return py::reinterpret_borrow<O>(value); }
O name(const py::dict &globals, const char *key) {
    py::str k(key);
    if (globals.contains(k)) return globals[k];
    return graphpy::builtin(key);
}
bool equal(const O &a, const O &b) {
    return graphpy::truth(checked(PyObject_RichCompare(a.ptr(), b.ptr(), Py_EQ)));
}
O item(const O &value, py::ssize_t index) { return value[py::int_(index)]; }
bool flag(const O &value, const char *key) { return graphpy::truth(value.attr(key)); }

struct RecursiveCall {
    RecursiveCall() {
        if (Py_EnterRecursiveCall(" while traversing image files")) throw py::error_already_set();
    }
    ~RecursiveCall() { Py_LeaveRecursiveCall(); }
};

struct Entry { O name; bool directory; bool hidden; bool link; };
struct Match { O path; bool directory; };
using Sink = std::function<void(const Match &)>;

struct ActiveIteration {
    bool &running;
    explicit ActiveIteration(bool &value) : running(value) {
        if (running) throw py::value_error("generator already executing");
        running = true;
    }
    ~ActiveIteration() { running = false; }
};

class Walker {
    O config_;
    O os_;
    O path_;
    O root_;
    O empty_;
    O current_;
    O sep_;
    O specials_;
    O seen_;
    bool absolute_ = false;
    bool case_sensitive_;
    bool dot_;
    bool follow_;

    O join(const O &a, const O &b) { return path_.attr("join")(a, b); }
    bool special(const O &value) { return graphpy::contains(specials_, value); }
    bool literal(const O &value) { return PyUnicode_Check(value.ptr()) || PyBytes_Check(value.ptr()); }
    bool match(const O &target, const O &filename) {
        if (target.is_none()) return true;
        if (!literal(target)) return graphpy::truth(target.attr("match")(filename));
        if (case_sensitive_) return equal(filename, target);
        O lowered_target = target.attr("lower")();
        O lowered_name = filename.attr("lower")();
        return equal(lowered_name, lowered_target);
    }
    bool exists(const O &value) {
        O full = absolute_ ? value : join(root_, value);
        return graphpy::truth(path_.attr("lexists")(full));
    }
    bool excluded(O value, bool directory) {
        O patterns = config_.attr("npatterns");
        if (!graphpy::truth(patterns)) return false;
        if (directory && !graphpy::truth(value.attr("endswith")(sep_)))
            value = checked(PyNumber_InPlaceAdd(value.ptr(), sep_.ptr()));
        for (py::handle pattern : py::reinterpret_borrow<py::iterable>(patterns))
            if (graphpy::truth(own(pattern).attr("fullmatch")(value))) return true;
        return false;
    }
    std::vector<Entry> entries(const O &current, bool directory_only) {
        std::vector<Entry> result;
        // The original yields these before opening scandir, including on OSError.
        for (py::handle value : py::reinterpret_borrow<py::iterable>(specials_))
            result.push_back({own(value), true, true, false});
        try {
            O scan_path;
            if (absolute_ && graphpy::truth(current)) scan_path = current;
            else scan_path = graphpy::truth(current) ? join(root_, current) : root_;
            O scan = os_.attr("scandir")(scan_path);
            O entered = scan.attr("__enter__")();
            try {
                for (py::handle handle : py::reinterpret_borrow<py::iterable>(entered)) {
                    try {
                        O entry = own(handle);
                        O filename = entry.attr("name");
                        bool hidden = false;
                        if (!dot_) {
                            O first = filename[py::slice(py::int_(0), py::int_(1), py::none())];
                            hidden = equal(first, item(specials_, 0));
                        }
                        bool directory = graphpy::truth(entry.attr("is_dir")());
                        bool link = directory && graphpy::truth(entry.attr("is_symlink")());
                        if (!directory_only || directory)
                            result.push_back({filename, directory, hidden, link});
                    } catch (const py::error_already_set &error) {
                        if (!error.matches(PyExc_OSError)) throw;
                    }
                }
            } catch (const py::error_already_set &error) {
                bool suppressed = graphpy::truth(scan.attr("__exit__")(
                    error.type(), error.value(), error.trace()));
                if (!suppressed) throw;
            } catch (...) {
                scan.attr("__exit__")(py::none(), py::none(), py::none());
                throw;
            }
            scan.attr("__exit__")(py::none(), py::none(), py::none());
        } catch (const py::error_already_set &error) {
            if (!error.matches(PyExc_OSError)) throw;
        }
        return result;
    }
    void directory(const O &current, const O &target,
                   bool directory_only, bool deep, const Sink &sink) {
        RecursiveCall recursion;
        auto files = entries(current, directory_only);
        for (const auto &entry : files) {
            if (special(entry.name)) {
                if (!target.is_none() && match(target, entry.name))
                    sink({join(current, entry.name), true});
                continue;
            }
            O full = join(current, entry.name);
            // 11.0.1's _glob_dir yields an unfollowed link too; it only skips
            // descending into it (globstar_follow needs GLOBSTARLONG, which Load
            // Images' flags never set).
            bool selected = target.is_none() ? !entry.hidden : match(target, entry.name);
            if (selected) sink({full, entry.directory});
            bool follow = !entry.link || follow_;
            if (deep && !entry.hidden && entry.directory && follow) {
                directory(full, target, directory_only, deep, sink);
            }
        }
    }
    void glob(const O &current, const std::vector<O> &parts, size_t at, const Sink &sink) {
        RecursiveCall recursion;
        const O &part = parts.at(at);
        bool is_magic = flag(part, "is_magic");
        bool directory_only = flag(part, "dir_only");
        O target = part.attr("pattern");
        if (is_magic && flag(part, "is_globstar")) {
            size_t next = at + 1;
            bool globstar_end = next == parts.size();
            if (!globstar_end) {
                directory_only = flag(parts[next], "dir_only");
                target = parts[next].attr("pattern");
                ++next;
            } else target = py::none();
            if (globstar_end && graphpy::truth(current))
                sink({join(current, empty_), true});
            directory(current, target, directory_only, true, [&](const Match &candidate) {
                if (next < parts.size()) {
                    glob(candidate.path, parts, next, sink);
                } else sink(candidate);
            });
        } else if (!directory_only) {
            directory(current, target, false, false, sink);
        } else {
            directory(current, target, true, false, [&](const Match &candidate) {
                if (at + 1 < parts.size()) {
                    glob(candidate.path, parts, at + 1, sink);
                } else sink(candidate);
            });
        }
    }
    void emit(py::list &result, O value, bool directory, bool directory_only) {
        if (excluded(value, directory)) return;
        if (directory_only || (flag(config_, "mark") && directory)) value = join(value, empty_);
        if (!flag(config_, "nounique")) {
            O key = case_sensitive_ ? value : O(value.attr("lower")());
            if (graphpy::contains(seen_, key)) return;
            // Preserve WCMatch 11.0.1's check-lower/add-original behavior.
            if (PySet_Add(seen_.ptr(), value.ptr()) < 0) throw py::error_already_set();
        }
        result.append(value);
    }
public:
    Walker(const O &configuration, const O &os)
        : config_(configuration), os_(os), path_(os.attr("path")),
          root_(configuration.attr("root_dir")), empty_(configuration.attr("empty")),
          current_(configuration.attr("current")), sep_(configuration.attr("sep")),
          specials_(configuration.attr("specials")), seen_(configuration.attr("seen")),
          case_sensitive_(flag(configuration, "case_sensitive")), dot_(flag(configuration, "dot")),
          follow_(flag(configuration, "follow_links")) {}

    py::list run() {
        py::list result;
        for (py::handle pattern : py::reinterpret_borrow<py::iterable>(config_.attr("pattern"))) {
            // 11.0.1 starts every pattern at the root (8.4.1 carried the previous
            // pattern's literal start into the next one).
            O current = current_;
            std::vector<O> parts;
            for (py::handle part : py::reinterpret_borrow<py::iterable>(pattern)) parts.push_back(own(part));
            if (parts.empty()) continue;
            bool directory_only = flag(parts.back(), "dir_only");
            absolute_ = flag(parts.front(), "is_drive");
            if (!flag(parts.front(), "is_magic")) {
                O first = parts.front();
                current = item(first, 0);
                if (!graphpy::truth(current) || (absolute_ && !exists(current))) continue;
                std::vector<Match> starts;
                bool parent = equal(current, item(specials_, 1));
                bool self = equal(current, item(specials_, 0)) || equal(current, sep_);
                if (!absolute_ && !parent && !self) {
                    for (const auto &entry : entries(py::none(), directory_only))
                        if (!special(entry.name) && match(current, entry.name))
                            starts.push_back({entry.name, entry.directory});
                } else starts.push_back({current, true});
                if (flag(first, "dir_only")) {
                    for (const auto &start : starts) {
                        if (parts.size() > 1) {
                            glob(start.path, parts, 1, [&](const Match &candidate) {
                                emit(result, candidate.path, candidate.directory, directory_only);
                            });
                        } else emit(result, start.path, start.directory, directory_only);
                    }
                } else {
                    for (const auto &candidate : starts)
                        if (exists(candidate.path))
                            emit(result, candidate.path, candidate.directory, directory_only);
                }
            } else {
                O base = equal(current, current_) ? empty_ : current;
                glob(base, parts, 0, [&](const Match &candidate) {
                    emit(result, candidate.path, candidate.directory, directory_only);
                });
            }
        }
        return result;
    }
};

O alphanumeric(const O &value) {
    O upper = value.attr("upper")();
    if (!PyUnicode_Check(upper.ptr()))
        graphpy::raise(PyExc_TypeError, py::str("natural sort requires Unicode text"));
    if (PyUnicode_READY(upper.ptr()) < 0) throw py::error_already_set();
    const auto length = PyUnicode_GET_LENGTH(upper.ptr());
    const int kind = PyUnicode_KIND(upper.ptr());
    const void *data = PyUnicode_DATA(upper.ptr());
    py::list result;
    Py_ssize_t previous = 0;
    Py_ssize_t index = 0;
    while (index < length) {
        if (!Py_UNICODE_ISDECIMAL(PyUnicode_READ(kind, data, index))) { ++index; continue; }
        result.append(checked(PyUnicode_Substring(upper.ptr(), previous, index)));
        const Py_ssize_t begin = index++;
        while (index < length && Py_UNICODE_ISDECIMAL(PyUnicode_READ(kind, data, index))) ++index;
        O digits = checked(PyUnicode_Substring(upper.ptr(), begin, index));
        result.append(checked(PyLong_FromUnicodeObject(digits.ptr(), 10)));
        previous = index;
    }
    result.append(checked(PyUnicode_Substring(upper.ptr(), previous, length)));
    return result;
}

O extension_filter(const O &extensions) {
    O joined = py::str("|").attr("join")(extensions);
    O first = checked(PyUnicode_Concat(py::str("**/*@(").ptr(), joined.ptr()));
    return checked(PyUnicode_Concat(first.ptr(), py::str(")").ptr()));
}
O flags(const O &glob) {
    O value = glob.attr("EXTGLOB");
    for (const char *key : {"BRACE", "GLOBSTAR", "NEGATE", "DOTGLOB", "NEGATEALL"})
        value = checked(PyNumber_Or(value.ptr(), glob.attr(key).ptr()));
    return value;
}

O list_glob(const py::dict &globals, const O &directory, const O &expression, const O &extensions) {
    O extension_expression = extension_filter(extensions);
    O glob = name(globals, "glob");
    O glob_flags = flags(glob);
    py::list discovered;
    if (PyUnicode_Check(expression.ptr()) || PyBytes_Check(expression.ptr()) || graphpy::truth(expression)) {
        O configuration = glob.attr("Glob")(
            expression, py::arg("root_dir") = directory, py::arg("flags") = glob_flags);
        Walker walker(configuration, name(globals, "os"));
        discovered = walker.run();
    }
    O insensitive = checked(PyNumber_Or(glob_flags.ptr(), glob.attr("IGNORECASE").ptr()));
    O transformed = glob.attr("_flag_transform")(insensitive);
    O compiler = glob.attr("_wcparse").attr("compile");
    O compiled = compiler(extension_expression, transformed);
    // globfilter's flags here do not include REALPATH. No library walker/match
    // wrapper executes: apply its compiled include/exclude regexes directly.
    if (flag(compiled, "_real"))
        graphpy::raise(PyExc_RuntimeError, py::str("unexpected WCMatch real-path extension filter"));
    O include = compiled.attr("_include");
    O exclude = compiled.attr("_exclude");
    O unique = checked(PySet_New(nullptr));
    for (py::handle filename : discovered) {
        bool matched = false;
        for (py::handle pattern : py::reinterpret_borrow<py::iterable>(include)) {
            if (graphpy::truth(own(pattern).attr("fullmatch")(filename))) { matched = true; break; }
        }
        if (matched && !exclude.is_none()) {
            for (py::handle pattern : py::reinterpret_borrow<py::iterable>(exclude)) {
                if (graphpy::truth(own(pattern).attr("fullmatch")(filename))) { matched = false; break; }
            }
        }
        if (matched) {
            O path = checked(PyNumber_TrueDivide(directory.ptr(), filename.ptr()));
            O text = checked(PyObject_Str(path.ptr()));
            if (PySet_Add(unique.ptr(), text.ptr()) < 0) throw py::error_already_set();
        }
    }
    // Preserve CPython set iteration and stable equal-key sort ties, rather than
    // introducing C++ hash order or an unsolicited lexical tie breaker.
    O ordered = checked(PySequence_List(unique.ptr()));
    ordered.attr("sort")(py::arg("key") = py::cpp_function(&alphanumeric));
    py::list result;
    for (py::handle path : py::reinterpret_borrow<py::iterable>(ordered))
        result.append(name(globals, "Path")(path));
    return result;
}

class FileIterator {
    py::dict globals_;
    O directory_;
    O paths_;
    O cursor_ = py::none();
    O index_ = py::int_(0);
    bool closed_ = false;
    bool running_ = false;

    // Advances the cursor and returns the token (path, index). The end and any
    // error close the iterator, as a generator. Callers hold the running_ guard.
    O describe_step() {
        if (closed_) exhausted();
        O path;
        O index = index_;
        try {
            if (cursor_.is_none()) cursor_ = checked(PyObject_GetIter(paths_.ptr()));
            PyObject *raw = PyIter_Next(cursor_.ptr());
            if (!raw) {
                closed_ = true;
                if (PyErr_Occurred()) throw py::error_already_set();
                exhausted();
            }
            path = py::reinterpret_steal<O>(raw);
            index_ = checked(PyNumber_Add(index_.ptr(), py::int_(1).ptr()));
        } catch (...) {
            closed_ = true;
            throw;
        }
        return py::make_tuple(path, index);
    }
public:
    FileIterator(const py::dict &globals, const O &directory, const O &paths)
        : globals_(globals), directory_(directory), paths_(paths) {}
    O describe() {
        ActiveIteration active(running_);
        return describe_step();
    }
    // Loads a described item: a function of the token, the directory and the
    // node module's globals only, so it needs no guard and may run on any thread.
    // An Exception is the item's value; anything else propagates, and only
    // next() closes the iterator for it.
    O materialize(const O &token) const {
        O described = graphpy::unpack(token, 2);
        O path = item(described, 0), index = item(described, 1);
        try {
            O callback = name(globals_, "load_image_node");
            O loaded = graphpy::unpack(callback(path), 3);
            O image = item(loaded, 0), image_directory = item(loaded, 1), basename = item(loaded, 2);
            O relative = name(globals_, "os").attr("path").attr("relpath")(image_directory, directory_);
            return py::make_tuple(image, relative, basename, index);
        } catch (const py::error_already_set &error) {
            if (error.matches(PyExc_Exception)) return error.value();
            throw;
        }
    }
    O next() {
        ActiveIteration active(running_);
        O token = describe_step();
        try {
            return materialize(token);
        } catch (const py::error_already_set &) {
            closed_ = true;
            throw;
        }
    }
    void close() {
        if (running_) throw py::value_error("generator already executing");
        closed_ = true;
        cursor_ = py::none();
    }
};

O load_images(const py::dict &globals, const O &directory, const O &use_glob,
              const O &recursive, O expression, const O &use_limit, const O &limit, const O &fail_fast) {
    O extensions = name(globals, "get_available_image_formats")();
    if (!graphpy::truth(use_glob)) expression = py::str(graphpy::truth(recursive) ? "**/*" : "*");
    O paths = name(globals, "list_glob")(directory, expression, extensions);
    Py_ssize_t length = PyObject_Length(paths.ptr());
    if (length < 0) throw py::error_already_set();
    if (!length) {
        O formatted = checked(PyObject_Format(directory.ptr(), nullptr));
        O message = checked(PyUnicode_Concat(formatted.ptr(), py::str(" has no valid images.").ptr()));
        graphpy::raise(PyExc_FileNotFoundError, message);
    }
    if (graphpy::truth(use_limit)) paths = paths[py::slice(py::none(), limit, py::none())];
    // list_glob is a native-backed public helper; retain its callable boundary.
    length = PyObject_Length(paths.ptr());
    if (length < 0) throw py::error_already_set();
    py::cpp_function supplier([globals, directory, paths]() {
        return std::make_shared<FileIterator>(globals, directory, paths);
    });
    O generator = name(globals, "Generator")(supplier, py::int_(length));
    O configured = generator.attr("with_fail_fast")(fail_fast);
    // The post-limit snapshot the iterator walks (read-ahead safety).
    configured.attr("source_paths") = py::tuple(paths);
    return py::make_tuple(configured, directory);
}
}

void cn_bind_file_sequence(py::module_ &module) {
    py::class_<FileIterator, std::shared_ptr<FileIterator>>(module, "_FileSequenceIterator")
        .def("__iter__", [](FileIterator &self) -> FileIterator & { return self; },
             py::return_value_policy::reference_internal)
        .def("__next__", &FileIterator::next)
        .def("describe", &FileIterator::describe)
        .def("materialize", &FileIterator::materialize, py::arg("token"))
        .def("close", &FileIterator::close);
    module.def("file_sequence_extension_filter", &extension_filter);
    module.def("file_sequence_alphanumeric", &alphanumeric);
    module.def("file_sequence_list_glob", &list_glob);
    module.def("file_sequence_load_images", &load_images);
}
