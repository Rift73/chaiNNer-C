/* chaiNNer file-sequence application control (GPL-3.0).
 * Traversal adapted from WCMatch 11.0.1 glob.py, MIT license,
 * Copyright (c) 2018 - 2026 Isaac Muse. Its complete notice ships with the bundled
 * wcmatch distribution (wcmatch-11.0.1.dist-info/licenses/LICENSE.md).
 * WCMatch/bracex pattern parsing remains an explicit library dependency.
 * Directory walking, pattern application, deduplication, natural-key assembly,
 * snapshot/limit policy and lazy per-file iterator state execute in C++. Load Images'
 * own expressions (non-recursive and recursive) list with FindFirstFileExW, without
 * the GIL. On Linux every expression takes the walker over os.scandir (WCMatch's POSIX
 * rules: case-sensitive names, no drives), so the natural sort alone orders the list.
 */
#include "graph_python.hpp"
#include "graph_unpack.hpp"
#include <sys/stat.h>
#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#endif

namespace py = pybind11;
namespace {
using graphpy::O;
O checked(PyObject *value) {
    if (!value) throw py::error_already_set();
    return py::reinterpret_steal<O>(value);
}
// A str as the UTF-16 Windows takes (lone surrogates kept, as os.scandir reports them).
std::wstring to_wide(const O &text) {
    Py_ssize_t size = 0;
    wchar_t *buffer = PyUnicode_AsWideCharString(text.ptr(), &size);
    if (!buffer) throw py::error_already_set();
    std::wstring result(buffer, static_cast<size_t>(size));
    PyMem_Free(buffer);
    return result;
}

// Directory identities: os.path.normcase(os.path.realpath(path)) as UTF-16.
using Visited = std::unordered_set<std::wstring>;
#ifdef _WIN32
// Below the classic limit for a directory path, Windows opens a path without the
// long-path setting, so realpath returns the final path without its \\?\ prefix.
constexpr size_t short_directory_path = MAX_PATH - 12;

// A name Windows keeps as it is in a path: no trailing dot or space (stripped), no
// control character and no device name (CON, NUL, COM1, ...).
bool plain_name(std::wstring_view name) {
    if (name.empty() || name.back() == L'.' || name.back() == L' ') return false;
    for (wchar_t c : name)
        if (c < L' ') return false;
    std::wstring_view stem = name.substr(0, name.find(L'.'));
    while (!stem.empty() && stem.back() == L' ') stem.remove_suffix(1);
    std::wstring upper;
    for (wchar_t c : stem) upper += c >= L'a' && c <= L'z' ? static_cast<wchar_t>(c - 32) : c;
    for (const wchar_t *device : {L"CON", L"PRN", L"AUX", L"NUL", L"CONIN$", L"CONOUT$"})
        if (upper == device) return false;
    if (upper.size() == 4 && (upper.starts_with(L"COM") || upper.starts_with(L"LPT"))) {
        const wchar_t digit = upper.back();
        if ((digit >= L'0' && digit <= L'9') || digit == L'\u00b9' || digit == L'\u00b2' || digit == L'\u00b3')
            return false;
    }
    return true;
}

// The identity of a folder without a reparse point, named `name` in the folder whose
// identity is `parent`, at `path` (os.path.join(root_dir, relative path)): the parent's
// final path and the name as the listing reports it, lowercased as normcase does
// (LCMapStringEx maps each UTF-16 unit on its own, so a path lowercases part by part).
// None when only realpath can tell: an unusual name or a long path.
std::optional<std::wstring> child_identity(const std::wstring &parent, std::wstring_view name,
                                           size_t path) {
    if (parent.empty() || path >= short_directory_path || !plain_name(name)) return std::nullopt;
    std::wstring result = parent;
    if (result.back() != L'\\') result += L'\\';
    const size_t start = result.size();
    const int size = static_cast<int>(name.size());
    result.resize(start + name.size());
    if (LCMapStringEx(LOCALE_NAME_INVARIANT, LCMAP_LOWERCASE, name.data(), size, result.data() + start,
                      size, nullptr, nullptr, 0) != size || result.size() >= short_directory_path)
        return std::nullopt;
    return result;
}
#else
// The identity of a folder that is not a link, named `name` in the folder whose identity
// is `parent`: realpath keeps such a name as it is, and normcase changes nothing on
// POSIX. Any path length (posixpath.realpath has no limit); `path` is not needed.
std::optional<std::wstring> child_identity(const std::wstring &parent, std::wstring_view name, size_t) {
    if (parent.empty()) return std::nullopt;
    std::wstring result = parent;
    if (result.back() != L'/') result += L'/';
    result += name;
    return result;
}
#endif
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

struct Entry { O name; bool directory; bool hidden; bool link; bool reparse; };
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
    O full_path(const O &value) { return absolute_ ? value : join(root_, value); }
    bool exists(const O &value) { return graphpy::truth(path_.attr("lexists")(full_path(value))); }
    // A directory's identity: its final path, with every link and junction resolved.
    std::wstring identity(const O &value) {
        return to_wide(path_.attr("normcase")(path_.attr("realpath")(full_path(value))));
    }
    // A child folder's identity from its parent's, without opening the folder; realpath
    // for a junction, link or other reparse point and where child_identity cannot tell.
    std::wstring identity(const std::wstring &parent, const Entry &entry, const O &value) {
        if (!entry.reparse) {
            const size_t path = to_wide(full_path(value)).size();
            if (auto derived = child_identity(parent, to_wide(entry.name), path)) return *derived;
        }
        return identity(value);
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
            result.push_back({own(value), true, true, false, false});
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
#ifdef _WIN32
                        // The entry's own attributes: its cached lstat, no system call.
                        bool reparse = directory &&
                            (entry.attr("stat")(py::arg("follow_symlinks") = false)
                                 .attr("st_file_attributes").cast<unsigned long>() &
                             FILE_ATTRIBUTE_REPARSE_POINT);
#else
                        // POSIX has no junctions: a link is the one folder realpath renames.
                        bool reparse = link;
#endif
                        if (!directory_only || directory)
                            result.push_back({filename, directory, hidden, link, reparse});
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
    // visited: the identities a deep walk entered, null when not deep; key: the identity
    // of current in a deep walk. A deep walk enters each directory once: a junction or
    // link back to a directory it already entered (a cycle, or a second way to the
    // same folder) is not entered.
    void directory(const O &current, const O &target, bool directory_only,
                   const Sink &sink, Visited *visited, const std::wstring &key) {
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
            // Images' flags never set). A junction is not a link to it.
            bool selected = target.is_none() ? !entry.hidden : match(target, entry.name);
            if (selected) sink({full, entry.directory});
            bool follow = !entry.link || follow_;
            if (visited && !entry.hidden && entry.directory && follow) {
                std::wstring child = identity(key, entry, full);
                if (visited->insert(child).second)
                    directory(full, target, directory_only, sink, visited, child);
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
            const std::wstring start = identity(current);
            Visited visited{start};
            directory(current, target, directory_only, [&](const Match &candidate) {
                if (next < parts.size()) {
                    glob(candidate.path, parts, next, sink);
                } else sink(candidate);
            }, &visited, start);
        } else if (!directory_only) {
            directory(current, target, false, sink, nullptr, {});
        } else {
            directory(current, target, true, [&](const Match &candidate) {
                if (at + 1 < parts.size()) {
                    glob(candidate.path, parts, at + 1, sink);
                } else sink(candidate);
            }, nullptr, {});
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

#ifdef _WIN32
// ---- Load Images' own expressions ("*" and "**/*") without WCMatch ----
// The Walker above with those patterns' flags (DOTGLOB: nothing hidden, no FOLLOW, no
// MARK, unique results; "*" matches every name but a drive-like "X:..."), then
// list_glob's extension filter and natural sort, in C++.

O from_wide(std::wstring_view text) {
    return checked(PyUnicode_FromWideChar(text.data(), static_cast<Py_ssize_t>(text.size())));
}
bool ascii(std::wstring_view text) {
    for (wchar_t c : text)
        if (c >= 0x80) return false;
    return true;
}

// What only the WCMatch walker reproduces: a name with a colon (drive-like to the
// patterns and to pathlib), a path longer than os.scandir accepts (a ValueError), a
// folder whose path os.scandir rewrites (below).
struct Unsupported {};
constexpr size_t scandir_limit = 32767;

// os.scandir lists a folder by its path normalized as a final part, which drops the
// trailing dots and spaces of that part (and reads a device name as the device): a
// "\*.*" after it would keep them, so such a folder is not listed here.
bool rewritten(std::wstring_view directory) {
    if (directory.empty() || directory.back() == L'\\' || directory.back() == L'/' || directory.back() == L':')
        return false;
    return !plain_name(directory.substr(directory.find_last_of(L"\\/") + 1));
}

// An entry as the Walker sees os.scandir's DirEntry on Windows.
struct Found { std::wstring name; bool directory; bool link; bool reparse; };

// os.scandir's join_path_filenameW: a separator unless the path ends in one or a colon.
std::wstring joined(const std::wstring &directory, std::wstring_view name) {
    std::wstring result = directory;
    if (!result.empty() && result.back() != L'\\' && result.back() != L'/' && result.back() != L':')
        result += L'\\';
    result += name;
    return result;
}

class PlainWalk {
public:
    // A relative path's identity (realpath; the root's is that of "").
    using Resolve = std::function<std::wstring(const std::wstring &)>;
    // A symlink entry's is_dir(), which follows it; None when it raises an OSError.
    using Follow = std::function<std::optional<bool>(const std::wstring &)>;

    // root: root_dir, listed as given; prefix: os.path.join(root_dir, "x") without the
    // "x", which the Walker's joins of root_dir and a relative path come to.
    PlainWalk(std::wstring root, std::wstring prefix, Resolve resolve, Follow follow)
        : root_(std::move(root)), prefix_(std::move(prefix)), resolve_(std::move(resolve)),
          follow_(std::move(follow)) {}

    // The relative paths the Walker yields, in its order: a folder's listing, each
    // entry followed (recursive) by the walk of its folder, each directory entered once
    // by identity. Iterative, so depth costs no stack.
    std::vector<std::wstring> run(bool recursive) {
        struct Frame { std::vector<Found> entries; size_t next; std::wstring relative; std::wstring identity; };
        std::vector<std::wstring> listed;
        Visited visited;
        std::vector<Frame> stack;
        if (recursive) stack.push_back({{}, 0, {}, *visited.insert(resolve_({})).first});
        else stack.emplace_back();
        stack.back().entries = list(root_);
        while (!stack.empty()) {
            Frame &frame = stack.back();
            if (frame.next == frame.entries.size()) {
                stack.pop_back();
                continue;
            }
            const Found &entry = frame.entries[frame.next++];
            std::wstring relative = frame.relative.empty() ? entry.name : frame.relative + L'\\' + entry.name;
            listed.push_back(relative);
            if (!recursive || !entry.directory || entry.link) continue;
            const std::wstring path = prefix_ + relative;
            std::optional<std::wstring> derived;
            if (!entry.reparse) derived = child_identity(frame.identity, entry.name, path.size());
            std::wstring identity = derived ? std::move(*derived) : resolve_(relative);
            if (!visited.insert(identity).second) continue;
            std::vector<Found> entries = list(path);
            stack.push_back({std::move(entries), 0, std::move(relative), std::move(identity)});
        }
        return listed;
    }

private:
    std::wstring root_;
    std::wstring prefix_;
    Resolve resolve_;
    Follow follow_;

    // os.scandir on Windows ("<directory>\*.*" without "." and ".."). A listing that
    // cannot start is empty and one that fails later keeps what it read, as the
    // Walker's OSError handling has it; an entry whose is_dir() raises is left out.
    std::vector<Found> list(const std::wstring &directory) {
        if (directory.size() > scandir_limit || rewritten(directory)) throw Unsupported{};
        std::vector<Found> found;
        WIN32_FIND_DATAW data;
        const std::wstring pattern = joined(directory, L"*.*");
        HANDLE handle = FindFirstFileExW(pattern.c_str(), FindExInfoBasic, &data, FindExSearchNameMatch,
                                         nullptr, FIND_FIRST_EX_LARGE_FETCH);
        if (handle == INVALID_HANDLE_VALUE) return found;
        const std::unique_ptr<void, decltype(&FindClose)> close(handle, &FindClose);
        do {
            const std::wstring_view name(data.cFileName);
            if (name == L"." || name == L"..") continue;
            if (name.find(L':') != std::wstring_view::npos) throw Unsupported{};
            const bool reparse = (data.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
            const bool symlink = reparse && data.dwReserved0 == IO_REPARSE_TAG_SYMLINK;
            bool directory_entry = (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
            if (symlink) {
                const std::wstring path = joined(directory, name);
                if (path.size() > scandir_limit) throw Unsupported{};
                const std::optional<bool> target = follow_(path);
                if (!target) continue;
                directory_entry = *target;
            }
            found.push_back({std::wstring(name), directory_entry, directory_entry && symlink, reparse});
        } while (FindNextFileW(handle, &data));
        return found;
    }
};

// One character of the extension pattern compiled with IGNORECASE: Python's re matches
// an ASCII letter to its other case and to U+0130/U+0131 (i), U+212A (k), U+017F (s).
bool same_letter(wchar_t name, wchar_t extension) {
    if (name == extension) return true;
    const wchar_t lower = static_cast<wchar_t>(extension | 0x20);
    if (lower < L'a' || lower > L'z') return false;
    if ((name | 0x20) == lower) return true;
    switch (lower) {
    case L'i': return name == 0x130 || name == 0x131;
    case L'k': return name == 0x212A;
    case L's': return name == 0x17F;
    default: return false;
    }
}
bool plain_extension(std::wstring_view extension) {
    if (extension.empty()) return false;
    for (wchar_t c : extension)
        if (c != L'.' && !(c >= L'0' && c <= L'9') && !((c | 0x20) >= L'a' && (c | 0x20) <= L'z')) return false;
    return true;
}
// "**/*@(<extensions>)" on a walked path: its name ends with one of the extensions.
bool image(std::wstring_view relative, const std::vector<std::wstring> &extensions) {
    const std::wstring_view name = relative.substr(relative.rfind(L'\\') + 1);
    for (const auto &extension : extensions) {
        if (extension.size() > name.size()) continue;
        const std::wstring_view tail = name.substr(name.size() - extension.size());
        if (std::equal(tail.begin(), tail.end(), extension.begin(), same_letter)) return true;
    }
    return false;
}

// Python's lower() and upper() of a walked path: ASCII here, other text through Python.
std::wstring lowered(const std::wstring &text) {
    if (!ascii(text)) return to_wide(from_wide(text).attr("lower")());
    std::wstring result = text;
    for (wchar_t &c : result)
        if (c >= L'A' && c <= L'Z') c = static_cast<wchar_t>(c + 32);
    return result;
}
std::u32string uppered(const std::wstring &text) {
    if (!ascii(text)) {
        O upper = from_wide(text).attr("upper")();
        Py_UCS4 *buffer = PyUnicode_AsUCS4Copy(upper.ptr());
        if (!buffer) throw py::error_already_set();
        std::u32string result(buffer, buffer + PyUnicode_GET_LENGTH(upper.ptr()));
        PyMem_Free(buffer);
        return result;
    }
    std::u32string result(text.begin(), text.end());
    for (char32_t &c : result)
        if (c >= U'a' && c <= U'z') c -= 32;
    return result;
}

// A Unicode decimal digit's value, else -1 (re's \d and int()).
int decimal(char32_t c) {
    if (c < 0x80) return c >= U'0' && c <= U'9' ? static_cast<int>(c - U'0') : -1;
    return Py_UNICODE_TODECIMAL(c);
}
// Python's comparison of two alphanumeric keys from uppercased text: text runs by code
// point, digit runs by value, and a key that ends first sorts first.
int natural_compare(std::u32string_view a, std::u32string_view b) {
    size_t i = 0, j = 0;
    while (true) {
        const size_t text_a = i, text_b = j;
        while (i < a.size() && decimal(a[i]) < 0) ++i;
        while (j < b.size() && decimal(b[j]) < 0) ++j;
        if (const int order = a.substr(text_a, i - text_a).compare(b.substr(text_b, j - text_b))) return order;
        if (i == a.size() || j == b.size()) return (i < a.size()) - (j < b.size());
        size_t digits_a = i, digits_b = j;
        while (i < a.size() && decimal(a[i]) >= 0) ++i;
        while (j < b.size() && decimal(b[j]) >= 0) ++j;
        while (digits_a < i && decimal(a[digits_a]) == 0) ++digits_a;
        while (digits_b < j && decimal(b[digits_b]) == 0) ++digits_b;
        if (i - digits_a != j - digits_b) return i - digits_a < j - digits_b ? -1 : 1;
        for (; digits_a < i; ++digits_a, ++digits_b)
            if (const int order = decimal(a[digits_a]) - decimal(b[digits_b])) return order;
    }
}

// list_glob for "*" and "**/*" with plain extensions ([.A-Za-z0-9]) under a
// WindowsPath; None when the arguments need WCMatch. The walk runs without the GIL,
// taking it back only for a junction's realpath and a symlink's is_dir().
std::optional<py::list> plain_listing(const py::dict &globals, const O &directory, const O &expression,
                                      const O &extensions) {
    if (!PyUnicode_CheckExact(expression.ptr())) return std::nullopt;
    const bool recursive = PyUnicode_CompareWithASCIIString(expression.ptr(), "**/*") == 0;
    if (!recursive && PyUnicode_CompareWithASCIIString(expression.ptr(), "*") != 0) return std::nullopt;
    O windows_path = py::module_::import("pathlib").attr("WindowsPath");
    if (Py_TYPE(directory.ptr()) != reinterpret_cast<PyTypeObject *>(windows_path.ptr())) return std::nullopt;
    if (!PyList_CheckExact(extensions.ptr()) && !PyTuple_CheckExact(extensions.ptr())) return std::nullopt;
    std::vector<std::wstring> suffixes;
    for (py::handle extension : extensions) {
        if (!PyUnicode_CheckExact(extension.ptr())) return std::nullopt;
        suffixes.push_back(to_wide(own(extension)));
        if (!plain_extension(suffixes.back())) return std::nullopt;
    }
    if (suffixes.empty()) return std::nullopt;

    O os = name(globals, "os");
    O path = os.attr("path");
    O root = os.attr("fspath")(directory);
    std::wstring root_text = to_wide(root);
    if (root_text.find(L'\0') != std::wstring::npos) return std::nullopt;
    std::wstring prefix = to_wide(path.attr("join")(root, py::str("x")));
    prefix.pop_back();
    // str(directory / relative) is base + relative; the sort compares the relative
    // parts alone, as base ends before any digit run (in a separator or a colon).
    O probe = checked(PyObject_Str(checked(PyNumber_TrueDivide(directory.ptr(), py::str("x").ptr())).ptr()));
    O base = checked(PyUnicode_Substring(probe.ptr(), 0, PyUnicode_GET_LENGTH(probe.ptr()) - 1));
    const Py_ssize_t base_length = PyUnicode_GET_LENGTH(base.ptr());
    if (base_length && Py_UNICODE_ISDECIMAL(PyUnicode_READ_CHAR(base.ptr(), base_length - 1))) return std::nullopt;

    O join = path.attr("join"), realpath = path.attr("realpath"), normcase = path.attr("normcase");
    O stat = py::module_::import("nt").attr("stat");
    const auto resolve = [&](const std::wstring &relative) {
        py::gil_scoped_acquire acquire;
        return to_wide(normcase(realpath(join(root, from_wide(relative)))));
    };
    const auto follow = [&](const std::wstring &entry) -> std::optional<bool> {
        py::gil_scoped_acquire acquire;
        try {
            const auto mode = stat(from_wide(entry)).attr("st_mode").cast<unsigned long>();
            return (mode & _S_IFMT) == _S_IFDIR;
        } catch (const py::error_already_set &error) {
            if (error.matches(PyExc_FileNotFoundError)) return false;
            if (error.matches(PyExc_OSError)) return std::nullopt;
            throw;
        }
    };
    std::vector<std::wstring> listed;
    try {
        py::gil_scoped_release release;
        listed = PlainWalk(std::move(root_text), std::move(prefix), resolve, follow).run(recursive);
    } catch (const Unsupported &) {
        return std::nullopt;
    }

    // WCMatch's unique results: a path whose lower() was already yielded is not yielded
    // again (it keeps the yielded paths as they are). Then the extension filter.
    std::unordered_set<std::wstring> seen;
    std::vector<O> paths;
    std::vector<std::u32string> keys;
    O unique = checked(PySet_New(nullptr));
    for (const std::wstring &relative : listed) {
        if (seen.contains(lowered(relative))) continue;
        seen.insert(relative);
        if (!image(relative, suffixes)) continue;
        O text = checked(PyUnicode_Concat(base.ptr(), from_wide(relative).ptr()));
        if (PySet_Add(unique.ptr(), text.ptr()) < 0) throw py::error_already_set();
        paths.push_back(std::move(text));
        keys.push_back(uppered(relative));
    }
    // sorted(set, key=alphanumeric_sort): stable over the set's iteration order.
    std::unordered_map<PyObject *, size_t> position;
    for (size_t index = 0; index < paths.size(); ++index) position.emplace(paths[index].ptr(), index);
    std::vector<size_t> order;
    for (py::handle text : unique) order.push_back(position.at(text.ptr()));
    {
        py::gil_scoped_release release;
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t a, size_t b) { return natural_compare(keys[a], keys[b]) < 0; });
    }
    O path_type = name(globals, "Path");
    py::list result;
    for (size_t index : order) result.append(path_type(paths[index]));
    return result;
}
#endif

O list_glob(const py::dict &globals, const O &directory, const O &expression, const O &extensions) {
    O extension_expression = extension_filter(extensions);
#ifdef _WIN32
    if (auto listed = plain_listing(globals, directory, expression, extensions)) return std::move(*listed);
#endif
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
