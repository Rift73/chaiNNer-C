/* Native entry points of utility text and regex nodes, Directory to Text and Parse Number (pattern:
 * image_utility.cpp).
 *
 * The Python bodies are str operations on code points (utility_text.cpp's units and CPython's str methods). Text
 * arrives as UTF-8 std::string; a str holding a lone surrogate crosses as a PyRef, as does an int beyond int64, and
 * both decline to the Python body (RunPython), as does every error whose text needs CPython's repr. Regex nodes run
 * chainner_ext's engine (RustRegex's) through its capsule API (regex_api.hpp). */
#include "run_engine.hpp"

#include "regex_api.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace {

using namespace cn::run;

const std::string &text(const Value &value)
{
    if (const auto *string = std::get_if<std::string>(&value)) return *string;
    throw RunPython{};  // a lone-surrogate str
}

int64_t integer(const Value &value)
{
    if (const auto *number = std::get_if<int64_t>(&value)) return *number;
    throw RunPython{};  // an int beyond int64
}

bool none(const Value &value) { return std::holds_alternative<std::monostate>(value); }

/* Code points of UTF-8 text (every byte but continuation bytes starts one). */
int64_t length(std::string_view utf8)
{
    int64_t count = 0;
    for (char byte : utf8) count += (static_cast<unsigned char>(byte) & 0xC0) != 0x80;
    return count;
}

/* The byte offset of code point index (0 <= index <= length). */
size_t offset(std::string_view utf8, int64_t index)
{
    size_t at = 0;
    for (; index > 0; --index) {
        do ++at;
        while (at < utf8.size() && (static_cast<unsigned char>(utf8[at]) & 0xC0) == 0x80);
    }
    return at;
}

/* text[start:stop] with Python's step-1 slice rules (PySlice_AdjustIndices); stop nullopt is the end. */
std::string slice(std::string_view utf8, int64_t start, std::optional<int64_t> stop)
{
    int64_t size = length(utf8);
    auto adjust = [size](int64_t index) {
        if (index < 0) return index + size < 0 ? 0 : index + size;
        return index > size ? size : index;
    };
    int64_t first = adjust(start);
    int64_t last = stop ? adjust(*stop) : size;
    if (last <= first) return {};
    size_t begin = offset(utf8, first);
    return std::string(utf8.substr(begin, offset(utf8.substr(begin), last - first)));
}

/* chainner:utility:note (text/note.py, execution_ops.cpp execution_note): no outputs, whatever the inputs. */
void note(NodeContext &, std::span<const Value>, std::vector<Value> &) {}

/* chainner:utility:text_append (text/text_append.py): separator.join of the inputs that are not None. */
void text_append(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const std::string &separator = text(inputs[0]);
    std::string result;
    bool first = true;
    for (const Value &input : inputs.subspan(1)) {
        if (none(input)) continue;
        if (!first) result += separator;
        result += text(input);
        first = false;
    }
    outputs.push_back(std::move(result));
}

/* chainner:utility:text_length (text/text_length.py): len(text). */
void text_length(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    outputs.push_back(length(text(inputs[0])));
}

/* chainner:utility:text_padding (utility_text.cpp text_padding): str.rjust / ljust / center (unicodeobject.c pad
 * and center: left = marg / 2 + (marg & width & 1)). */
void text_padding(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const std::string &value = text(inputs[0]);
    int64_t width = integer(inputs[1]);
    const std::string &fill = text(inputs[2]);
    const std::string &alignment = std::get<std::string>(inputs[3]);
    if (alignment != "start" && alignment != "end" && alignment != "center") throw RunPython{};
    if (length(fill) != 1) throw RunPython{};  // TypeError: TextInput's length bounds admit one character only
    // A huge width is MemoryError or a huge allocation in CPython; leave both to it.
    if (width > (int64_t(1) << 26)) throw RunPython{};
    int64_t size = length(value);
    if (size >= width) {
        outputs.push_back(inputs[0]);
        return;
    }
    int64_t margin = width - size;
    int64_t left = alignment == "start" ? margin : alignment == "end" ? 0 : margin / 2 + (margin & width & 1);
    std::string result;
    result.reserve(size_t(margin) * fill.size() + value.size());
    for (int64_t i = 0; i < left; ++i) result += fill;
    result += value;
    for (int64_t i = left; i < margin; ++i) result += fill;
    outputs.push_back(std::move(result));
}

/* chainner:utility:text_replace (utility_text.cpp text_replace): str.replace(old, new[, 1]). An empty old inserts new
 * before each code point and at the end (count of them at most); a non-empty one matches bytes, which in valid
 * UTF-8 start and end on code points. */
void text_replace(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const std::string &value = text(inputs[0]);
    const std::string &old = text(inputs[1]);
    const std::string &replacement = text(inputs[2]);
    int64_t mode = std::get<int64_t>(inputs[3]);
    int64_t count = mode == 0 ? std::numeric_limits<int64_t>::max() : 1;
    std::string result;
    if (old.empty()) {
        size_t at = 0;
        for (int64_t inserted = 0;; ++inserted) {
            if (inserted < count) result += replacement;
            if (at == value.size()) break;
            size_t next = offset(std::string_view(value).substr(at), 1) + at;
            result.append(value, at, next - at);
            at = next;
        }
    } else {
        size_t at = 0;
        for (int64_t replaced = 0; replaced < count; ++replaced) {
            size_t found = value.find(old, at);
            if (found == std::string::npos) break;
            result.append(value, at, found - at);
            result += replacement;
            at = found + old.size();
        }
        result.append(value, at, std::string::npos);
    }
    outputs.push_back(std::move(result));
}

/* chainner:utility:text_slice (utility_text.cpp text_slice); only the inputs the operation reads must be ints. */
void text_slice(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const std::string &value = text(inputs[0]);
    switch (std::get<int64_t>(inputs[1])) {
    case 0: outputs.push_back(slice(value, integer(inputs[2]), std::nullopt)); return;
    case 1: {
        int64_t start = std::max(integer(inputs[2]), -length(value));
        int64_t size = integer(inputs[3]);
        // Python's start + length is unbounded; any stop past the end slices as the end.
        int64_t stop = start > std::numeric_limits<int64_t>::max() - size ? std::numeric_limits<int64_t>::max()
                                                                         : start + size;
        // A negative start whose length reaches the end slices to the end (utility_text.cpp text_slice).
        if (start < 0 && stop >= 0) outputs.push_back(slice(value, start, std::nullopt));
        else outputs.push_back(slice(value, start, stop));
        return;
    }
    case 2: {
        int64_t maximum = integer(inputs[4]);
        if (maximum == 0) {
            outputs.push_back(std::string());
            return;
        }
        const std::string &alignment = std::get<std::string>(inputs[5]);
        if (alignment == "start") outputs.push_back(slice(value, 0, maximum));
        else if (alignment == "end") outputs.push_back(slice(value, -maximum, std::nullopt));
        else outputs.push_back(std::monostate{});
        return;
    }
    default: outputs.push_back(std::monostate{}); return;
    }
}

/* ReplacementString (nodes/utils/replacement.py, utility_text.cpp replacement_init/replace): literals and
 * interpolation names alternate, literals.size() == names.size() + 1. */
struct Replacement {
    std::vector<std::string> literals{std::string()};
    std::vector<std::string> names;
};

/* A dict[str, str] in insertion order: assigning an existing key keeps its place. */
class Replacements {
public:
    void set(std::string key, std::string value)
    {
        for (auto &entry : entries_) {
            if (entry.first == key) {
                entry.second = std::move(value);
                return;
            }
        }
        entries_.emplace_back(std::move(key), std::move(value));
    }
    const std::string *find(std::string_view key) const
    {
        for (const auto &entry : entries_)
            if (entry.first == key) return &entry.second;
        return nullptr;
    }
    std::string keys() const  // ", ".join(replacements.keys())
    {
        std::string joined;
        for (size_t i = 0; i < entries_.size(); ++i) joined += (i ? ", " : "") + entries_[i].first;
        return joined;
    }

private:
    std::vector<std::pair<std::string, std::string>> entries_;
};

/* replacement_init's scan: '{{' is a literal '{', '{name}' an interpolation, any other '{' a literal. Names allow
 * letters, digits and '_' (Py_UNICODE_ISALNUM); a non-ASCII character decides through CPython's Unicode tables
 * unless an invalid ASCII character already fails the name, so that case declines. */
Replacement parse_replacement(std::string_view pattern)
{
    Replacement parsed;
    size_t cursor = 0, last = 0;
    while (cursor < pattern.size()) {
        if (pattern[cursor] != '{') {
            ++cursor;
            continue;
        }
        size_t end = cursor + 1;
        bool escaped = end < pattern.size() && pattern[end] == '{';
        if (escaped) {
            ++end;
        } else {
            while (end < pattern.size() && pattern[end] != '{' && pattern[end] != '}') ++end;
            if (end == pattern.size() || pattern[end] != '}') {
                ++cursor;
                continue;
            }
            ++end;
        }
        parsed.literals.back().append(pattern.substr(last, cursor - last));
        last = end;
        if (escaped) {
            parsed.literals.back() += '{';
        } else {
            std::string_view name = pattern.substr(cursor + 1, end - cursor - 2);
            if (name.empty())
                throw NodeError("ValueError", "Invalid replacement pattern. {} is not a valid replacement. Either "
                                              "specify a name or id number, or escape a single \"{\" as \"{{\". "
                                              "Full pattern: " + std::string(pattern));
            bool unicode = false, invalid = false;
            for (char c : name) {
                auto byte = static_cast<unsigned char>(c);
                if (byte >= 0x80) unicode = true;
                else if (c != '_' && !(c >= '0' && c <= '9') && !(c >= 'a' && c <= 'z') && !(c >= 'A' && c <= 'Z'))
                    invalid = true;
            }
            if (invalid)
                throw NodeError("ValueError", "Invalid replacement pattern. {" + std::string(name) + "} is not a valid "
                                              "replacement. Names and ids only allow letters and digits. Full "
                                              "pattern: " + std::string(pattern));
            if (unicode) throw RunPython{};
            parsed.names.emplace_back(name);
            parsed.literals.emplace_back();
        }
        cursor = end;
    }
    parsed.literals.back().append(pattern.substr(last));
    return parsed;
}

/* ReplacementString.replace: each name's value, an error for the first name without one. */
std::string replace(const Replacement &parsed, const Replacements &replacements)
{
    std::string result = parsed.literals[0];
    for (size_t i = 0; i < parsed.names.size(); ++i) {
        const std::string *value = replacements.find(parsed.names[i]);
        if (!value)
            throw NodeError("ValueError", "Unknown replacement. There is no replacement with the name or id " +
                                              parsed.names[i] + ". Available replacements: " + replacements.keys() +
                                              ".");
        result += *value;
        result += parsed.literals[i + 1];
    }
    return result;
}

/* chainner:utility:text_pattern (utility_text.cpp text_pattern): {1}..{9} from the inputs that are not None. */
void text_pattern(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const std::string &pattern = text(inputs[0]);
    Replacements replacements;
    for (size_t i = 1; i < inputs.size(); ++i)
        if (!none(inputs[i])) replacements.set(std::to_string(i), text(inputs[i]));
    outputs.push_back(replace(parse_replacement(pattern), replacements));
}

/* A compiled regex, shared by the cache and the searches using it. */
using CompiledRegex = std::shared_ptr<const chn_regex>;

/* text_cache's lru_cache(maxsize=128) of compiled patterns: compiling is deterministic, so a hit stands for a fresh
 * compile. A chn_regex is immutable and searchable from several threads at once. */
class RegexCache {
public:
    CompiledRegex find(const std::string &pattern)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto entry = entries_.begin(); entry != entries_.end(); ++entry) {
            if (entry->first == pattern) {
                entries_.splice(entries_.begin(), entries_, entry);
                return entry->second;
            }
        }
        return nullptr;
    }
    void add(const std::string &pattern, CompiledRegex compiled)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.emplace_front(pattern, std::move(compiled));
        if (entries_.size() > 128) entries_.pop_back();
    }

private:
    std::mutex mutex_;
    std::list<std::pair<std::string, CompiledRegex>> entries_;  // most recently used first
};

/* RustRegex(pattern) (compile_regex): the engine's "Invalid regex: ..." as ValueError; an engine failure declines
 * (Python raises it as MemoryError or a PanicException). */
CompiledRegex compile(const chn_regex_api &api, const std::string &pattern)
{
    static RegexCache *cache = new RegexCache;  // never destroyed: no frees during interpreter finalization
    if (CompiledRegex hit = cache->find(pattern)) return hit;
    chn_regex *re = nullptr;
    chn_regex_string error{nullptr, 0};
    chn_regex_status status = api.compile(pattern.data(), pattern.size(), nullptr, &re, &error);
    std::string message(error.ptr ? error.ptr : "", error.len);
    api.string_drop(&error);
    if (status == CHN_REGEX_ERROR_INVALID) throw NodeError("ValueError", message);
    if (status != CHN_REGEX_OK) throw RunPython{};
    CompiledRegex compiled(re, [free = api.regex_free](const chn_regex *held) { free(const_cast<chn_regex *>(held)); });
    cache->add(pattern, compiled);
    return compiled;
}

const chn_regex_api &regex_api()
{
    const chn_regex_api *api = cn_regex_api();
    if (!api) throw RunPython{};  // chainner_ext could not provide its engine
    return *api;
}

/* A search's group spans (start, end byte pairs, CHN_REGEX_NO_SLOT for a group that did not take part). regex-py
 * reports character positions through its PosTranslator, and the Python body slices by them; on code point
 * boundaries that is the same text as the byte span. A boundary inside a character (where the translator may
 * panic) declines. */
std::vector<size_t> spans(const std::string &value, std::vector<size_t> slots)
{
    for (size_t at : slots) {
        if (at == CHN_REGEX_NO_SLOT) continue;
        if (at > value.size() || (at < value.size() && (static_cast<unsigned char>(value[at]) & 0xC0) == 0x80))
            throw RunPython{};
    }
    return slots;
}

/* The capture map (engine_capture_map): str(i) for every group, then every named group, each the group's text or
 * "" when it did not take part. */
Replacements capture_map(const chn_regex_api &api, const chn_regex *re, const std::string &value, const size_t *slots)
{
    size_t groups = api.captures_len(re);
    auto group = [&](size_t i) {
        size_t start = slots[2 * i], end = slots[2 * i + 1];
        return start == CHN_REGEX_NO_SLOT || end == CHN_REGEX_NO_SLOT ? std::string() : value.substr(start, end - start);
    };
    Replacements replacements;
    for (size_t i = 0; i < groups; ++i) replacements.set(std::to_string(i), group(i));
    for (size_t i = 0; i < groups; ++i) {
        size_t size = 0;
        if (const char *name = api.capture_name(re, i, &size)) replacements.set(std::string(name, size), group(i));
    }
    return replacements;
}

/* RustRegex.search(text): the first match's spans, empty for None. */
std::vector<size_t> first_match(const chn_regex_api &api, const chn_regex *re, const std::string &value)
{
    std::vector<size_t> slots(2 * api.captures_len(re));
    chn_regex_string panic{nullptr, 0};
    size_t start = api.to_byte_pos(value.data(), value.size(), 0);
    chn_regex_status status = api.search_at(re, value.data(), value.size(), start, slots.data(), slots.size(), &panic);
    api.string_drop(&panic);
    if (status == CHN_REGEX_NO_MATCH) return {};
    if (status != CHN_REGEX_OK) throw RunPython{};
    return spans(value, std::move(slots));
}

/* RustRegex.findall(text): every match's spans, 2 * captures_len per match. */
std::vector<size_t> all_matches(const chn_regex_api &api, const chn_regex *re, const std::string &value)
{
    std::vector<size_t> found, slots(2 * api.captures_len(re));
    chn_regex_iter iter;
    api.iter_init(&iter, re, value.data(), value.size());
    for (;;) {
        chn_regex_status status = api.iter_next_captures(&iter, slots.data(), slots.size());
        if (status == CHN_REGEX_NO_MATCH) return spans(value, std::move(found));
        if (status != CHN_REGEX_OK) throw RunPython{};
        found.insert(found.end(), slots.begin(), slots.end());
    }
}

/* chainner:utility:regex_find (utility_text.cpp regex_find, installed): the first match, whole (OutputMode 0) or
 * through the output pattern (1). */
void regex_find(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const std::string &value = text(inputs[0]);
    const std::string &pattern = text(inputs[1]);
    int64_t mode = std::get<int64_t>(inputs[2]);
    const chn_regex_api &api = regex_api();
    CompiledRegex re = compile(api, pattern);
    std::vector<size_t> match = first_match(api, re.get(), value);
    if (match.empty())
        throw NodeError("RuntimeError", "No match found. Unable to find the pattern '" + pattern + "' in the text.");
    if (mode == 0) {
        outputs.push_back(value.substr(match[0], match[1] - match[0]));
    } else if (mode == 1) {
        Replacements replacements = capture_map(api, re.get(), value, match.data());
        outputs.push_back(replace(parse_replacement(text(inputs[3])), replacements));
    } else {
        outputs.push_back(std::monostate{});
    }
}

/* chainner:utility:regex_replace (utility_text.cpp regex_replace): every match (ReplacementMode 0) or the first (1)
 * replaced through the replacement pattern, parsed before the search. */
void regex_replace(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const std::string &value = text(inputs[0]);
    const std::string &pattern = text(inputs[1]);
    const std::string &replacement_pattern = text(inputs[2]);
    bool first = std::get<int64_t>(inputs[3]) == 1;
    const chn_regex_api &api = regex_api();
    CompiledRegex re = compile(api, pattern);
    Replacement replacement = parse_replacement(replacement_pattern);
    std::vector<size_t> found = first ? first_match(api, re.get(), value) : all_matches(api, re.get(), value);
    if (found.empty()) {
        outputs.push_back(inputs[0]);
        return;
    }
    size_t width = 2 * api.captures_len(re.get());
    size_t count = first ? 1 : found.size() / width;
    std::string result;
    size_t last = 0;
    for (size_t m = 0; m < count; ++m) {
        const size_t *slots = found.data() + m * width;
        result.append(value, last, slots[0] - last);
        result += replace(replacement, capture_map(api, re.get(), value, slots));
        last = slots[1];
    }
    result.append(value, last, std::string::npos);
    outputs.push_back(std::move(result));
}

/* chainner:utility:directory_to_text (directory/directory_to_text.py): str(directory), the path's UTF-16 as UTF-8;
 * a lone surrogate (which UTF-8 cannot hold) declines. On Linux the path is bytes that str() decodes with the file
 * system encoding: ASCII names are their bytes in any such encoding, and any other byte declines. */
void directory_to_text(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
#ifdef _WIN32
    const std::wstring &wide = std::get<Path>(inputs[0]).value.native();
    std::string result;
    result.reserve(wide.size());
    for (size_t i = 0; i < wide.size(); ++i) {
        uint32_t unit = wide[i];
        if (unit >= 0xD800 && unit <= 0xDFFF) {
            if (unit >= 0xDC00 || i + 1 == wide.size() || wide[i + 1] < 0xDC00 || wide[i + 1] > 0xDFFF)
                throw RunPython{};
            unit = 0x10000 + ((unit - 0xD800) << 10) + (uint32_t(wide[++i]) - 0xDC00);
        }
        if (unit < 0x80) {
            result += char(unit);
        } else if (unit < 0x800) {
            result += char(0xC0 | (unit >> 6));
            result += char(0x80 | (unit & 0x3F));
        } else if (unit < 0x10000) {
            result += char(0xE0 | (unit >> 12));
            result += char(0x80 | ((unit >> 6) & 0x3F));
            result += char(0x80 | (unit & 0x3F));
        } else {
            result += char(0xF0 | (unit >> 18));
            result += char(0x80 | ((unit >> 12) & 0x3F));
            result += char(0x80 | ((unit >> 6) & 0x3F));
            result += char(0x80 | (unit & 0x3F));
        }
    }
    outputs.push_back(std::move(result));
#else
    const std::string &bytes = std::get<Path>(inputs[0]).value.native();
    for (char byte : bytes) {
        if (static_cast<unsigned char>(byte) >= 0x80) throw RunPython{};
    }
    outputs.push_back(bytes);
#endif
}

/* chainner:utility:parse_number (utility_scalar.cpp parse_number): int(text, base)'s success path for ASCII text
 * (PyLong_FromString: whitespace " \t\n\v\f\r" around, a sign, the base's 0x/0o/0b prefix with one optional '_'
 * after it, digits with single '_' between them) whose value fits int64. Non-ASCII text (Unicode digits and
 * spaces), more than 640 characters (the shortest int_max_str_digits limit), errors (repr in their text) and big
 * ints decline. */
void parse_number(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const std::string &value = text(inputs[0]);
    int64_t base = integer(inputs[1]);
    if (value.size() > 640 || base < 2 || base > 36) throw RunPython{};
    for (char c : value)
        if (static_cast<unsigned char>(c) >= 0x7F) throw RunPython{};
    auto space = [](char c) { return c == ' ' || (c >= '\t' && c <= '\r'); };
    size_t at = 0, end = value.size();
    while (at < end && space(value[at])) ++at;
    while (end > at && space(value[end - 1])) --end;
    bool negative = false;
    if (at < end && (value[at] == '+' || value[at] == '-')) negative = value[at++] == '-';
    if (end - at >= 2 && value[at] == '0') {
        char kind = char(value[at + 1] | 0x20);
        if ((base == 16 && kind == 'x') || (base == 8 && kind == 'o') || (base == 2 && kind == 'b')) {
            at += 2;
            if (at < end && value[at] == '_') ++at;
        }
    }
    if (at == end) throw RunPython{};
    uint64_t magnitude = 0;
    const uint64_t limit = negative ? uint64_t(1) << 63 : (uint64_t(1) << 63) - 1;
    bool underscore = true;  // a leading '_' is invalid, as is a doubled or trailing one
    for (; at < end; ++at) {
        char c = value[at];
        if (c == '_') {
            if (underscore) throw RunPython{};
            underscore = true;
            continue;
        }
        int digit = c >= '0' && c <= '9' ? c - '0' : (c | 0x20) >= 'a' && (c | 0x20) <= 'z' ? (c | 0x20) - 'a' + 10 : 99;
        if (digit >= base) throw RunPython{};
        if (magnitude > (limit - uint64_t(digit)) / uint64_t(base)) throw RunPython{};
        magnitude = magnitude * uint64_t(base) + uint64_t(digit);
        underscore = false;
    }
    if (underscore) throw RunPython{};
    outputs.push_back(negative ? int64_t(0 - magnitude) : int64_t(magnitude));
}

}  // namespace

CN_RUN_NODES_DECLARE(utility_text)
{
    registry.add("chainner:utility:directory_to_text", directory_to_text);
    registry.add("chainner:utility:note", note);
    registry.add("chainner:utility:parse_number", parse_number);
    registry.add("chainner:utility:regex_find", regex_find);
    registry.add("chainner:utility:regex_replace", regex_replace);
    registry.add("chainner:utility:text_append", text_append);
    registry.add("chainner:utility:text_length", text_length);
    registry.add("chainner:utility:text_padding", text_padding);
    registry.add("chainner:utility:text_pattern", text_pattern);
    registry.add("chainner:utility:text_replace", text_replace);
    registry.add("chainner:utility:text_slice", text_slice);
}
