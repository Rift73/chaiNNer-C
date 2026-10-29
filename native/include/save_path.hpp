/* The rule for a writer's Subdirectory Path and file name (Save Image, Save DDS Texture), checked before any join, by
 * the value's text alone (links and junctions do not change the answer). A value that would put the file outside the
 * output directory is refused: a ".." component (split on / and \), a rooted value (a leading / or \) or, on Windows,
 * one with a drive (as ntpath.splitdrive: a first character followed by ':'). On Windows a component that is a device
 * name (ntpath.isreserved's: CON, PRN, AUX, NUL, CONIN$, CONOUT$, COM and LPT with 1-9 or a superscript 1, 2 or 3;
 * any case, with or without an extension) is refused too: older Windows sends such a file to the device, as texconv
 * still does on Windows 11. "." components, ".." inside a name ("..foo") and separators in a name stay allowed.
 * image_io.cpp full_path raises the error; the engine's Save Image (run/nodes/io.cpp) declines such a value to it. */
#pragma once

#include <string>
#include <string_view>

namespace cn::save_path {

struct Refusal {
    enum Kind { none, leaves, reserved } kind = none;
    std::string_view part;  // reserved: the component that is a device name
};

#ifdef _WIN32
/* Whether a component names a device, as ntpath.isreserved reads it: its text before the first '.', trailing spaces
 * removed, in capitals, is one of the device names above. */
inline bool device(std::string_view part)
{
    std::string_view base = part.substr(0, part.find('.'));
    while (!base.empty() && base.back() == ' ') base.remove_suffix(1);
    std::string name(base);
    for (char &c : name)
        if (c >= 'a' && c <= 'z') c = char(c - 'a' + 'A');
    if (name == "CON" || name == "PRN" || name == "AUX" || name == "NUL" || name == "CONIN$" || name == "CONOUT$")
        return true;
    if (name.size() < 4 || (name.compare(0, 3, "COM") != 0 && name.compare(0, 3, "LPT") != 0)) return false;
    const std::string_view digit = std::string_view(name).substr(3);  // 1-9, or a superscript 1, 2 or 3 (UTF-8)
    return (digit.size() == 1 && digit[0] >= '1' && digit[0] <= '9') || digit == "\xC2\xB9" ||
           digit == "\xC2\xB2" || digit == "\xC2\xB3";
}
#endif

/* Why the UTF-8 value is refused, if it is. */
inline Refusal check(std::string_view value)
{
    if (value.empty()) return {};
    if (value.front() == '/' || value.front() == '\\') return {Refusal::leaves, {}};
#ifdef _WIN32
    const unsigned char lead = static_cast<unsigned char>(value.front());
    const size_t first = lead >= 0xF0 ? 4u : lead >= 0xE0 ? 3u : lead >= 0xC0 ? 2u : 1u;  // its first code point
    if (value.size() > first && value[first] == ':') return {Refusal::leaves, {}};
#endif
    Refusal refusal;
    for (size_t start = 0;;) {
        const size_t end = value.find_first_of("/\\", start);
        const std::string_view part = value.substr(start, end == std::string_view::npos ? end : end - start);
        if (part == "..") return {Refusal::leaves, {}};
#ifdef _WIN32
        if (refusal.kind == Refusal::none && device(part)) refusal = {Refusal::reserved, part};
#endif
        if (end == std::string_view::npos) return refusal;
        start = end + 1;
    }
}

}  // namespace cn::save_path
