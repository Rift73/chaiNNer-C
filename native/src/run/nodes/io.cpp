/* Native entry points of the image I/O nodes (Load Image, Save Image), over the native codecs (image_decode.hpp,
 * image_codecs.h, image_encode.hpp) and atomic writes (atomic_file.hpp). An entry point takes a file only where the
 * codecs give exactly today's result; every other case (a format that stays on Pillow/OpenCV/texconv, an error, a
 * variant the codecs decline) goes to the node's Python body (RunPython), which gives today's result or error. A
 * declined save has written nothing (writes are atomic), so the Python body starts afresh. */
#include "../kernel_status.hpp"

#include "atomic_file.hpp"
#include "codec_output.hpp"
#include "image_codecs.h"
#include "image_decode.hpp"
#include "image_encode.hpp"
#include "save_path.hpp"

#include <array>
#include <memory>
#include <new>
#include <string>
#include <string_view>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace {

using namespace cn::run;

#ifdef _WIN32
/* nodes/utils/utils.py split_file_path: os.path.splitext, then os.path.split of the base (ntpath, both separators). */
struct SplitPath {
    std::wstring directory, name, extension;
};
SplitPath split_file_path(const std::wstring &p)
{
    const auto separator = [](wchar_t c) { return c == L'\\' || c == L'/'; };
    // genericpath._splitext: the last dot after the last separator, unless the name is all dots before it.
    std::wstring base = p, extension;
    const size_t slash = p.find_last_of(L"\\/"), dot = p.rfind(L'.');
    if (dot != std::wstring::npos && (slash == std::wstring::npos || dot > slash)) {
        for (size_t i = slash == std::wstring::npos ? 0 : slash + 1; i < dot; ++i)
            if (p[i] != L'.') {
                base = p.substr(0, dot);
                extension = p.substr(dot);
                break;
            }
    }
    // ntpath.split: the drive and root stay with the head, whose trailing separators are stripped.
    size_t root = 0;
    if (base.size() >= 2 && separator(base[0]) && separator(base[1])) {
        // \\server\share (or \\?\...): the root ends after the share component and its separator.
        size_t server = base.find_first_of(L"\\/", 2);
        size_t share = server == std::wstring::npos ? std::wstring::npos : base.find_first_of(L"\\/", server + 1);
        root = share == std::wstring::npos ? base.size() : share + 1;
    } else {
        if (base.size() >= 2 && base[1] == L':') root = 2;
        if (root < base.size() && separator(base[root])) ++root;
    }
    size_t i = base.size();
    while (i > root && !separator(base[i - 1])) --i;
    std::wstring head = base.substr(root, i - root);
    while (!head.empty() && separator(head.back())) head.pop_back();
    return {base.substr(0, root) + head, base.substr(i), extension};
}

/* The UTF-8 of a wide string; false for one that is not valid UTF-16 (a lone surrogate). */
bool utf8(const std::wstring &text, std::string &out)
{
    if (text.empty()) {
        out.clear();
        return true;
    }
    const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), int(text.size()), nullptr, 0, nullptr, nullptr);
    if (size <= 0) return false;
    out.resize(size_t(size));
    return WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), int(text.size()), out.data(), size, nullptr,
                               nullptr) == size;
}

/* get_ext: the extension, lowercased; false for a non-ASCII one, which matches no format list. */
bool ascii_lower(const std::wstring &extension, std::string &out)
{
    out.clear();
    for (wchar_t c : extension) {
        if (c > 0x7F) return false;
        out.push_back(char(c >= L'A' && c <= L'Z' ? c - L'A' + L'a' : c));
    }
    return true;
}
#else
/* nodes/utils/utils.py split_file_path: os.path.splitext, then os.path.split of the base (posixpath: '/' alone, no
 * drive; the head keeps a root of slashes and otherwise loses its trailing ones). On the path's file system bytes. */
struct SplitPath {
    std::string directory, name, extension;
};
SplitPath split_file_path(const std::string &p)
{
    // genericpath._splitext: the last dot after the last separator, unless the name is all dots before it.
    std::string base = p, extension;
    const size_t slash = p.rfind('/'), dot = p.rfind('.');
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
        for (size_t i = slash == std::string::npos ? 0 : slash + 1; i < dot; ++i)
            if (p[i] != '.') {
                base = p.substr(0, dot);
                extension = p.substr(dot);
                break;
            }
    }
    const size_t i = base.rfind('/') + 1; // 0 without a separator
    std::string head = base.substr(0, i);
    if (head.find_first_not_of('/') != std::string::npos)
        while (head.back() == '/') head.pop_back();
    return {head, base.substr(i), extension};
}

/* The name as the engine's UTF-8 string; false for bytes that are not UTF-8, which Python decodes with surrogate
 * escapes the engine's strings cannot hold. */
bool utf8(const std::string &text, std::string &out)
{
    const auto *bytes = reinterpret_cast<const unsigned char *>(text.data());
    for (size_t i = 0, n = text.size(); i < n;) {
        const unsigned char lead = bytes[i];
        const size_t length = lead < 0x80 ? 1 : lead < 0xC2 ? 0 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : lead < 0xF5 ? 4 : 0;
        if (!length || i + length > n) return false;
        for (size_t k = 1; k < length; ++k)
            if ((bytes[i + k] & 0xC0) != 0x80) return false;
        // The second byte's range rules out overlong forms, surrogates and code points past U+10FFFF.
        const unsigned char second = length > 1 ? bytes[i + 1] : 0;
        if ((lead == 0xE0 && second < 0xA0) || (lead == 0xED && second > 0x9F) || (lead == 0xF0 && second < 0x90) ||
            (lead == 0xF4 && second > 0x8F))
            return false;
        i += length;
    }
    out = text;
    return true;
}

/* get_ext: the extension, lowercased; false for a non-ASCII one, which matches no format list. */
bool ascii_lower(const std::string &extension, std::string &out)
{
    out.clear();
    for (char c : extension) {
        if (static_cast<unsigned char>(c) > 0x7F) return false;
        out.push_back(c >= 'A' && c <= 'Z' ? char(c - 'A' + 'a') : c);
    }
    return true;
}
#endif

bool listed(std::string_view extension, std::initializer_list<std::string_view> names)
{
    for (std::string_view name : names)
        if (extension == name) return true;
    return false;
}

/* load_image.py's decoder order: pil-jpeg (.jpg .jpeg), pil-tiff-alpha (.tif .tiff), cv (OpenCV's list),
 * texconv-dds, pil (Pillow's list). The worker's Image.MAX_IMAGE_PIXELS (image_limits.py) bounds the Pillow routes. */
constexpr uint64_t pillow_max_pixels = CN_MAX_IMAGE_PIXELS; // backend/src/image_limits.py's Image.MAX_IMAGE_PIXELS

/* normalize(): the decoded samples to float32 through cn_pixels_convert_checked, as ImageOutput.enforce does (the
 * type ids are native_buffers._TYPES); a floating-point event, which NumPy would report, declines. */
Image normalized(const uint8_t *pixels, const cn::image_decode::Shape &shape)
{
    using cn::image_decode::Sample;
    int type = 0;
    switch (shape.sample) {
    case Sample::f32: type = 0; break;
    case Sample::f64: type = 1; break;
    case Sample::u8: type = 2; break;
    case Sample::u16: type = 3; break;
    case Sample::i8: type = 4; break;
    case Sample::i16: type = 5; break;
    case Sample::i32: type = 6; break;
    case Sample::u32: type = 7; break;
    case Sample::i64: type = 8; break;
    case Sample::u64: type = 9; break;
    }
    MutableImage image = MutableImage::allocate(int32_t(shape.height), int32_t(shape.width), int32_t(shape.channels));
    int events = 0;
    check(cn_pixels_convert_checked(pixels, image.data(), image.height() * size_t(image.width()) * size_t(image.channels()),
                                    type, 0, 1, &events));
    if (events) throw RunPython{};
    return std::move(image).publish();
}

/* chainner:image:load (image/io/load_image.py, image_io.cpp load_image): the image, its directory and its name. */
void load_image(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
#ifdef _WIN32
    const std::wstring &text = std::get<Path>(inputs[0]).value.native();
    if (text.starts_with(L"\\\\?\\") || text.starts_with(L"\\\\.\\")) throw RunPython{}; // device paths: ntpath's own roots
#else
    const std::string &text = std::get<Path>(inputs[0]).value.native();
#endif
    const SplitPath parts = split_file_path(text);
    std::string extension, name;
    if (!ascii_lower(parts.extension, extension) || !utf8(parts.name, name)) throw RunPython{};
    const bool pillow = listed(extension, {".jpg", ".jpeg", ".jfif"});
    // A TIFF goes first to the straight-alpha decoder, whose native half takes only its plainest files; when it
    // declines, the Python decoder may still take the file, so the node's Python body runs.
    const bool tiff = listed(extension, {".tif", ".tiff"});
    // OpenCV's list without the formats no native codec takes (JPEG 2000, Sun raster, Radiance HDR), which the Python
    // body reads once.
    const bool opencv = listed(extension, {".bmp", ".dib", ".jpe", ".png", ".webp", ".pbm", ".pgm", ".ppm", ".pxm", ".pnm",
                                           ".exr"});
    if ((!pillow && !tiff && !opencv) || !cn::image_decode::enabled()) throw RunPython{};
    std::vector<uint8_t> file;
    // Not zeroed: the codecs write every sample, and a file found corrupt after its header touches none of the pages.
    std::unique_ptr<uint8_t[]> pixels;
    cn::image_decode::Shape shape;
    // A failed allocation declines (the C codecs call this; nothing may unwind through them): the node's Python body
    // runs.
    const cn::image_decode::Allocator allocate = [&](const cn::image_decode::Shape &s) noexcept -> uint8_t * {
        shape = s;
        pixels.reset(new (std::nothrow) uint8_t[s.bytes()]);
        return pixels.get();
    };
    if (!cn::image_decode::read_file(std::filesystem::path(text), file) ||
        !(pillow ? cn::image_decode::decode_pillow_jpeg(file, pillow_max_pixels, allocate)
          : tiff ? cn::image_decode::decode_pillow_tiff_alpha(file, pillow_max_pixels, allocate)
                 : cn::image_decode::decode_opencv(file, allocate)))
        throw RunPython{};
    outputs.push_back(normalized(pixels.get(), shape));
    outputs.push_back(Path{std::filesystem::path(parts.directory)});
    outputs.push_back(std::move(name));
}

/* to_uint8 / to_uint16 of an enforced image (cn_pixels_convert_checked: float32 to 8 or 16 bits, clamped, half to
 * even); a floating-point event, which NumPy would report, declines. */
std::vector<uint8_t> quantized(const Image &image, int bits)
{
    std::vector<uint8_t> out(image.values() * size_t(bits / 8));
    int events = 0;
    check(cn_pixels_convert_checked(image.data, out.data(), image.values(), 0, bits == 8 ? 1 : 2, 1, &events));
    if (events) throw RunPython{};
    return out;
}

/* Save Image's file bytes for the formats whose encoders are native (save_image.py encoding() and image_io.cpp
 * encode()): PNG (chaiNNer-C's encoder for 8 bits, OpenCV's PngEncoder for 16), JPG, WEBP (lossless with alpha as
 * Pillow's exact save), BMP, TIFF (8/16-bit with its compression, or float32 as it is; then the ExtraSamples fix). */
std::vector<uint8_t> save_bytes(const Image &image, std::span<const Value> inputs)
{
    const std::string &format = std::get<std::string>(inputs[4]);
    const std::string &depth = std::get<std::string>(inputs[format == "tiff" ? 10 : 5]);
    const uint32_t height = uint32_t(image.height), width = uint32_t(image.width), channels = uint32_t(image.channels);
    const bool floats = format == "tiff" && depth == "f32";
    const bool sixteen = (format == "png" || format == "tiff") && depth == "u16";
    std::vector<uint8_t> pixels = floats ? std::vector<uint8_t>() : quantized(image, sixteen ? 16 : 8);
    const uint32_t sample_bytes = floats ? 4 : sixteen ? 2 : 1;
    const cn_codec_image view{floats ? reinterpret_cast<const uint8_t *>(image.data) : pixels.data(), height, width,
                              channels, sample_bytes, size_t(width) * channels * sample_bytes};
    std::vector<uint8_t> bytes;
    if (format == "png" && !sixteen) {
        const ptrdiff_t c = ptrdiff_t(channels);
        const auto result = cn::image_encode::png(pixels.data(), height, width, c, ptrdiff_t(width) * c, c, 1, bytes);
        if (result == cn::image_encode::Png::encoded) return bytes;
        // The Python body raises the same encoder's error (MemoryError or RuntimeError).
        if (result != cn::image_encode::Png::declined) throw RunPython{};
    }
    const bool webp_lossless = std::get<bool>(inputs[6]);
    const int64_t quality = std::get<int64_t>(inputs[7]), compression = std::get<int64_t>(inputs[11]);
    if (format == "webp" && webp_lossless && channels == 4)
        for (size_t i = 0; i < pixels.size(); i += 4) std::swap(pixels[i], pixels[i + 2]); // BGRA2RGBA for Pillow
    {
        cn::codecs::Output output;
        auto &out = output.value;
        cn_codec_status status = CN_CODEC_FALLBACK;
        if (format == "png") status = cn_png_encode(&view, &out);
        else if (format == "jpg" && quality >= 0 && quality <= 100)
            status = cn_jpeg_encode(&view, int(quality), uint32_t(std::get<int64_t>(inputs[8])), std::get<bool>(inputs[9]) ? 1 : 0, &out);
        else if (format == "webp" && webp_lossless && channels == 4) status = cn_webp_encode_exact(&view, &out);
        else if (format == "webp" && quality <= 100) status = cn_webp_encode(&view, webp_lossless ? 101 : int(quality), &out);
        else if (format == "bmp") status = cn_bmp_encode(&view, &out);
        else if (format == "tiff" && (floats || compression == 1 || compression == 5 || compression == 8))
            status = cn_tiff_encode(&view, floats ? 0 : int(compression), &out);
        if (status != CN_CODEC_OK) throw RunPython{};
        bytes.assign(out.data, out.data + out.size);
    }
    if (std::vector<uint8_t> fixed; format == "tiff" && cn::image_encode::tiff_extra_samples(bytes, fixed)) return fixed;
    return bytes;
}

/* A path from an engine string (UTF-8). */
std::filesystem::path utf8_path(const std::string &text)
{
    return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t *>(text.data()), text.size()));
}

/* image_io.cpp full_path: base / relative (unless empty or ".") / f"{filename}.{extension}", absolutised as
 * ntpath.abspath does (GetFullPathNameW) instead of Path.resolve(), which names the same file (ruling, lane I Q3). */
std::filesystem::path full_path(std::span<const Value> inputs)
{
    std::filesystem::path path = std::get<Path>(inputs[1]).value;
    const auto *relative = std::get_if<std::string>(&inputs[2]);
    const std::string &name = std::get<std::string>(inputs[3]);
    // A value that leaves the output directory or names a Windows device takes the Python body's error (save_path.hpp).
    if ((relative && cn::save_path::check(*relative).kind != cn::save_path::Refusal::none) ||
        cn::save_path::check(name).kind != cn::save_path::Refusal::none)
        throw RunPython{};
    if (relative && !relative->empty() && *relative != ".") path /= utf8_path(*relative);
    path /= utf8_path(name + "." + std::get<std::string>(inputs[4]));
#ifdef _WIN32
    const std::wstring &text = path.native();
    const DWORD size = GetFullPathNameW(text.c_str(), 0, nullptr, nullptr);
    if (!size) throw RunPython{};
    std::wstring full(size, L'\0');
    const DWORD written = GetFullPathNameW(text.c_str(), size, full.data(), nullptr);
    if (!written || written >= size) throw RunPython{};
    full.resize(written);
    return std::filesystem::path(full);
#else
    // Path.resolve() (posixpath.realpath) itself: links are common on Linux, and a ".." after one leaves the link's
    // target, which an absolutised spelling would not. The existing part resolved, the rest made normal. A link named
    // by the path itself takes today's path, as on Windows, where the path stays lexical and keeps it.
    std::error_code error, absent;
    std::filesystem::path full = std::filesystem::absolute(path, error);
    if (!error && std::filesystem::is_symlink(std::filesystem::symlink_status(full, absent))) throw RunPython{};
    if (!error) full = std::filesystem::weakly_canonical(full, error);
    if (error) throw RunPython{};
    return full;
#endif
}

/* chainner:image:save (image/io/save_image.py, image_io.cpp save_image): skip-existing before the image is read,
 * the parent folders made, the image read, encoded and written atomically. No outputs. */
void save_image(NodeContext &, std::span<const Value> inputs, std::vector<Value> &)
{
    const std::string &format = std::get<std::string>(inputs[4]);
    if ((format != "png" && format != "jpg" && format != "webp" && format != "bmp" && format != "tiff") ||
        !cn::image_decode::enabled())
        throw RunPython{}; // GIF, TGA, AVIF (Pillow), DDS (texconv)
    std::filesystem::path path = full_path(inputs);
#ifdef _WIN32
    const DWORD attributes = GetFileAttributesW(path.c_str());
    if (attributes != INVALID_FILE_ATTRIBUTES) {
        // A link takes today's path (Path.exists() follows it, and a dangling one does not exist there).
        if (attributes & FILE_ATTRIBUTE_REPARSE_POINT) throw RunPython{};
        if (std::get<bool>(inputs[24])) return; // skip existing files
        // Path.resolve() names an existing file by its on-disk spelling, which an overwrite keeps; a folder in the
        // way takes today's path.
        if (attributes & FILE_ATTRIBUTE_DIRECTORY) throw RunPython{};
        WIN32_FIND_DATAW found;
        const HANDLE search = FindFirstFileW(path.c_str(), &found);
        if (search == INVALID_HANDLE_VALUE) throw RunPython{};
        FindClose(search);
        path.replace_filename(found.cFileName);
    } else {
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) throw RunPython{};
    }
#else
    std::error_code error;
    const std::filesystem::file_status status = std::filesystem::symlink_status(path, error);
    if (status.type() != std::filesystem::file_type::not_found) {
        if (error) throw RunPython{};
        if (std::get<bool>(inputs[24])) return; // skip existing files (full_path declined a link)
        // A folder in the way takes today's path; Linux names keep their case, so an overwrite keeps the path's.
        if (std::filesystem::is_directory(status)) throw RunPython{};
    } else {
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) throw RunPython{};
    }
#endif
    const Value &value = std::get<std::shared_ptr<LazyValue>>(inputs[0])->value();
    const std::vector<uint8_t> bytes = save_bytes(std::get<Image>(value), inputs);
    try {
        cn::files::write_atomic(path, bytes);
    } catch (const cn::files::FileError &) {
        throw RunPython{}; // nothing written; the Python body raises today's error
    }
}

}  // namespace

CN_RUN_NODES_DECLARE(io)
{
    registry.add("chainner:image:load", load_image);
    registry.add("chainner:image:save", save_image);
}
