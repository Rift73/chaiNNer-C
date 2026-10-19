/* Native image decoding with today's results (GPL-3.0); see native/include/image_decode.hpp. */
#include "image_decode.hpp"

#include "image_codecs.h"

#include <algorithm>
#include <cstring>
#include <string>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace cn::image_decode {
namespace {

// The C codec's allocation callback over an Allocator, and the orientation the decoded pixels still need: an
// oriented image decodes into scratch and is turned into the caller's buffer afterwards.
struct Target {
    const Allocator &allocate;
    int orientation;
    Shape shape;
    std::vector<uint8_t> scratch;
    uint64_t max_pixels = 0; // Image.MAX_IMAGE_PIXELS where Pillow checks the size at load; 0 for none
    uint64_t max_width = 0, max_height = 0; // OpenCV's side limits on its route; 0 for none
};

// It runs inside the C codecs, so nothing may unwind out of it: a failed allocation declines the image.
uint8_t *allocate_target(void *context, uint32_t height, uint32_t width, uint32_t channels, uint32_t sample) noexcept {
    auto &target = *static_cast<Target *>(context);
    if ((target.max_pixels && uint64_t(height) * width > target.max_pixels) ||
        (target.max_width && (width > target.max_width || height > target.max_height)))
        return nullptr;
    target.shape = Shape{height, width, channels, Sample(sample)};
    if (target.orientation == 1) return target.allocate(target.shape);
    try {
        target.scratch.resize(target.shape.bytes());
    } catch (...) {
        return nullptr;
    }
    return target.scratch.data();
}

// read_cv's orient(): ImageOps.exif_transpose's flips and transpose (5-8 swap height and width), as
// np.ascontiguousarray of image[rows, columns].swapaxes(0, 1) gives them.
bool finish(Target &target) {
    if (target.orientation == 1) return true;
    const Shape &from = target.shape;
    const bool swap = target.orientation >= 5;
    const Shape to{swap ? from.width : from.height, swap ? from.height : from.width, from.channels, from.sample};
    uint8_t *out = target.allocate(to);
    if (!out) return false;
    const int o = target.orientation;
    const bool rows = o == 3 || o == 4 || o == 6 || o == 7, columns = o == 2 || o == 3 || o == 7 || o == 8;
    const size_t pixel = size_t(from.channels) * from.sample_bytes(), stride = size_t(from.width) * pixel;
    const uint8_t *in = target.scratch.data();
    for (uint32_t i = 0; i < to.height; ++i)
        for (uint32_t j = 0; j < to.width; ++j) {
            // The view before the transpose has from's shape; out[i][j] = view[j][i] when swapped.
            const uint32_t vy = swap ? j : i, vx = swap ? i : j;
            const uint32_t y = rows ? from.height - 1 - vy : vy, x = columns ? from.width - 1 - vx : vx;
            std::memcpy(out, in + y * stride + x * pixel, pixel);
            out += pixel;
        }
    return true;
}

uint32_t be16(const uint8_t *p) { return uint32_t(p[0]) << 8 | p[1]; }

// JpegImagePlugin._open (Pillow 12.3) over the file's bytes: false wherever Image.open would raise, for an MPO
// (jpeg_factory's other path), and for DHP (libjpeg refuses it); else the SOF's layers and size and info["exif"].
struct PillowJpeg {
    int layers = 0;
    uint32_t width = 0, height = 0;
    std::string exif;
};
bool pillow_jpeg_header(std::span<const uint8_t> f, PillowJpeg &out) {
    if (f.size() < 3 || f[0] != 0xFF || f[1] != 0xD8 || f[2] != 0xFF) return false;
    size_t at = 3;
    bool exif = false;
    std::vector<std::string> icc;
    // s = fp.read(n): fewer bytes at the end.
    const auto read = [&](size_t n) {
        const auto s = f.subspan(at, std::min(n, f.size() - at));
        at += s.size();
        return s;
    };
    // n = i16(fp.read(2)) - 2 and ImageFile._safe_read(fp, n); false for struct.error or "Truncated File Read".
    const auto segment = [&](std::span<const uint8_t> &s) {
        const auto length = read(2);
        if (length.size() < 2) return false;
        const long n = long(be16(length.data())) - 2;
        if (n <= 0) {
            s = {};
            return true;
        }
        s = read(size_t(n));
        return s.size() == size_t(n);
    };
    const auto starts = [](std::span<const uint8_t> s, std::string_view prefix) {
        return s.size() >= prefix.size() && std::memcmp(s.data(), prefix.data(), prefix.size()) == 0;
    };
    // s = b"\xff"; each turn: i = s[0] (IndexError when s is empty), a marker reads one more byte for i16(s).
    int byte = 0xFF;
    for (;;) {
        if (byte < 0) return false;
        if (byte != 0xFF) {
            const auto s = read(1);
            byte = s.empty() ? -1 : s[0];
            continue;
        }
        const auto second = read(1);
        if (second.empty()) return false;
        const uint32_t marker = 0xFF00 | second[0];
        if (marker == 0xFFFF) continue; // padding: s = b"\xff"
        if (marker == 0xFF00) {          // an escaped 0xFF: s = fp.read(1)
            const auto s = read(1);
            byte = s.empty() ? -1 : s[0];
            continue;
        }
        if (marker < 0xFFC0) return false; // "no marker found"
        std::span<const uint8_t> s;
        const bool sof = (marker >= 0xFFC0 && marker <= 0xFFCF && marker != 0xFFC4 && marker != 0xFFC8 &&
                          marker != 0xFFCC);
        if (marker == 0xFFDE) return false; // DHP (SOF's handler in Pillow, an error in libjpeg)
        if (sof) {
            if (!segment(s) || s.size() < 6 || s[0] != 8) return false;
            out.height = be16(&s[1]);
            out.width = be16(&s[3]);
            out.layers = s[5];
            if (out.layers != 1 && out.layers != 3 && out.layers != 4) return false;
            if (!icc.empty()) {
                // icclist.sort(); icclist[0][13] raises IndexError on a short first fragment.
                std::sort(icc.begin(), icc.end());
                if (icc.front().size() < 14) return false;
                icc.clear();
            }
            if ((s.size() - 6) % 3) return false; // the layer loop's t[1] or t[2]
        } else if (marker == 0xFFC4 || marker == 0xFFCC || marker == 0xFFDA || marker == 0xFFDC ||
                   marker == 0xFFDD || marker == 0xFFDF) {
            if (!segment(s)) return false; // Skip
            if (marker == 0xFFDA) return out.layers != 0;
        } else if (marker == 0xFFDB) {
            if (!segment(s)) return false;
            while (!s.empty()) { // DQT: 1 + 64 or 1 + 128 bytes per table
                const size_t length = s[0] / 16 == 0 ? 65 : 129;
                if (s.size() < length) return false;
                s = s.subspan(length);
            }
        } else if (marker >= 0xFFE0 && marker <= 0xFFEF) {
            if (!segment(s)) return false;
            if (marker == 0xFFE0 && starts(s, "JFIF") && s.size() < 7) return false; // i16(s, 5)
            if (marker == 0xFFE1 && starts(s, std::string_view("Exif\0\0", 6))) {
                if (exif) out.exif.append(reinterpret_cast<const char *>(s.data()) + 6, s.size() - 6);
                else out.exif.assign(reinterpret_cast<const char *>(s.data()), s.size());
                exif = true;
            }
            if (marker == 0xFFE2 && starts(s, std::string_view("ICC_PROFILE\0", 12)))
                icc.emplace_back(reinterpret_cast<const char *>(s.data()), s.size());
            if (marker == 0xFFE2 && starts(s, std::string_view("MPF\0", 4))) return false;
            if (marker == 0xFFED && starts(s, std::string_view("Photoshop 3.0\0", 14))) {
                // The 8BIM walk: struct.error ends it quietly, s[offset] past the end is an IndexError.
                size_t offset = 14;
                while (offset + 4 <= s.size() && std::memcmp(&s[offset], "8BIM", 4) == 0) {
                    offset += 4;
                    if (offset + 2 > s.size()) break;
                    const uint32_t code = be16(&s[offset]);
                    offset += 2;
                    if (offset >= s.size()) return false;
                    offset += 1 + s[offset];
                    offset += offset & 1;
                    if (offset + 4 > s.size()) break;
                    const size_t size = size_t(be16(&s[offset])) << 16 | be16(&s[offset + 2]);
                    offset += 4;
                    const size_t available = offset < s.size() ? std::min(size, s.size() - offset) : 0;
                    if (code == 0x03ED && available < 14) break; // ResolutionInfo's i32/i16 reads
                    offset += size;
                    offset += offset & 1;
                }
            }
            if (marker == 0xFFEE && starts(s, "Adobe") && s.size() < 7) return false; // i16(s, 5)
        } else if (marker == 0xFFFE) {
            if (!segment(s)) return false; // COM
        }
        // RSTn, SOI, EOI, JPG and JPGn have no handler.
        const auto next = read(1);
        byte = next.empty() ? -1 : next[0];
    }
}

// Whether Image.open of a JPEG warns (a Python UserWarning) or fails in JpegImagePlugin._read_dpi_from_exif, which
// loads info["exif"] as Image.Exif: IFD0's loader (TiffImagePlugin.ImageFileDirectory_v2.load, Pillow 12.3) warns when
// the entry count, an entry, an out-of-line value or the next-IFD offset is cut short, and reading ResolutionUnit
// (296), then XResolution (282), warns when the tag holds more than one value; with both read, x_resolution[1] raises
// an IndexError the read does not catch for a BYTE, ASCII or UNDEFINED XResolution of one value (taken as any count
// here). Conservative where the JFIF density would skip the read: such a file declines either way.
bool pillow_exif_declines(std::string_view exif) {
    using namespace std::string_view_literals;
    while (exif.starts_with("Exif\0\0"sv)) exif.remove_prefix(6);
    // Image.Exif reads an 8-byte header; a shorter one, an unknown prefix or BigTIFF raises an error the read catches.
    if (exif.size() < 8) return false;
    const auto *p = reinterpret_cast<const uint8_t *>(exif.data());
    const std::string_view prefix = exif.substr(0, 4);
    if (prefix != "MM\0*"sv && prefix != "II*\0"sv && prefix != "MM*\0"sv && prefix != "II\0*"sv && prefix != "MM\0+"sv)
        return false;
    const bool little = p[0] == 'I';
    const auto u16 = [&](size_t at) { return little ? uint32_t(p[at] | p[at + 1] << 8) : uint32_t(p[at] << 8 | p[at + 1]); };
    const auto u32 = [&](size_t at) { return little ? u16(at) | u16(at + 2) << 16 : u16(at) << 16 | u16(at + 2); };
    const size_t size = exif.size();
    const auto available = [&](uint64_t at, uint64_t count) { return at <= size && size - at >= count; };
    uint64_t at = u32(4);
    if (!available(at, 2)) return true;
    const uint32_t entries = u16(size_t(at));
    at += 2;
    struct Entry {
        bool present = false;
        uint32_t type = 0, count = 0;
    } unit, resolution;
    for (uint32_t i = 0; i < entries; ++i, at += 12) {
        if (!available(at, 12)) return true;
        const uint32_t tag = u16(size_t(at)), type = u16(size_t(at) + 2), count = u32(size_t(at) + 4);
        static constexpr uint8_t unit_sizes[17] = {0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8, 4, 0, 0, 8};
        const uint64_t bytes = type < std::size(unit_sizes) ? uint64_t(count) * unit_sizes[type] : 0;
        if (!bytes) continue; // an unsupported type, or no data: not stored
        if (bytes > 4 && !available(u32(size_t(at) + 8), bytes)) return true;
        if (tag == 296) unit = {true, type, count};
        if (tag == 282) resolution = {true, type, count};
    }
    if (!available(at, 4)) return true;
    // BYTE, ASCII and UNDEFINED load as one value; the other types as one value per count.
    const auto several = [](const Entry &e) { return e.present && e.type != 1 && e.type != 2 && e.type != 7 && e.count > 1; };
    const bool indexed = resolution.present && (resolution.type == 1 || resolution.type == 2 || resolution.type == 7);
    return unit.present && (several(unit) || several(resolution) || indexed);
}

// imdecode's codec by signature (loadsave.cpp findDecoder, in registration order: BMP, GIF, HDR, JPEG, WebP, Sun
// raster, PxM, PAM, PFM, TIFF, PNG, JPEG 2000 and the rest): a file a codec not ported here claims declines. The
// signature is the first 32 bytes (the WebP decoder's length, the longest); a shorter file pads it with spaces, so
// it declines. PxM, PAM and PFM test isspace(byte 2), which depends on the CRT locale above ASCII: such a byte is
// taken as a space, so the file declines whichever way the test goes.
enum class OpenCvCodec { other, bmp, jpeg, webp, pxm, tiff, png };
OpenCvCodec opencv_codec(std::span<const uint8_t> f) {
    if (f.size() < 32) return OpenCvCodec::other;
    const auto starts = [&](std::string_view prefix) { return std::memcmp(f.data(), prefix.data(), prefix.size()) == 0; };
    const auto space = [](uint8_t c) { return c == ' ' || (c >= '\t' && c <= '\r') || c >= 0x80; };
    using namespace std::string_view_literals;
    if (starts("BM")) return OpenCvCodec::bmp;
    if (starts("GIF") || starts("#?RGBE") || starts("#?RADIANCE")) return OpenCvCodec::other;
    if (starts("\xFF\xD8\xFF")) return OpenCvCodec::jpeg;
    if (cn_webp_signature(f.data(), f.size())) return OpenCvCodec::webp;
    if (starts("\x59\xA6\x6A\x95")) return OpenCvCodec::other; // Sun raster
    if (f[0] == 'P' && ((f[1] >= '1' && f[1] <= '7') || f[1] == 'f' || f[1] == 'F') && space(f[2]))
        return f[1] <= '6' && f[2] < 0x80 ? OpenCvCodec::pxm : OpenCvCodec::other; // PxM; PAM, PFM
    if (starts("II\x2a\0"sv) || starts("MM\0\x2a"sv) || starts("II\x2b\0"sv) || starts("MM\0\x2b"sv))
        return OpenCvCodec::tiff;
    if (starts("\x89PNG\r\n\x1a\n")) return OpenCvCodec::png;
    return OpenCvCodec::other;
}

// One of OpenCV's image size limits (loadsave.cpp reads them through utils::getConfigurationParameterSizeT at
// import): unset is the default, a plain decimal integer (what image_limits.py sets) is its value. False for every
// other value, whose reading by OpenCV is not reproduced here, and the file declines.
bool opencv_limit(const wchar_t *name, uint64_t fallback, uint64_t &limit) {
    wchar_t value[64];
    SetLastError(ERROR_SUCCESS);
    const DWORD length = GetEnvironmentVariableW(name, value, DWORD(std::size(value)));
    if (length == 0 && GetLastError() == ERROR_ENVVAR_NOT_FOUND) {
        limit = fallback;
        return true;
    }
    if (length == 0 || length >= std::size(value)) return false;
    size_t digits = 0;
    uint64_t v = 0;
    for (; digits < length && value[digits] >= L'0' && value[digits] <= L'9'; ++digits) {
        const uint64_t digit = uint64_t(value[digits] - L'0');
        if (v > (UINT64_MAX - digit) / 10) return false;
        v = v * 10 + digit;
    }
    if (!digits || digits != length) return false;
    limit = v;
    return true;
}

} // namespace

bool enabled() {
    wchar_t value[4];
    const DWORD length = GetEnvironmentVariableW(L"CHAINNER_C_NATIVE_CODECS", value, 4);
    return !(length == 1 && value[0] == L'0');
}

bool read_file(const std::filesystem::path &path, std::vector<uint8_t> &data) {
    const HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                    nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER size;
    bool ok = GetFileSizeEx(file, &size) != 0 && size.QuadPart >= 0 && uint64_t(size.QuadPart) <= SIZE_MAX;
    if (ok) {
        data.resize(size_t(size.QuadPart));
        for (size_t done = 0; ok && done < data.size();) {
            DWORD got = 0;
            const DWORD want = DWORD(std::min<size_t>(data.size() - done, 1u << 30));
            ok = ReadFile(file, data.data() + done, want, &got, nullptr) != 0 && got != 0;
            done += got;
        }
    }
    CloseHandle(file);
    return ok;
}

bool decode_pillow_jpeg(std::span<const uint8_t> file, uint64_t max_pixels, const Allocator &allocate) {
    PillowJpeg header;
    if (!pillow_jpeg_header(file, header) || !header.width || !header.height) return false;
    if (max_pixels && uint64_t(header.width) * header.height > max_pixels) return false;
    if (pillow_exif_declines(header.exif)) return false;
    Target target{allocate, exif_orientation(reinterpret_cast<const uint8_t *>(header.exif.data()), header.exif.size()),
                  {}, {}};
    if (cn_jpeg_decode(file.data(), file.size(), CN_JPEG_PILLOW, header.layers, allocate_target, &target) !=
            CN_CODEC_OK ||
        target.shape.width != header.width || target.shape.height != header.height)
        return false;
    return finish(target);
}

bool decode_opencv(std::span<const uint8_t> file, const Allocator &allocate) {
    const OpenCvCodec codec = opencv_codec(file);
    uint64_t width = 0, height = 0, pixels = 0;
    if (codec == OpenCvCodec::other || !opencv_limit(L"OPENCV_IO_MAX_IMAGE_WIDTH", 1 << 20, width) ||
        !opencv_limit(L"OPENCV_IO_MAX_IMAGE_HEIGHT", 1 << 20, height) ||
        !opencv_limit(L"OPENCV_IO_MAX_IMAGE_PIXELS", 1 << 30, pixels) || !width || !height || !pixels)
        return false; // a limit of 0 refuses every image
    const std::string_view exif = exif_block(file.data(), file.size());
    // validateInputImageSize's limits, checked as the codec asks for the buffer.
    Target target{allocate, exif_orientation(reinterpret_cast<const uint8_t *>(exif.data()), exif.size()), {}, {},
                  pixels, width, height};
    cn_codec_status status = CN_CODEC_FALLBACK;
    switch (codec) {
    case OpenCvCodec::bmp: status = cn_bmp_decode(file.data(), file.size(), allocate_target, &target); break;
    case OpenCvCodec::jpeg: status = cn_jpeg_decode(file.data(), file.size(), CN_JPEG_OPENCV, 0, allocate_target, &target); break;
    case OpenCvCodec::webp: status = cn_webp_decode(file.data(), file.size(), allocate_target, &target); break;
    case OpenCvCodec::pxm: status = cn_pxm_decode(file.data(), file.size(), allocate_target, &target); break;
    case OpenCvCodec::tiff: status = cn_tiff_decode(file.data(), file.size(), allocate_target, &target); break;
    case OpenCvCodec::png: status = cn_png_decode(file.data(), file.size(), allocate_target, &target); break;
    case OpenCvCodec::other: break;
    }
    return status == CN_CODEC_OK && finish(target);
}

bool decode_pillow_tiff_alpha(std::span<const uint8_t> file, uint64_t max_pixels, const Allocator &allocate) {
    Target target{allocate, 1, {}, {}, max_pixels};
    return cn_tiff_decode_pillow_alpha(file.data(), file.size(), allocate_target, &target, &target.orientation) ==
               CN_CODEC_OK &&
           finish(target);
}

int exif_orientation(const uint8_t *p, size_t n) {
    if (n >= 6 && std::memcmp(p, "Exif\0\0", 6) == 0) {
        p += 6;
        n -= 6;
    }
    const ClassicTiff tiff(p, n);
    if (!tiff.valid) return 1;
    for (size_t entry = tiff.ifd + 2, end = entry + size_t(tiff.u16(tiff.ifd)) * 12; entry < end && entry + 12 <= n;
         entry += 12) {
        if (tiff.u16(entry) != 274) continue;
        const uint32_t type = tiff.u16(entry + 2);
        const uint32_t value = tiff.u32(entry + 4) != 1 ? 0 : type == 3 ? tiff.u16(entry + 8) : type == 4 ? tiff.u32(entry + 8) : 0;
        return value >= 2 && value <= 8 ? int(value) : 1;
    }
    return 1;
}

std::string_view exif_block(const uint8_t *p, size_t n) {
    const auto be32 = [&](size_t at) { return size_t(p[at]) << 24 | size_t(p[at + 1]) << 16 | size_t(p[at + 2]) << 8 | p[at + 3]; };
    const auto view = [&](size_t at, size_t length) { return std::string_view(reinterpret_cast<const char *>(p + at), length); };
    if (n >= 4 && p[0] == 0xFF && p[1] == 0xD8) {
        for (size_t at = 2; at + 4 <= n && p[at] == 0xFF;) {
            const uint8_t marker = p[at + 1];
            if (marker == 0xFF) { // fill byte
                ++at;
                continue;
            }
            if (marker == 0xDA || marker == 0xD9) break;
            if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
                at += 2;
                continue;
            }
            const size_t length = size_t(p[at + 2]) << 8 | p[at + 3];
            if (length < 2 || length > n - at - 2) break;
            if (marker == 0xE1 && length >= 8 && std::memcmp(p + at + 4, "Exif\0\0", 6) == 0) return view(at + 4, length - 2);
            at += 2 + length;
        }
    } else if (n >= 8 && std::memcmp(p, "\x89PNG\r\n\x1a\n", 8) == 0) {
        for (size_t at = 8; at + 12 <= n;) {
            const size_t length = be32(at);
            if (length > n - at - 12 || std::memcmp(p + at + 4, "IEND", 4) == 0) break;
            if (std::memcmp(p + at + 4, "eXIf", 4) == 0) return view(at + 8, length);
            at += 12 + length;
        }
    } else if (n >= 12 && std::memcmp(p, "RIFF", 4) == 0 && std::memcmp(p + 8, "WEBP", 4) == 0) {
        for (size_t at = 12; at + 8 <= n;) {
            const size_t length = size_t(p[at + 4]) | size_t(p[at + 5]) << 8 | size_t(p[at + 6]) << 16 | size_t(p[at + 7]) << 24;
            if (length > n - at - 8) break;
            if (std::memcmp(p + at, "EXIF", 4) == 0) return view(at + 8, length);
            at += 8 + length + (length & 1);
        }
    }
    return {};
}

} // namespace cn::image_decode
