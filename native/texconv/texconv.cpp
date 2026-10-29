// texconv for Linux: the DDS-to-DDS conversions chaiNNer asks of Microsoft's texconv.exe (nodes/impl/dds/texconv.py,
// native/src/image_io.cpp), on DirectXTex may2026, the library texconv.exe 2026.5.8.1 is built from. Each step is
// texconv's own (DirectXTex's Texconv/texconv.cpp, MIT) in its order, with its DirectXTex calls and flags: load,
// decompress, -alpha, the 16384 size cap, conversion, mip levels, -pmalpha, compression, the alpha mode and the
// header. Only the options chaiNNer passes are accepted, with texconv's spelling and checks; any other stops with an
// error. BC6H and BC7 use DirectXTex's CPU codecs, as texconv.exe with -nogpu. Windows texconv runs some
// conversions and mip filters through WIC (wic.cpp); its DirectXTex is built with MSVC's /fp:fast, whose arithmetic
// CMakeLists.txt reproduces. native/tests/test_texconv.py checks the results against texconv.exe's.
// The standard headers come first: on Linux DirectXMath includes sal.h, whose macros break libstdc++'s headers.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "wic.h"

using namespace DirectX;

namespace {

enum Option : uint32_t {
    OPT_NOLOGO = 1,
    OPT_OVERWRITE,
    OPT_USE_DX10,
    OPT_USE_DX9,
    OPT_SEPALPHA,
    OPT_PREMUL_ALPHA,
    OPT_DEMUL_ALPHA,
    OPT_NOGPU,
    OPT_FLAGS_MAX,
    OPT_MIPLEVELS,
    OPT_FORMAT,
    OPT_FILTER,
    OPT_SRGBI,
    OPT_SRGBO,
    OPT_SRGB,
    OPT_OUTPUTDIR,
    OPT_FILETYPE,
    OPT_BC_COMPRESS,
};

template <typename T> struct Named {
    const char* name;
    T value;
};

constexpr Named<uint32_t> options[] = {
    {"m", OPT_MIPLEVELS}, {"f", OPT_FORMAT}, {"if", OPT_FILTER}, {"srgbi", OPT_SRGBI}, {"srgbo", OPT_SRGBO},
    {"srgb", OPT_SRGB}, {"o", OPT_OUTPUTDIR}, {"y", OPT_OVERWRITE}, {"ft", OPT_FILETYPE}, {"dx10", OPT_USE_DX10},
    {"dx9", OPT_USE_DX9}, {"nologo", OPT_NOLOGO}, {"sepalpha", OPT_SEPALPHA}, {"pmalpha", OPT_PREMUL_ALPHA},
    {"alpha", OPT_DEMUL_ALPHA}, {"nogpu", OPT_NOGPU}, {"bc", OPT_BC_COMPRESS},
};

#define FORMAT(name) {#name, DXGI_FORMAT_##name}
// Save Image's formats (every one of texconv's but the video formats and XR_BIAS), which include the intermediates.
constexpr Named<DXGI_FORMAT> formats[] = {
    FORMAT(R32G32B32A32_FLOAT), FORMAT(R32G32B32A32_UINT), FORMAT(R32G32B32A32_SINT), FORMAT(R32G32B32_FLOAT),
    FORMAT(R32G32B32_UINT), FORMAT(R32G32B32_SINT), FORMAT(R16G16B16A16_FLOAT), FORMAT(R16G16B16A16_UNORM),
    FORMAT(R16G16B16A16_UINT), FORMAT(R16G16B16A16_SNORM), FORMAT(R16G16B16A16_SINT), FORMAT(R32G32_FLOAT),
    FORMAT(R32G32_UINT), FORMAT(R32G32_SINT), FORMAT(R10G10B10A2_UNORM), FORMAT(R10G10B10A2_UINT),
    FORMAT(R11G11B10_FLOAT), FORMAT(R8G8B8A8_UNORM), FORMAT(R8G8B8A8_UNORM_SRGB), FORMAT(R8G8B8A8_UINT),
    FORMAT(R8G8B8A8_SNORM), FORMAT(R8G8B8A8_SINT), FORMAT(R16G16_FLOAT), FORMAT(R16G16_UNORM), FORMAT(R16G16_UINT),
    FORMAT(R16G16_SNORM), FORMAT(R16G16_SINT), FORMAT(R32_FLOAT), FORMAT(R32_UINT), FORMAT(R32_SINT),
    FORMAT(R8G8_UNORM), FORMAT(R8G8_UINT), FORMAT(R8G8_SNORM), FORMAT(R8G8_SINT), FORMAT(R16_FLOAT),
    FORMAT(R16_UNORM), FORMAT(R16_UINT), FORMAT(R16_SNORM), FORMAT(R16_SINT), FORMAT(R8_UNORM), FORMAT(R8_UINT),
    FORMAT(R8_SNORM), FORMAT(R8_SINT), FORMAT(A8_UNORM), FORMAT(R9G9B9E5_SHAREDEXP), FORMAT(R8G8_B8G8_UNORM),
    FORMAT(G8R8_G8B8_UNORM), FORMAT(BC1_UNORM), FORMAT(BC1_UNORM_SRGB), FORMAT(BC2_UNORM), FORMAT(BC2_UNORM_SRGB),
    FORMAT(BC3_UNORM), FORMAT(BC3_UNORM_SRGB), FORMAT(BC4_UNORM), FORMAT(BC4_SNORM), FORMAT(BC5_UNORM),
    FORMAT(BC5_SNORM), FORMAT(B5G6R5_UNORM), FORMAT(B5G5R5A1_UNORM), FORMAT(B8G8R8A8_UNORM), FORMAT(B8G8R8X8_UNORM),
    FORMAT(B8G8R8A8_UNORM_SRGB), FORMAT(B8G8R8X8_UNORM_SRGB), FORMAT(BC6H_UF16), FORMAT(BC6H_SF16),
    FORMAT(BC7_UNORM), FORMAT(BC7_UNORM_SRGB), FORMAT(B4G4R4A4_UNORM), {"A4B4G4R4_UNORM", DXGI_FORMAT(191)},
};
#undef FORMAT

// The filters chaiNNer's Mip Filter offers.
constexpr Named<TEX_FILTER_FLAGS> filters[] = {
    {"POINT", TEX_FILTER_POINT}, {"LINEAR", TEX_FILTER_LINEAR}, {"CUBIC", TEX_FILTER_CUBIC},
    {"FANT", TEX_FILTER_FANT}, {"BOX", TEX_FILTER_BOX}, {"TRIANGLE", TEX_FILTER_TRIANGLE},
};

// texconv's _wcsicmp on ASCII names.
bool same(std::string_view a, std::string_view b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
        return (x >= 'a' && x <= 'z' ? x - 32 : x) == (y >= 'a' && y <= 'z' ? y - 32 : y);
    });
}

template <typename T, size_t N> T lookup(std::string_view name, const Named<T> (&table)[N]) {
    for (const auto& entry : table)
        if (same(name, entry.name)) return entry.value;
    return T(0);
}

// texconv's DXT1-DXT5 aliases; the other aliases and special formats are not chaiNNer's.
DXGI_FORMAT format_named(std::string_view name) {
    if (const DXGI_FORMAT format = lookup(name, formats)) return format;
    if (same(name, "DXT1")) return DXGI_FORMAT_BC1_UNORM;
    if (same(name, "DXT2") || same(name, "DXT3")) return DXGI_FORMAT_BC2_UNORM;
    if (same(name, "DXT4") || same(name, "DXT5")) return DXGI_FORMAT_BC3_UNORM;
    return DXGI_FORMAT_UNKNOWN;
}

constexpr bool ispow2(size_t x) { return x != 0 && !(x & (x - 1)); }

size_t count_mips(size_t width, size_t height, size_t depth) {
    size_t levels = 1;
    while (height > 1 || width > 1 || depth > 1) {
        if (height > 1) height >>= 1;
        if (width > 1) width >>= 1;
        if (depth > 1) depth >>= 1;
        ++levels;
    }
    return levels;
}

struct Failure {
    std::string step;
    HRESULT hr;
};

void check(HRESULT hr, const char* step) {
    if (FAILED(hr)) throw Failure{step, hr};
}

// Block compression as DirectXTex's Compress, each image's block rows split over the CPU's threads: a block's
// encoding reads only its own 4x4 pixels, as texconv's OpenMP loop (CompressBC_Parallel) relies on.
void compress(const ScratchImage& image, DXGI_FORMAT format, TEX_COMPRESS_FLAGS flags, float threshold,
    ScratchImage& result) {
    TexMetadata metadata = image.GetMetadata();
    metadata.format = format;
    check(result.Initialize(metadata), "compress");
    constexpr size_t rows_per_task = 16; // block rows
    struct Task {
        size_t image, first, count;
    };
    std::vector<Task> tasks;
    for (size_t i = 0; i < image.GetImageCount(); ++i) {
        const size_t blocks = (image.GetImages()[i].height + 3) / 4;
        for (size_t first = 0; first < blocks; first += rows_per_task)
            tasks.push_back({i, first, std::min(rows_per_task, blocks - first)});
    }
    std::atomic<size_t> next{0};
    std::atomic<HRESULT> outcome{S_OK};
    const auto work = [&] {
        for (size_t t = next++; t < tasks.size() && SUCCEEDED(outcome.load()); t = next++) {
            const Task& task = tasks[t];
            const Image& source = image.GetImages()[task.image];
            const Image& target = result.GetImages()[task.image];
            Image strip = source;
            strip.height = std::min(task.count * 4, source.height - task.first * 4);
            strip.pixels = source.pixels + task.first * 4 * source.rowPitch;
            strip.slicePitch = strip.height * source.rowPitch;
            ScratchImage compressed;
            HRESULT hr = Compress(strip, format, flags, threshold, compressed);
            if (SUCCEEDED(hr)) {
                const Image& blocks = *compressed.GetImage(0, 0, 0);
                for (size_t row = 0; row < task.count; ++row)
                    std::memcpy(target.pixels + (task.first + row) * target.rowPitch,
                        blocks.pixels + row * blocks.rowPitch, target.rowPitch);
            } else
                outcome = hr;
        }
    };
    const size_t threads = std::min<size_t>(std::max(1u, std::thread::hardware_concurrency()), tasks.size());
    std::vector<std::thread> pool;
    for (size_t i = 1; i < threads; ++i) pool.emplace_back(work);
    work();
    for (auto& thread : pool) thread.join();
    check(outcome, "compress");
}

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw Failure{"open", E_FAIL};
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (file.bad()) throw Failure{"read", E_FAIL};
    return data;
}

struct Settings {
    uint64_t flags = 0;
    size_t mip_levels = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    TEX_FILTER_FLAGS filter = TEX_FILTER_DEFAULT, srgb = TEX_FILTER_DEFAULT, filter_options = TEX_FILTER_DEFAULT;
    TEX_COMPRESS_FLAGS compress = TEX_COMPRESS_DEFAULT;
    std::filesystem::path output_directory;
    bool has(Option option) const { return (flags & (uint64_t(1) << option)) != 0; }
};

// One file through texconv's steps, as texconv's main loop converts it to a DDS file.
void convert(const Settings& s, const std::filesystem::path& source) {
    std::printf("reading %s", source.c_str());
    std::fflush(stdout);
    const std::string extension = source.extension().string();
    if (!same(extension, ".dds") && !same(extension, ".ddx")) throw Failure{"load (DDS input only)", E_FAIL};
    TexMetadata info;
    auto image = std::make_unique<ScratchImage>();
    {
        const std::vector<uint8_t> data = read_file(source);
        check(LoadFromDDSMemory(data.data(), data.size(), DDS_FLAGS_ALLOW_LARGE_FILES, &info, *image), "load");
    }
    if (IsTypeless(info.format)) throw Failure{"load (typeless format)", E_FAIL};
    size_t tMips = (!s.mip_levels && info.mipLevels > 1) ? info.mipLevels : s.mip_levels;
    if (IsPlanar(info.format)) {
        auto timage = std::make_unique<ScratchImage>();
        check(ConvertToSinglePlane(image->GetImages(), image->GetImageCount(), info, *timage), "converttosingleplane");
        info.format = timage->GetMetadata().format;
        image.swap(timage);
    }
    const DXGI_FORMAT tformat = s.format == DXGI_FORMAT_UNKNOWN ? info.format : s.format;

    // Decompress, keeping the original for a target of its own format.
    std::unique_ptr<ScratchImage> cimage;
    if (IsCompressed(info.format)) {
        auto timage = std::make_unique<ScratchImage>();
        check(Decompress(image->GetImages(), image->GetImageCount(), info, DXGI_FORMAT_UNKNOWN, *timage), "decompress");
        info.format = timage->GetMetadata().format;
        cimage = std::move(image);
        image = std::move(timage);
    }

    // -alpha: undo premultiplied alpha.
    if (s.has(OPT_DEMUL_ALPHA) && HasAlpha(info.format) && info.format != DXGI_FORMAT_A8_UNORM &&
        info.GetAlphaMode() != TEX_ALPHA_MODE_STRAIGHT && info.IsPMAlpha()) {
        auto timage = std::make_unique<ScratchImage>();
        check(PremultiplyAlpha(image->GetImages(), image->GetImageCount(), info, TEX_PMALPHA_REVERSE | s.srgb, *timage),
            "demultiply alpha");
        info.miscFlags2 = timage->GetMetadata().miscFlags2;
        image.swap(timage);
        cimage.reset();
    }

    // texconv's default feature level (11.0) caps the size at 16384.
    constexpr size_t max_size = 16384;
    const size_t twidth = std::min(info.width, max_size), theight = std::min(info.height, max_size);
    if (info.width != twidth || info.height != theight) {
        auto timage = std::make_unique<ScratchImage>();
        check(wic::Resize(image->GetImages(), image->GetImageCount(), image->GetMetadata(), twidth, theight,
                  s.filter | s.filter_options, *timage),
            "resize");
        info.width = twidth;
        info.height = theight;
        info.mipLevels = 1;
        image.swap(timage);
        cimage.reset();
        if (tMips > 0) tMips = std::min(tMips, count_mips(info.width, info.height, info.depth > 1 ? info.depth : 1));
    }

    if (info.format != tformat && !IsCompressed(tformat)) {
        auto timage = std::make_unique<ScratchImage>();
        check(wic::Convert(image->GetImages(), image->GetImageCount(), image->GetMetadata(), tformat,
                  s.filter | s.filter_options | s.srgb, TEX_THRESHOLD_DEFAULT, *timage),
            "convert");
        info.format = tformat;
        image.swap(timage);
        cimage.reset();
    }

    // Mip levels: a single top level when the count changes, then the new chain.
    TEX_FILTER_FLAGS filter3d = s.filter;
    if ((!ispow2(info.width) || !ispow2(info.height) || !ispow2(info.depth)) &&
        info.dimension == TEX_DIMENSION_TEXTURE3D)
        filter3d = TEX_FILTER_TRIANGLE; // texconv's choice for non-power-of-2 volumes
    if ((!tMips || info.mipLevels != tMips) && info.mipLevels != 1) {
        auto timage = std::make_unique<ScratchImage>();
        TexMetadata mdata = info;
        mdata.mipLevels = 1;
        check(timage->Initialize(mdata), "copy to single level");
        const bool volume = info.dimension == TEX_DIMENSION_TEXTURE3D;
        for (size_t i = 0; i < (volume ? info.depth : info.arraySize); ++i)
            check(CopyRectangle(*image->GetImage(0, volume ? 0 : i, volume ? i : 0), Rect(0, 0, info.width, info.height),
                      *timage->GetImage(0, volume ? 0 : i, volume ? i : 0), TEX_FILTER_DEFAULT, 0, 0),
                "copy to single level");
        image.swap(timage);
        info.mipLevels = 1;
        if (cimage && tMips == 1) {
            // The original compressed top level stays usable.
            mdata = cimage->GetMetadata();
            mdata.mipLevels = 1;
            check(timage->Initialize(mdata), "copy compressed to single level");
            const bool cvolume = mdata.dimension == TEX_DIMENSION_TEXTURE3D;
            for (size_t i = 0; i < (cvolume ? mdata.depth : mdata.arraySize); ++i) {
                const Image* from = cimage->GetImage(0, cvolume ? 0 : i, cvolume ? i : 0);
                const Image* to = timage->GetImage(0, cvolume ? 0 : i, cvolume ? i : 0);
                std::memcpy(to->pixels, from->pixels, std::min(to->slicePitch, from->slicePitch));
            }
            cimage.swap(timage);
        } else
            cimage.reset();
    }
    if ((!tMips || info.mipLevels != tMips) && (info.width > 1 || info.height > 1 || info.depth > 1)) {
        auto timage = std::make_unique<ScratchImage>();
        if (info.dimension == TEX_DIMENSION_TEXTURE3D)
            check(GenerateMipMaps3D(image->GetImages(), image->GetImageCount(), image->GetMetadata(),
                      filter3d | s.filter_options, tMips, *timage),
                "mipmaps");
        else
            check(wic::GenerateMipMaps(image->GetImages(), image->GetImageCount(), image->GetMetadata(),
                      s.filter | s.filter_options, tMips, *timage),
                "mipmaps");
        info.mipLevels = timage->GetMetadata().mipLevels;
        image.swap(timage);
        cimage.reset();
    }

    // -pmalpha
    if (s.has(OPT_PREMUL_ALPHA) && HasAlpha(info.format) && info.format != DXGI_FORMAT_A8_UNORM && !info.IsPMAlpha()) {
        auto timage = std::make_unique<ScratchImage>();
        check(PremultiplyAlpha(image->GetImages(), image->GetImageCount(), info, TEX_PMALPHA_DEFAULT | s.srgb, *timage),
            "premultiply alpha");
        info.miscFlags2 = timage->GetMetadata().miscFlags2;
        image.swap(timage);
        cimage.reset();
    }

    if (IsCompressed(tformat)) {
        if (cimage && cimage->GetMetadata().format == tformat) {
            image = std::move(cimage);
            info.format = tformat;
        } else {
            cimage.reset();
            auto timage = std::make_unique<ScratchImage>();
            compress(*image, tformat, s.compress | s.srgb, TEX_THRESHOLD_DEFAULT, *timage);
            info.format = tformat;
            image.swap(timage);
        }
    }
    cimage.reset();

    if (HasAlpha(info.format) && info.format != DXGI_FORMAT_A8_UNORM) {
        if (image->IsAlphaAllOpaque())
            info.SetAlphaMode(TEX_ALPHA_MODE_OPAQUE);
        else if (info.IsPMAlpha()) {
        } else if (s.has(OPT_SEPALPHA))
            info.SetAlphaMode(TEX_ALPHA_MODE_CUSTOM);
        else if (info.GetAlphaMode() == TEX_ALPHA_MODE_UNKNOWN)
            info.SetAlphaMode(TEX_ALPHA_MODE_STRAIGHT);
    } else
        info.SetAlphaMode(TEX_ALPHA_MODE_UNKNOWN);

    std::filesystem::path destination = s.output_directory;
    destination /= source.stem();
    destination += ".dds";
    std::printf("\nwriting %s", destination.c_str());
    std::fflush(stdout);
    if (!s.has(OPT_OVERWRITE) && std::filesystem::exists(destination))
        throw Failure{"write (the output file exists; -y overwrites it)", E_FAIL};
    DDS_FLAGS dds = DDS_FLAGS_NONE;
    if (s.has(OPT_USE_DX10)) dds |= DDS_FLAGS_FORCE_DX10_EXT | DDS_FLAGS_FORCE_DX10_EXT_MISC2;
    else if (s.has(OPT_USE_DX9)) dds |= DDS_FLAGS_FORCE_DX9_LEGACY;
    Blob blob;
    check(SaveToDDSMemory(image->GetImages(), image->GetImageCount(), info, dds, blob), "save");
    std::ofstream file(destination, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(blob.GetConstBufferPointer()), std::streamsize(blob.GetBufferSize()));
    file.close();
    if (!file) {
        std::error_code ignored;
        std::filesystem::remove(destination, ignored); // texconv deletes a file it could not finish
        throw Failure{"write", E_FAIL};
    }
    std::printf("\n");
}

int usage_error(const char* message, const char* value) {
    std::printf("%s%s\n", message, value);
    return 1;
}

} // namespace

int main(int argc, char* argv[]) {
    Settings s;
    std::vector<std::filesystem::path> files;
    bool allow_options = true;
    for (int i = 1; i < argc; ++i) {
        std::string argument = argv[i];
        // Only '-' starts an option: '/' starts a path on Linux.
        if (!allow_options || argument.empty() || argument[0] != '-') {
            files.emplace_back(argument);
            continue;
        }
        if (argument == "--") {
            allow_options = false;
            continue;
        }
        if (argument[1] == '-') return usage_error("ERROR: Unknown option: ", argument.c_str());
        std::string name = argument.substr(1), value;
        const size_t separator = name.find_first_of(":=");
        const bool inline_value = separator != std::string::npos;
        if (inline_value) {
            value = name.substr(separator + 1);
            name.resize(separator);
        }
        const uint32_t option = lookup(name, options);
        if (!option) return usage_error("ERROR: Unknown option: ", argument.c_str());
        if (option < OPT_FLAGS_MAX) {
            if (s.flags & (uint64_t(1) << option)) return usage_error("ERROR: Duplicate option: ", argument.c_str());
            s.flags |= uint64_t(1) << option;
        }
        const bool valued = option == OPT_MIPLEVELS || option == OPT_FORMAT || option == OPT_FILTER ||
            option == OPT_OUTPUTDIR || option == OPT_FILETYPE || option == OPT_BC_COMPRESS;
        if (valued && !inline_value) {
            if (i + 1 >= argc) return usage_error("ERROR: Missing value for -", name.c_str());
            value = argv[++i];
        }
        switch (option) {
        case OPT_MIPLEVELS: {
            char* end = nullptr;
            s.mip_levels = std::strtoull(value.c_str(), &end, 10);
            if (value.empty() || *end) return usage_error("Invalid value specified with -m ", value.c_str());
            break;
        }
        case OPT_FORMAT:
            s.format = format_named(value);
            if (!s.format) return usage_error("Invalid value specified with -f ", value.c_str());
            break;
        case OPT_FILTER:
            s.filter = lookup(value, filters);
            if (!s.filter) return usage_error("Invalid value specified with -if ", value.c_str());
            break;
        case OPT_SRGBI: s.srgb |= TEX_FILTER_SRGB_IN; break;
        case OPT_SRGBO: s.srgb |= TEX_FILTER_SRGB_OUT; break;
        case OPT_SRGB: s.srgb |= TEX_FILTER_SRGB; break;
        case OPT_SEPALPHA: s.filter_options |= TEX_FILTER_SEPARATE_ALPHA; break;
        case OPT_OUTPUTDIR: s.output_directory = value; break;
        case OPT_FILETYPE:
            if (!same(value, "dds")) return usage_error("Invalid value specified with -ft (DDS only) ", value.c_str());
            break;
        case OPT_PREMUL_ALPHA:
        case OPT_DEMUL_ALPHA:
            if (s.has(OPT_PREMUL_ALPHA) && s.has(OPT_DEMUL_ALPHA))
                return usage_error("Can't use -pmalpha and -alpha at same time", "");
            break;
        case OPT_USE_DX10:
        case OPT_USE_DX9:
            if (s.has(OPT_USE_DX10) && s.has(OPT_USE_DX9)) return usage_error("Can't use -dx9 and -dx10 at same time", "");
            break;
        case OPT_BC_COMPRESS: {
            s.compress = TEX_COMPRESS_DEFAULT;
            bool found = false;
            for (const auto& [letter, flag] : {std::pair{'u', TEX_COMPRESS_UNIFORM}, {'d', TEX_COMPRESS_DITHER},
                     {'q', TEX_COMPRESS_BC7_QUICK}, {'x', TEX_COMPRESS_BC7_USE_3SUBSETS}})
                if (value.find(letter) != std::string::npos) {
                    s.compress |= flag;
                    found = true;
                }
            if ((s.compress & TEX_COMPRESS_BC7_QUICK) && (s.compress & TEX_COMPRESS_BC7_USE_3SUBSETS))
                return usage_error("Can't use -bc x (max) and -bc q (quick) at same time", "");
            if (!found) return usage_error("Invalid value specified for -bc, missing d, u, q, or x: ", value.c_str());
            break;
        }
        default: break;
        }
    }
    if (files.empty()) return usage_error("Usage: texconv <options> [--] <files>", "");
    if (!s.has(OPT_NOLOGO)) std::printf("texconv for chaiNNer-C on DirectXTex may2026\n\n");
    int result = 0;
    for (const auto& file : files) {
        try {
            convert(s, file);
        } catch (const Failure& failure) {
            std::printf(" FAILED [%s] (%08X)\n", failure.step.c_str(), static_cast<unsigned>(failure.hr));
            result = 1;
        } catch (const std::exception& error) {
            std::printf(" FAILED (%s)\n", error.what());
            result = 1;
        }
    }
    return result;
}
