/* chainner_trt_vs.dll: the TensorRT runner as a VapourSynth API 4 plugin (namespace
 * chainner_trt), for AI Upscale's TensorRT backend and any script that loads it:
 *
 *   core.chainner_trt.Upscale(clip, engine, streams=1, device=0)
 *
 * `clip` is GRAY or RGB of one format and size, integer (8-16 bits) or float (16 or 32
 * bits); integer planes use the colour_info range convention, by _ColorRange (else limited
 * for gray and full for RGB). `engine` is a plain engine file's path, or the engine itself
 * as bytes (chaiNNer passes compressed engines decompressed). As the AI Upscale bridge
 * (nodes/impl/vapoursynth/chainner_vs/upscale.py): frames are clipped to 0-1 before the
 * engine and after it, properties carry over, and when the engine changes the colour
 * family _Matrix becomes the new family's (0 on RGB, none on gray). */
#include "frame.h"
#include "runner.h"

#include <VapourSynth4.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace cn_trt;

struct Filter {
    VSNode* node = nullptr;
    VSVideoInfo vi{};
    std::unique_ptr<Runner> runner;
    Sample sample = Sample::F32;
    int bits = 32;
    int planes = 0;
    int result_planes = 0;
    std::atomic<bool> warned{false};
};

std::vector<char> engine_bytes(VSMap const* in, VSAPI const* vsapi) {
    char const* data = vsapi->mapGetData(in, "engine", 0, nullptr);
    int const size = vsapi->mapGetDataSize(in, "engine", 0, nullptr);
    if (vsapi->mapGetDataTypeHint(in, "engine", 0, nullptr) == dtBinary) return {data, data + size};
    std::u8string const name(reinterpret_cast<char8_t const*>(data), static_cast<std::size_t>(size));
    std::ifstream file(std::filesystem::path(name), std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open the engine file " + std::string(data, static_cast<std::size_t>(size)) + ".");
    std::vector<char> bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    // compressed_file's XZ_HEADER and ZSTD_HEADER
    std::string const head(bytes.data(), std::min<std::size_t>(bytes.size(), 6));
    if (head.starts_with("\xfd" "7zXZ") || head.starts_with("\x28\xb5\x2f\xfd"))
        throw std::runtime_error("The engine file is compressed (Save Engine's Compress): pass its bytes "
                                 "decompressed, as AI Upscale does, or save the engine uncompressed.");
    return bytes;
}

VSFrame const* VS_CC get_frame(int n, int reason, void* data, void**, VSFrameContext* context, VSCore* core,
                               VSAPI const* vsapi) {
    auto& f = *static_cast<Filter*>(data);
    if (reason == arInitial) {
        vsapi->requestFrameFilter(n, f.node, context);
        return nullptr;
    }
    if (reason != arAllFramesReady) return nullptr;
    VSFrame const* src = vsapi->getFrameFilter(n, f.node, context);
    VSFrame* dst = vsapi->newVideoFrame(&f.vi.format, f.vi.width, f.vi.height, src, core);
    try {
        int const bytes = sample_bytes(f.sample);
        HostPlane in[3], out[3];
        for (int p = 0; p < f.planes; ++p)
            in[p] = {const_cast<uint8_t*>(vsapi->getReadPtr(src, p)), vsapi->getStride(src, p), bytes};
        for (int p = 0; p < f.result_planes; ++p) out[p] = {vsapi->getWritePtr(dst, p), vsapi->getStride(dst, p), bytes};
        int error = 0;
        int64_t const range = vsapi->mapGetInt(vsapi->getFramePropertiesRO(src), "_ColorRange", 0, &error);
        Format const format{f.sample, f.bits, error ? f.planes == 3 : range == 0};
        Runner& r = *f.runner;
        FrameJob const job = describe_frame(in, f.planes, out, f.result_planes, vsapi->getFrameWidth(src, 0),
                                            vsapi->getFrameHeight(src, 0), format, r.bounds(), r.scale(),
                                            r.input_channels(), r.output_channels());
        r.upscale(job, in, out, true);
        if (f.result_planes != f.planes) {
            VSMap* props = vsapi->getFramePropertiesRW(dst);
            if (f.result_planes == 3) {
                vsapi->mapSetInt(props, "_Matrix", 0, maReplace);
            } else {
                vsapi->mapDeleteKey(props, "_Matrix");
            }
        }
        std::string const failure = r.capture_failure();
        if (!failure.empty() && !f.warned.exchange(true)) vsapi->logMessage(mtWarning, failure.c_str(), core);
    } catch (std::exception const& e) {
        vsapi->freeFrame(src);
        vsapi->freeFrame(dst);
        vsapi->setFilterError((std::string("Upscale: ") + e.what()).c_str(), context);
        return nullptr;
    }
    vsapi->freeFrame(src);
    return dst;
}

void VS_CC free_filter(void* data, VSCore* core, VSAPI const* vsapi) {
    std::unique_ptr<Filter> const f(static_cast<Filter*>(data));
    try {
        f->runner->close();
    } catch (std::exception const& e) {
        vsapi->logMessage(mtWarning, (std::string("Upscale: freeing the TensorRT runner: ") + e.what()).c_str(), core);
    }
    vsapi->freeNode(f->node);
}

void VS_CC create(VSMap const* in, VSMap* out, void*, VSCore* core, VSAPI const* vsapi) {
    auto f = std::make_unique<Filter>();
    f->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    try {
        VSVideoInfo const& vi = *vsapi->getVideoInfo(f->node);
        VSVideoFormat const& format = vi.format;
        if (format.colorFamily == cfUndefined || vi.width == 0 || vi.height == 0)
            throw std::invalid_argument("The clip needs one format and one frame size.");
        if (format.colorFamily != cfGray && format.colorFamily != cfRGB)
            throw std::invalid_argument("The clip must be GRAY or RGB: convert it first.");
        if (format.sampleType == stInteger && format.bytesPerSample <= 2) {
            f->sample = format.bytesPerSample == 1 ? Sample::U8 : Sample::U16;
        } else if (format.sampleType == stFloat && (format.bitsPerSample == 16 || format.bitsPerSample == 32)) {
            f->sample = format.bitsPerSample == 16 ? Sample::F16 : Sample::F32;
        } else {
            throw std::invalid_argument("The clip's samples must be 8-16 bit integers or 16/32-bit floats.");
        }
        f->bits = format.bitsPerSample;
        f->planes = format.numPlanes;
        int error = 0;
        int64_t streams = vsapi->mapGetInt(in, "streams", 0, &error);
        if (error) streams = 1;
        int64_t device = vsapi->mapGetInt(in, "device", 0, &error);
        if (error) device = 0;
        if (streams < 1 || streams > max_streams || device < 0 || device > 1024)
            throw std::invalid_argument("streams must be 1 to " + std::to_string(max_streams) +
                                        " and device a GPU index.");
        std::vector<char> const engine = engine_bytes(in, vsapi);
        f->runner = std::make_unique<Runner>(engine.data(), engine.size(), static_cast<int>(device), static_cast<int>(streams));
        Runner const& r = *f->runner;
        f->result_planes = result_planes(f->planes, r.input_channels(), r.output_channels());
        f->vi = vi;
        f->vi.width = vi.width * r.scale();
        f->vi.height = vi.height * r.scale();
        if (!vsapi->queryVideoFormat(&f->vi.format, f->result_planes == 3 ? cfRGB : cfGray, format.sampleType,
                                     format.bitsPerSample, 0, 0, core))
            throw std::invalid_argument("VapourSynth has no output format for this clip.");
    } catch (std::exception const& e) {
        vsapi->mapSetError(out, (std::string("Upscale: ") + e.what()).c_str());
        vsapi->freeNode(f->node);
        return;
    }
    VSFilterDependency const dependencies[] = {{f->node, rpStrictSpatial}};
    Filter* const filter = f.release();
    vsapi->createVideoFilter(out, "Upscale", &filter->vi, get_frame, free_filter, fmParallel, dependencies, 1, filter, core);
}

}  // namespace

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, VSPLUGINAPI const* vspapi) {
    vspapi->configPlugin("org.chainner-c.trt", "chainner_trt", "chaiNNer-C's TensorRT runner", VS_MAKE_VERSION(1, 0),
                         VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("Upscale", "clip:vnode;engine:data;streams:int:opt;device:int:opt;", "clip:vnode;",
                             create, nullptr, plugin);
}
