/* libFuzzer harness over Load Image's native decoders (GPL-3.0; README.md here): image_decode.cpp's three routes
 * (OpenCV's by signature, Pillow's JPEG, Pillow's straight-alpha TIFF) on each input. Every route runs twice, into
 * buffers filled with 0x00 and then 0xFF, so a sample a decoder leaves unwritten (image_codecs.h: "The decoder writes
 * every sample") shows as a difference; a route that succeeds once and declines once is nondeterministic. Both abort,
 * which libFuzzer reports as a crash; AddressSanitizer reports the memory errors. */
#include "image_codecs.h"
#include "image_decode.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using namespace cn::image_decode;

// Larger images decline here, as a failed allocation does in the application, to keep each run short and small.
constexpr size_t max_bytes = size_t(16) << 20;

struct Result {
    bool ok = false;
    std::vector<std::vector<uint8_t>> buffers; // every buffer the route asked for, the image last
};

[[noreturn]] void fail(const char *route, const char *what) {
    std::fprintf(stderr, "fuzz_image_decode: %s: %s\n", route, what);
    std::abort();
}

template <typename Decode> Result run(const char *route, uint8_t fill, Decode decode) {
    Result result;
    const Allocator allocate = [&](const Shape &shape) -> uint8_t * {
        const size_t sample = shape.sample_bytes();
        if (!shape.height || !shape.width || shape.channels < 1 || shape.channels > 4 ||
            (sample != 1 && sample != 2 && sample != 4 && sample != 8))
            fail(route, "an allocation of an impossible shape");
        if (uint64_t(shape.height) * shape.width > CN_MAX_IMAGE_PIXELS) fail(route, "an image above CN_MAX_IMAGE_PIXELS");
        if (shape.bytes() > max_bytes) return nullptr;
        result.buffers.emplace_back(shape.bytes(), fill);
        return result.buffers.back().data();
    };
    result.ok = decode(allocate);
    return result;
}

template <typename Decode> void check(const char *route, Decode decode) {
    const Result zeros = run(route, 0x00, decode), ones = run(route, 0xFF, decode);
    if (zeros.ok != ones.ok) fail(route, "succeeded once and declined once");
    if (zeros.ok && zeros.buffers.back() != ones.buffers.back()) fail(route, "left samples unwritten");
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    const std::span<const uint8_t> file(data, size);
    check("cv", [&](const Allocator &allocate) { return decode_opencv(file, allocate); });
    check("pil-jpeg", [&](const Allocator &allocate) { return decode_pillow_jpeg(file, CN_MAX_IMAGE_PIXELS, allocate); });
    check("pil-tiff-alpha",
          [&](const Allocator &allocate) { return decode_pillow_tiff_alpha(file, CN_MAX_IMAGE_PIXELS, allocate); });
    return 0;
}
