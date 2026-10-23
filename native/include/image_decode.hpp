/* Native image decoding with today's results (GPL-3.0): Load Image's decoders `pil-jpeg`/`pil` (Pillow 12.3) and
 * `cv` (OpenCV 5.0 imdecode under IMREAD_UNCHANGED) for the files the vendored codecs take exactly as those paths do
 * (native/include/image_codecs.h). Each decode gives today's array before normalisation (dtype, shape, channel
 * order, bytes, EXIF orientation applied) or declines; a declined file goes through today's path, which gives the
 * image or the error. Nothing here touches Python, so the run engine's threads call it as they are. */
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

namespace cn::image_decode {

/* A sample's type; the values are image_codecs.h's cn_codec_sample (the byte count in the low byte). */
enum class Sample : uint16_t {
    u8 = 1, u16 = 2, u32 = 4, u64 = 8, i8 = 0x101, i16 = 0x102, i32 = 0x104, i64 = 0x108, f32 = 0x204, f64 = 0x208
};

/* Today's array: height x width x channels, C-contiguous; one channel is the 2-D (height, width) array. */
struct Shape {
    uint32_t height = 0, width = 0, channels = 0;
    Sample sample = Sample::u8;
    size_t sample_bytes() const noexcept { return size_t(sample) & 0xFF; }
    size_t bytes() const noexcept { return size_t(height) * width * channels * sample_bytes(); }
};

/* Returns the buffer of `shape.bytes()` bytes the image goes into, or nullptr to decline it (a failed allocation
 * too). It must not throw: the C codecs call it. */
using Allocator = std::function<uint8_t *(const Shape &shape)>;

/* Whether native decoding runs at all: CHAINNER_C_NATIVE_CODECS=0 (a diagnostic switch, read per call) sends every
 * file through today's path, the oracle of the codec corpus. */
bool enabled();

/* The whole file, or false when it cannot be read (today's path then reports the error). */
bool read_file(const std::filesystem::path &path, std::vector<uint8_t> &data);

/* Today's `pil` decoder for a JPEG (Image.open sniffs FF D8 FF first): JpegImagePlugin's marker walk, its mode from
 * the SOF's layers, libjpeg-turbo as JpegDecode runs it, CMYK as convert("RGB"), BGR, then info["exif"]'s
 * orientation. max_pixels is Image.MAX_IMAGE_PIXELS (0 for None): larger images decline (Pillow warns or refuses).
 * MPO files (an MPF segment) and files whose EXIF makes Image.open warn or fail decline. */
bool decode_pillow_jpeg(std::span<const uint8_t> file, uint64_t max_pixels, const Allocator &allocate);

/* Today's `cv` decoder for the bytes of a file: imdecode's codec by signature, decoded by the ported ones (BMP, JPEG,
 * WebP, PxM, TIFF, PNG) and by OpenEXRCore for OpenEXR (cn_exr_decode: OpenCV's ExrDecoder, which the runtime's
 * OpenCV lacks), then the file's EXIF orientation as read_cv applies it. Other codecs decline, and so do images above
 * OpenCV's size limits (OPENCV_IO_MAX_IMAGE_WIDTH, _HEIGHT, _PIXELS; a value other than a plain integer declines
 * every file) or above CN_MAX_IMAGE_PIXELS. `ported` false (CHAINNER_C_NATIVE_CODECS=0) declines every file but an
 * OpenEXR one, which has no other decoder. */
bool decode_opencv(std::span<const uint8_t> file, const Allocator &allocate, bool ported = true);

/* Today's `pil-tiff-alpha` decoder (load_image.py's _read_tiff_straight_alpha) for the plain 8-bit RGBA and
 * grey+alpha TIFFs cn_tiff_decode_pillow_alpha takes: BGRA, turned as its Orientation tag (or, without one, its XMP
 * packet's tiff:Orientation, which Pillow turns by) says. max_pixels is
 * Image.MAX_IMAGE_PIXELS (0 for None), as for decode_pillow_jpeg: larger images decline (Pillow warns or refuses at
 * load). Other files decline and the Python decoder runs. */
bool decode_pillow_tiff_alpha(std::span<const uint8_t> file, uint64_t max_pixels, const Allocator &allocate);

/* EXIF Orientation (tag 274) of an EXIF block, the TIFF header and IFD0 after an optional "Exif\0\0": 2-8, or 1 when
 * it is absent, malformed or out of range. */
int exif_orientation(const uint8_t *p, size_t n);

/* The EXIF block of an encoded JPEG (the first Exif APP1 before the scan), PNG (eXIf, before or after IDAT) or WebP
 * (EXIF chunk) file, found by walking its segments or chunks without decoding; empty for every other format. */
std::string_view exif_block(const uint8_t *p, size_t n);

/* A classic TIFF structure ("II" little-endian or "MM" big-endian, magic 42): valid when the header and its first
 * IFD's entry count lie in the n bytes; the IFD's entries are 12 bytes each from ifd+2. */
struct ClassicTiff {
    const uint8_t *p;
    size_t n;
    bool little = false, valid = false;
    size_t ifd = 0;
    ClassicTiff(const uint8_t *bytes, size_t size) : p(bytes), n(size) {
        if (n < 8 || !((p[0] == 'I' && p[1] == 'I') || (p[0] == 'M' && p[1] == 'M'))) return;
        little = p[0] == 'I';
        ifd = u32(4);
        valid = u16(2) == 42 && ifd <= n - 2;
    }
    uint32_t u16(size_t at) const { return little ? uint32_t(p[at] | p[at + 1] << 8) : uint32_t(p[at] << 8 | p[at + 1]); }
    uint32_t u32(size_t at) const { return little ? u16(at) | u16(at + 2) << 16 : u16(at) << 16 | u16(at + 2); }
};

} // namespace cn::image_decode
