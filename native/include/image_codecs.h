#ifndef CHAINNER_IMAGE_CODECS_H
#define CHAINNER_IMAGE_CODECS_H

/* chaiNNer-C's calls into the vendored codec libraries (native/cmake/codecs.cmake), with the settings today's Pillow
 * 12.3.0 and OpenCV 5.0.0 paths use, so pixels and file bytes stay identical to theirs.
 *
 * A decoder returns CN_CODEC_OK with the pixels in the buffer it obtained from `allocate`, or CN_CODEC_FALLBACK for
 * every file it does not take exactly as today's path would (an error, a warning, a suspension at the end of the
 * data, a size today's path refuses, a variant not ported): the caller then runs today's path, which gives today's
 * image or today's error. A decoder never reports an error of its own. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cn_codec_status { CN_CODEC_OK = 0, CN_CODEC_FALLBACK = 1 } cn_codec_status;

/* The one image-size limit of every decode route (2026-10-19): 40,000 x 40,000 pixels in all, the value
 * backend/src/image_limits.py gives Pillow and OpenCV. The decoders take no larger image; OpenCV's own limits
 * (OPENCV_IO_MAX_IMAGE_*) are checked where the buffer is allocated (image_decode.cpp). */
#define CN_MAX_IMAGE_PIXELS UINT64_C(1600000000)

/* A decoded sample's type: its size in bytes in the low byte and its kind (unsigned, signed, float) above, so
 * CN_SAMPLE_U8 and CN_SAMPLE_U16 are the byte counts 1 and 2. */
typedef enum cn_codec_sample {
    CN_SAMPLE_U8 = 1,
    CN_SAMPLE_U16 = 2,
    CN_SAMPLE_U32 = 4,
    CN_SAMPLE_U64 = 8,
    CN_SAMPLE_I8 = 0x101,
    CN_SAMPLE_I16 = 0x102,
    CN_SAMPLE_I32 = 0x104,
    CN_SAMPLE_I64 = 0x108,
    CN_SAMPLE_F32 = 0x204,
    CN_SAMPLE_F64 = 0x208
} cn_codec_sample;

/* The buffer of a decoded image, height x width x channels samples of the cn_codec_sample `sample` each,
 * C-contiguous; NULL refuses the image (the decoder then falls back). The decoder writes every sample. */
typedef uint8_t *(*cn_codec_allocate)(void *context, uint32_t height, uint32_t width, uint32_t channels,
                                      uint32_t sample);

typedef enum cn_jpeg_flavour {
    /* Pillow 12.3's JpegDecode with JpegImagePlugin's rawmode: grey, BGR (Pillow's RGB, swapped), or CMYK through
     * Pillow's CMYK;I unpacking and cmyk2rgb, as BGR. Any warning, error or suspension falls back. */
    CN_JPEG_PILLOW = 0,
    /* OpenCV 5.0's JpegDecoder under IMREAD_UNCHANGED: grey for one component, else BGR, CMYK through
     * icvCvt_CMYK2BGR_8u_C4C3R; an error or suspension in jpeg_finish_decompress is ignored, as OpenCV does. */
    CN_JPEG_OPENCV = 1
} cn_jpeg_flavour;

/* Decodes a JPEG file of `size` bytes with libjpeg-turbo 3.1.4.1. `components` is the component count the caller
 * expects (Pillow's mode from the SOF it parsed), 0 for any; another count falls back. Grey decodes to 1 channel,
 * everything else to 3 (BGR), 8-bit samples. */
cn_codec_status cn_jpeg_decode(const uint8_t *data, size_t size, cn_jpeg_flavour flavour, int components,
                               cn_codec_allocate allocate, void *context);

/* Decodes a still PNG with libpng 1.6.57 as OpenCV 5.0's PngDecoder does under IMREAD_UNCHANGED: its chunk walk to
 * the first IDAT, its type rule (RGB or palette with tRNS and grey+alpha give BGRA, grey ignores tRNS, 16-bit stays
 * 16-bit, native-endian) and its transforms. An animated PNG falls back. */
cn_codec_status cn_png_decode(const uint8_t *data, size_t size, cn_codec_allocate allocate, void *context);

/* Decodes a WebP with libwebp 1.6.0 as OpenCV 5.0's WebPDecoder does under IMREAD_UNCHANGED: the features of the
 * first 32 bytes decide BGR or BGRA; a still image through WebPDecodeBGRInto/WebPDecodeBGRAInto, an animation
 * through WebPAnimDecoder (MODE_BGRA, first canvas, alpha dropped for BGR). */
cn_codec_status cn_webp_decode(const uint8_t *data, size_t size, cn_codec_allocate allocate, void *context);

/* OpenCV's WebPDecoder::checkSignature: WebPGetFeatures accepts the first 32 bytes. */
int cn_webp_signature(const uint8_t *data, size_t size);

/* Decodes a BMP ("BM") as OpenCV 5.0's BmpDecoder does under IMREAD_UNCHANGED: Windows headers from 36 bytes (V3,
 * V4, V5) and OS/2's 12-byte one (always grey), 1/4/8-bit palettes (1 channel when every entry is grey), RLE4 and
 * RLE8, 15/16-bit (5-5-5 or 5-6-5 bitfields only), 24-bit, and 32-bit (BGR, or BGRA with bitfields), 8-bit. */
cn_codec_status cn_bmp_decode(const uint8_t *data, size_t size, cn_codec_allocate allocate, void *context);

/* Decodes a PBM, PGM or PPM (P1-P6) as OpenCV 5.0's PxMDecoder does under IMREAD_UNCHANGED: grey or BGR, 16-bit
 * (native-endian) when maxval is above 255. A byte above ASCII where the header or ASCII samples are parsed falls
 * back (OpenCV's isdigit and isspace follow the CRT locale there). */
cn_codec_status cn_pxm_decode(const uint8_t *data, size_t size, cn_codec_allocate allocate, void *context);

/* Decodes a TIFF's first page as OpenCV 5.0's TiffDecoder does under IMREAD_UNCHANGED, through libtiff 4.7.1 built
 * as OpenCV builds it: its sample type rule (palette gives 3 channels, grey 1, signed and float samples their own
 * types); 8-bit images through libtiff's RGBA interface (palette, YCbCr, min-is-white, CMYK, 1/2/4-bit, unassociated
 * alpha premultiplied) as grey, BGR or BGRA; 10- to 16-bit and 32/64-bit samples as stored, RGB(A) swapped to
 * BGR(A); then OpenCV's orientation fix. LogLuv, the large-single-strip scanline path, raw planar-separate samples
 * and every file libtiff reports an error for fall back. */
cn_codec_status cn_tiff_decode(const uint8_t *data, size_t size, cn_codec_allocate allocate, void *context);

/* Decodes what Load Image's straight-alpha TIFF decoder (load_image.py's _read_tiff_straight_alpha, Pillow 12.3)
 * gives for its plainest files: 8-bit RGBA or grey+alpha with ExtraSamples (2,) in contiguous strips, uncompressed
 * (Pillow's raw decoder) or LZW, Deflate or PackBits (libtiff), as BGRA with the grey copied to B, G and R, before
 * the orientation turn; *orientation is set to the Orientation tag, or without one to the XMP packet's
 * tiff:Orientation as Pillow's getexif() reads it (1 outside 2-8), before `allocate` is called. Any other tag, type,
 * count or value, and anything Pillow would warn about (an Exif IFD it reads with a warning too), falls back to the
 * Python decoder. */
cn_codec_status cn_tiff_decode_pillow_alpha(const uint8_t *data, size_t size, cn_codec_allocate allocate,
                                            void *context, int *orientation);

/* Decodes an OpenEXR file with OpenEXRCore 3.5.2 as OpenCV 5.0's ExrDecoder does under IMREAD_UNCHANGED on a build
 * with OpenEXR (the runtime's OpenCV has none): part 0 of a multipart file, level 0 of a tiled one, every
 * compression OpenEXRCore has; the data window as the image (the display window is not used); float32 samples (half
 * widened exactly, uint32 converted, no clipping); R/G/B[/A] as BGR(A) with a missing colour channel 0; else channels
 * "0".."n-1" as they are (n of 1, 3 or 4); else Y (or Z) as grey, with RY/BY converted to BGR(A) after repeating the
 * subsampled chroma, by OpenEXR's luminance weights for the file's chromaticities; Y with A as BGRA with the grey in B,
 * G and R (OpenCV gives two channels, which chaiNNer's images do not have). Deep parts, other channel sets, images
 * above CN_MAX_IMAGE_PIXELS and every file OpenEXRCore reports an error for fall back. */
cn_codec_status cn_exr_decode(const uint8_t *data, size_t size, cn_codec_allocate allocate, void *context);

/* OpenCV's ExrDecoder::checkSignature: the magic number 76 2F 31 01. */
int cn_exr_signature(const uint8_t *data, size_t size);

/* Encoders. Each writes today's file bytes for an image into `out`, or returns CN_CODEC_FALLBACK (and leaves `out`
 * to be freed) where today's encoder would fail or warn, so today's path runs instead. */

/* An image to encode: height x width x channels samples of sample_bytes each (1 or 2: unsigned; 4: float32, which
 * only the encoders that say so take), rows `stride` bytes apart, in OpenCV's channel order (grey, BGR, BGRA) unless
 * an encoder says otherwise. */
typedef struct cn_codec_image {
    const uint8_t *data;
    uint32_t height, width, channels, sample_bytes;
    size_t stride;
} cn_codec_image;

/* A growing byte buffer from `allocate`/`release` (malloc and free when null); `failed` is set once an allocation
 * failed. Encoders reserve their expected size first, so most images take one block. */
typedef struct cn_codec_output {
    uint8_t *data;
    size_t size, capacity;
    int failed;
    void *(*allocate)(size_t size);
    void (*release)(void *data);
} cn_codec_output;

void cn_codec_output_reserve(cn_codec_output *out, size_t capacity);
void cn_codec_output_append(cn_codec_output *out, const void *bytes, size_t count);
void cn_codec_output_free(cn_codec_output *out);

/* OpenCV 5.0's JpegEncoder with Save Image's parameters (IMWRITE_JPEG_QUALITY `quality` clamped to 0..100,
 * IMWRITE_JPEG_SAMPLING_FACTOR `sampling` (0x111111, 0x121111, 0x211111, 0x221111 or 0x411111), IMWRITE_JPEG_PROGRESSIVE
 * `progressive`), through libjpeg-turbo 3.1.4.1: grey, BGR, or BGRX (alpha dropped), 8-bit. */
cn_codec_status cn_jpeg_encode(const cn_codec_image *image, int quality, uint32_t sampling, int progressive,
                               cn_codec_output *out);

/* OpenCV 5.0's PngEncoder without parameters through libpng 1.6.57 and zlib 1.3.2: SUB filter, level 1 (Z_BEST_SPEED),
 * Z_RLE, depth 8 or 16 (byte-swapped from native order), grey, BGR or BGRA, no ancillary chunks. */
cn_codec_status cn_png_encode(const cn_codec_image *image, cn_codec_output *out);

/* OpenCV 5.0's WebPEncoder with IMWRITE_WEBP_QUALITY `quality` (below 1 is 1, above 100 lossless): WebPEncodeBGR /
 * WebPEncodeBGRA, or WebPEncodeLosslessBGR / WebPEncodeLosslessBGRA; grey is copied to BGR first. 8-bit. */
cn_codec_status cn_webp_encode(const cn_codec_image *image, int quality, cn_codec_output *out);

/* Pillow 12.3's WebP save of an RGBA image with lossless=True, exact=True (WebPImagePlugin._save, _webp.WebPEncode):
 * quality 80, alpha_quality 100, method 4, ARGB import, then WebPMux reassembly. `image` is RGBA, 8-bit. */
cn_codec_status cn_webp_encode_exact(const cn_codec_image *image, cn_codec_output *out);

/* OpenCV 5.0's BmpEncoder without parameters: grey with a 256-entry grey palette, 24-bit BGR, or 32-bit BGRA with a
 * BITMAPV5HEADER (BI_BITFIELDS, sRGB); bottom-up rows padded to 4 bytes. 8-bit. */
cn_codec_status cn_bmp_encode(const cn_codec_image *image, cn_codec_output *out);

/* OpenCV 5.0's TiffEncoder as Save Image calls it, through libtiff 4.7.1 and zlib 1.3.2 into memory (classic
 * little-endian TIFF, one strip per 8 KiB of rows): `compression` is IMWRITE_TIFF_COMPRESSION (1 none, 5 LZW,
 * 8 Adobe Deflate) or 0 without parameters (LZW, none for float32), with the horizontal predictor for LZW and
 * Deflate; 8- or 16-bit unsigned samples, or float32 uncompressed; grey, BGR or BGRA, written as RGB(A). */
cn_codec_status cn_tiff_encode(const cn_codec_image *image, int compression, cn_codec_output *out);

#ifdef __cplusplus
}
#endif

#endif
