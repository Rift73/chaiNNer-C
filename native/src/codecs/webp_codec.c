/* WebP through libwebp 1.6.0 with OpenCV 5.0's WebPDecoder (grfmt_webp.cpp) under IMREAD_UNCHANGED (GPL-3.0). */
#include "image_codecs.h"

#include <stdlib.h>
#include <string.h>

#include "webp/decode.h"
#include "webp/demux.h"
#include "webp/encode.h"
#include "webp/mux.h"

/* readHeader reads the features of the first WEBP_HEADER_SIZE bytes only. */
#define WEBP_HEADER_SIZE 32

cn_codec_status cn_webp_decode(const uint8_t *data, size_t size, cn_codec_allocate allocate, void *context) {
    WebPBitstreamFeatures features;
    if (size < WEBP_HEADER_SIZE || WebPGetFeatures(data, WEBP_HEADER_SIZE, &features) != VP8_STATUS_OK ||
        features.width <= 0 || features.height <= 0)
        return CN_CODEC_FALLBACK;
    const uint32_t width = (uint32_t)features.width, height = (uint32_t)features.height;
    const uint32_t channels = features.has_alpha ? 4 : 3;
    if (!features.has_animation) {
        uint8_t *const pixels = allocate(context, height, width, channels, 1);
        if (!pixels) return CN_CODEC_FALLBACK;
        const size_t stride = (size_t)width * channels;
        const uint8_t *decoded =
            channels == 3 ? WebPDecodeBGRInto(data, size, pixels, stride * height, (int)stride)
                          : WebPDecodeBGRAInto(data, size, pixels, stride * height, (int)stride);
        return decoded == pixels ? CN_CODEC_OK : CN_CODEC_FALLBACK;
    }
    /* An animation: WebPAnimDecoder in MODE_BGRA, its first canvas, BGRA -> BGR without alpha (cvtColor). */
    WebPAnimDecoderOptions options;
    if (!WebPAnimDecoderOptionsInit(&options)) return CN_CODEC_FALLBACK;
    options.color_mode = MODE_BGRA;
    const WebPData webp = {data, size};
    WebPAnimDecoder *decoder = WebPAnimDecoderNew(&webp, &options);
    if (!decoder) return CN_CODEC_FALLBACK;
    cn_codec_status status = CN_CODEC_FALLBACK;
    WebPAnimInfo info;
    uint8_t *frame;
    int timestamp;
    if (WebPAnimDecoderGetInfo(decoder, &info) && info.canvas_width == width && info.canvas_height == height &&
        WebPAnimDecoderGetNext(decoder, &frame, &timestamp)) {
        uint8_t *out = allocate(context, height, width, channels, 1);
        if (out) {
            const size_t pixels = (size_t)width * height;
            if (channels == 4) {
                memcpy(out, frame, pixels * 4);
            } else {
                for (size_t i = 0; i < pixels; ++i, frame += 4, out += 3) memcpy(out, frame, 3);
            }
            status = CN_CODEC_OK;
        }
    }
    WebPAnimDecoderDelete(decoder);
    return status;
}

int cn_webp_signature(const uint8_t *data, size_t size) {
    WebPBitstreamFeatures features;
    return size >= WEBP_HEADER_SIZE && WebPGetFeatures(data, WEBP_HEADER_SIZE, &features) == VP8_STATUS_OK;
}

cn_codec_status cn_webp_encode(const cn_codec_image *image, int quality, cn_codec_output *out) {
    const uint32_t channels = image->channels, width = image->width, height = image->height;
    if (image->sample_bytes != 1 || (channels != 1 && channels != 3 && channels != 4) || width > INT32_MAX / 4 ||
        height > INT32_MAX || image->stride > INT32_MAX)
        return CN_CODEC_FALLBACK;
    const int lossless = quality > 100;
    const float factor = quality < 1 ? 1.0f : (float)quality; /* OpenCV logs the clamp to 1 on stderr */
    const uint8_t *pixels = image->data;
    size_t stride = image->stride;
    uint8_t *grey = NULL;
    if (channels == 1) { /* cvtColor(GRAY2BGR) */
        stride = (size_t)width * 3;
        grey = (uint8_t *)malloc(stride * height);
        if (!grey) return CN_CODEC_FALLBACK;
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x)
                memset(grey + y * stride + x * 3, image->data[y * image->stride + x], 3);
        pixels = grey;
    }
    uint8_t *encoded = NULL;
    size_t size;
    if (channels == 4)
        size = lossless ? WebPEncodeLosslessBGRA(pixels, (int)width, (int)height, (int)stride, &encoded)
                        : WebPEncodeBGRA(pixels, (int)width, (int)height, (int)stride, factor, &encoded);
    else
        size = lossless ? WebPEncodeLosslessBGR(pixels, (int)width, (int)height, (int)stride, &encoded)
                        : WebPEncodeBGR(pixels, (int)width, (int)height, (int)stride, factor, &encoded);
    free(grey);
    if (size) cn_codec_output_append(out, encoded, size);
    WebPFree(encoded);
    return size && !out->failed ? CN_CODEC_OK : CN_CODEC_FALLBACK;
}

cn_codec_status cn_webp_encode_exact(const cn_codec_image *image, cn_codec_output *out) {
    if (image->sample_bytes != 1 || image->channels != 4 || image->width > INT32_MAX || image->height > INT32_MAX)
        return CN_CODEC_FALLBACK;
    WebPConfig config;
    if (!WebPConfigInit(&config)) return CN_CODEC_FALLBACK;
    config.lossless = 1;
    config.quality = 80.0f;
    config.alpha_quality = 100.0f;
    config.method = 4;
    config.exact = 1;
    WebPPicture picture;
    if (!WebPValidateConfig(&config) || !WebPPictureInit(&picture)) return CN_CODEC_FALLBACK;
    picture.width = (int)image->width;
    picture.height = (int)image->height;
    picture.use_argb = 1;
    if (!WebPPictureAlloc(&picture)) return CN_CODEC_FALLBACK;
    for (uint32_t y = 0; y < image->height; ++y) { /* import_frame_libwebp: RGBA to ARGB words */
        const uint8_t *in = image->data + (size_t)y * image->stride;
        uint32_t *argb = picture.argb + (size_t)picture.argb_stride * y;
        for (uint32_t x = 0; x < image->width; ++x, in += 4)
            argb[x] = (uint32_t)in[2] | (uint32_t)in[1] << 8 | (uint32_t)in[0] << 16 | (uint32_t)in[3] << 24;
    }
    WebPMemoryWriter writer;
    WebPMemoryWriterInit(&writer);
    picture.writer = WebPMemoryWrite;
    picture.custom_ptr = &writer;
    const int ok = WebPEncode(&config, &picture);
    WebPPictureFree(&picture);
    cn_codec_status status = CN_CODEC_FALLBACK;
    if (ok) {
        /* Pillow always reassembles the bitstream through WebPMux (no ICC, EXIF or XMP here). */
        WebPMux *mux = WebPMuxNew();
        const WebPData bitstream = {writer.mem, writer.size};
        WebPData assembled = {NULL, 0};
        if (mux && WebPMuxSetImage(mux, &bitstream, 0) == WEBP_MUX_OK && WebPMuxAssemble(mux, &assembled) == WEBP_MUX_OK &&
            assembled.size) {
            cn_codec_output_append(out, assembled.bytes, assembled.size);
            status = out->failed ? CN_CODEC_FALLBACK : CN_CODEC_OK;
        }
        WebPDataClear(&assembled);
        WebPMuxDelete(mux);
    }
    WebPMemoryWriterClear(&writer);
    return status;
}
