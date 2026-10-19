/* PBM, PGM and PPM (P1-P6) as OpenCV 5.0's own PxMDecoder reads them (grfmt_pxm.cpp; utils.cpp's FillGrayRow1 and
 * FillGrayRow8) (GPL-3.0; the format logic follows OpenCV's Apache-2.0 code). */
#include "image_codecs.h"

#include <limits.h>
#include <string.h>

/* RLByteStream over the file in memory. */
typedef struct pxm_stream {
    const uint8_t *data;
    size_t size, at;
} pxm_stream;

/* getByte: -1 past the end, where OpenCV throws RBS_THROW_EOS. */
static int pxm_byte(pxm_stream *s) { return s->at < s->size ? s->data[s->at++] : -1; }

static int pxm_digit(int c) { return c >= '0' && c <= '9'; }

static int pxm_space(int c) { return c == ' ' || (c >= '\t' && c <= '\r'); }

/* ReadNumber: a decimal number after whitespace and '#' comments (each to a CR or LF), its terminating byte read
 * unless `maxdigits` (0 or 1) digits end it first; -1 where OpenCV throws (the end of the data, any other byte, a
 * value above INT_MAX) and where isdigit or isspace would test a byte above ASCII, which the CRT locale classifies. */
static int64_t pxm_number(pxm_stream *s, int maxdigits) {
    int c = pxm_byte(s);
    while (!pxm_digit(c)) {
        if (c < 0 || c >= 0x80) return -1;
        if (c == '#') {
            do {
                c = pxm_byte(s);
                if (c < 0) return -1;
            } while (c != '\n' && c != '\r');
            c = pxm_byte(s);
        } else if (pxm_space(c)) {
            while (pxm_space(c)) c = pxm_byte(s);
        } else {
            return -1; /* "PXM: Unexpected code in ReadNumber()" */
        }
    }
    int64_t value = 0;
    for (int digits = 1;; ++digits) {
        value = value * 10 + (c - '0');
        if (value > INT_MAX) return -1;
        if (digits == maxdigits) break;
        c = pxm_byte(s);
        if (c < 0 || c >= 0x80) return -1;
        if (!pxm_digit(c)) break;
    }
    return value;
}

static void pxm_put(uint8_t *out, size_t index, uint32_t value, uint32_t sample_bytes) {
    if (sample_bytes == 1) {
        out[index] = (uint8_t)value;
    } else {
        const uint16_t sample = (uint16_t)value; /* native byte order */
        memcpy(out + index * 2, &sample, 2);
    }
}

cn_codec_status cn_pxm_decode(const uint8_t *data, size_t size, cn_codec_allocate allocate, void *context) {
    /* readHeader */
    pxm_stream s = {data, size, 0};
    if (pxm_byte(&s) != 'P') return CN_CODEC_FALLBACK;
    const int kind = pxm_byte(&s);
    if (kind < '1' || kind > '6') return CN_CODEC_FALLBACK;
    const uint32_t bpp = kind == '1' || kind == '4' ? 1 : kind == '2' || kind == '5' ? 8 : 24;
    const int binary = kind >= '4';
    const int64_t width = pxm_number(&s, 0), height = pxm_number(&s, 0), maxval = bpp == 1 ? 1 : pxm_number(&s, 0);
    if (width <= 0 || height <= 0 || maxval <= 0 || maxval > 65535) return CN_CODEC_FALLBACK;
    /* imdecode's validateInputImageSize */
    if (width > 1 << 20 || height > 1 << 20 || width * height > 1 << 30) return CN_CODEC_FALLBACK;

    /* readData: maxval above 255 is 16-bit; 8-bit ASCII samples go through i * 255 / maxval (PBM's inverted), while
     * binary samples are taken as they are; ASCII samples above maxval are maxval. RGB is stored as BGR. */
    const uint32_t channels = bpp == 24 ? 3 : 1, sample_bytes = maxval > 255 ? 2 : 1;
    const uint32_t w = (uint32_t)width, h = (uint32_t)height, samples = w * channels;
    const size_t pitch = ((size_t)w * bpp * sample_bytes + 7) / 8;
    uint8_t grey[256] = {0};
    if (sample_bytes == 1)
        for (uint32_t i = 0; i <= (uint32_t)maxval; ++i)
            grey[i] = (uint8_t)((i * 255 / (uint32_t)maxval) ^ (bpp == 1 ? 255 : 0));
    if (binary && (uint64_t)pitch * h > size - s.at) return CN_CODEC_FALLBACK; /* getBytes reaches the end */
    uint8_t *const out = allocate(context, h, w, channels, sample_bytes);
    if (!out) return CN_CODEC_FALLBACK;
    const uint8_t *const pixels = data + s.at; /* the binary rows */
    size_t index = 0;
    for (uint32_t y = 0; y < h; ++y, index += samples) {
        const uint8_t *src = binary ? pixels + (size_t)y * pitch : NULL;
        if (bpp == 1) {
            for (uint32_t x = 0; x < w; ++x) {
                if (binary) {
                    out[index + x] = grey[(src[x >> 3] >> (7 - (x & 7))) & 1];
                } else {
                    const int64_t bit = pxm_number(&s, 1);
                    if (bit < 0) return CN_CODEC_FALLBACK;
                    out[index + x] = grey[bit != 0];
                }
            }
        } else if (!binary) {
            for (uint32_t i = 0; i < samples; ++i) {
                int64_t code = pxm_number(&s, 0);
                if (code < 0) return CN_CODEC_FALLBACK;
                if (code > maxval) code = maxval;
                const size_t at = channels == 3 ? index + i + 2 - 2 * (i % 3) : index + i; /* icvCvt_RGB2BGR */
                pxm_put(out, at, sample_bytes == 1 ? grey[code] : (uint32_t)code, sample_bytes);
            }
        } else if (sample_bytes == 1 && channels == 1) {
            memcpy(out + index, src, w);
        } else if (sample_bytes == 1) {
            for (uint8_t *dst = out + index, *end = dst + samples; dst < end; dst += 3, src += 3) {
                dst[0] = src[2];
                dst[1] = src[1];
                dst[2] = src[0];
            }
        } else { /* big-endian samples */
            for (uint32_t i = 0; i < samples; ++i, src += 2)
                pxm_put(out, channels == 3 ? index + i + 2 - 2 * (i % 3) : index + i, (uint32_t)src[0] << 8 | src[1],
                        2);
        }
    }
    return CN_CODEC_OK;
}
