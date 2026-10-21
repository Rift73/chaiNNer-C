/* BMP as OpenCV 5.0's own codec reads and writes it (grfmt_bmp.cpp; utils.cpp's FillGrayPalette, IsColorPalette,
 * CvtPaletteToGray, FillUniColor/FillUniGray and the row fillers) (GPL-3.0; the format logic follows OpenCV's
 * Apache-2.0 code). */
#include "image_codecs.h"

#include <limits.h>
#include <string.h>

/* RLByteStream over the file in memory: skips and seeks may pass the end, a read past it fails (RBS_THROW_EOS). */
typedef struct bmp_stream {
    const uint8_t *data;
    size_t size;
    uint64_t at;
} bmp_stream;

static const uint8_t *bmp_take(bmp_stream *s, uint64_t count) {
    if (s->at > s->size || s->size - s->at < count) {
        s->at = UINT64_MAX; /* every later read fails too */
        return NULL;
    }
    const uint8_t *bytes = s->data + s->at;
    s->at += count;
    return bytes;
}

static void bmp_skip(bmp_stream *s, uint64_t count) {
    if (s->at != UINT64_MAX) s->at += count;
}

static uint32_t bmp_word(bmp_stream *s) {
    const uint8_t *p = bmp_take(s, 2);
    return p ? (uint32_t)p[0] | (uint32_t)p[1] << 8 : 0;
}

static uint32_t bmp_dword(bmp_stream *s) {
    const uint8_t *p = bmp_take(s, 4);
    return p ? (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24 : 0;
}

/* icvCvt_BGR2Gray_8u_C3C1R: descale(b * cB + g * cG + r * cR, 14) with cR 4899, cG 9617, cB 1868. */
static uint8_t bmp_grey(const uint8_t *bgr) {
    return (uint8_t)((bgr[0] * 1868 + bgr[1] * 9617 + bgr[2] * 4899 + 8192) >> 14);
}

static void bmp_put(uint8_t *out, const uint8_t *colour, uint32_t channels) {
    out[0] = colour[0];
    if (channels == 3) {
        out[1] = colour[1];
        out[2] = colour[2];
    }
}

/* The RLE decoders' cursor, OpenCV's data and line_end pointers as byte x of row y in decode order (bottom-up
 * files decode their last row first); `row_bytes` is width3, width times 1 or 3 channels. */
typedef struct bmp_rle {
    uint8_t *out;
    size_t row_bytes, x;
    uint32_t height, y, channels;
    int bottom_up;
} bmp_rle;

static uint8_t *bmp_rle_at(const bmp_rle *r) {
    return r->out + (size_t)(r->bottom_up ? r->height - 1 - r->y : r->y) * r->row_bytes + r->x;
}

/* FillUniColor/FillUniGray: `count` bytes of one colour from the cursor on, a row's end moving to the next row,
 * until the last row is done. */
static void bmp_rle_fill(bmp_rle *r, uint64_t count, const uint8_t *colour) {
    do {
        const size_t end = count < r->row_bytes - r->x ? r->x + (size_t)count : r->row_bytes;
        count -= end - r->x;
        for (uint8_t *p = bmp_rle_at(r); r->x < end; r->x += r->channels, p += r->channels)
            bmp_put(p, colour, r->channels);
        if (r->x >= r->row_bytes) {
            r->x = 0;
            if (++r->y >= r->height) break;
        }
    } while (count > 0);
}

/* readData's RLE8 loop: 1 once the image is filled, 0 where OpenCV fails (a run past its row's end, the end of
 * the data). An encoded run that ends its row moves to the next one, and an end of line right after it is ignored;
 * a delta and the end of the bitmap fill the pixels they skip with palette entry 0. */
static int bmp_rle8(bmp_stream *s, bmp_rle *r, uint8_t (*colours)[3]) {
    int line_end = 0;
    for (;;) {
        const uint8_t *op = bmp_take(s, 2);
        if (!op) return 0;
        const uint32_t len = op[0], code = op[1];
        if (len) {
            const uint32_t y = r->y;
            if (r->x + len * r->channels > r->row_bytes) return 0;
            bmp_rle_fill(r, len * r->channels, colours[code]);
            line_end = r->y != y;
            if (r->y >= r->height) return 1;
        } else if (code > 2) { /* absolute mode */
            if (r->x + code * r->channels > r->row_bytes) return 0;
            const uint8_t *indices = bmp_take(s, (code + 1) & ~1u);
            if (!indices) return 0;
            uint8_t *p = bmp_rle_at(r);
            for (uint32_t i = 0; i < code; ++i, p += r->channels) bmp_put(p, colours[indices[i]], r->channels);
            r->x += code * r->channels;
            line_end = 0;
        } else { /* 0 end of line, 1 end of bitmap, 2 delta */
            if (code || !line_end || r->x > 0) {
                uint64_t shift = r->row_bytes - r->x, rows = r->height - r->y;
                if (code == 2) {
                    const uint8_t *delta = bmp_take(s, 2);
                    if (!delta) return 0;
                    shift = (uint64_t)delta[0] * r->channels;
                    rows = delta[1];
                }
                if (code) shift += rows * r->row_bytes;
                bmp_rle_fill(r, shift, colours[0]);
                if (r->y >= r->height) return 1;
            }
            line_end = 0;
        }
    }
}

/* readData's RLE4 loop: runs never move to the next row themselves; the end of the bitmap is one more end of
 * line, and a delta moves right only (its vertical offset is read and dropped), filling with palette entry 0. */
static int bmp_rle4(bmp_stream *s, bmp_rle *r, uint8_t (*colours)[3]) {
    for (;;) {
        const uint8_t *op = bmp_take(s, 2);
        if (!op) return 0;
        const uint32_t len = op[0], code = op[1];
        if (len) {
            if (r->x + len * r->channels > r->row_bytes) return 0;
            const uint8_t *pair[2] = {colours[code >> 4], colours[code & 15]};
            uint8_t *p = bmp_rle_at(r);
            for (uint32_t i = 0; i < len; ++i, p += r->channels) bmp_put(p, pair[i & 1], r->channels);
            r->x += len * r->channels;
        } else if (code > 2) { /* absolute mode */
            if (r->x + code * r->channels > r->row_bytes) return 0;
            const uint8_t *indices = bmp_take(s, (((code + 1) >> 1) + 1) & ~1u);
            if (!indices) return 0;
            uint8_t *p = bmp_rle_at(r);
            for (uint32_t i = 0; i < code; ++i, p += r->channels)
                bmp_put(p, colours[(indices[i >> 1] >> (i & 1 ? 0 : 4)) & 15], r->channels);
            r->x += code * r->channels;
        } else { /* 0 end of line, 1 end of bitmap, 2 delta */
            uint64_t shift = r->row_bytes - r->x;
            if (code == 2) {
                const uint8_t *delta = bmp_take(s, 2);
                if (!delta) return 0;
                shift = (uint64_t)delta[0] * r->channels;
            }
            bmp_rle_fill(r, shift, colours[0]);
            if (r->y >= r->height) return 1;
        }
    }
}

/* maskBGRA's channel: (pixel & mask) >> offset times 255.0f / (mask >> offset), truncated. */
static uint8_t bmp_mask(uint32_t pixel, uint32_t mask, int offset, float scale) {
    return (uint8_t)((float)((pixel & mask) >> offset) * scale);
}

cn_codec_status cn_bmp_decode(const uint8_t *data, size_t size, cn_codec_allocate allocate, void *context) {
    /* readHeader */
    bmp_stream s = {data, size, 10};
    const uint32_t offset = bmp_dword(&s), header = bmp_dword(&s);
    if (offset > INT32_MAX || header == 0 || header > INT32_MAX) return CN_CODEC_FALLBACK; /* m_offset < 0, size > 0 */
    uint32_t width, height_field, bpp, compression = 0, colours_used = 0;
    uint32_t masks[4] = {0, 0, 0, 0}; /* red, green, blue, alpha */
    int offsets[4] = {-1, -1, -1, -1};
    float scales[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    uint8_t palette[256][4]; /* b, g, r, a */
    memset(palette, 0, sizeof palette);
    int colour = 0, valid = 0;
    if (header >= 36) {
        width = bmp_dword(&s);
        height_field = bmp_dword(&s);
        bpp = bmp_dword(&s) >> 16;
        compression = bmp_dword(&s);
        bmp_skip(&s, 12);
        colours_used = bmp_dword(&s);
        if (compression > 3) return CN_CODEC_FALLBACK; /* BMP_RGB, BMP_RLE8, BMP_RLE4, BMP_BITFIELDS */
        if (bpp == 32 && compression == 3 && header >= 56) {
            bmp_skip(&s, 4);
            for (int i = 0; i < 4; ++i) {
                uint32_t mask = masks[i] = bmp_dword(&s);
                if (!mask) continue;
                int shift = 0;
                for (; !(mask & 1); mask >>= 1) ++shift;
                offsets[i] = shift;
                scales[i] = 255.0f / (float)mask;
            }
            bmp_skip(&s, header - 56);
        } else {
            bmp_skip(&s, header - 36);
        }
        if (width - 1 < INT32_MAX && height_field &&
            (((bpp == 1 || bpp == 4 || bpp == 8 || bpp == 24 || bpp == 32) && compression == 0) ||
             ((bpp == 16 || bpp == 32) && (compression == 0 || compression == 3)) || (bpp == 4 && compression == 2) ||
             (bpp == 8 && compression == 1))) {
            colour = valid = 1;
            if (bpp <= 8) {
                if (colours_used > 256) return CN_CODEC_FALLBACK;
                const uint32_t entries = colours_used ? colours_used : 1u << bpp;
                const uint8_t *entry = bmp_take(&s, (uint64_t)entries * 4);
                if (!entry) return CN_CODEC_FALLBACK;
                memcpy(palette, entry, (size_t)entries * 4);
                colour = 0; /* IsColorPalette over 1 << bpp entries */
                for (uint32_t i = 0; i < 1u << bpp; ++i)
                    if (palette[i][0] != palette[i][1] || palette[i][0] != palette[i][2]) colour = 1;
            } else if (bpp == 16 && compression == 3) {
                const uint32_t red = bmp_dword(&s), green = bmp_dword(&s), blue = bmp_dword(&s);
                if (blue == 0x1f && green == 0x3e0 && red == 0x7c00) bpp = 15;
                else if (!(blue == 0x1f && green == 0x7e0 && red == 0xf800)) valid = 0;
            } else if (bpp == 16 && compression == 0) {
                bpp = 15;
            }
        }
    } else if (header == 12) { /* OS/2 BITMAPCOREHEADER: the image is always grey (iscolor stays false) */
        width = bmp_word(&s);
        height_field = bmp_word(&s);
        bpp = bmp_dword(&s) >> 16;
        if (width && height_field && (bpp == 1 || bpp == 4 || bpp == 8 || bpp == 24 || bpp == 32)) {
            if (bpp <= 8) {
                const uint8_t *entry = bmp_take(&s, (uint64_t)3 << bpp);
                if (!entry) return CN_CODEC_FALLBACK;
                for (uint32_t i = 0; i < 1u << bpp; ++i, entry += 3) memcpy(palette[i], entry, 3);
            }
            valid = 1;
        }
    } else {
        return CN_CODEC_FALLBACK;
    }
    if (!valid || s.at == UINT64_MAX || height_field == 0x80000000u) return CN_CODEC_FALLBACK;
    const uint32_t channels = !colour ? 1 : bpp == 32 && compression != 0 ? 4 : 3;
    const int bottom_up = height_field <= INT32_MAX;
    const uint32_t height = bottom_up ? height_field : 0u - height_field;
    /* imdecode's validateInputImageSize */
    if (width > 1u << 20 || height > 1u << 20 || (uint64_t)width * height > CN_MAX_IMAGE_PIXELS) return CN_CODEC_FALLBACK;

    /* readData */
    const uint32_t nch = channels > 1 ? 3 : 1, unit = bpp == 15 ? 16 : bpp;
    const uint64_t pitch = (((uint64_t)width * unit + 7) / 8 + 3) & ~UINT64_C(3);
    const int rle = compression == 1 || compression == 2;
    if (offset > size || (!rle && pitch * height > size - offset)) return CN_CODEC_FALLBACK; /* reads past the end */
    uint8_t colours[256][3]; /* the palette as written: BGR, or CvtPaletteToGray's grey */
    memset(colours, 0, sizeof colours);
    if (bpp <= 8)
        for (uint32_t i = 0; i < 1u << bpp; ++i) {
            if (colour) memcpy(colours[i], palette[i], 3);
            else colours[i][0] = bmp_grey(palette[i]);
        }
    uint8_t *const out = allocate(context, height, width, channels, 1);
    if (!out) return CN_CODEC_FALLBACK;
    const size_t row_bytes = (size_t)width * channels;
    if (rle) {
        s.at = offset;
        bmp_rle cursor = {out, row_bytes, 0, height, 0, nch, bottom_up};
        const int done = compression == 1 ? bmp_rle8(&s, &cursor, colours) : bmp_rle4(&s, &cursor, colours);
        return done ? CN_CODEC_OK : CN_CODEC_FALLBACK;
    }
    const int has_masks = offsets[0] >= 0 && offsets[1] >= 0 && offsets[2] >= 0;
    for (uint32_t y = 0; y < height; ++y) {
        const uint8_t *src = data + offset + y * pitch;
        uint8_t *dst = out + (size_t)(bottom_up ? height - 1 - y : y) * row_bytes;
        switch (bpp) {
        case 1:
            for (uint32_t x = 0; x < width; ++x, dst += nch)
                bmp_put(dst, colours[(src[x >> 3] >> (7 - (x & 7))) & 1], nch);
            break;
        case 4:
            for (uint32_t x = 0; x < width; ++x, dst += nch)
                bmp_put(dst, colours[(src[x >> 1] >> (x & 1 ? 0 : 4)) & 15], nch);
            break;
        case 8:
            for (uint32_t x = 0; x < width; ++x, dst += nch) bmp_put(dst, colours[src[x]], nch);
            break;
        case 15:
        case 16: /* icvCvt_BGR5552BGR_8u_C2C3R, icvCvt_BGR5652BGR_8u_C2C3R */
            for (uint32_t x = 0; x < width; ++x, src += 2, dst += 3) {
                const uint32_t v = (uint32_t)src[0] | (uint32_t)src[1] << 8;
                dst[0] = (uint8_t)((v << 3) & 0xf8);
                dst[1] = (uint8_t)(bpp == 15 ? (v >> 2) & 0xf8 : (v >> 3) & 0xfc);
                dst[2] = (uint8_t)(bpp == 15 ? (v >> 7) & 0xf8 : (v >> 8) & 0xf8);
            }
            break;
        case 24:
            if (channels == 3) memcpy(dst, src, (size_t)width * 3);
            else
                for (uint32_t x = 0; x < width; ++x, src += 3) dst[x] = bmp_grey(src);
            break;
        default: /* 32 */
            if (channels == 1) {
                for (uint32_t x = 0; x < width; ++x, src += 4) dst[x] = bmp_grey(src); /* icvCvt_BGRA2Gray_8u_C4C1R */
            } else if (channels == 3) {
                for (uint32_t x = 0; x < width; ++x, src += 4, dst += 3)
                    memcpy(dst, src, 3); /* icvCvt_BGRA2BGR_8u_C4C3R */
            } else if (!has_masks) {
                memcpy(dst, src, (size_t)width * 4);
            } else {
                for (uint32_t x = 0; x < width; ++x, src += 4, dst += 4) {
                    const uint32_t v =
                        (uint32_t)src[0] | (uint32_t)src[1] << 8 | (uint32_t)src[2] << 16 | (uint32_t)src[3] << 24;
                    dst[0] = bmp_mask(v, masks[2], offsets[2], scales[2]);
                    dst[1] = bmp_mask(v, masks[1], offsets[1], scales[1]);
                    dst[2] = bmp_mask(v, masks[0], offsets[0], scales[0]);
                    dst[3] = offsets[3] >= 0 ? bmp_mask(v, masks[3], offsets[3], scales[3]) : 255;
                }
            }
            break;
        }
    }
    return CN_CODEC_OK;
}

static void put32(cn_codec_output *out, uint32_t value) {
    const uint8_t bytes[4] = {(uint8_t)value, (uint8_t)(value >> 8), (uint8_t)(value >> 16), (uint8_t)(value >> 24)};
    cn_codec_output_append(out, bytes, 4);
}

static void put16(cn_codec_output *out, uint32_t value) {
    const uint8_t bytes[2] = {(uint8_t)value, (uint8_t)(value >> 8)};
    cn_codec_output_append(out, bytes, 2);
}

cn_codec_status cn_bmp_encode(const cn_codec_image *image, cn_codec_output *out) {
    const uint32_t channels = image->channels, width = image->width, height = image->height;
    if (image->sample_bytes != 1 || (channels != 1 && channels != 3 && channels != 4)) return CN_CODEC_FALLBACK;
    const uint64_t row = (uint64_t)width * channels, step = (row + 3) & ~UINT64_C(3);
    const int bitfields = channels == 4; /* IMWRITE_BMP_COMPRESSION's default: BITFIELDS, for 32-bit pixels only */
    const uint32_t header = bitfields ? 124 : 40, palette = channels > 1 ? 0 : 1024;
    const uint32_t offset = 14 + header + palette;
    const uint64_t size = step * height + offset;
    if (size > INT_MAX || width > INT_MAX || height > INT_MAX) return CN_CODEC_FALLBACK; /* validateToInt */
    cn_codec_output_reserve(out, (size_t)size);
    cn_codec_output_append(out, "BM", 2);
    put32(out, (uint32_t)size);
    put32(out, 0);
    put32(out, offset);
    put32(out, header);
    put32(out, width);
    put32(out, height);
    put16(out, 1);
    put16(out, channels << 3);
    put32(out, bitfields ? 3 : 0); /* BMP_BITFIELDS or BMP_RGB */
    for (int i = 0; i < 5; ++i) put32(out, 0);
    if (bitfields) {
        put32(out, 0x00FF0000); /* red, green, blue and alpha masks */
        put32(out, 0x0000FF00);
        put32(out, 0x000000FF);
        put32(out, 0xFF000000);
        cn_codec_output_append(out, "BGRs", 4); /* bV5CSType: sRGB */
        for (int i = 0; i < 9 + 3 + 4; ++i) put32(out, 0); /* endpoints, gammas, intent, profile, reserved */
    }
    if (channels == 1) {
        for (uint32_t i = 0; i < 256; ++i) {
            const uint8_t entry[4] = {(uint8_t)i, (uint8_t)i, (uint8_t)i, 0}; /* i * 255 / 255 */
            cn_codec_output_append(out, entry, 4);
        }
    }
    static const uint8_t zeros[3] = {0, 0, 0};
    for (uint32_t y = height; y-- > 0;) {
        cn_codec_output_append(out, image->data + (size_t)y * image->stride, (size_t)row);
        cn_codec_output_append(out, zeros, (size_t)(step - row));
    }
    return out->failed ? CN_CODEC_FALLBACK : CN_CODEC_OK;
}
