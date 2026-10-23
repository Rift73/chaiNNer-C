/* TIFF through libtiff 4.7.1 built as OpenCV 5.0 builds it (third_party/libtiff/chainner), with OpenCV 5.0's
 * TiffDecoder under IMREAD_UNCHANGED and its TiffEncoder (grfmt_tiff.cpp), both over memory with OpenCV's client
 * procs (GPL-3.0). libtiff reports errors through a handler, not a jump: any error it reports (and any libjpeg
 * warning), even one OpenCV would read past, falls back, so today's path decides those files. */
#include "image_codecs.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tiffio.h"

/* OpenCV's handlers log; here an error only marks the file (and the save) for today's path. */
static int tiff_error(TIFF *tif, void *user_data, const char *module, const char *format, va_list arguments) {
    (void)tif;
    (void)module;
    (void)format;
    (void)arguments;
    *(int *)user_data = 1;
    return 1;
}

/* Warnings change nothing, except libjpeg's (tif_jpeg.c reports them as "JPEGLib"): on corrupt data its SIMD and C
 * transforms can differ, and OpenCV's libjpeg-turbo always runs its SIMD while this one heeds JSIMD_FORCENONE. */
static int tiff_warning(TIFF *tif, void *user_data, const char *module, const char *format, va_list arguments) {
    (void)tif;
    (void)format;
    (void)arguments;
    if (module && strcmp(module, "JPEGLib") == 0) *(int *)user_data = 1;
    return 1;
}

/* cv_tiffClientOpen: TIFFClientOpenExt with cv_tiffCreateOptions' options (the handlers, and libtiff 4.7.1's
 * warnings about unknown tags, which change nothing but the warnings). */
static TIFF *tiff_open(const char *mode, thandle_t handle, TIFFReadWriteProc read, TIFFReadWriteProc write,
                       TIFFSeekProc seek, TIFFCloseProc close, TIFFSizeProc size, TIFFMapFileProc map, int *failed) {
    TIFFOpenOptions *options = TIFFOpenOptionsAlloc();
    if (!options) return NULL;
    TIFFOpenOptionsSetErrorHandlerExtR(options, tiff_error, failed);
    TIFFOpenOptionsSetWarningHandlerExtR(options, tiff_warning, failed);
    TIFFOpenOptionsSetWarnAboutUnknownTags(options, 1);
    TIFF *tif = TIFFClientOpenExt("", mode, handle, read, write, seek, close, size, map, NULL, options);
    TIFFOpenOptionsFree(options);
    return tif;
}

/* Decoding: TiffDecoderBufHelper's procs over the file's bytes. Reads stop at the end, seeks clamp to it; its map
 * proc fills in the buffer but returns 0, so libtiff reads through read and seek (not mapped). */
typedef struct tiff_source {
    const uint8_t *data;
    uint64_t size, at;
} tiff_source;

static tmsize_t source_read(thandle_t handle, void *buffer, tmsize_t n) {
    tiff_source *source = (tiff_source *)handle;
    const tmsize_t size = (tmsize_t)source->size, at = (tmsize_t)source->at;
    if (n > size - at) n = size - at;
    memcpy(buffer, source->data + at, (size_t)n);
    source->at += (uint64_t)n;
    return n;
}

static tmsize_t source_write(thandle_t handle, void *buffer, tmsize_t n) {
    (void)handle;
    (void)buffer;
    (void)n;
    return 0;
}

static toff_t source_seek(thandle_t handle, toff_t offset, int whence) {
    tiff_source *source = (tiff_source *)handle;
    toff_t at = source->at;
    switch (whence) {
    case SEEK_SET: at = offset; break;
    case SEEK_CUR: at += offset; break;
    case SEEK_END: at = source->size + offset; break;
    default: break;
    }
    if (at > source->size) at = source->size;
    source->at = at;
    return at;
}

static int source_map(thandle_t handle, void **base, toff_t *size) {
    tiff_source *source = (tiff_source *)handle;
    *base = (void *)(uintptr_t)source->data;
    *size = source->size;
    return 0;
}

static toff_t source_size(thandle_t handle) { return ((tiff_source *)handle)->size; }

static int source_close(thandle_t handle) {
    (void)handle;
    return 0;
}

/* utils.cpp's grey weights (SCALE 14) as icvCvt_*2Gray applies them with swap_rb: R, G, B. */
#define GREY_R 4899
#define GREY_G 9617
#define GREY_B 1868
#define GREY(r, g, b) (((r) * GREY_R + (g) * GREY_G + (b) * GREY_B + (1 << 13)) >> 14)

/* One row of TIFFReadRGBA*'s raster (R, G, B, A bytes) as readData converts it: icvCvt_BGRA2RGBA_8u_C4R,
 * icvCvt_BGRA2BGR_8u_C4C3R or icvCvt_BGRA2Gray_8u_C4C1R, each with R and B swapped. */
static void rgba_row(const uint8_t *in, uint8_t *out, uint32_t width, uint32_t channels) {
    for (uint32_t x = 0; x < width; ++x, in += 4) {
        if (channels == 1) {
            *out++ = (uint8_t)GREY(in[0], in[1], in[2]);
            continue;
        }
        out[0] = in[2];
        out[1] = in[1];
        out[2] = in[0];
        if (channels == 4) out[3] = in[3];
        out += channels;
    }
}

/* OpenCV's channel swaps between RGB(A) and BGR(A) (cvtColor, extend_cvtColor): R and B exchanged, every sample
 * copied as its bits, so float NaN payloads stay. */
#define SWAP_RED_BLUE(type)                                                                                        \
    do {                                                                                                           \
        const type *from = (const type *)in;                                                                       \
        type *to = (type *)out;                                                                                    \
        for (uint32_t x = 0; x < width; ++x, from += channels, to += channels) {                                  \
            const type red = from[0];                                                                              \
            to[0] = from[2];                                                                                       \
            to[1] = from[1];                                                                                       \
            to[2] = red;                                                                                           \
            if (channels == 4) to[3] = from[3];                                                                    \
        }                                                                                                          \
    } while (0)

static void swap_red_blue(const uint8_t *in, uint8_t *out, uint32_t width, uint32_t channels, size_t sample_bytes) {
    switch (sample_bytes) {
    case 1: SWAP_RED_BLUE(uint8_t); break;
    case 2: SWAP_RED_BLUE(uint16_t); break;
    case 4: SWAP_RED_BLUE(uint32_t); break;
    default: SWAP_RED_BLUE(uint64_t); break;
    }
}

/* One row of 16-bit samples (ncn per pixel) as readData's dst_bpp 16 case converts it: grey copied, or grey from the
 * first three samples (icvCvt_BGRA2Gray_16u_CnC1R), RGB(A) swapped to BGR(A) (channels == ncn). */
static void raw16_row(const uint16_t *in, uint16_t *out, uint32_t width, uint32_t ncn, uint32_t channels) {
    if (channels > 1) {
        swap_red_blue((const uint8_t *)in, (uint8_t *)out, width, channels, 2);
    } else if (ncn == 1) {
        memcpy(out, in, (size_t)width * 2);
    } else {
        for (uint32_t x = 0; x < width; ++x, in += ncn) out[x] = (uint16_t)GREY((int)in[0], (int)in[1], (int)in[2]);
    }
}

/* _unpack10To16, _unpack12To16 and _unpack14To16: `count` big-endian packed samples of `bits` bits from one row
 * (bytes up to end), the missing tail as zero bits. */
static void unpack_to16(const uint8_t *in, const uint8_t *end, uint16_t *out, size_t count, uint32_t bits) {
    const uint64_t mask = ((uint64_t)1 << bits) - 1;
    uint64_t buffer = 0;
    uint32_t held = 0;
    for (size_t i = 0; i < count; ++i) {
        while (held < bits) {
            buffer = buffer << 8 | (in < end ? *in++ : 0);
            held += 8;
        }
        held -= bits;
        out[i] = (uint16_t)(buffer >> held & mask);
    }
}

/* OpenCV's flip(img, img, code) in place on rows x columns pixels: 0 turns the rows over, 1 the columns, -1 both. */
static void flip_pixels(uint8_t *p, uint32_t rows, uint32_t columns, size_t pixel, int code) {
    const size_t stride = (size_t)columns * pixel;
    uint8_t swap[32];
    if (code == 0 || code == -1) {
        for (uint32_t y = 0; y < rows / 2; ++y) {
            uint8_t *a = p + (size_t)y * stride, *b = p + (size_t)(rows - 1 - y) * stride;
            for (size_t i = 0; i < stride; i += pixel) {
                memcpy(swap, a + i, pixel);
                memcpy(a + i, b + i, pixel);
                memcpy(b + i, swap, pixel);
            }
        }
    }
    if (code == 1 || code == -1) {
        for (uint32_t y = 0; y < rows; ++y) {
            uint8_t *row = p + (size_t)y * stride;
            for (uint32_t x = 0; x < columns / 2; ++x) {
                uint8_t *a = row + x * pixel, *b = row + (size_t)(columns - 1 - x) * pixel;
                memcpy(swap, a, pixel);
                memcpy(a, b, pixel);
                memcpy(b, swap, pixel);
            }
        }
    }
}

/* fixOrientationFull (true) and fixOrientationPartial (false, after TIFFReadRGBA*) as a flip, a transpose and a
 * flip; NO_FLIP is none. */
#define NO_FLIP 2
typedef struct tiff_orientation {
    int before, transpose, after;
} tiff_orientation;

static tiff_orientation orientation_fix(uint16_t orientation, int full) {
    tiff_orientation fix = {NO_FLIP, 0, NO_FLIP};
    if (full) {
        switch (orientation) {
        case ORIENTATION_TOPRIGHT: fix.before = 1; break;
        case ORIENTATION_BOTRIGHT: fix.before = -1; break;
        case ORIENTATION_BOTLEFT: fix.before = 0; break;
        case ORIENTATION_LEFTTOP: fix.transpose = 1; break;
        case ORIENTATION_RIGHTTOP: fix.after = 1; break;
        case ORIENTATION_RIGHTBOT: fix.after = -1; break;
        case ORIENTATION_LEFTBOT: fix.after = 0; break;
        default: break;
        }
        if (fix.after != NO_FLIP) fix.transpose = 1;
    } else if (orientation == ORIENTATION_RIGHTTOP || orientation == ORIENTATION_LEFTBOT) {
        fix.before = -1;
        fix.transpose = 1;
    } else if (orientation == ORIENTATION_LEFTTOP || orientation == ORIENTATION_RIGHTBOT) {
        fix.transpose = 1;
    }
    return fix;
}

/* OpenCV's limits: imdecode's image size (validateInputImageSize), readData's tile and buffer sizes. */
#define OPENCV_MAX_SIDE (UINT32_C(1) << 20)
#define OPENCV_MAX_PIXELS CN_MAX_IMAGE_PIXELS
#define TILE_MAX_SIDE (UINT32_C(1) << 24)
#define MAX_TILE_SIZE (UINT64_C(1) << 30)
/* readData reads a strip or tile of this many bytes or more (as 32-bit RGBA for 8-bit images) scanline by scanline
 * where it can; that path is not ported, and every buffer this large declines. */
#define LARGE_TILE_SIZE (MAX_TILE_SIZE * 95 / 100)

/* What readHeader and readData find about the first page: the output's size and cn_codec_sample, the samples as
 * stored (bpp, ncn), the orientation tag and the read path (tiles, output bits). */
typedef struct tiff_image {
    uint32_t width, height, channels, sample;
    uint16_t bpp, ncn, orientation;
    int tiled, dst_bits;
} tiff_image;

static int tiff_header(TIFF *tif, tiff_image *image) {
    uint32_t width = 0, height = 0;
    uint16_t photometric = 0;
    if (!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width) || !TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height) ||
        !TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photometric))
        return 0;
    const int grey = photometric == PHOTOMETRIC_MINISWHITE || photometric == PHOTOMETRIC_MINISBLACK;
    uint16_t bpp = 8, ncn = grey ? 1 : 3;
    if (!TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bpp)) bpp = 1; /* bi-level images may lack it */
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &ncn);
    TIFFNumberOfDirectories(tif); /* m_frame_count: walks every IFD */
    /* LogLuv (m_hdr: XYZ floats, cvtColor's XYZ2BGR) is not ported; neither are the other CIE spaces' float
     * conversions in the RGBA interface. */
    if (photometric == PHOTOMETRIC_LOGLUV || photometric == PHOTOMETRIC_LOGL || photometric == PHOTOMETRIC_CIELAB ||
        photometric == PHOTOMETRIC_ICCLAB || photometric == PHOTOMETRIC_ITULAB)
        return 0;
    if (bpp > 8 && (photometric > 2 || (ncn != 1 && ncn != 3 && ncn != 4))) bpp = 8;
    uint16_t format = SAMPLEFORMAT_UINT;
    TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &format);
    if (ncn < 1 || ncn > 4) return 0; /* normalizeChannelsNumber */
    const int integer = format == SAMPLEFORMAT_UINT || format == SAMPLEFORMAT_INT;
    const int is_signed = format == SAMPLEFORMAT_INT;
    uint32_t channels = !grey ? ncn : 1, sample;
    switch (bpp) {
    case 1:
    case 4:
    case 8:
        /* 4 bits only for a palette; a 1-bit palette stays one channel (grey through the RGBA interface) */
        if (!integer || (bpp == 4 && photometric != PHOTOMETRIC_PALETTE)) return 0;
        if (bpp != 1 && photometric == PHOTOMETRIC_PALETTE) channels = 3;
        sample = is_signed ? CN_SAMPLE_I8 : CN_SAMPLE_U8;
        break;
    case 10:
    case 12:
    case 14:
    case 16:
        if (!integer) return 0;
        sample = is_signed ? CN_SAMPLE_I16 : CN_SAMPLE_U16;
        break;
    case 32:
    case 64:
        if (!integer && format != SAMPLEFORMAT_IEEEFP) return 0;
        channels = ncn;
        sample = format == SAMPLEFORMAT_IEEEFP ? (bpp == 32 ? CN_SAMPLE_F32 : CN_SAMPLE_F64)
                 : is_signed                   ? (bpp == 32 ? CN_SAMPLE_I32 : CN_SAMPLE_I64)
                                               : (bpp == 32 ? CN_SAMPLE_U32 : CN_SAMPLE_U64);
        break;
    default: return 0;
    }
    if (!width || !height || width > OPENCV_MAX_SIDE || height > OPENCV_MAX_SIDE ||
        (uint64_t)width * height > OPENCV_MAX_PIXELS)
        return 0;
    image->width = width;
    image->height = height;
    image->channels = channels;
    image->sample = sample;
    return 1;
}

/* readData from the tag reads up to the loops, for the paths ported: the RGBA interface for 8 bits, raw strips or
 * tiles for 16, 32 and 64. Returns the buffer bytes per tile row and the tile size, or 0. */
static int tiff_layout(TIFF *tif, tiff_image *image, uint32_t *tile_width, uint32_t *tile_height, size_t *row_bytes) {
    uint16_t photometric = 0;
    if (!TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photometric)) return 0;
    image->tiled = TIFFIsTiled(tif) != 0;
    const int grey = photometric == PHOTOMETRIC_MINISWHITE || photometric == PHOTOMETRIC_MINISBLACK;
    image->bpp = 8;
    image->ncn = grey ? 1 : 3;
    if (!TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &image->bpp)) image->bpp = 1;
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &image->ncn);
    image->orientation = ORIENTATION_TOPLEFT;
    TIFFGetField(tif, TIFFTAG_ORIENTATION, &image->orientation);
    image->dst_bits = (int)(image->sample & 0xFF) * 8;
    uint32_t tw = image->width, th = 0;
    if (image->tiled) {
        if (!TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tw) || !TIFFGetField(tif, TIFFTAG_TILELENGTH, &th)) return 0;
    } else {
        TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &th);
    }
    if (tw == 0) tw = image->width;
    if (th == 0 || (!image->tiled && th == UINT32_MAX)) th = image->height;
    if (tw > TILE_MAX_SIDE || th > TILE_MAX_SIDE || image->ncn > 4 || image->bpp > 64) return 0;
    if (!image->tiled) {
        /* The strips libtiff reads must be the ones the loops expect (an absent tag defaults elsewhere). */
        uint32_t rows = 0;
        TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rows);
        if (rows != th && (rows < image->height || th < image->height)) return 0;
    }
    if (image->dst_bits == 8) {
        if ((uint64_t)tw * th * 4 >= LARGE_TILE_SIZE) return 0;
        char message[1024];
        if (!TIFFRGBAImageOK(tif, message)) return 0;
        image->bpp = 8;
        image->ncn = 4;
        /* readData's CV_CheckEQ for the 8-bit conversions */
        if (image->channels == 2) return 0;
    } else {
        uint16_t planar = PLANARCONFIG_CONTIG;
        TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &planar);
        /* Raw samples come from grey or RGB(A) files only (readHeader turns every other one with more than 8 bits
         * into 8). OpenCV reads the first plane's strips as if they held every sample: not ported; neither are
         * signed 10- to 14-bit samples, which its scaling saturates. */
        if (planar == PLANARCONFIG_SEPARATE && image->ncn > 1) return 0;
        if (image->bpp < image->dst_bits && image->sample == CN_SAMPLE_I16) return 0;
        if (image->dst_bits == 16) {
            const uint32_t bytes = image->bpp / 8 > 1 ? image->bpp / 8u : 1u;
            if ((uint64_t)tw * th * image->ncn * bytes >= LARGE_TILE_SIZE) return 0;
        } else if (image->sample == CN_SAMPLE_F32 || image->sample == CN_SAMPLE_F64) {
            if (!TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP)) return 0;
        }
    }
    /* divUp(static_cast<size_t>(ncn * tile_width0 * bpp), 8): the product is 32-bit in OpenCV. */
    const uint64_t bits = (uint64_t)image->ncn * tw * image->bpp;
    if (bits > UINT32_MAX) return 0;
    *row_bytes = (size_t)((bits + 7) / 8);
    if ((uint64_t)th * *row_bytes >= LARGE_TILE_SIZE) return 0;
    *tile_width = tw;
    *tile_height = th;
    return 1;
}

/* readData's loops over the strips or tiles into `out` (height x width x channels, rows as stored). */
static int tiff_read(TIFF *tif, const tiff_image *image, uint32_t tw, uint32_t th, size_t row_bytes, uint8_t *out) {
    const uint32_t width = image->width, height = image->height, channels = image->channels;
    const size_t sample_bytes = image->sample & 0xFF, pixel = channels * sample_bytes;
    const size_t stride = (size_t)width * pixel, buffer_size = (size_t)th * row_bytes;
    const uint16_t o = image->orientation;
    const int vertical_flip = image->dst_bits == 8 && (o == ORIENTATION_BOTRIGHT || o == ORIENTATION_RIGHTBOT ||
                                                      o == ORIENTATION_BOTLEFT || o == ORIENTATION_LEFTBOT);
    const int unpack = image->bpp < image->dst_bits;
    const size_t unpacked_bytes = (size_t)image->ncn * tw * 2;
    uint8_t *buffer = (uint8_t *)malloc(buffer_size ? buffer_size : 1);
    uint16_t *unpacked = unpack ? (uint16_t *)malloc(unpacked_bytes) : NULL;
    int ok = buffer && (!unpack || unpacked);
    uint32_t index = 0;
    for (uint32_t y = 0; ok && y < height; y += th) {
        const uint32_t rows = th < height - y ? th : height - y;
        const uint32_t image_y = vertical_flip ? height - y - rows : y;
        for (uint32_t x = 0; ok && x < width; x += tw, ++index) {
            const uint32_t columns = tw < width - x ? tw : width - x;
            if (image->dst_bits == 8) {
                const uint8_t *start = buffer;
                if (!image->tiled) {
                    ok = TIFFReadRGBAStrip(tif, y, (uint32_t *)buffer);
                } else {
                    ok = TIFFReadRGBATile(tif, x, y, (uint32_t *)buffer);
                    start += (size_t)(th - rows) * tw * 4; /* tiles fill the raster from the bottom up */
                }
                for (uint32_t i = 0; ok && i < rows; ++i)
                    rgba_row(start + (size_t)i * tw * 4, out + (size_t)(image_y + rows - i - 1) * stride + x * pixel,
                             columns, channels);
                continue;
            }
            const tmsize_t read = image->tiled ? TIFFReadEncodedTile(tif, index, buffer, (tmsize_t)buffer_size)
                                               : TIFFReadEncodedStrip(tif, index, buffer, (tmsize_t)buffer_size);
            /* Fewer bytes than the rows used would leave OpenCV's buffer partly unset. */
            if (read < 0 || (uint64_t)read < (uint64_t)rows * row_bytes) {
                ok = 0;
                break;
            }
            for (uint32_t i = 0; i < rows; ++i) {
                const uint8_t *row = buffer + (size_t)i * row_bytes;
                uint8_t *target = out + (size_t)(image_y + i) * stride + x * pixel;
                if (image->dst_bits == 16) {
                    const uint16_t *samples = (const uint16_t *)row;
                    if (unpack) {
                        unpack_to16(row, row + row_bytes, unpacked, (size_t)image->ncn * tw, image->bpp);
                        samples = unpacked;
                    }
                    raw16_row(samples, (uint16_t *)target, columns, image->ncn, channels);
                } else if (channels >= 3) {
                    swap_red_blue(row, target, columns, channels, sample_bytes); /* extend_cvtColor */
                } else {
                    memcpy(target, row, columns * pixel);
                }
            }
        }
    }
    free(buffer);
    free(unpacked);
    if (ok && unpack) {
        /* img *= 1 << (16 - bpp): exact for unsigned samples below 1 << bpp */
        uint16_t *samples = (uint16_t *)out;
        for (size_t i = 0, n = (size_t)height * width * channels; i < n; ++i)
            samples[i] = (uint16_t)(samples[i] << (16 - image->bpp));
    }
    return ok;
}

static void transpose_pixels(const uint8_t *in, uint8_t *out, uint32_t rows, uint32_t columns, size_t pixel) {
    for (uint32_t x = 0; x < columns; ++x)
        for (uint32_t y = 0; y < rows; ++y, out += pixel) memcpy(out, in + ((size_t)y * columns + x) * pixel, pixel);
}

cn_codec_status cn_tiff_decode(const uint8_t *data, size_t size, cn_codec_allocate allocate, void *context) {
    tiff_source source = {data, size, 0};
    int failed = 0;
    TIFF *tif = tiff_open("r", &source, source_read, source_write, source_seek, source_close, source_size,
                          source_map, &failed);
    if (!tif) return CN_CODEC_FALLBACK;
    cn_codec_status status = CN_CODEC_FALLBACK;
    tiff_image image;
    uint32_t tw, th;
    size_t row_bytes;
    if (!failed && tiff_header(tif, &image) && !failed && tiff_layout(tif, &image, &tw, &th, &row_bytes) &&
        !failed) {
        const tiff_orientation fix = orientation_fix(image.orientation, image.dst_bits != 8);
        const size_t pixel = image.channels * (size_t)(image.sample & 0xFF);
        const uint32_t height = image.height, width = image.width;
        uint8_t *work = fix.transpose ? (uint8_t *)malloc((size_t)height * width * pixel)
                                      : allocate(context, height, width, image.channels, image.sample);
        if (work && tiff_read(tif, &image, tw, th, row_bytes, work) && !failed) {
            if (fix.before != NO_FLIP) flip_pixels(work, height, width, pixel, fix.before);
            if (!fix.transpose) {
                status = CN_CODEC_OK;
            } else {
                uint8_t *out = allocate(context, width, height, image.channels, image.sample);
                if (out) {
                    transpose_pixels(work, out, height, width, pixel);
                    if (fix.after != NO_FLIP) flip_pixels(out, width, height, pixel, fix.after);
                    status = CN_CODEC_OK;
                }
            }
        }
        if (fix.transpose) free(work);
    }
    TIFFClose(tif);
    return failed ? CN_CODEC_FALLBACK : status;
}

/* The straight-alpha decoder (load_image.py's _read_tiff_straight_alpha through Pillow 12.3's TiffImagePlugin) for
 * the plainest of the files it takes: IFD0 as ImageFileDirectory_v2.load reads it without a warning, only the tags
 * below, each with the type and count Pillow and libtiff agree on. Anything else falls back to the Python decoder. */
typedef struct ifd_reader {
    const uint8_t *data;
    size_t size;
    int little, big;
} ifd_reader;

static uint64_t ifd_read(const ifd_reader *r, size_t at, unsigned bytes) {
    uint64_t value = 0;
    for (unsigned i = 0; i < bytes; ++i) value |= (uint64_t)r->data[at + (r->little ? i : bytes - 1 - i)] << (8 * i);
    return value;
}

/* A tag's type, count and where its values are. */
typedef struct ifd_entry {
    uint32_t type;
    uint64_t count;
    size_t at;
} ifd_entry;

enum {
    TAG_SUBFILE, TAG_WIDTH, TAG_LENGTH, TAG_BITS, TAG_COMPRESSION, TAG_PHOTOMETRIC, TAG_FILL, TAG_OFFSETS,
    TAG_ORIENTATION, TAG_SAMPLES, TAG_ROWS, TAG_COUNTS, TAG_XRES, TAG_YRES, TAG_PLANAR, TAG_UNIT, TAG_PREDICTOR,
    TAG_EXTRA, TAG_FORMAT, TAG_XMP, TAG_KINDS
};
static const uint16_t straight_alpha_tags[TAG_KINDS] = {254, 256, 257, 258, 259, 262, 266, 273, 274, 277,
                                                        278, 279, 282, 283, 284, 296, 317, 338, 339, 700};

/* A value's size in bytes for the types Pillow reads (ImageFileDirectory_v2's _load_dispatch), 0 for the others. */
static unsigned ifd_unit(uint32_t type) {
    switch (type) {
    case 1: case 2: case 6: case 7: return 1; /* BYTE, ASCII, SBYTE, UNDEFINED */
    case 3: case 8: return 2;                 /* SHORT, SSHORT */
    case 4: case 9: case 11: case 13: return 4; /* LONG, SLONG, FLOAT, IFD */
    case 5: case 10: case 12: case 16: return 8; /* RATIONAL, SRATIONAL, DOUBLE, LONG8 */
    default: return 0;
    }
}

/* Value i of a SHORT, LONG or LONG8 entry. */
static uint64_t ifd_value(const ifd_reader *r, const ifd_entry *e, uint64_t i) {
    const unsigned unit = ifd_unit(e->type);
    return ifd_read(r, e->at + (size_t)i * unit, unit);
}

/* Whether the entry has one of the types in `types` (a 0-terminated list) and `count` values (0: any count). */
static int ifd_is(const ifd_entry *e, const uint32_t *types, uint64_t count) {
    if (count && e->count != count) return 0;
    for (; *types; ++types)
        if (e->type == *types) return 1;
    return 0;
}

/* Every value of an entry equal to `value`. */
static int ifd_all(const ifd_reader *r, const ifd_entry *e, uint64_t value) {
    for (uint64_t i = 0; i < e->count; ++i)
        if (ifd_value(r, e, i) != value) return 0;
    return 1;
}

static const uint32_t TYPE_SHORT[] = {3, 0}, TYPE_LONG[] = {4, 0}, TYPE_INTEGER[] = {3, 4, 0},
                      TYPE_OFFSET[] = {3, 4, 16, 0}, TYPE_RATIONAL[] = {5, 0}, TYPE_ASCII[] = {2, 0},
                      TYPE_BYTE[] = {1, 0};

/* Whether Pillow loads the Exif IFD at `ifd` without a warning: TiffImageFile.load_end reads it (getexif's get_ifd),
 * and ImageFileDirectory_v2.load warns when its entry count, an entry, a value stored apart or the next IFD's offset
 * is cut short. Several numbers for a tag Pillow's Exif table gives one value (ExifVersion, FlashpixVersion, the
 * Interoperability IFD, CFAPattern) warn too ("Metadata Warning"); every other tag of the group takes any count.
 * Types Pillow does not read are skipped, as it skips them. */
static int exif_ifd_quiet(const ifd_reader *r, uint64_t ifd) {
    const unsigned word = r->big ? 8 : 4, count_bytes = r->big ? 8 : 2, entry_size = r->big ? 20 : 12;
    if (ifd > r->size || r->size - ifd < count_bytes) return 0;
    const uint64_t entries = ifd_read(r, (size_t)ifd, count_bytes), room = r->size - ifd - count_bytes;
    if (entries > room / entry_size || room - entries * entry_size < word) return 0;
    for (uint64_t i = 0; i < entries; ++i) {
        const size_t at = (size_t)(ifd + count_bytes + i * entry_size);
        const uint32_t tag = (uint32_t)ifd_read(r, at, 2), type = (uint32_t)ifd_read(r, at + 2, 2);
        const uint64_t count = ifd_read(r, at + 4, word);
        const unsigned unit = ifd_unit(type);
        if (!unit) continue;
        if (count > 1 && type != 1 && type != 2 && type != 7 &&
            (tag == 36864 || tag == 40960 || tag == 40965 || tag == 41730))
            return 0;
        if (count > r->size / unit) return 0;
        if (count * unit > word) {
            const uint64_t offset = ifd_read(r, at + 4 + word, word);
            if (offset > r->size || count * unit > r->size - offset) return 0;
        }
    }
    return 1;
}

/* The orientation Pillow's getexif() takes from the XMP packet when IFD0 has no Orientation tag: the digit after
 * the first `tiff:Orientation="` or `tiff:Orientation>` (re.search of rb'tiff:Orientation(="|>)([0-9])'), else 1.
 * TiffImageFile.load_end turns the image by it (exif_transpose). */
static uint64_t xmp_orientation(const ifd_reader *r, const ifd_entry *xmp) {
    static const char key[] = "tiff:Orientation";
    const size_t key_length = sizeof key - 1, size = (size_t)xmp->count;
    const uint8_t *p = r->data + xmp->at;
    for (size_t i = 0; size - i > key_length; ++i) {
        if (memcmp(p + i, key, key_length) != 0) continue;
        size_t at = i + key_length;
        if (p[at] == '=' && size - at > 1 && p[at + 1] == '"') at += 2;
        else if (p[at] == '>') at += 1;
        else continue;
        if (at < size && p[at] >= '0' && p[at] <= '9') return (uint64_t)(p[at] - '0');
    }
    return 1;
}

/* Reads IFD0 into `tags` (absent tags have count 0): false for anything Pillow would warn about, skip, or that this
 * decoder does not take. */
static int straight_alpha_ifd(const ifd_reader *r, ifd_entry *tags) {
    const uint8_t *p = r->data;
    if (r->size < 8 || !((p[0] == 'I' && p[1] == 'I') || (p[0] == 'M' && p[1] == 'M'))) return 0;
    /* Pillow takes a file for BigTIFF when its third byte is 43, so only little-endian BigTIFF reads as one. */
    const unsigned magic = (unsigned)ifd_read(r, 2, 2);
    if (magic == 43 && (!r->little || r->size < 16 || ifd_read(r, 4, 2) != 8 || ifd_read(r, 6, 2) != 0)) return 0;
    if (magic != 42 && magic != 43) return 0;
    /* Classic: 2-byte entry count, 12-byte entries, 4-byte offsets; BigTIFF: 8, 20 and 8. */
    const int big = r->big;
    const unsigned word = big ? 8 : 4, count_bytes = big ? 8 : 2, entry_size = big ? 20 : 12;
    const uint64_t ifd = ifd_read(r, big ? 8 : 4, word);
    if (ifd > r->size || r->size - ifd < count_bytes) return 0;
    const uint64_t entries = ifd_read(r, (size_t)ifd, count_bytes);
    if (entries > 4096) return 0;
    const uint64_t end = ifd + count_bytes + entries * entry_size;
    if (end > r->size || r->size - end < word) return 0; /* the next IFD's offset */
    memset(tags, 0, sizeof(ifd_entry) * TAG_KINDS);
    uint32_t previous = 0;
    for (uint64_t i = 0; i < entries; ++i) {
        const size_t at = (size_t)(ifd + count_bytes + i * entry_size);
        const uint32_t tag = (uint32_t)ifd_read(r, at, 2), type = (uint32_t)ifd_read(r, at + 2, 2);
        const uint64_t count = ifd_read(r, at + 4, word);
        if (i && tag <= previous) return 0; /* ascending, no duplicates */
        previous = tag;
        const unsigned unit = ifd_unit(type);
        if (!unit || !count || count > (UINT64_C(1) << 32)) return 0;
        const uint64_t size = count * unit;
        size_t values = at + 4 + word;
        if (size > word) {
            const uint64_t offset = ifd_read(r, values, word);
            if (offset > r->size || size > r->size - offset) return 0;
            values = (size_t)offset;
        }
        const ifd_entry e = {type, count, values};
        /* Tags Pillow's _setup never reads (or, ICC, reads as the bytes they are), in their usual types: text,
         * PageNumber, white point and primaries, IPTC, Photoshop, the Exif IFD's offset (its IFD as Pillow's load_end
         * reads it without a warning), ICC profile, Photoshop layers. */
        if ((tag == 318 && ifd_is(&e, TYPE_RATIONAL, 2)) || (tag == 319 && ifd_is(&e, TYPE_RATIONAL, 6))) continue;
        if (tag == 269 || tag == 270 || tag == 271 || tag == 272 || tag == 285 || tag == 305 || tag == 306 ||
            tag == 315) {
            if (!ifd_is(&e, TYPE_ASCII, 0)) return 0;
            continue;
        }
        if (tag == 297 || tag == 33723 || tag == 34377 || tag == 34665 || tag == 34675 || tag == 37724) {
            static const uint32_t TYPE_UNDEFINED[] = {7, 0}, TYPE_BLOB[] = {1, 7, 0}, TYPE_IPTC[] = {1, 4, 7, 0},
                                  TYPE_IFD[] = {4, 13, 0};
            const int usual = tag == 297     ? ifd_is(&e, TYPE_SHORT, 2)
                              : tag == 33723 ? ifd_is(&e, TYPE_IPTC, 0)
                              : tag == 34377 ? ifd_is(&e, TYPE_BLOB, 0)
                              : tag == 34665 ? ifd_is(&e, TYPE_IFD, 1) && exif_ifd_quiet(r, ifd_value(r, &e, 0))
                                             : ifd_is(&e, TYPE_UNDEFINED, 0);
            if (!usual) return 0;
            continue;
        }
        int known = 0;
        for (int k = 0; k < TAG_KINDS; ++k)
            if (straight_alpha_tags[k] == tag) {
                tags[k] = e;
                known = 1;
            }
        if (!known) return 0;
    }
    return 1;
}

cn_codec_status cn_tiff_decode_pillow_alpha(const uint8_t *data, size_t size, cn_codec_allocate allocate,
                                            void *context, int *orientation) {
    ifd_reader r = {data, size, size >= 2 && data[0] == 'I', size >= 3 && data[2] == 43};
    ifd_entry tags[TAG_KINDS];
    if (!straight_alpha_ifd(&r, tags)) return CN_CODEC_FALLBACK;
    const ifd_entry *t = tags;
    /* The image: 8-bit RGBA or grey+alpha with ExtraSamples (2,) (Pillow's modes RGBA and LA, raw modes RGBA and
     * LA), contiguous strips, a compression Pillow and libtiff decode losslessly. */
    if (!ifd_is(&t[TAG_WIDTH], TYPE_INTEGER, 1) || !ifd_is(&t[TAG_LENGTH], TYPE_INTEGER, 1) ||
        !ifd_is(&t[TAG_SAMPLES], TYPE_SHORT, 1) || !ifd_is(&t[TAG_PHOTOMETRIC], TYPE_SHORT, 1) ||
        !ifd_is(&t[TAG_EXTRA], TYPE_SHORT, 1) || !ifd_is(&t[TAG_ROWS], TYPE_INTEGER, 1) ||
        !ifd_is(&t[TAG_BITS], TYPE_SHORT, 0))
        return CN_CODEC_FALLBACK;
    const uint64_t width = ifd_value(&r, &t[TAG_WIDTH], 0), height = ifd_value(&r, &t[TAG_LENGTH], 0);
    const uint64_t samples = ifd_value(&r, &t[TAG_SAMPLES], 0), photometric = ifd_value(&r, &t[TAG_PHOTOMETRIC], 0);
    const uint64_t rows = ifd_value(&r, &t[TAG_ROWS], 0);
    if (!width || !height || width > INT32_MAX || height > INT32_MAX || !rows ||
        !((photometric == PHOTOMETRIC_RGB && samples == 4) || (photometric == PHOTOMETRIC_MINISBLACK && samples == 2)) ||
        ifd_value(&r, &t[TAG_EXTRA], 0) != EXTRASAMPLE_UNASSALPHA ||
        (t[TAG_BITS].count != 1 && t[TAG_BITS].count != samples) || !ifd_all(&r, &t[TAG_BITS], 8))
        return CN_CODEC_FALLBACK;
    const uint64_t compression = t[TAG_COMPRESSION].count ? ifd_value(&r, &t[TAG_COMPRESSION], 0) : COMPRESSION_NONE;
    if ((t[TAG_COMPRESSION].count && !ifd_is(&t[TAG_COMPRESSION], TYPE_SHORT, 1)) ||
        (compression != COMPRESSION_NONE && compression != COMPRESSION_LZW && compression != COMPRESSION_ADOBE_DEFLATE &&
         compression != COMPRESSION_DEFLATE && compression != COMPRESSION_PACKBITS))
        return CN_CODEC_FALLBACK;
    /* Optional tags with the one value (or the values) taken. */
    if ((t[TAG_FILL].count && (!ifd_is(&t[TAG_FILL], TYPE_SHORT, 1) || !ifd_all(&r, &t[TAG_FILL], 1))) ||
        (t[TAG_PLANAR].count && (!ifd_is(&t[TAG_PLANAR], TYPE_SHORT, 1) || !ifd_all(&r, &t[TAG_PLANAR], 1))) ||
        (t[TAG_SUBFILE].count && (!ifd_is(&t[TAG_SUBFILE], TYPE_LONG, 1) || !ifd_all(&r, &t[TAG_SUBFILE], 0))) ||
        (t[TAG_FORMAT].count && (!ifd_is(&t[TAG_FORMAT], TYPE_SHORT, 0) || !ifd_all(&r, &t[TAG_FORMAT], 1) ||
                                 (t[TAG_FORMAT].count != 1 && t[TAG_FORMAT].count != samples))) ||
        (t[TAG_ORIENTATION].count && !ifd_is(&t[TAG_ORIENTATION], TYPE_SHORT, 1)) ||
        (t[TAG_XMP].count && !ifd_is(&t[TAG_XMP], TYPE_BYTE, 0)) ||
        (t[TAG_UNIT].count && (!ifd_is(&t[TAG_UNIT], TYPE_SHORT, 1) || ifd_value(&r, &t[TAG_UNIT], 0) < 1 ||
                               ifd_value(&r, &t[TAG_UNIT], 0) > 3)))
        return CN_CODEC_FALLBACK;
    if (t[TAG_PREDICTOR].count) {
        const uint64_t predictor = ifd_value(&r, &t[TAG_PREDICTOR], 0);
        if (!ifd_is(&t[TAG_PREDICTOR], TYPE_SHORT, 1) || (predictor != PREDICTOR_NONE && predictor != PREDICTOR_HORIZONTAL) ||
            (predictor != PREDICTOR_NONE && compression == COMPRESSION_NONE))
            return CN_CODEC_FALLBACK;
    }
    for (int k = TAG_XRES; k <= TAG_YRES; ++k) /* rationals Pillow turns into the dpi: no zero denominator */
        if (t[k].count && (!ifd_is(&t[k], TYPE_RATIONAL, 1) || !ifd_read(&r, t[k].at + 4, 4))) return CN_CODEC_FALLBACK;
    const uint64_t strips = (height + rows - 1) / rows;
    if (!ifd_is(&t[TAG_OFFSETS], TYPE_OFFSET, strips) || !ifd_is(&t[TAG_COUNTS], TYPE_OFFSET, strips))
        return CN_CODEC_FALLBACK;
    const size_t row_bytes = (size_t)width * samples;
    const uint64_t value = t[TAG_ORIENTATION].count ? ifd_value(&r, &t[TAG_ORIENTATION], 0)
                           : t[TAG_XMP].count       ? xmp_orientation(&r, &t[TAG_XMP])
                                                    : 1;
    *orientation = value >= 2 && value <= 8 ? (int)value : 1; /* image_io_orient's range, exif_transpose's */
    uint8_t *const out = allocate(context, (uint32_t)height, (uint32_t)width, 4, CN_SAMPLE_U8);
    if (!out) return CN_CODEC_FALLBACK;
    /* Rows as Pillow gives them after convert("RGBA") (grey copied to R, G and B) and cvtColor's RGBA2BGRA. */
#define STRAIGHT_ALPHA_ROWS(source, count, first)                                                                  \
    for (uint64_t y = 0; y < (count); ++y) {                                                                       \
        const uint8_t *in = (source) + y * row_bytes;                                                              \
        uint8_t *to = out + ((first) + y) * width * 4;                                                             \
        for (uint64_t x = 0; x < width; ++x, in += samples, to += 4) {                                             \
            to[0] = samples == 4 ? in[2] : in[0];                                                                  \
            to[1] = samples == 4 ? in[1] : in[0];                                                                  \
            to[2] = in[0];                                                                                         \
            to[3] = in[samples - 1];                                                                               \
        }                                                                                                          \
    }
    if (compression == COMPRESSION_NONE) {
        /* Pillow's raw decoder: each strip's rows from its offset (StripByteCounts unused), truncation an error */
        for (uint64_t i = 0; i < strips; ++i) {
            const uint64_t first = i * rows, count = rows < height - first ? rows : height - first;
            const uint64_t offset = ifd_value(&r, &t[TAG_OFFSETS], i);
            if (offset > size || count * row_bytes > size - offset) return CN_CODEC_FALLBACK;
            STRAIGHT_ALPHA_ROWS(data + offset, count, first)
        }
        return CN_CODEC_OK;
    }
    /* Pillow's libtiff decoder (TiffDecode.c _decodeStrip): TIFFReadEncodedStrip strip by strip. */
    tiff_source source = {data, size, 0};
    int failed = 0;
    TIFF *tif = tiff_open("r", &source, source_read, source_write, source_seek, source_close, source_size,
                          source_map, &failed);
    if (!tif) return CN_CODEC_FALLBACK;
    uint32_t libtiff_rows = 0;
    TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &libtiff_rows);
    const tmsize_t strip_size = TIFFStripSize(tif);
    uint8_t *buffer = !failed && !TIFFIsTiled(tif) && TIFFNumberOfStrips(tif) == strips && libtiff_rows == rows &&
                              TIFFScanlineSize(tif) == (tmsize_t)row_bytes && strip_size > 0
                          ? (uint8_t *)malloc((size_t)strip_size)
                          : NULL;
    int ok = buffer != NULL;
    for (uint64_t i = 0; ok && i < strips; ++i) {
        const uint64_t first = i * rows, count = rows < height - first ? rows : height - first;
        const tmsize_t read = TIFFReadEncodedStrip(tif, (uint32_t)i, buffer, strip_size);
        ok = read >= 0 && (uint64_t)read >= count * row_bytes && !failed;
        if (ok) STRAIGHT_ALPHA_ROWS(buffer, count, first)
    }
#undef STRAIGHT_ALPHA_ROWS
    free(buffer);
    TIFFClose(tif);
    return ok && !failed ? CN_CODEC_OK : CN_CODEC_FALLBACK;
}

/* Encoding: TiffEncoderBufHelper's procs over the output. A write past the end grows it (a gap reads as zeros),
 * seeks are not clamped; a read past the end would throw in OpenCV and falls back here. */
typedef struct tiff_sink {
    cn_codec_output *out;
    uint64_t at;
    int failed;
} tiff_sink;

static tmsize_t sink_read(thandle_t handle, void *buffer, tmsize_t n) {
    tiff_sink *sink = (tiff_sink *)handle;
    const uint64_t end = sink->at + (uint64_t)n;
    if (sink->out->size <= end) {
        sink->failed = 1;
        return 0;
    }
    memcpy(buffer, sink->out->data + sink->at, (size_t)n);
    sink->at = end;
    return n;
}

static tmsize_t sink_write(thandle_t handle, void *buffer, tmsize_t n) {
    tiff_sink *sink = (tiff_sink *)handle;
    cn_codec_output *out = sink->out;
    const uint64_t end = sink->at + (uint64_t)n;
    if (out->failed || end > SIZE_MAX / 2) {
        sink->failed = 1;
        return 0;
    }
    if (end > out->size) {
        if (end > out->capacity) cn_codec_output_reserve(out, end > out->capacity * 2 ? (size_t)end : out->capacity * 2);
        if (out->failed) {
            sink->failed = 1;
            return 0;
        }
        memset(out->data + out->size, 0, (size_t)end - out->size);
        out->size = (size_t)end;
    }
    memcpy(out->data + sink->at, buffer, (size_t)n);
    sink->at = end;
    return n;
}

static toff_t sink_seek(thandle_t handle, toff_t offset, int whence) {
    tiff_sink *sink = (tiff_sink *)handle;
    switch (whence) {
    case SEEK_SET: sink->at = offset; break;
    case SEEK_CUR: sink->at += offset; break;
    case SEEK_END: sink->at = sink->out->size + offset; break;
    default: break;
    }
    return sink->at;
}

static toff_t sink_size(thandle_t handle) { return ((tiff_sink *)handle)->out->size; }

static int sink_close(thandle_t handle) {
    (void)handle;
    return 0;
}

cn_codec_status cn_tiff_encode(const cn_codec_image *image, int compression, cn_codec_output *out) {
    const uint32_t channels = image->channels, width = image->width, height = image->height;
    const int is_float = image->sample_bytes == 4;
    if ((image->sample_bytes != 1 && image->sample_bytes != 2 && !is_float) ||
        (channels != 1 && channels != 3 && channels != 4) || !width || !height || width > INT32_MAX ||
        height > INT32_MAX)
        return CN_CODEC_FALLBACK;
    /* writeLibTiff's defaults: LZW (none for 32F) and the horizontal predictor; Save Image's float32 has no
     * parameters, so other compressions with float stay with OpenCV. */
    if (compression == 0) compression = is_float ? COMPRESSION_NONE : COMPRESSION_LZW;
    if (is_float ? compression != COMPRESSION_NONE
                 : compression != COMPRESSION_NONE && compression != COMPRESSION_LZW &&
                       compression != COMPRESSION_ADOBE_DEFLATE)
        return CN_CODEC_FALLBACK;
    const uint32_t bits = image->sample_bytes * 8;
    /* fileStep = (width * channels * bitsPerChannel) / 8, in int */
    if ((uint64_t)width * channels * bits > INT32_MAX) return CN_CODEC_FALLBACK;
    const size_t file_step = (size_t)width * channels * bits / 8;
    size_t rows_per_strip = (size_t)(1 << 13) / file_step;
    if (rows_per_strip > height) rows_per_strip = height;
    if (rows_per_strip < 1) rows_per_strip = 1;
    cn_codec_output_reserve(out, file_step * height + 4096);
    tiff_sink sink = {out, 0, 0};
    int failed = 0;
    TIFF *tif = tiff_open("w", &sink, sink_read, sink_write, sink_seek, sink_close, sink_size, NULL, &failed);
    if (!tif) return CN_CODEC_FALLBACK;
    int ok = TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width) && TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height) &&
             TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bits) && TIFFSetField(tif, TIFFTAG_COMPRESSION, compression) &&
             TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, channels > 1 ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK) &&
             TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, channels) &&
             TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG) &&
             TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, (uint32_t)rows_per_strip) &&
             TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, is_float ? SAMPLEFORMAT_IEEEFP : SAMPLEFORMAT_UINT);
    if (ok && compression != COMPRESSION_NONE) ok = TIFFSetField(tif, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
    /* A row buffer, because TIFFWriteScanline changes the data it is given. */
    const tmsize_t scanline = ok ? TIFFScanlineSize(tif) : 0;
    uint8_t *row = scanline > 0 && (size_t)scanline == file_step ? (uint8_t *)malloc((size_t)scanline + 32) : NULL;
    ok = ok && row;
    for (uint32_t y = 0; ok && y < height; ++y) {
        const uint8_t *in = image->data + (size_t)y * image->stride;
        if (channels == 1) {
            memcpy(row, in, file_step);
        } else {
            swap_red_blue(in, row, width, channels, image->sample_bytes); /* cvtColor's BGR2RGB / BGRA2RGBA */
        }
        ok = TIFFWriteScanline(tif, row, y, 0) == 1;
    }
    ok = ok && TIFFWriteDirectory(tif);
    TIFFClose(tif);
    free(row);
    return ok && !failed && !sink.failed && !out->failed ? CN_CODEC_OK : CN_CODEC_FALLBACK;
}
