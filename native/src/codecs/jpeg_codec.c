/* JPEG through libjpeg-turbo 3.1.4.1 with the decoder settings of Pillow 12.3's JpegDecode.c and OpenCV 5.0's
 * grfmt_jpeg.cpp (GPL-3.0). Both leave libjpeg's defaults (ISLOW IDCT, fancy upsampling, block smoothing, scale
 * 1/1, the colour space libjpeg guesses from the JFIF and Adobe markers) and read from a memory source that
 * suspends at the end of its data; only the output colour space and the CMYK formula differ. */
#include "image_codecs.h"

#include <setjmp.h>
#include <stdio.h>
#include <string.h>

#include "jpeglib.h"
#include "jerror.h"

typedef struct jpeg_failure {
    struct jpeg_error_mgr manager;
    jmp_buf jump;
} jpeg_failure;

static void jpeg_fail(j_common_ptr info) { longjmp(((jpeg_failure *)info->err)->jump, 1); }

/* Warnings are counted (a warning falls back), trace messages dropped, nothing printed. */
static void jpeg_message(j_common_ptr info, int level) {
    if (level < 0) info->err->num_warnings++;
}

static void source_none(j_decompress_ptr info) { (void)info; }

/* The whole file is in the buffer: needing more suspends, as the suspending sources of Pillow and OpenCV do once
 * their data is gone. */
static boolean source_suspend(j_decompress_ptr info) {
    (void)info;
    return FALSE;
}

static void source_skip(j_decompress_ptr info, long count) {
    struct jpeg_source_mgr *source = info->src;
    if (count <= 0) return;
    const size_t skipped = (size_t)count < source->bytes_in_buffer ? (size_t)count : source->bytes_in_buffer;
    source->next_input_byte += skipped;
    source->bytes_in_buffer -= skipped;
}

/* OpenCV's default side limit (loadsave.cpp validateInputImageSize) and the decoders' pixel limit. */
#define OPENCV_MAX_SIDE (UINT32_C(1) << 20)
#define OPENCV_MAX_PIXELS CN_MAX_IMAGE_PIXELS
#define ROWS_PER_READ 16

/* Pillow: CMYK;I inverts each sample, cmyk2rgb gives CLIP8(nk - MULDIV255(c, nk)) with nk = 255 - k, here
 * k - MULDIV255(255 - c, k), which never needs the clip. */
static uint8_t pillow_cmyk(unsigned value, unsigned k) {
    const unsigned t = (255u - value) * k + 128u;
    return (uint8_t)(k - (((t >> 8) + t) >> 8));
}

/* OpenCV: icvCvt_CMYK2BGR_8u_C4C3R's c = k - ((255 - c) * k >> 8). */
static uint8_t opencv_cmyk(unsigned value, unsigned k) { return (uint8_t)(k - (((255u - value) * k) >> 8)); }

cn_codec_status cn_jpeg_decode(const uint8_t *data, size_t size, cn_jpeg_flavour flavour, int components,
                               cn_codec_allocate allocate, void *context) {
    struct jpeg_decompress_struct info;
    jpeg_failure failure;
    struct jpeg_source_mgr source;
    volatile cn_codec_status status = CN_CODEC_FALLBACK;
    info.err = jpeg_std_error(&failure.manager);
    failure.manager.error_exit = jpeg_fail;
    failure.manager.emit_message = jpeg_message;
    if (setjmp(failure.jump)) {
        jpeg_destroy_decompress(&info);
        return status;
    }
    jpeg_create_decompress(&info);
    source.init_source = source_none;
    source.fill_input_buffer = source_suspend;
    source.skip_input_data = source_skip;
    source.resync_to_restart = jpeg_resync_to_restart;
    source.term_source = source_none;
    source.next_input_byte = data;
    source.bytes_in_buffer = size;
    info.src = &source;
    if (flavour == CN_JPEG_OPENCV) {
        jpeg_save_markers(&info, JPEG_APP0 + 1, 0xffff);
        jpeg_save_markers(&info, JPEG_APP0 + 2, 0xffff);
    }
    if (jpeg_read_header(&info, TRUE) != JPEG_HEADER_OK || info.data_precision != 8 ||
        (components && info.num_components != components))
        goto fallback;
    const int cmyk = info.num_components == 4;
    const int grey = flavour == CN_JPEG_PILLOW ? info.num_components == 1 : info.num_components <= 1;
    info.out_color_space = cmyk ? JCS_CMYK : grey ? JCS_GRAYSCALE : JCS_EXT_BGR;
    jpeg_calc_output_dimensions(&info);
    if (flavour == CN_JPEG_OPENCV &&
        (!info.output_width || !info.output_height || info.output_width > OPENCV_MAX_SIDE ||
         info.output_height > OPENCV_MAX_SIDE ||
         (uint64_t)info.output_width * info.output_height > OPENCV_MAX_PIXELS))
        goto fallback;
    if (!jpeg_start_decompress(&info) || failure.manager.num_warnings) goto fallback;
    const uint32_t width = info.output_width, height = info.output_height, channels = grey ? 1 : 3;
    uint8_t *const pixels = allocate(context, height, width, channels, 1);
    if (!pixels) goto fallback;
    const size_t stride = (size_t)width * channels;
    JSAMPARRAY scratch = cmyk ? (*info.mem->alloc_sarray)((j_common_ptr)&info, JPOOL_IMAGE,
                                                          (JDIMENSION)width * 4, ROWS_PER_READ)
                              : NULL;
    JSAMPROW rows[ROWS_PER_READ];
    while (info.output_scanline < height) {
        const uint32_t first = info.output_scanline;
        const uint32_t count = height - first < ROWS_PER_READ ? height - first : ROWS_PER_READ;
        for (uint32_t i = 0; i < count; ++i) rows[i] = cmyk ? scratch[i] : pixels + (size_t)(first + i) * stride;
        const JDIMENSION read = jpeg_read_scanlines(&info, rows, count);
        if (!read || failure.manager.num_warnings) goto fallback;
        if (!cmyk) continue;
        for (JDIMENSION i = 0; i < read; ++i) {
            const uint8_t *in = scratch[i];
            uint8_t *out = pixels + (size_t)(first + i) * stride;
            for (uint32_t x = 0; x < width; ++x, in += 4, out += 3) {
                if (flavour == CN_JPEG_PILLOW) {
                    out[0] = pillow_cmyk(in[2], in[3]);
                    out[1] = pillow_cmyk(in[1], in[3]);
                    out[2] = pillow_cmyk(in[0], in[3]);
                } else {
                    out[0] = opencv_cmyk(in[2], in[3]);
                    out[1] = opencv_cmyk(in[1], in[3]);
                    out[2] = opencv_cmyk(in[0], in[3]);
                }
            }
        }
    }
    if (flavour == CN_JPEG_OPENCV) {
        /* OpenCV sets its result before jpeg_finish_decompress and keeps it through anything there. */
        status = CN_CODEC_OK;
        jpeg_finish_decompress(&info);
    } else {
        /* Pillow is done once every row is out, also when jpeg_finish_decompress suspends; an error there fails. */
        jpeg_finish_decompress(&info);
        status = failure.manager.num_warnings ? CN_CODEC_FALLBACK : CN_CODEC_OK;
    }
fallback:
    jpeg_destroy_decompress(&info);
    return status;
}

/* Encoding: OpenCV 5.0's JpegEncoder::write. Its destination fills a 4 KiB buffer and appends it to the output; this
 * one writes into the output's free space directly (the same bytes). */
typedef struct jpeg_output {
    struct jpeg_destination_mgr manager;
    cn_codec_output *out;
} jpeg_output;

/* Points the destination at the output's free space, growing it when less than 4 KiB is left. */
static void output_space(j_compress_ptr info) {
    jpeg_output *output = (jpeg_output *)info->dest;
    cn_codec_output *out = output->out;
    if (out->capacity - out->size < 4096) cn_codec_output_reserve(out, out->capacity ? out->capacity * 2 : 65536);
    if (out->failed) {
        info->err->msg_code = JERR_OUT_OF_MEMORY;
        (*info->err->error_exit)((j_common_ptr)info);
    }
    output->manager.next_output_byte = out->data + out->size;
    output->manager.free_in_buffer = out->capacity - out->size;
}

static void output_start(j_compress_ptr info) { output_space(info); }

static boolean output_flush(j_compress_ptr info) {
    jpeg_output *output = (jpeg_output *)info->dest;
    output->out->size = output->out->capacity;
    output_space(info);
    return TRUE;
}

static void output_finish(j_compress_ptr info) {
    jpeg_output *output = (jpeg_output *)info->dest;
    output->out->size = (size_t)(output->manager.next_output_byte - output->out->data);
}

cn_codec_status cn_jpeg_encode(const cn_codec_image *image, int quality, uint32_t sampling, int progressive,
                               cn_codec_output *out) {
    const uint32_t channels = image->channels;
    if (image->sample_bytes != 1 || (channels != 1 && channels != 3 && channels != 4) ||
        (sampling != 0x111111 && sampling != 0x121111 && sampling != 0x211111 && sampling != 0x221111 &&
         sampling != 0x411111) ||
        quality < 0 || quality > 100 || (progressive != 0 && progressive != 1))
        return CN_CODEC_FALLBACK; /* values OpenCV warns about */
    struct jpeg_compress_struct info;
    jpeg_failure failure;
    jpeg_output output;
    volatile cn_codec_status status = CN_CODEC_FALLBACK;
    info.err = jpeg_std_error(&failure.manager);
    failure.manager.error_exit = jpeg_fail;
    failure.manager.emit_message = jpeg_message;
    if (setjmp(failure.jump)) {
        jpeg_destroy_compress(&info);
        return status;
    }
    jpeg_create_compress(&info);
    output.manager.init_destination = output_start;
    output.manager.empty_output_buffer = output_flush;
    output.manager.term_destination = output_finish;
    output.out = out;
    info.dest = &output.manager;
    cn_codec_output_reserve(out, (size_t)image->width * image->height * channels / 2 + 4096);
    info.image_width = image->width;
    info.image_height = image->height;
    info.input_components = (int)channels;
    info.in_color_space = channels == 1 ? JCS_GRAYSCALE : channels == 3 ? JCS_EXT_BGR : JCS_EXT_BGRX;
    jpeg_set_defaults(&info);
    info.restart_interval = 0;
    jpeg_set_quality(&info, quality, TRUE);
    if (progressive) jpeg_simple_progression(&info);
    if (channels > 1) {
        info.comp_info[0].h_samp_factor = (int)(sampling >> 20 & 0xF);
        info.comp_info[0].v_samp_factor = (int)(sampling >> 16 & 0xF);
        info.comp_info[1].h_samp_factor = (int)(sampling >> 12 & 0xF);
        info.comp_info[1].v_samp_factor = (int)(sampling >> 8 & 0xF);
        info.comp_info[2].h_samp_factor = (int)(sampling >> 4 & 0xF);
        info.comp_info[2].v_samp_factor = (int)(sampling & 0xF);
    }
    jpeg_start_compress(&info, TRUE);
    while (info.next_scanline < info.image_height) {
        JSAMPROW row = (JSAMPROW)(image->data + (size_t)info.next_scanline * image->stride);
        jpeg_write_scanlines(&info, &row, 1);
    }
    jpeg_finish_compress(&info);
    status = out->failed || failure.manager.num_warnings ? CN_CODEC_FALLBACK : CN_CODEC_OK;
    jpeg_destroy_compress(&info);
    return status;
}
