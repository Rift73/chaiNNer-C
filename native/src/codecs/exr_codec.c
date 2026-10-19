/* OpenEXR images as OpenCV 5.0's ExrDecoder reads them under IMREAD_UNCHANGED (grfmt_exr.cpp), decoded by OpenEXRCore
 * 3.5.2 (GPL-3.0; the channel rules follow OpenCV's Apache-2.0 code). OpenCV reads through OpenEXR's C++ InputFile
 * into float slices; the runtime's OpenCV is built without OpenEXR, so this is Load Image's EXR decoder. */
#include "image_codecs.h"
#include "parallel.h"

#include <openexr.h>

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

/* The file in memory, read by OpenEXRCore's custom stream callbacks. */
typedef struct exr_stream {
    const uint8_t *data;
    size_t size;
} exr_stream;

static int64_t exr_stream_read(exr_const_context_t ctxt, void *userdata, void *buffer, uint64_t size, uint64_t offset,
                               exr_stream_error_func_ptr_t error_cb) {
    (void)ctxt;
    (void)error_cb;
    const exr_stream *stream = (const exr_stream *)userdata;
    if (offset >= stream->size) return 0;
    const uint64_t count = stream->size - offset < size ? stream->size - offset : size;
    memcpy(buffer, stream->data + offset, (size_t)count);
    return (int64_t)count;
}

static int64_t exr_stream_size(exr_const_context_t ctxt, void *userdata) {
    (void)ctxt;
    return (int64_t)((const exr_stream *)userdata)->size;
}

/* OpenEXRCore prints its errors to stderr by default; a failed file declines quietly as every codec here does. */
static void exr_quiet(exr_const_context_t ctxt, exr_result_t code, const char *message) {
    (void)ctxt;
    (void)code;
    (void)message;
}

/* exr_openjph.cpp: OpenJPH's message streams off, once per process. */
void cn_exr_quiet_openjph(void);

#define EXR_MAX_CHANNELS 4

/* What part 0's channels become (readHeader and readData under IMREAD_UNCHANGED): the output's channel count and,
 * per channel of the file's list, its output slot or -1 when OpenCV does not read it. */
typedef struct exr_plan {
    int channels;
    int chroma;     /* Y, RY, BY: slots 1, 2, 0 hold Y, RY, BY until converted to G, R, B */
    int grey_alpha; /* Y with A: Y in slot 0 is copied to 1 and 2 (OpenCV gives two channels) */
    int *slot;
    int filled[EXR_MAX_CHANNELS];
    exr_attr_chromaticities_t chromaticities;
} exr_plan;

static int exr_find(const exr_attr_chlist_t *list, const char *name) {
    for (int i = 0; i < list->num_channels; ++i)
        if (strcmp(list->entries[i].name.str, name) == 0) return i;
    return -1;
}

static void exr_assign(exr_plan *plan, int channel, int slot) {
    if (channel < 0) return;
    plan->slot[channel] = slot;
    plan->filled[slot] = 1;
}

/* readHeader's channel rule: R, G or B (with A) is colour; else channels "0", "1", ... are copied as they are (the
 * multispectral case, taken when chaiNNer can use the count: 1, 3 or 4); else Y (or Z) is luminance, with RY or BY
 * luminance/chroma. Anything else is not an image OpenCV reads. */
static int exr_classify(const exr_attr_chlist_t *list, exr_plan *plan) {
    const int r = exr_find(list, "R"), g = exr_find(list, "G"), b = exr_find(list, "B"), a = exr_find(list, "A");
    for (int i = 0; i < list->num_channels; ++i) plan->slot[i] = -1;
    if (r >= 0 || g >= 0 || b >= 0) {
        plan->channels = a >= 0 ? 4 : 3;
        exr_assign(plan, b, 0);
        exr_assign(plan, g, 1);
        exr_assign(plan, r, 2);
        exr_assign(plan, a, 3);
        return 1;
    }
    if (exr_find(list, "0") >= 0) {
        const int count = list->num_channels;
        if (count != 1 && count != 3 && count != 4) return 0;
        for (int i = 0; i < count; ++i) {
            char name[2] = {(char)('0' + i), 0};
            const int channel = exr_find(list, name);
            if (channel < 0) return 0;
            exr_assign(plan, channel, i);
        }
        plan->channels = count;
        return 1;
    }
    int y = exr_find(list, "Y");
    if (y < 0) y = exr_find(list, "Z");
    if (y < 0) return 0;
    const int ry = exr_find(list, "RY"), by = exr_find(list, "BY");
    if (ry >= 0 || by >= 0) {
        plan->channels = a >= 0 ? 4 : 3;
        plan->chroma = 1;
        exr_assign(plan, by, 0);
        exr_assign(plan, y, 1);
        exr_assign(plan, ry, 2);
        exr_assign(plan, a, 3);
    } else if (a >= 0) {
        plan->channels = 4;
        plan->grey_alpha = 1;
        exr_assign(plan, y, 0);
        exr_assign(plan, a, 3);
    } else {
        plan->channels = 1;
        exr_assign(plan, y, 0);
    }
    return 1;
}

/* One decode of part 0's level-(0, 0) chunks: whole-resolution channels go straight into the output's slots,
 * subsampled ones (scanline images only) into a plane each, widened afterwards. */
typedef struct exr_job {
    exr_const_context_t ctxt;
    const exr_attr_chlist_t *list;
    const exr_plan *plan;
    float *out;
    float **planes;
    int64_t min_x, min_y;
    uint32_t width, height;
    int tiled;
    int32_t lines_per_chunk, tiles_x;
    uint32_t tile_width, tile_height;
    atomic_int failed;
} exr_job;

/* Floor division, for data-window coordinates below zero. */
static int64_t exr_floor_div(int64_t a, int64_t b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

static void exr_decode_chunks(void *context, size_t begin, size_t end) {
    exr_job *job = (exr_job *)context;
    const uint32_t channels = (uint32_t)job->plan->channels;
    exr_decode_pipeline_t decoder;
    memset(&decoder, 0, sizeof decoder); /* EXR_DECODE_PIPELINE_INITIALIZER's zeros */
    decoder.pipe_size = sizeof decoder;
    int started = 0;
    for (size_t chunk = begin; chunk < end && !atomic_load(&job->failed); ++chunk) {
        exr_chunk_info_t info;
        size_t x0 = 0, y0;
        exr_result_t result;
        if (job->tiled) {
            const int32_t tx = (int32_t)(chunk % (size_t)job->tiles_x), ty = (int32_t)(chunk / (size_t)job->tiles_x);
            result = exr_read_tile_chunk_info(job->ctxt, 0, tx, ty, 0, 0, &info);
            x0 = (size_t)tx * job->tile_width;
            y0 = (size_t)ty * job->tile_height;
        } else {
            const int64_t y = job->min_y + (int64_t)chunk * job->lines_per_chunk;
            result = exr_read_scanline_chunk_info(job->ctxt, 0, (int)y, &info);
            y0 = (size_t)(y - job->min_y);
        }
        if (result == EXR_ERR_SUCCESS)
            result = started ? exr_decoding_update(job->ctxt, 0, &info, &decoder)
                             : exr_decoding_initialize(job->ctxt, 0, &info, &decoder);
        if (result == EXR_ERR_SUCCESS && (x0 + (size_t)info.width > job->width || y0 + (size_t)info.height > job->height))
            result = EXR_ERR_CORRUPT_CHUNK;
        if (result != EXR_ERR_SUCCESS) {
            atomic_store(&job->failed, 1);
            break;
        }
        for (int c = 0; c < decoder.channel_count; ++c) {
            exr_coding_channel_info_t *channel = &decoder.channels[c];
            const int slot = job->plan->slot[c];
            if (slot < 0) {
                channel->decode_to_ptr = NULL;
                continue;
            }
            channel->user_data_type = EXR_PIXEL_FLOAT;
            channel->user_bytes_per_element = 4;
            const exr_attr_chlist_entry_t *entry = &job->list->entries[c];
            if (job->planes[slot]) {
                const size_t plane_width = (job->width + (uint32_t)entry->x_sampling - 1) / (uint32_t)entry->x_sampling;
                /* generic_unpack writes sampled line y at (y / ys - start_y / ys) lines past the pointer. */
                const int64_t row = info.start_y / entry->y_sampling - job->min_y / entry->y_sampling;
                channel->decode_to_ptr = (uint8_t *)(job->planes[slot] + (size_t)row * plane_width);
                channel->user_pixel_stride = 4;
                channel->user_line_stride = (int32_t)(plane_width * 4);
            } else {
                channel->decode_to_ptr = (uint8_t *)(job->out + ((y0 * job->width + x0) * channels + (size_t)slot));
                channel->user_pixel_stride = (int32_t)(channels * 4);
                channel->user_line_stride = (int32_t)((size_t)job->width * channels * 4);
            }
        }
        if (!started) {
            result = exr_decoding_choose_default_routines(job->ctxt, 0, &decoder);
            started = 1;
        }
        if (result == EXR_ERR_SUCCESS) result = exr_decoding_run(job->ctxt, 0, &decoder);
        if (result != EXR_ERR_SUCCESS) {
            atomic_store(&job->failed, 1);
            break;
        }
    }
    exr_decoding_destroy(job->ctxt, &decoder);
}

/* OpenEXR's luminance weights for its chromaticities (ImfRgbaYca.cpp computeYw: row Y of RGBtoXYZ(cr, 1), normalised).
 * OpenCV's ChromaToBGR uses the primaries' y coordinates instead, which tints grey (0.33 + 0.60 + 0.06 = 0.99, and
 * green is weighted 0.60, not 0.7152, for Rec. 709); OpenEXR's own reconstruction is followed here. */
static void exr_luminance_weights(const exr_attr_chromaticities_t *c, float yw[3]) {
    const double rx = c->red_x, ry = c->red_y, gx = c->green_x, gy = c->green_y, bx = c->blue_x, by = c->blue_y;
    const double wx = c->white_x, wy = c->white_y;
    const double X = wx / wy, Y = 1.0, Z = (1.0 - wx - wy) / wy;
    const double d = rx * (by - gy) + bx * (gy - ry) + gx * (ry - by);
    const double sr = (X * (by - gy) - gx * (Y * (by - 1) + by * (X + Z)) + bx * (Y * (gy - 1) + gy * (X + Z))) / d;
    const double sg = (X * (ry - by) + rx * (Y * (by - 1) + by * (X + Z)) - bx * (Y * (ry - 1) + ry * (X + Z))) / d;
    const double sb = (X * (gy - ry) - rx * (Y * (gy - 1) + gy * (X + Z)) + gx * (Y * (ry - 1) + ry * (X + Z))) / d;
    const double r = sr * ry, g = sg * gy, b = sb * by, sum = r + g + b;
    yw[0] = (float)(r / sum);
    yw[1] = (float)(g / sum);
    yw[2] = (float)(b / sum);
}

/* After the decode: subsampled channels widened by repetition (UpSample), missing colour channels 0 (OpenCV's slice
 * fill value), luminance/chroma to BGR (ChromaToBGR's arithmetic), grey copied under its alpha. */
static void exr_complete(const exr_job *job, const exr_attr_chlist_t *list) {
    const exr_plan *plan = job->plan;
    const size_t channels = (size_t)plan->channels, width = job->width, height = job->height;
    float *out = job->out;
    for (int c = 0; c < list->num_channels; ++c) {
        const int slot = plan->slot[c];
        if (slot < 0 || !job->planes[slot]) continue;
        const size_t xs = (size_t)list->entries[c].x_sampling, ys = (size_t)list->entries[c].y_sampling;
        const size_t plane_width = (width + xs - 1) / xs;
        for (size_t y = 0; y < height; ++y)
            for (size_t x = 0; x < width; ++x)
                out[(y * width + x) * channels + (size_t)slot] = job->planes[slot][(y / ys) * plane_width + x / xs];
    }
    /* Slots no channel fills; under a grey alpha, 1 and 2 are copies of 0. */
    int zero[EXR_MAX_CHANNELS] = {0};
    int any_zero = 0;
    for (size_t s = 0; s < channels; ++s)
        any_zero |= zero[s] = !plan->filled[s] && !(plan->grey_alpha && (s == 1 || s == 2));
    if (any_zero)
        for (size_t i = 0; i < width * height; ++i)
            for (size_t s = 0; s < channels; ++s)
                if (zero[s]) out[i * channels + s] = 0.0f;
    if (plan->chroma) {
        float yw[3];
        exr_luminance_weights(&plan->chromaticities, yw);
        for (size_t i = 0; i < width * height; ++i) {
            float *p = out + i * channels;
            const double luminance = p[1], r = ((double)p[2] + 1) * luminance, b = ((double)p[0] + 1) * luminance;
            p[0] = (float)b;
            p[1] = (float)((luminance - b * yw[2] - r * yw[0]) / yw[1]);
            p[2] = (float)r;
        }
    }
    if (plan->grey_alpha)
        for (size_t i = 0; i < width * height; ++i) out[i * 4 + 1] = out[i * 4 + 2] = out[i * 4];
}

int cn_exr_signature(const uint8_t *data, size_t size) {
    return size >= 4 && data[0] == 0x76 && data[1] == 0x2F && data[2] == 0x31 && data[3] == 0x01;
}

cn_codec_status cn_exr_decode(const uint8_t *data, size_t size, cn_codec_allocate allocate, void *context) {
    if (!cn_exr_signature(data, size)) return CN_CODEC_FALLBACK;
    cn_exr_quiet_openjph();
    exr_stream stream = {data, size};
    exr_context_initializer_t init = EXR_DEFAULT_CONTEXT_INITIALIZER;
    init.user_data = &stream;
    init.read_fn = exr_stream_read;
    init.size_fn = exr_stream_size;
    init.error_handler_fn = exr_quiet;
    exr_context_t ctxt = NULL;
    if (exr_start_read(&ctxt, "", &init) != EXR_ERR_SUCCESS) {
        exr_finish(&ctxt);
        return CN_CODEC_FALLBACK;
    }
    cn_codec_status status = CN_CODEC_FALLBACK;
    int *slots = NULL;
    float *planes[EXR_MAX_CHANNELS] = {NULL};
    exr_storage_t storage;
    exr_attr_box2i_t window;
    const exr_attr_chlist_t *list = NULL;
    int parts = 0;
    /* InputFile reads part 0 of a multipart file; deep parts are not images it reads. */
    if (exr_get_count(ctxt, &parts) != EXR_ERR_SUCCESS || parts < 1 ||
        exr_get_storage(ctxt, 0, &storage) != EXR_ERR_SUCCESS ||
        (storage != EXR_STORAGE_SCANLINE && storage != EXR_STORAGE_TILED) ||
        exr_get_data_window(ctxt, 0, &window) != EXR_ERR_SUCCESS || exr_get_channels(ctxt, 0, &list) != EXR_ERR_SUCCESS ||
        !list || list->num_channels < 1)
        goto done;
    /* The data window is the image (m_datawindow); the display window is not used. */
    const int64_t width = (int64_t)window.max.x - window.min.x + 1, height = (int64_t)window.max.y - window.min.y + 1;
    if (width < 1 || height < 1 || (uint64_t)width * (uint64_t)height > CN_MAX_IMAGE_PIXELS) goto done;
    exr_plan plan = {0};
    slots = (int *)malloc(sizeof(int) * (size_t)list->num_channels);
    if (!slots) goto done;
    plan.slot = slots;
    /* OpenEXRCore's line strides are 32-bit. */
    if (!exr_classify(list, &plan) || width * plan.channels * 4 > INT32_MAX) goto done;
    /* Rec. 709, OpenEXR's default, when the header has no chromaticities. */
    if (exr_attr_get_chromaticities(ctxt, 0, "chromaticities", &plan.chromaticities) != EXR_ERR_SUCCESS)
        plan.chromaticities = (exr_attr_chromaticities_t){0.64f, 0.33f, 0.30f, 0.60f, 0.15f, 0.06f, 0.3127f, 0.3290f};
    exr_job job = {.ctxt = ctxt, .list = list, .plan = &plan, .planes = planes, .min_x = window.min.x,
                   .min_y = window.min.y, .width = (uint32_t)width, .height = (uint32_t)height,
                   .tiled = storage == EXR_STORAGE_TILED};
    for (int c = 0; c < list->num_channels; ++c) {
        const exr_attr_chlist_entry_t *entry = &list->entries[c];
        if (plan.slot[c] < 0 || (entry->x_sampling == 1 && entry->y_sampling == 1)) continue;
        /* Subsampled samples sit where x % xs == 0 and y % ys == 0; the window must start on one. */
        if (job.tiled || entry->x_sampling < 1 || entry->y_sampling < 1 ||
            exr_floor_div(window.min.x, entry->x_sampling) * entry->x_sampling != window.min.x ||
            exr_floor_div(window.min.y, entry->y_sampling) * entry->y_sampling != window.min.y)
            goto done;
        const size_t plane = ((size_t)width + (size_t)entry->x_sampling - 1) / (size_t)entry->x_sampling *
                             (((size_t)height + (size_t)entry->y_sampling - 1) / (size_t)entry->y_sampling);
        planes[plan.slot[c]] = (float *)malloc(plane * sizeof(float));
        if (!planes[plan.slot[c]]) goto done;
    }
    size_t chunks;
    if (job.tiled) {
        exr_tile_level_mode_t level;
        exr_tile_round_mode_t round;
        int32_t tiles_y = 0;
        if (exr_get_tile_descriptor(ctxt, 0, &job.tile_width, &job.tile_height, &level, &round) != EXR_ERR_SUCCESS ||
            exr_get_tile_counts(ctxt, 0, 0, 0, &job.tiles_x, &tiles_y) != EXR_ERR_SUCCESS || job.tiles_x < 1 ||
            tiles_y < 1 || !job.tile_width || !job.tile_height)
            goto done;
        chunks = (size_t)job.tiles_x * (size_t)tiles_y;
    } else {
        if (exr_get_scanlines_per_chunk(ctxt, 0, &job.lines_per_chunk) != EXR_ERR_SUCCESS || job.lines_per_chunk < 1)
            goto done;
        chunks = ((size_t)height + (size_t)job.lines_per_chunk - 1) / (size_t)job.lines_per_chunk;
    }
    job.out = (float *)allocate(context, (uint32_t)height, (uint32_t)width, (uint32_t)plan.channels, CN_SAMPLE_F32);
    if (!job.out) goto done;
    atomic_init(&job.failed, 0);
    /* Chunks decode in parallel, each range with its own pipeline over the shared read context. */
    if (cn_parallel_for(chunks, 1, exr_decode_chunks, &job) != CN_OK || atomic_load(&job.failed)) goto done;
    exr_complete(&job, list);
    status = CN_CODEC_OK;
done:
    for (int s = 0; s < EXR_MAX_CHANNELS; ++s) free(planes[s]);
    free(slots);
    exr_finish(&ctxt);
    return status;
}
