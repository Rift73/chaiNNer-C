/* Bulk pixel boundaries and tiled-upscaler arithmetic. No model/GPU calls. */
#include "convert_shared.h"
#include "isa.h"
#include "parallel.h"
#include "run_kernels.h"
#include <fenv.h>
#include <math.h>
#include <string.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif

/* Whether the caller's rounding mode, which the pool's helpers take, is round to
 * nearest (MXCSR RC 00 on x86). */
static int rounds_to_nearest(void) {
#ifdef CN_CONVERT_SSE
    return (_mm_getcsr() & 0x6000u) == 0;
#else
    return fegetround() == FE_TONEAREST;
#endif
}

/* ORs a chunk's event bits into its call's. */
static void merge_events(volatile long *target, long events) {
    if (events) {
#ifdef _MSC_VER
        _InterlockedOr(target, events);
#else
        __atomic_fetch_or(target, events, __ATOMIC_RELAXED);
#endif
    }
}

/* One loop per chunk (convert_shared.h). NumPy 1.24.4's pinned MSVC float->u8/u16
 * cast goes through int32; scaled_integer defines its overflow sentinel. */
static void convert_range(void *opaque, size_t begin, size_t end) {
    convert_job *j = opaque;
    long events;
#if defined(_MSC_VER) && defined(_M_X64)
    if (j->isa >= CN_ISA_AVX2 && ((j->type == 2 && j->output == 0) || j->type == 0))
        events = cn_convert_avx2(j, begin, end);
    else
#endif
        events = convert_scalar(j, begin, end);
    merge_events(&j->events, events);
}

typedef struct {
    const float *src;
    uint16_t *out;
    size_t pixels;
    cn_isa_level isa;
    volatile long events;
} gbr_planes_job;

static void gbr_planes_range(void *opaque, size_t begin, size_t end) {
    gbr_planes_job *j = opaque;
    long events;
#if defined(_MSC_VER) && defined(_M_X64)
    if (j->isa >= CN_ISA_AVX2)
        events = cn_gbr_planes_avx2(j->src, j->out, j->pixels, begin, end);
    else
#endif
        events = gbr_planes_scalar(j->src, j->out, j->pixels, begin, end);
    merge_events(&j->events, events);
}

/* Save Video's and Frames to Clip's RGB frames (image_utils.to_uint16_gbr_planes):
 * a C-contiguous BGR float32 image of `pixels` pixels to its G, B and R planes in
 * uint16, plane after plane in out (3 x pixels), each element and the event bits as
 * cn_pixels_convert_checked's (type 0, output 2) give them, so the planes equal its
 * output rearranged; one pass instead of the conversion and a gather. */
CN_EXPORT cn_status cn_pixels_gbr_planes_u16(const float *src, uint16_t *out, size_t pixels,
                                            int *events) {
    if (!src || !out) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / (3 * sizeof(float))) return CN_SIZE_OVERFLOW;
    gbr_planes_job job = {src, out, pixels, cn_isa_current(), 0};
    /* cn_pixels_convert_checked's grain of 262,144 elements, in pixels of three. */
    cn_status status = cn_parallel_for(pixels, 87382, gbr_planes_range, &job);
    if (events) *events = (int)job.events;
    return status;
}

/* src and out may alias exactly (src == out) for (type 0, output 0, normalize 1)
 * only: the output enforce's clip run in place (native_buffers.freeze_normalized
 * with clamp). Every path reads each element before its own store and never after
 * (scalar loop, 8-lane groups, disjoint chunks), so neither pointer is restrict,
 * here or in convert_shared.h and convert_avx2.c. Any other overlap is unsupported
 * (at normalize 0 the scalar float32 copy is a memcpy). */
CN_EXPORT cn_status cn_pixels_convert_checked(const void *src, void *out, size_t count,
                                     int type, int output, int normalize, int *events) {
    if (!src || !out || type < 0 || type > 10 || output < 0 || output > 2 ||
        normalize < 0 || normalize > 1) return CN_INVALID_ARGUMENT;
    if (count > SIZE_MAX / sizeof(double)) return CN_SIZE_OVERFLOW;
    convert_job job = {src, out, type, output, normalize, cn_isa_current(),
                       rounds_to_nearest(), 0};
    cn_status status = cn_parallel_for(count, 262144, convert_range, &job);
    if (events) *events = (int)job.events;
    return status;
}
CN_EXPORT cn_status cn_pixels_convert(const void *src, void *out, size_t count,
                                     int type, int output, int normalize) {
    return cn_pixels_convert_checked(src, out, count, type, output, normalize, NULL);
}

/* A borrow is safe only if clipping would leave every bit unchanged. Inspect
 * integer representations so negative zero and all nonfinite values take the
 * existing conversion path (including its floating-point exception behavior).
 * At ISA level avx2 or above (read once) cn_pixels_normalized_avx2 applies the same
 * bit tests in 8-lane groups. */
CN_EXPORT cn_status cn_pixels_normalized_f32(const float *src, size_t count, int *same) {
    if (!src || !same) return CN_INVALID_ARGUMENT;
    if (count > SIZE_MAX / sizeof(float)) return CN_SIZE_OVERFLOW;
#if defined(_MSC_VER) && defined(_M_X64)
    if (cn_isa_current() >= CN_ISA_AVX2) {
        *same = cn_pixels_normalized_avx2(src, count);
        return CN_OK;
    }
#endif
    *same = 0;
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits;
        memcpy(&bits, src + i, sizeof(bits));
        if (bits > UINT32_C(0x3f800000) ||
            (bits != 0 && bits < UINT32_C(0x00800000))) return CN_OK;
    }
    *same = 1;
    return CN_OK;
}

typedef struct {
    const float *a, *b, *weights;
    float *out;
    size_t width, channels, a_row, b_row;
    int axis;
} mix_job;

/* Output elements [begin, end), row by row: the row, pixel and channel of the first
 * element are divided out once, then counted (no division per element). Each element
 * is a * (1 - w) + b * w as b * w + a - a * w, three roundings in that order. */
static void mix_range(void *opaque, size_t begin, size_t end) {
    mix_job *j = opaque;
    const size_t row_width = j->width * j->channels;
    size_t y = begin / row_width, x = begin % row_width;
    for (size_t i = begin; i < end; ++y, x = 0) {
        const float *a = j->a + y * j->a_row, *b = j->b + y * j->b_row;
        const size_t stop = end - i < row_width - x ? x + (end - i) : row_width;
        size_t pixel = x / j->channels, channel = x % j->channels;
        for (; x < stop; ++x, ++i) {
            float weight = j->weights[j->axis == 0 ? i : j->axis == 1 ? pixel : y];
            float value = b[x] * weight;
            value += a[x];
            float subtract = a[x] * weight;
            j->out[i] = value - subtract;
            if (++channel == j->channels) {
                channel = 0;
                ++pixel;
            }
        }
    }
}

CN_EXPORT cn_status cn_tile_mix(const float *a, const float *b, const float *weights,
                               float *out, size_t height, size_t width,
                               size_t channels, int axis, size_t a_row, size_t b_row) {
    if (!a || !b || !weights || !out || !width || !channels || axis < 0 || axis > 2)
        return CN_INVALID_ARGUMENT;
    if (height > SIZE_MAX / width || height * width > SIZE_MAX / channels / sizeof(float))
        return CN_SIZE_OVERFLOW;
    if (width > SIZE_MAX / channels / sizeof(float)) return CN_SIZE_OVERFLOW;
    size_t row_width = width * channels;
    if (a_row < row_width || b_row < row_width) return CN_INVALID_ARGUMENT;
    if (a_row > SIZE_MAX / sizeof(float) || b_row > SIZE_MAX / sizeof(float) ||
        (height && (height - 1 > (SIZE_MAX / sizeof(float) - row_width) / a_row ||
                    height - 1 > (SIZE_MAX / sizeof(float) - row_width) / b_row)))
        return CN_SIZE_OVERFLOW;
    mix_job job = {a, b, weights, out, width, channels, a_row, b_row, axis};
    return cn_parallel_for(height * width * channels, 65536, mix_range, &job);
}

typedef struct {
    const float *src;
    float *out, *white;
    size_t channels;
    int background;
} alpha_job;

/* denoise_and_flatten_alpha of one pixel: its channels' min, max and mean, then
 * max * mean + min * (1 - mean), clipped to [0, 1] (NaN passes). A NaN channel makes
 * max the last NaN channel, and the result is that NaN, quieted, as this kernel has
 * always given it; it is set explicitly because which NaN an arithmetic operation
 * keeps follows the compiler's operand order, which differs between callers. */
static float flattened_alpha(const float *pixel, size_t channels) {
    float low = pixel[0], high = pixel[0], sum = -0.0f;
    for (size_t c = 0; c < channels; ++c) {
        if (pixel[c] < low || isnan(pixel[c])) low = pixel[c];
        if (pixel[c] > high || isnan(pixel[c])) high = pixel[c];
        sum += pixel[c];
    }
    float mean = sum / (float)channels;
    float part = low * (1.0f - mean);
    float result = high * mean + part;
    if (isnan(high)) {
        uint32_t bits;
        memcpy(&bits, &high, sizeof(bits));
        bits |= UINT32_C(0x00400000);
        memcpy(&result, &bits, sizeof(result));
    }
    return result <= 0 ? 0 : result >= 1 ? 1 : result;
}

static void alpha_range(void *opaque, size_t begin, size_t end) {
    alpha_job *j = opaque;
    for (size_t p = begin; p < end; ++p) {
        const float *pixel = j->src + p * j->channels;
        if (j->background) {
            for (size_t c = 0; c < 3; ++c) {
                j->out[p * 3 + c] = pixel[c] * pixel[3];
                float white = (pixel[c] - 1.0f) * pixel[3];
                j->white[p * 3 + c] = white + 1.0f;
            }
        } else {
            j->out[p] = flattened_alpha(pixel, j->channels);
        }
    }
}

typedef struct {
    const float *white, *black;
    float *out;
    volatile long events[2];
} recovered_job;

#define CN_RECOVER_BLOCK 256

/* FP conditions raised since the last call, cleared, in cn_framework_shared's event
 * bits (1 divide by zero, 2 overflow, 4 underflow, 8 invalid). */
static long raised_events(void) {
    int state = fetestexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INVALID);
    feclearexcept(FE_ALL_EXCEPT);
    return ((state & FE_DIVBYZERO) ? 1 : 0) | ((state & FE_OVERFLOW) ? 2 : 0) |
           ((state & FE_UNDERFLOW) ? 4 : 0) | ((state & FE_INVALID) ? 8 : 0);
}

/* Pixels [begin, end) in blocks: the candidates 1 - (white - black) (the difference,
 * then the subtraction from 1, each stage's conditions tested apart as NumPy's two
 * ufunc calls), then the pixel's black colour and its flattened candidates' alpha.
 * The flatten reports nothing, as flatten_alpha does. */
static void recovered_range(void *opaque, size_t begin, size_t end) {
    recovered_job *j = opaque;
    float candidates[CN_RECOVER_BLOCK * 3];
    long events[2] = {0, 0};
    fenv_t previous;
    fegetenv(&previous);
    feclearexcept(FE_ALL_EXCEPT);
    for (size_t first = begin; first < end; first += CN_RECOVER_BLOCK) {
        size_t count = end - first < CN_RECOVER_BLOCK ? end - first : CN_RECOVER_BLOCK;
        const float *white = j->white + first * 3, *black = j->black + first * 3;
        for (size_t k = 0; k < count * 3; ++k) candidates[k] = white[k] - black[k];
        events[0] |= raised_events();
        for (size_t k = 0; k < count * 3; ++k) candidates[k] = 1.0f - candidates[k];
        events[1] |= raised_events();
        for (size_t p = 0; p < count; ++p) {
            float *pixel = j->out + (first + p) * 4;
            memcpy(pixel, black + p * 3, 3 * sizeof(float));
            pixel[3] = flattened_alpha(candidates + p * 3, 3);
        }
        feclearexcept(FE_ALL_EXCEPT);
    }
    fesetenv(&previous);
    for (size_t stage = 0; stage < 2; ++stage) {
        if (!events[stage]) continue;
#ifdef _MSC_VER
        _InterlockedOr(&j->events[stage], events[stage]);
#else
        __atomic_fetch_or(&j->events[stage], events[stage], __ATOMIC_RELAXED);
#endif
    }
}

/* convenient_upscale's transparency path for (pixels, 3) white and black upscales:
 * out (pixels, 4) is the black colour with the alpha that recover_alpha, then
 * denoise_and_flatten_alpha give, in one pass; events[0..1]: the two subtractions'. */
CN_EXPORT cn_status cn_upscale_alpha_recover(const float *white, const float *black,
                                            float *out, size_t pixels, int *events) {
    if (!white || !black || !out || !events) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / 4 / sizeof(float)) return CN_SIZE_OVERFLOW;
    recovered_job job = {white, black, out, {0, 0}};
    cn_status status = cn_parallel_for(pixels, 16384, recovered_range, &job);
    events[0] = (int)job.events[0];
    events[1] = (int)job.events[1];
    return status;
}

CN_EXPORT cn_status cn_upscale_alpha(const float *src, float *out, float *white,
                                    size_t pixels, size_t channels, int background) {
    if (!src || !out || !channels || channels > 7 || background < 0 || background > 1 ||
        (background && (channels != 4 || !white))) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / channels / sizeof(float)) return CN_SIZE_OVERFLOW;
    alpha_job job = {src, out, white, channels, background};
    return cn_parallel_for(pixels, 65536, alpha_range, &job);
}

CN_EXPORT cn_status cn_constant_alpha(const float *src, size_t pixels, int *constant) {
    if (!src || !constant || !pixels) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / 4 / sizeof(float)) return CN_SIZE_OVERFLOW;
    float first = src[3];
    *constant = 1;
    for (size_t p = 1; p < pixels; ++p) {
        float value = src[p * 4 + 3];
        if (value != first && !(isnan(value) && isnan(first))) { *constant = 0; break; }
    }
    return CN_OK;
}

typedef struct {
    const float *image;
    const uint8_t *caption;
    float *out;
    size_t width, image_height, caption_height, channels;
    int top;
} caption_job;

/* Output pixels [begin, end). The caption's rows and the image's rows are each one run
 * of pixels (the caption's first when top), so the chunk's image pixels are one copy and
 * its caption pixels read the raster in order. */
static void caption_range(void *opaque, size_t begin, size_t end) {
    caption_job *j = opaque;
    size_t channels = j->channels;
    size_t caption_pixels = j->caption_height * j->width;
    size_t image_pixels = j->image_height * j->width;
    size_t caption_begin = j->top ? 0 : image_pixels;
    size_t image_begin = j->top ? caption_pixels : 0;
    size_t first = begin > image_begin ? begin : image_begin;
    size_t last = end < image_begin + image_pixels ? end : image_begin + image_pixels;
    if (first < last)
        memcpy(j->out + first * channels, j->image + (first - image_begin) * channels,
               (last - first) * channels * sizeof(float));
    first = begin > caption_begin ? begin : caption_begin;
    last = end < caption_begin + caption_pixels ? end : caption_begin + caption_pixels;
    for (size_t p = first; p < last; ++p) {
        float value = (float)j->caption[p - caption_begin] / 255.0f;
        float *target = j->out + p * channels;
        for (size_t c = 0; c < channels; ++c) target[c] = c == 3 ? 1.0f : value;
    }
}

CN_EXPORT cn_status cn_caption_compose(const float *image, const uint8_t *caption,
                                      float *out, size_t height, size_t width,
                                      size_t channels, size_t caption_height, int top) {
    if (!image || !caption || !out || !height || !width || !caption_height ||
        (channels != 1 && channels != 3 && channels != 4) || top < 0 || top > 1)
        return CN_INVALID_ARGUMENT;
    if (height > SIZE_MAX - caption_height ||
        width > SIZE_MAX / channels / sizeof(float) ||
        height + caption_height > SIZE_MAX / sizeof(float) / channels / width)
        return CN_SIZE_OVERFLOW;
    caption_job job = {image, caption, out, width, height, caption_height, channels, top};
    return cn_parallel_for((height + caption_height) * width, 16384, caption_range, &job);
}

typedef struct { const uint8_t *mask; float *out; float ink[3]; } text_job;
static void text_range(void *opaque, size_t begin, size_t end) {
    text_job *j = opaque;
    for (size_t p = begin; p < end; ++p) {
        uint8_t coverage = j->mask[p];
        for (size_t c = 0; c < 3; ++c) j->out[p * 4 + c] = coverage ? j->ink[c] : 0;
        j->out[p * 4 + 3] = (float)coverage / 255.0f;
    }
}
CN_EXPORT cn_status cn_text_raster(const uint8_t *mask, const uint8_t *ink,
                                  float *out, size_t pixels) {
    if (!mask || !ink || !out) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / 4 / sizeof(float)) return CN_SIZE_OVERFLOW;
    text_job job = {mask, out, {(float)ink[0] / 255.0f,
                              (float)ink[1] / 255.0f, (float)ink[2] / 255.0f}};
    return cn_parallel_for(pixels, 16384, text_range, &job);
}
