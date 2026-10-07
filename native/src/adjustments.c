#include "chainner.h"
#include "cn_crt_math.h"
#include "parallel.h"
#include "numpy_clip_f32.h"
#include <math.h>

/* Readable and writable extents must be disjoint or exactly in-place. A
 * shifted alias would otherwise race across independent pixel work ranges. */
static cn_status checked_buffers(const float *src, size_t source_count,
        float *dst, size_t output_count) {
    uintptr_t a = (uintptr_t)src, b = (uintptr_t)dst;
    if (!src || !dst || a % _Alignof(float) || b % _Alignof(float))
        return CN_INVALID_ARGUMENT;
    if (source_count > SIZE_MAX / sizeof(float) || output_count > SIZE_MAX / sizeof(float))
        return CN_SIZE_OVERFLOW;
    size_t source_bytes = source_count * sizeof(float), output_bytes = output_count * sizeof(float);
    if (a > UINTPTR_MAX - source_bytes || b > UINTPTR_MAX - output_bytes)
        return CN_SIZE_OVERFLOW;
    if (a < b + output_bytes && b < a + source_bytes
        && (a != b || source_bytes != output_bytes)) return CN_INVALID_ARGUMENT;
    return CN_OK;
}

/* The channel-aware kernels run flat over gapless memory in any axis order (D-16:
 * np.clip's order-K layout). With channel stride s in elements, the memory holds
 * runs of s elements of one channel that cycle through the channels: element i is
 * channel (i / s) % channels, and its pixel's channel k sits at i + (k - c) * s.
 * Interleaved memory has s = 1; a planar (c, h, w) base has s = h * w. Every run
 * is whole, so the pixel count is a multiple of s. */
typedef struct adjustment_job {
    const float *src;
    float *dst;
    size_t channels, stride;
    int op;
    float a, b;
    int should_clip;
} adjustment_job;

/* Element i of channel c becomes x: parameters have already undergone the
   original Python scalar arithmetic, and individual float operations intentionally
   match NumPy's float32 stages. ALPHA is the same pixel's channel 3, read only by
   ops 5 and 6. A statement macro, so each loop below compiles as written: an
   inline function costs the interleaved clamp about 7 %. */
#define ADJUST_ELEMENT(job, x, c, a, b, ALPHA)                                    \
    switch ((job)->op) {                                                        \
    case 0: x += a; break;                                                      \
    case 1: x *= a; break;                                                      \
    case 2: if (c < 3) x = 1.0f - x; break;                                     \
    case 3:                                                                     \
        if (c < 3) {                                                            \
            if (a != 1.0f) x *= a;                                              \
            x += b;                                                             \
            if ((job)->should_clip) x = cn_numpy_clip_f32(x, 0.0f, 1.0f);       \
        }                                                                       \
        break;                                                                  \
    case 4: x = cn_numpy_clip_f32(x, a, b); break;                              \
    case 5: if (c < 3) x *= (ALPHA); break;                                     \
    case 6: if (c < 3) x /= (ALPHA); break;                                     \
    case 7: x = (x - a) / b; break;                                             \
    }

static void adjustment_range(void *context, size_t begin, size_t end) {
    adjustment_job *job = (adjustment_job *)context;
    const float *src = job->src;
    float *dst = job->dst;
    size_t channels = job->channels;
    float a = job->a, b = job->b;
    for (size_t i = begin; i < end; ++i) {
        float x = src[i];
        size_t c = i % channels;
        ADJUST_ELEMENT(job, x, c, a, b, src[i - c + 3])
        dst[i] = x;
    }
}

/* Channel stride above 1: one channel per run, the alpha run 3 - c runs ahead. */
static void adjustment_runs(void *context, size_t begin, size_t end) {
    adjustment_job *job = (adjustment_job *)context;
    const float *src = job->src;
    float *dst = job->dst;
    size_t channels = job->channels, stride = job->stride;
    float a = job->a, b = job->b;
    for (size_t i = begin; i < end;) {
        size_t run = i / stride, c = run % channels;
        size_t stop = (run + 1) * stride < end ? (run + 1) * stride : end;
        for (; i < stop; ++i) {
            float x = src[i];
            ADJUST_ELEMENT(job, x, c, a, b, src[i + (3 - c) * stride])
            dst[i] = x;
        }
    }
}

#undef ADJUST_ELEMENT

CN_EXPORT cn_status cn_adjust_f32(const float *src, float *dst, size_t pixels,
                                size_t channels, size_t channel_stride, int op,
                                float a, float b, int should_clip) {
    if (!src || !dst || !channels || op < 0 || op > 7 || (should_clip != 0 && should_clip != 1))
        return CN_INVALID_ARGUMENT;
    if ((op == 5 || op == 6) && channels != 4) return CN_INVALID_ARGUMENT;
    if (!channel_stride || pixels % channel_stride) return CN_INVALID_ARGUMENT;
    if (channels > SIZE_MAX / sizeof(float) ||
        pixels > SIZE_MAX / sizeof(float) / channels) return CN_SIZE_OVERFLOW;
    cn_status status = checked_buffers(src, pixels * channels, dst, pixels * channels);
    if (status != CN_OK) return status;
    adjustment_job job = {src, dst, channels, channel_stride, op, a, b, should_clip};
    return cn_parallel_for(pixels * channels, 65536,
        channel_stride == 1 ? adjustment_range : adjustment_runs, &job);
}

static void opacity_range(void *context, size_t begin, size_t end) {
    adjustment_job *job = (adjustment_job *)context;
    const float *src = job->src;
    float *dst = job->dst;
    size_t channels = job->channels;
    for (size_t p = begin; p < end; ++p) {
        for (size_t c = 0; c < 3; ++c)
            dst[p * 4 + c] = src[p * channels + (channels == 1 ? 0 : c)];
        dst[p * 4 + 3] = (channels == 4 ? src[p * 4 + 3] : 1.0f) * job->a;
    }
}

CN_EXPORT cn_status cn_opacity_f32(const float *src, float *dst, size_t pixels,
                                 size_t channels, float opacity) {
    if (!src || !dst || (channels != 1 && channels != 3 && channels != 4))
        return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / sizeof(float) / 4) return CN_SIZE_OVERFLOW;
    cn_status status = checked_buffers(src, pixels * channels, dst, pixels * 4);
    if (status != CN_OK) return status;
    adjustment_job job = {.src = src, .dst = dst, .channels = channels, .a = opacity};
    return cn_parallel_for(pixels, 65536, opacity_range, &job);
}

typedef struct levels_job {
    const float *src;
    float *dst;
    size_t channels, stride;
    unsigned int mask;
    float black, span, exponent, out_black, out_span;
} levels_job;

/* mask uses image memory order B,G,R,A. Grayscale always processes its value.
 * Upstream runs an unselected channel through identity levels (black 0, white 1,
 * gamma 1, output 0 to 1), the same float32 steps: its final + 0 turns -0 into +0. */
static inline float leveled(const levels_job *job, float x, size_t c) {
    if (job->channels == 1 || (job->mask & (1u << c))) {
        x = (x - job->black) / job->span;
        x = cn_numpy_clip_f32(x, 0.0f, 1.0f);
        x = cn_crt.powf(x, job->exponent);
        x *= job->out_span;
        x += job->out_black;
    } else {
        x = (x - 0.0f) / 1.0f;
        x = cn_numpy_clip_f32(x, 0.0f, 1.0f);
        x = cn_crt.powf(x, 1.0f);
        x *= 1.0f;
        x += 0.0f;
    }
    return x;
}

static void levels_range(void *context, size_t begin, size_t end) {
    const levels_job job = *(const levels_job *)context;
    for (size_t i = begin; i < end; ++i)
        job.dst[i] = leveled(&job, job.src[i], i % job.channels);
}

/* Channel stride above 1 (adjustment_job's comment): one channel per run. */
static void levels_runs(void *context, size_t begin, size_t end) {
    const levels_job job = *(const levels_job *)context;
    for (size_t i = begin; i < end;) {
        size_t run = i / job.stride, c = run % job.channels;
        size_t stop = (run + 1) * job.stride < end ? (run + 1) * job.stride : end;
        for (; i < stop; ++i) job.dst[i] = leveled(&job, job.src[i], c);
    }
}

CN_EXPORT cn_status cn_levels_f32(const float *src, float *dst, size_t pixels,
                                size_t channels, size_t channel_stride, unsigned int mask,
                                float black, float white, float gamma,
                                float out_black, float out_white) {
    if (!src || !dst || (channels != 1 && channels != 3 && channels != 4))
        return CN_INVALID_ARGUMENT;
    if (!channel_stride || pixels % channel_stride) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / sizeof(float) / channels) return CN_SIZE_OVERFLOW;
    cn_status status = checked_buffers(src, pixels * channels, dst, pixels * channels);
    if (status != CN_OK) return status;
    levels_job job = {src, dst, channels, channel_stride, mask,
        black, white - black, 1.0f / gamma, out_black, out_white - out_black};
    return cn_parallel_for(pixels * channels, 65536,
        channel_stride == 1 ? levels_range : levels_runs, &job);
}

typedef struct log_job {
    const float *src;
    float *dst;
    float white, gamma, offset, gain, inverse_scale;
    int invert;
} log_job;

static void log_range(void *context, size_t begin, size_t end) {
    log_job *job = (log_job *)context;
    for (size_t i = begin; i < end; ++i) {
        float x = job->src[i];
        if (!job->invert) {
            x *= 1023.0f;
            x -= job->white;
            x *= 0.002f;
            x /= job->gamma;
            x = cn_crt.powf(10.0f, x);
            x -= job->offset;
            x *= job->gain;
        } else {
            x /= job->gain;
            x += job->offset;
            x = cn_crt.log10f(x);
            x /= job->inverse_scale;
            x += job->white;
            x /= 1023.0f;
        }
        job->dst[i] = x;
    }
}

CN_EXPORT cn_status cn_log_linear_f32(const float *src, float *dst, size_t count,
                                    float white, float gamma, float offset,
                                    float gain, float inverse_scale, int invert) {
    if (!src || !dst || (invert != 0 && invert != 1)) return CN_INVALID_ARGUMENT;
    if (count > SIZE_MAX / sizeof(float)) return CN_SIZE_OVERFLOW;
    cn_status status = checked_buffers(src, count, dst, count);
    if (status != CN_OK) return status;
    log_job job = {src, dst, white, gamma, offset, gain, inverse_scale, invert};
    return cn_parallel_for(count, 65536, log_range, &job);
}
