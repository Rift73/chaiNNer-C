/* chaiNNer-rs clipboard quantization (MIT, Michael Schmidt, 2023).
 * C implementation keeps the original two float32 rounding operations and
 * uses the existing bounded CPU pool for independent output pixels. */
#include "clipboard_ops.h"
#include "parallel.h"
#include <math.h>
#include <string.h>

typedef struct clipboard_job {
    const unsigned char *source;
    uint8_t *output;
    size_t height, width, channels;
    ptrdiff_t row, column, channel;
    int rgb;
} clipboard_job;

static uint8_t quantize(const unsigned char *source) {
    float value;
    memcpy(&value, source, sizeof(value));
    const float scaled = value * 255.0f;
    const float shifted = scaled + 0.5f;
    if (!(shifted >= 0.0f)) return 0; /* NaN and negative values. */
    if (shifted >= 255.0f) return 255;
    return (uint8_t)floorf(shifted);
}

static void convert_pixels(void *context, size_t begin, size_t end) {
    const clipboard_job *job = (const clipboard_job *)context;
    for (size_t index = begin; index < end; ++index) {
        const size_t y = index / job->width, x = index % job->width;
        const unsigned char *pixel = job->source + (ptrdiff_t)y * job->row + (ptrdiff_t)x * job->column;
        uint8_t *out = job->output + ((job->height - 1 - y) * job->width + x) * 4;
        if (job->channels == 1) {
            const uint8_t gray = quantize(pixel);
            out[0] = gray; out[1] = gray; out[2] = gray; out[3] = 255;
        } else {
            const size_t blue = job->rgb ? 2 : 0, red = job->rgb ? 0 : 2;
            out[0] = quantize(pixel + (ptrdiff_t)blue * job->channel);
            out[1] = quantize(pixel + job->channel);
            out[2] = quantize(pixel + (ptrdiff_t)red * job->channel);
            out[3] = job->channels == 4 ? quantize(pixel + 3 * job->channel) : 255;
        }
    }
}

static int add_extent(size_t dimension, ptrdiff_t stride, ptrdiff_t *low, ptrdiff_t *high) {
    if (dimension <= 1 || stride == 0) return 1;
    const size_t magnitude = stride < 0 ? (size_t)(-(stride + 1)) + 1 : (size_t)stride;
    const size_t limit = stride < 0 ? (size_t)PTRDIFF_MAX + 1 : (size_t)PTRDIFF_MAX;
    if (dimension - 1 > limit / magnitude) return 0;
    const size_t extent = (dimension - 1) * magnitude;
    if (stride < 0) {
        const ptrdiff_t delta = extent == (size_t)PTRDIFF_MAX + 1 ? PTRDIFF_MIN : -(ptrdiff_t)extent;
        if (*low < PTRDIFF_MIN - delta) return 0;
        *low += delta;
    } else {
        const ptrdiff_t delta = (ptrdiff_t)extent;
        if (*high > PTRDIFF_MAX - delta) return 0;
        *high += delta;
    }
    return 1;
}

int cn_clipboard_pixels_f32(const void *source, size_t height, size_t width,
    size_t channels, ptrdiff_t row, ptrdiff_t column, ptrdiff_t channel,
    int rgb, uint8_t *output, size_t output_bytes) {
    if ((channels != 1 && channels != 3 && channels != 4) || (rgb != 0 && rgb != 1)) return CN_INVALID_ARGUMENT;
    if (width && height > SIZE_MAX / width) return CN_SIZE_OVERFLOW;
    const size_t count = width * height;
    if (count > SIZE_MAX / 4 || count * 4 != output_bytes) return CN_SIZE_OVERFLOW;
    if (!count) return CN_OK;
    if (!source || !output || height > (size_t)PTRDIFF_MAX || width > (size_t)PTRDIFF_MAX) return CN_INVALID_ARGUMENT;
    ptrdiff_t low = 0, high = 0;
    if (!add_extent(height, row, &low, &high) || !add_extent(width, column, &low, &high) ||
        !add_extent(channels, channel, &low, &high)) return CN_SIZE_OVERFLOW;
    const uintptr_t input = (uintptr_t)source, out = (uintptr_t)output;
    const size_t below = low < 0 ? (size_t)(-(low + 1)) + 1 : 0;
    if (input < below || (size_t)high > UINTPTR_MAX - input ||
        input + (size_t)high > UINTPTR_MAX - sizeof(float) || output_bytes > UINTPTR_MAX - out) return CN_SIZE_OVERFLOW;
    if (out < input + (size_t)high + sizeof(float) && input - below < out + output_bytes) return CN_INVALID_ARGUMENT;
    clipboard_job job = {(const unsigned char *)source, output, height, width, channels, row, column, channel, rgb};
    return cn_parallel_for(count, 16384, convert_pixels, &job);
}
