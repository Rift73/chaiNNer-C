/* chaiNNer text fitting/anchors and Pillow 9.2 ImageDraw multiline placement.
 * Pillow's HPND license is reproduced in chainner_native.LICENSE.txt.
 * Glyph shaping/rasterization remains the installed FreeType/RAQM primitive;
 * this C module owns application line scanning, fitting and placement.
 */
#include "chainner.h"
#include <math.h>

static cn_status span(const void *ptr, size_t count, size_t item) {
    if (!ptr || (uintptr_t)ptr % item) return CN_INVALID_ARGUMENT;
    if (count > SIZE_MAX / item || (uintptr_t)ptr > UINTPTR_MAX - count * item)
        return CN_SIZE_OVERFLOW;
    return CN_OK;
}

CN_EXPORT cn_status cn_text_scan(const uint32_t *text, size_t count, size_t *summary) {
    cn_status status = span(text, count, sizeof(uint32_t));
    if (status != CN_OK) return status;
    status = span(summary, 3, sizeof(size_t));
    if (status != CN_OK || count == SIZE_MAX) return status == CN_OK ? CN_SIZE_OVERFLOW : status;
    uintptr_t a = (uintptr_t)text, b = (uintptr_t)summary;
    if (a < b + 3 * sizeof(size_t) && b < a + count * sizeof(uint32_t))
        return CN_INVALID_ARGUMENT;
    size_t begin = 0, lines = 0, longest_begin = 0, longest_size = 0;
    for (size_t i = 0; i <= count; ++i) {
        if (i == count || text[i] == 10) {
            size_t length = i - begin;
            if (length > longest_size) { longest_begin = begin; longest_size = length; }
            ++lines; begin = i + 1;
        }
    }
    summary[0] = lines; summary[1] = longest_begin; summary[2] = longest_size;
    return CN_OK;
}

CN_EXPORT cn_status cn_text_ranges(const uint32_t *text, size_t count,
        size_t *ranges, size_t lines) {
    size_t summary[3];
    cn_status status = cn_text_scan(text, count, summary);
    if (status != CN_OK) return status;
    if (lines != summary[0]) return CN_INVALID_ARGUMENT;
    if (lines > SIZE_MAX / 2) return CN_SIZE_OVERFLOW;
    status = span(ranges, lines * 2, sizeof(size_t));
    if (status != CN_OK) return status;
    uintptr_t a = (uintptr_t)text, b = (uintptr_t)ranges;
    if (a < b + lines * 2 * sizeof(size_t) && b < a + count * sizeof(uint32_t))
        return CN_INVALID_ARGUMENT;
    size_t begin = 0, line = 0;
    for (size_t i = 0; i <= count; ++i)
        if (i == count || text[i] == 10) {
            ranges[line * 2] = begin; ranges[line * 2 + 1] = i - begin;
            ++line; begin = i + 1;
        }
    return CN_OK;
}

CN_EXPORT cn_status cn_text_fit(double width, double height, double ref_width,
        double ref_height, size_t lines, double *result) {
    cn_status status = span(result, 1, sizeof(double));
    if (status != CN_OK) return status;
    if (!(width > 0) || !(height > 0) || !(ref_width > 0) || !(ref_height > 0)
        || !lines || !isfinite(width) || !isfinite(height)
        || !isfinite(ref_width) || !isfinite(ref_height)) return CN_INVALID_ARGUMENT;
    double w = trunc(width * 100.0 / ref_width);
    double h = trunc(height * 100.0 / (ref_height * (double)lines));
    *result = w < h ? w : h;
    return CN_OK;
}

CN_EXPORT cn_status cn_caption_fit(double height, double width, double measured,
        double *result) {
    cn_status status = span(result, 1, sizeof(double));
    if (status != CN_OK) return status;
    if (!(height > 0) || !(width > 0) || !isfinite(height) || !isfinite(width)
        || !isfinite(measured)) return CN_INVALID_ARGUMENT;
    double initial = nearbyint(height * .8);
    *result = measured > width ? nearbyint(initial * width / measured) : initial;
    return CN_OK;
}

CN_EXPORT cn_status cn_text_anchor(double width, double height, double text_width,
        double text_height, int position, double *xy) {
    cn_status status = span(xy, 2, sizeof(double));
    if (status != CN_OK) return status;
    if (position < 0 || position > 8) return CN_INVALID_ARGUMENT;
    static const double image_factor[3] = {0, .5, 1};
    static const double text_factor[3] = {.5, 0, -.5};
    int row = position / 3, column = position % 3;
    xy[0] = nearbyint(width * image_factor[column] + text_width * text_factor[column]);
    xy[1] = nearbyint(height * image_factor[row] + text_height * text_factor[row]);
    return CN_OK;
}

CN_EXPORT cn_status cn_text_positions(const double *widths, size_t lines,
        double x, double y, double spacing, int alignment, double *xy) {
    if (!lines || lines > SIZE_MAX / 2 || alignment < 0 || alignment > 2
        || !isfinite(x) || !isfinite(y) || !isfinite(spacing)) return CN_INVALID_ARGUMENT;
    cn_status status = span(widths, lines, sizeof(double));
    if (status != CN_OK) return status;
    status = span(xy, lines * 2, sizeof(double));
    if (status != CN_OK) return status;
    uintptr_t a = (uintptr_t)widths, b = (uintptr_t)xy;
    if (a < b + lines * 2 * sizeof(double) && b < a + lines * sizeof(double))
        return CN_INVALID_ARGUMENT;
    double maximum = 0;
    for (size_t i = 0; i < lines; ++i) {
        if (!isfinite(widths[i])) return CN_INVALID_ARGUMENT;
        if (widths[i] > maximum) maximum = widths[i];
    }
    double top = y - (double)(lines - 1) * spacing / 2.0;
    for (size_t i = 0; i < lines; ++i) {
        double difference = maximum - widths[i], left = x - difference / 2.0;
        if (alignment == 1) left += difference / 2.0;
        if (alignment == 2) left += difference;
        xy[2 * i] = left; xy[2 * i + 1] = top;
        top += spacing;
    }
    return CN_OK;
}
