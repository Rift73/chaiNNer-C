/* Morphology (cn_morphology_complete), moved from morphology_ops.c. Shared by that
 * baseline unit and the /arch:AVX2 unit morphology_avx2.c: the context, and static
 * helpers compiled into each including unit, never inline, so the linker cannot fold a
 * VEX copy into baseline callers. Every including unit uses every helper (C4505). */
#ifndef CHAINNER_MORPHOLOGY_SHARED_H
#define CHAINNER_MORPHOLOGY_SHARED_H
#include "chainner.h"
#include "isa.h"
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#define CN_MORPHOLOGY_SSE 1
#endif

/* A deque entry of the ellipse's row extrema: the src value beside its index, so the
 * compares read no src (D11); defined in morph_entry.h, which morphology_ops.c includes
 * (morphology_avx2.c never touches an entry). */
struct morph_entry;

/* isa: the level the C entry read once for the call (cn_isa_current). The ellipse
 * combine runs the offsets dy in [offset, offset_end): src holds the rows' extrema of
 * one half-width, read for dy < zero_from; the rest, whose half-width is 0, read
 * current (the iteration's input) itself. */
typedef struct morphology_context {
    const float *src, *current;
    float *out;
    struct morph_entry *queue;
    const size_t *spans;
    size_t height, width, channels, radius, lane_half, offset, offset_end, zero_from;
    int maximum, shape, stage;
    float border;
    cn_isa_level isa;
} morphology_context;

/* OpenCV's MaxOp/MinOp as B3 executes them, maxss/minss everywhere (VERIFIED(dumpbin)
 * on B3, D11), written with the intrinsics so that every includer emits the same
 * instruction. The vector form (OpenCV's v_max/v_min) is maxss/minss(a, b), a > b or
 * a < b ? a : b; the scalar form (std::max/std::min) is maxss/minss(b, a), b > a or
 * b < a ? b : a. On equal or unordered operands each returns its second operand, and
 * under DAZ a denormal it returns comes back as the zero of its sign. Both forms equal
 * the installed chaiNNer's cv2.dilate/cv2.erode through the node path bit for bit at
 * the default MXCSR (task4-verdict.md). */
static float update(float a, float b, int maximum, int vector)
{
#ifdef CN_MORPHOLOGY_SSE
    __m128 first = _mm_set_ss(vector ? a : b), second = _mm_set_ss(vector ? b : a);
    return _mm_cvtss_f32(maximum ? _mm_max_ss(first, second) : _mm_min_ss(first, second));
#else
    if (vector) return maximum ? (a > b ? a : b) : (a < b ? a : b);
    return maximum ? (a < b ? b : a) : (b < a ? b : a);
#endif
}

/* The end of the combine's row segment that starts at i, inside [i, end), and its row
 * y: the rows dy above and below exist for every output of the segment or for none. */
static size_t combine_segment(const morphology_context *ctx, size_t i, size_t end, size_t *y)
{
    size_t row_width = ctx->width * ctx->channels, stop;
    *y = i / row_width;
    stop = (*y + 1) * row_width;
    return stop < end ? stop : end;
}

/* Output i (row y) of the combine over dy in [offset, offset_end): B3's
 * ellipse_combine sequence, one of its passes per dy, in ascending dy. It starts from
 * the border at dy = 0 (the fill, folded in: D11), else from out[i], and updates
 * (vector form) with the source i dy rows above when y >= dy, then dy rows below when
 * 0 < dy < height - y, the tests of B3's ellipse_combine. */
static float combine_one(const morphology_context *ctx, size_t i, size_t y)
{
    size_t row_width = ctx->width * ctx->channels, dy;
    float value = ctx->offset ? ctx->out[i] : ctx->border;
    for (dy = ctx->offset; dy < ctx->offset_end; ++dy) {
        const float *src = dy < ctx->zero_from ? ctx->src : ctx->current;
        size_t step = dy * row_width;
        if (y >= dy) value = update(value, src[i - step], ctx->maximum, 1);
        if (dy && dy < ctx->height - y) value = update(value, src[i + step], ctx->maximum, 1);
    }
    return value;
}

/* morphology_avx2.c: the combine over [begin, end), inside one chunk, for ISA level
 * avx2 or above: per row segment, groups of 8 run combine_one's sequence as
 * vmaxps/vminps(value, src) (B3's maxss/minss operand order), the rest combine_one. */
void cn_morphology_combine_avx2(const morphology_context *ctx, size_t begin, size_t end);
#endif
