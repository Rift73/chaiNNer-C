/* Lens Blur's component composition, moved from lens_complete_ops.c. Shared by
 * that baseline unit and the /arch:AVX2 unit lens_avx2.c: the context, and a
 * static helper compiled into each including unit, never inline, so the linker
 * cannot fold a VEX copy into baseline callers. Every including unit uses it
 * (C4505). */
#ifndef CHAINNER_LENS_SHARED_H
#define CHAINNER_LENS_SHARED_H
#include "chainner.h"
#include "isa.h"

/* isa: the level the C entry read once for the call (cn_isa_current). */
typedef struct compose_context {
    const float *f1, *f2, *f3, *f4;
    float *out, a, b;
    int accumulate;
    cn_isa_level isa;
} compose_context;

/* Element i of out: NumPy's float32 order for real + 1j * imag, then the
 * weighted sum, added to out[i] when accumulating. */
static float compose_one(const compose_context *ctx, size_t i)
{
    float real = ctx->f1[i] - ctx->f4[i];
    float imag = ctx->f2[i] + ctx->f3[i];
    /* Original expression: real + 1j * imag. Its zero products are
     * significant for infinities and signed zero and cannot be removed. */
    float complex_real = 0.0f * imag - 0.0f;
    float complex_imag = 0.0f + imag;
    real += complex_real;
    imag = 0.0f + complex_imag;
    float value = real * ctx->a + imag * ctx->b;
    if (ctx->accumulate) value = ctx->out[i] + value;
    return value;
}

/* lens_avx2.c: out[begin, end), inside one chunk, for ISA level avx2 or above:
 * groups of 8 run compose_one's operations in its order, the rest compose_one. */
void cn_lens_compose_avx2(const compose_context *ctx, size_t begin, size_t end);
#endif
