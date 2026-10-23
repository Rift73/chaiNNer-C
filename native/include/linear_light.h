/* Linear Light's transfer curves (Descale and Resize; Get Native Resolution's linear planes). Self-contained: C or
 * C++, no other chaiNNer header.
 *
 * cn_light_linearize takes an encoded value to linear light, cn_light_encode takes it back. Each is the standard
 * curve on |value| with the value's sign (copysign(f(|value|), value)), never clipped, so negative ringing and values
 * above 1 survive a round trip. The expressions are those Vernier 0.2 evaluates (media.cpp's linearize and
 * encodeLight), in its order and in double, so an analysis gives its numbers. native/src/linear_light_ops.c applies
 * them to float32 images (cn_linear_light, run_kernels.h).
 *
 * A unit may define CN_LIGHT_POW(a, g) (a >= 0, g > 0) before including this header to evaluate the powers its own
 * way; linear_light_ops.c does (linear_light_shared.h). The default is pow. */
#ifndef CHAINNER_LINEAR_LIGHT_H
#define CHAINNER_LINEAR_LIGHT_H
#include <math.h>

#ifndef CN_LIGHT_POW
#define CN_LIGHT_POW pow
#endif

/* The transfers, numbered as the Transfer dropdown's values (backend nodes/impl/linear_light.py, LightTransfer). */
enum {
    CN_LIGHT_GAMMA24 = 0, /* BT.1886 with a zero black level: a pure 2.4 power */
    CN_LIGHT_SRGB = 1,    /* IEC 61966-2-1, piecewise */
    CN_LIGHT_GAMMA22 = 2, /* a pure 2.2 power */
    CN_LIGHT_BT709 = 3    /* the BT.709 camera OETF, inverted to linearize */
};

/* Any other transfer leaves the value as it is. */
static inline double cn_light_linearize(double value, int transfer)
{
    const double a = fabs(value);
    double v = a;
    switch (transfer) {
    case CN_LIGHT_GAMMA24: v = CN_LIGHT_POW(a, 2.4); break;
    case CN_LIGHT_SRGB: v = a <= .04045 ? a / 12.92 : CN_LIGHT_POW((a + .055) / 1.055, 2.4); break;
    case CN_LIGHT_GAMMA22: v = CN_LIGHT_POW(a, 2.2); break;
    case CN_LIGHT_BT709: v = a < .081 ? a / 4.5 : CN_LIGHT_POW((a + .099) / 1.099, 1 / .45); break;
    }
    return copysign(v, value);
}

static inline double cn_light_encode(double value, int transfer)
{
    const double a = fabs(value);
    double v = a;
    switch (transfer) {
    case CN_LIGHT_GAMMA24: v = CN_LIGHT_POW(a, 1 / 2.4); break;
    case CN_LIGHT_SRGB: v = a <= .0031308 ? 12.92 * a : 1.055 * CN_LIGHT_POW(a, 1 / 2.4) - .055; break;
    case CN_LIGHT_GAMMA22: v = CN_LIGHT_POW(a, 1 / 2.2); break;
    case CN_LIGHT_BT709: v = a < .018 ? 4.5 * a : 1.099 * CN_LIGHT_POW(a, .45) - .099; break;
    }
    return copysign(v, value);
}
#endif
