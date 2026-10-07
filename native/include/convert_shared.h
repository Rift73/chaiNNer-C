/* Pixel conversions (cn_pixels_convert_checked), moved from buffers.c. Shared by
 * that baseline unit and the /arch:AVX2 unit convert_avx2.c: the job, and static
 * helpers compiled into each including unit, never inline, so the linker cannot
 * fold a VEX copy into baseline callers. Every including unit uses every helper
 * (C4505): convert_scalar reaches them all.
 *
 * Each element keeps B3's operation sequence; the clamp is np.clip on the reference
 * stack (clamp_unit), which NumPy 2.5.3 makes B3's again. No helper calls the CRT:
 * classification is by bit tests and the half-to-even rounding is rounded_even's
 * integer form. */
#ifndef CHAINNER_CONVERT_SHARED_H
#define CHAINNER_CONVERT_SHARED_H
#include "chainner.h"
#include "isa.h"
#include <string.h>
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#define CN_CONVERT_SSE 1
#endif

/* isa: the level the C entry read once for the call (cn_isa_current). nearest: the
 * caller's MXCSR, which the pool's helpers take, rounds to nearest (read once per
 * call). events: bit 1, a finite value whose scaled value overflowed; bit 2, a
 * scaled value that is NaN or outside int32; bit 4, a float32 signalling NaN
 * source. A chunk ORs its bits in once. */
typedef struct convert_job {
    const void *src;
    void *out;
    int type, output, normalize;
    cn_isa_level isa;
    int nearest;
    volatile long events;
} convert_job;

/* B3's u8 -> f32 quotients (float)k / 255.0f (divss), k 0-255, in round to
 * nearest only: under a directed mode convert_scalar divides, as B3 does. */
static const float u8_quotients[256] = {
    0.0f, 0.00392156886f, 0.00784313772f, 0.0117647061f, 0.0156862754f, 0.0196078438f,
    0.0235294122f, 0.0274509806f, 0.0313725509f, 0.0352941193f, 0.0392156877f, 0.0431372561f,
    0.0470588244f, 0.0509803928f, 0.0549019612f, 0.0588235296f, 0.0627451017f, 0.0666666701f,
    0.0705882385f, 0.0745098069f, 0.0784313753f, 0.0823529437f, 0.0862745121f, 0.0901960805f,
    0.0941176489f, 0.0980392173f, 0.101960786f, 0.105882354f, 0.109803922f, 0.113725491f,
    0.117647059f, 0.121568628f, 0.125490203f, 0.129411772f, 0.13333334f, 0.137254909f,
    0.141176477f, 0.145098045f, 0.149019614f, 0.152941182f, 0.156862751f, 0.160784319f,
    0.164705887f, 0.168627456f, 0.172549024f, 0.176470593f, 0.180392161f, 0.184313729f,
    0.188235298f, 0.192156866f, 0.196078435f, 0.200000003f, 0.203921571f, 0.20784314f,
    0.211764708f, 0.215686277f, 0.219607845f, 0.223529413f, 0.227450982f, 0.23137255f,
    0.235294119f, 0.239215687f, 0.243137255f, 0.247058824f, 0.250980407f, 0.254901975f,
    0.258823544f, 0.262745112f, 0.266666681f, 0.270588249f, 0.274509817f, 0.278431386f,
    0.282352954f, 0.286274523f, 0.290196091f, 0.294117659f, 0.298039228f, 0.301960796f,
    0.305882365f, 0.309803933f, 0.313725501f, 0.31764707f, 0.321568638f, 0.325490206f,
    0.329411775f, 0.333333343f, 0.337254912f, 0.34117648f, 0.345098048f, 0.349019617f,
    0.352941185f, 0.356862754f, 0.360784322f, 0.36470589f, 0.368627459f, 0.372549027f,
    0.376470596f, 0.380392164f, 0.384313732f, 0.388235301f, 0.392156869f, 0.396078438f,
    0.400000006f, 0.403921574f, 0.407843143f, 0.411764711f, 0.41568628f, 0.419607848f,
    0.423529416f, 0.427450985f, 0.431372553f, 0.435294122f, 0.43921569f, 0.443137258f,
    0.447058827f, 0.450980395f, 0.454901963f, 0.458823532f, 0.4627451f, 0.466666669f,
    0.470588237f, 0.474509805f, 0.478431374f, 0.482352942f, 0.486274511f, 0.490196079f,
    0.494117647f, 0.498039216f, 0.501960814f, 0.505882382f, 0.509803951f, 0.513725519f,
    0.517647088f, 0.521568656f, 0.525490224f, 0.529411793f, 0.533333361f, 0.53725493f,
    0.541176498f, 0.545098066f, 0.549019635f, 0.552941203f, 0.556862772f, 0.56078434f,
    0.564705908f, 0.568627477f, 0.572549045f, 0.576470613f, 0.580392182f, 0.58431375f,
    0.588235319f, 0.592156887f, 0.596078455f, 0.600000024f, 0.603921592f, 0.607843161f,
    0.611764729f, 0.615686297f, 0.619607866f, 0.623529434f, 0.627451003f, 0.631372571f,
    0.635294139f, 0.639215708f, 0.643137276f, 0.647058845f, 0.650980413f, 0.654901981f,
    0.65882355f, 0.662745118f, 0.666666687f, 0.670588255f, 0.674509823f, 0.678431392f,
    0.68235296f, 0.686274529f, 0.690196097f, 0.694117665f, 0.698039234f, 0.701960802f,
    0.70588237f, 0.709803939f, 0.713725507f, 0.717647076f, 0.721568644f, 0.725490212f,
    0.729411781f, 0.733333349f, 0.737254918f, 0.741176486f, 0.745098054f, 0.749019623f,
    0.752941191f, 0.75686276f, 0.760784328f, 0.764705896f, 0.768627465f, 0.772549033f,
    0.776470602f, 0.78039217f, 0.784313738f, 0.788235307f, 0.792156875f, 0.796078444f,
    0.800000012f, 0.80392158f, 0.807843149f, 0.811764717f, 0.815686285f, 0.819607854f,
    0.823529422f, 0.827450991f, 0.831372559f, 0.835294127f, 0.839215696f, 0.843137264f,
    0.847058833f, 0.850980401f, 0.854901969f, 0.858823538f, 0.862745106f, 0.866666675f,
    0.870588243f, 0.874509811f, 0.87843138f, 0.882352948f, 0.886274517f, 0.890196085f,
    0.894117653f, 0.898039222f, 0.90196079f, 0.905882359f, 0.909803927f, 0.913725495f,
    0.917647064f, 0.921568632f, 0.925490201f, 0.929411769f, 0.933333337f, 0.937254906f,
    0.941176474f, 0.945098042f, 0.949019611f, 0.952941179f, 0.956862748f, 0.960784316f,
    0.964705884f, 0.968627453f, 0.972549021f, 0.97647059f, 0.980392158f, 0.984313726f,
    0.988235295f, 0.992156863f, 0.996078432f, 1.0f,
};

static uint32_t float_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/* The enforce clamp (normalize, or any type but float32) as np.clip(x, 0, 1)
 * computes it on the reference stack (NumPy 2.5.3 clip.cpp): x < 0 gives +0, then
 * x > 1 gives 1, so NaN passes with its payload and -0 stays -0; -inf gives +0 and
 * +inf 1. NumPy's build runs it as maxps(0, x) then minps(1, x), which take a
 * denormal as a zero of its sign under DAZ: there a positive denormal gives +0 and a
 * negative one -0, while at the default MXCSR a positive denormal stays and a
 * negative one gives +0. maxss(+0, v) then minss(1, v), B3's form, give those bits. */
static float clamp_unit(float value)
{
#ifdef CN_CONVERT_SSE
    __m128 v = _mm_max_ss(_mm_setzero_ps(), _mm_set_ss(value));
    return _mm_cvtss_f32(_mm_min_ss(_mm_set_ss(1.0f), v));
#else
    value = value < 0.0f ? 0.0f : value;
    return value > 1.0f ? 1.0f : value;
#endif
}

/* Half to even of a float in [-2^31, 2^31): B3's rounded_even (floor, then the
 * remainder against 0.5 and fmod(low, 2)) for every such float, without CRT calls
 * and in any rounding mode. t truncates; f floors; frac is exact except for
 * scaled in (-1, 0), where f = -1 is odd and a frac rounded to 0.5 still gives 0. */
static int32_t rounded_even(float scaled)
{
    int32_t t = (int32_t)scaled;
    int32_t f = t - (scaled < (float)t);
    float frac = scaled - (float)f;
    return f + (frac > 0.5f || (frac == 0.5f && (f & 1)));
}

/* B3's float -> integer tail: scaled = value * scale; bit 1 when value is finite
 * and scaled is infinite; bit 2, and B3's INT32_MIN sentinel (NumPy 1.24.4's MSVC
 * cast through int32), when scaled is NaN or outside [-2^31, 2^31); else
 * scaled rounded half to even. */
static int32_t scaled_integer(float value, float scale, long *events)
{
    float scaled = value * scale;
    if ((float_bits(value) & 0x7f800000u) != 0x7f800000u &&
        (float_bits(scaled) & 0x7fffffffu) == 0x7f800000u) *events |= 1;
    if (!(scaled >= -2147483648.0f && scaled < 2147483648.0f)) {
        *events |= 2;
        return INT32_MIN;
    }
    return rounded_even(scaled);
}

/* Bit 4: a float32 signalling NaN source (exponent all ones, a mantissa, no quiet
 * bit). */
static long signalling_nan(uint32_t bits)
{
    return (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) &&
        !(bits & 0x00400000u) ? 4 : 0;
}

/* f32 -> u8 (type 0, output 1), with the clamp when normalize. */
static long convert_f32_u8(const convert_job *j, size_t begin, size_t end)
{
    const float *src = (const float *)j->src;
    uint8_t *out = (uint8_t *)j->out;
    int clamp = j->normalize;
    long events = 0;
    for (size_t i = begin; i < end; ++i) {
        float value = clamp ? clamp_unit(src[i]) : src[i];
        out[i] = (uint8_t)scaled_integer(value, 255.0f, &events);
        events |= signalling_nan(float_bits(src[i]));
    }
    return events;
}

#define CN_CONVERT_BLOCK 256

/* B3's load_value arithmetic for count elements from start, one loop per type. */
static void load_values(const convert_job *j, size_t start, size_t count, float *values)
{
    size_t k;
    switch (j->type) {
    case 0: memcpy(values, (const float *)j->src + start, count * sizeof(float)); break;
    case 1:
        for (k = 0; k < count; ++k) values[k] = (float)((const double *)j->src)[start + k];
        break;
    case 2:
        for (k = 0; k < count; ++k) values[k] = (float)((const uint8_t *)j->src)[start + k] / 255.0f;
        break;
    case 3:
        for (k = 0; k < count; ++k) values[k] = (float)((const uint16_t *)j->src)[start + k] / 65535.0f;
        break;
    case 4:
        for (k = 0; k < count; ++k) values[k] = (float)((const int8_t *)j->src)[start + k] / 127.0f;
        break;
    case 5:
        for (k = 0; k < count; ++k) values[k] = (float)((const int16_t *)j->src)[start + k] / 32767.0f;
        break;
    case 6:
        for (k = 0; k < count; ++k)
            values[k] = (float)((double)(float)((const int32_t *)j->src)[start + k] / 2147483647.0);
        break;
    case 7:
        for (k = 0; k < count; ++k)
            values[k] = (float)((double)(float)((const uint32_t *)j->src)[start + k] / 4294967295.0);
        break;
    case 8:
        for (k = 0; k < count; ++k)
            values[k] = (float)((double)(float)((const int64_t *)j->src)[start + k] / 9223372036854775807.0);
        break;
    case 9:
        for (k = 0; k < count; ++k)
            values[k] = (float)((double)(float)((const uint64_t *)j->src)[start + k] / 18446744073709551615.0);
        break;
    default:
        for (k = 0; k < count; ++k) values[k] = ((const uint8_t *)j->src)[start + k] ? 1.0f : 0.0f;
        break;
    }
}

/* Every combo without its own loop: blocks of load_values, the clamp, then the
 * store or the integer tail. */
static long convert_generic(const convert_job *j, size_t begin, size_t end)
{
    float values[CN_CONVERT_BLOCK];
    int clamp = j->normalize || j->type != 0;
    float scale = j->output == 1 ? 255.0f : 65535.0f;
    long events = 0;
    size_t start = begin, count, k;
    for (; start < end; start += count) {
        count = end - start < CN_CONVERT_BLOCK ? end - start : CN_CONVERT_BLOCK;
        load_values(j, start, count, values);
        if (clamp)
            for (k = 0; k < count; ++k) values[k] = clamp_unit(values[k]);
        if (j->output == 0) {
            memcpy((float *)j->out + start, values, count * sizeof(float));
            continue;
        }
        if (j->output == 1)
            for (k = 0; k < count; ++k)
                ((uint8_t *)j->out)[start + k] = (uint8_t)scaled_integer(values[k], scale, &events);
        else
            for (k = 0; k < count; ++k)
                ((uint16_t *)j->out)[start + k] = (uint16_t)scaled_integer(values[k], scale, &events);
        if (j->type == 0)
            for (k = 0; k < count; ++k)
                events |= signalling_nan(float_bits(((const float *)j->src)[start + k]));
    }
    return events;
}

/* The D7 loops over [begin, end), one per (type, output, normalize); returns the
 * event bits. u8 -> f32 (type 2, output 0) reads B3's quotients in round to
 * nearest and otherwise divides as B3 does; the clamp leaves the quotients as they
 * are in every mode. f32 -> f32 is the clamp, or a copy when normalize is 0. */
static long convert_scalar(const convert_job *j, size_t begin, size_t end)
{
    if (begin >= end) return 0;
    if (j->type == 2 && j->output == 0) {
        const uint8_t *src = (const uint8_t *)j->src;
        float *out = (float *)j->out;
        if (j->nearest)
            for (size_t i = begin; i < end; ++i) out[i] = u8_quotients[src[i]];
        else
            for (size_t i = begin; i < end; ++i) out[i] = (float)src[i] / 255.0f;
        return 0;
    }
    if (j->type == 0 && j->output == 0) {
        const float *src = (const float *)j->src;
        float *out = (float *)j->out;
        if (!j->normalize) {
            memcpy(out + begin, src + begin, (end - begin) * sizeof(float));
            return 0;
        }
        for (size_t i = begin; i < end; ++i) out[i] = clamp_unit(src[i]);
        return 0;
    }
    if (j->type == 0 && j->output == 1) return convert_f32_u8(j, begin, end);
    return convert_generic(j, begin, end);
}

/* convert_avx2.c: the combos with their own loops ((2, 0, *), (0, 0, *),
 * (0, 1, *)) over [begin, end), 8-element groups at indices = 0 (mod 8) and the
 * head and tail through convert_scalar; returns the event bits. The caller
 * ensures ISA level avx2 or above. */
long cn_convert_avx2(const convert_job *j, size_t begin, size_t end);

/* convert_avx2.c: cn_pixels_normalized_f32's scan of src[0, count): 1 when every
 * element is +0 or a normal value in (0, 1] (the baseline loop's bit tests), else
 * 0. The caller ensures ISA level avx2 or above. */
int cn_pixels_normalized_avx2(const float *src, size_t count);
#endif
