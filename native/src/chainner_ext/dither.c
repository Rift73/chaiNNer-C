/* Port of chainner_ext 0.3.10 crates/bindings/src/dither.rs, MIT OR Apache-2.0. */
/* UniformQuantization, PaletteQuantization (extract_unique_ndim's order, kept by the
 * ordered palette plans), DiffusionAlgorithm and the four dithering functions over
 * neighborhood_ops.c. The quantizer's channel rules and panics are dither.rs's. */
#include "ext.h"
#include "numeric.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

PyTypeObject *chn_uniform_type, *chn_palette_type, *chn_diffusion_type;

/* ---------------------------------------------------------------------------------- */
/* UniformQuantization */

typedef struct uniform_object {
    PyObject_HEAD
    uint32_t per_channel;
} uniform_object;

static PyObject *uniform_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    static const char *const names[] = {"colors_per_channel"};
    PyObject *values[1];
    uint32_t per_channel;
    if (chn_parse(args, kwargs, "UniformQuantization.__new__", names, 1, 1, values) ||
        chn_u32(values[0], "colors_per_channel", &per_channel)) return NULL;
    if (per_channel < 2) {
        PyErr_SetString(PyExc_ValueError, "Argument 'per_channel' must be at least 2.");
        return NULL;
    }
    uniform_object *self = PyObject_New(uniform_object, type);
    if (self) self->per_channel = per_channel;
    return (PyObject *)self;
}

static PyObject *uniform_colors_per_channel(PyObject *self, void *closure)
{
    (void)closure;
    return PyLong_FromUnsignedLong(((uniform_object *)self)->per_channel);
}

static PyGetSetDef uniform_getset[] = {
    {"colors_per_channel", uniform_colors_per_channel, NULL, "", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

/* ---------------------------------------------------------------------------------- */
/* PaletteQuantization: the unique colors, sorted by extract_unique_const's key. */

typedef struct palette_object {
    PyObject_HEAD
    float *colors;   /* count x channels */
    size_t count, channels;
    void *plans[5];  /* ordered plans by the image's channel count (1, 3, 4) */
} palette_object;

typedef struct palette_sort {
    uint32_t bits[4];
    uint32_t key; /* f32::total_cmp order of the key */
    size_t index;
} palette_sort;

static uint32_t total_order(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits ^ ((bits >> 31) ? UINT32_MAX : UINT32_C(0x80000000));
}

static int compare_bits(const void *a, const void *b)
{
    const palette_sort *x = a, *y = b;
    int order = memcmp(x->bits, y->bits, sizeof(x->bits));
    if (order) return order;
    return x->index < y->index ? -1 : x->index > y->index;
}

static int compare_key(const void *a, const void *b)
{
    const palette_sort *x = a, *y = b;
    if (x->key != y->key) return x->key < y->key ? -1 : 1;
    /* Equal keys: upstream's AHashSet order (a "tie" in the conformance manifest);
       the first occurrence is kept first, as the node's palettes do. */
    return x->index < y->index ? -1 : x->index > y->index;
}

/* extract_unique_const's key: numeric.h's for 1, 3 and 4 channels; two channels sum. */ float palette_key(const float *color, size_t channels)
{
    return channels == 2 ? 0.0f + color[0] + color[1] : cn_palette_key(color, channels);
}

static PyObject *palette_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    static const char *const names[] = {"palette"};
    PyObject *values[1];
    chn_image image;
    if (chn_parse(args, kwargs, "PaletteQuantization.__new__", names, 1, 1, values) ||
        chn_image_extract(values[0], "palette", &image)) return NULL;
    if (chn_image_load(&image)) { chn_image_release(&image); return NULL; }
    if (image.height != 1) {
        chn_image_release(&image);
        PyErr_SetString(PyExc_ValueError, "Argument 'palette' must have a height of 1.");
        return NULL;
    }
    size_t c = image.channels, n = image.width;
    if (c < 1 || c > 4) {
        chn_image_release(&image);
        PyErr_Format(PyExc_ValueError, "Argument 'palette' has an unsupported number of "
            "channels. Images with %zu channels are not supported.", c);
        return NULL;
    }
    palette_sort *entries = calloc(n ? n : 1, sizeof(palette_sort));
    float *colors = malloc((n ? n : 1) * c * sizeof(float));
    if (!entries || !colors) {
        free(entries); free(colors); chn_image_release(&image);
        return PyErr_NoMemory();
    }
    for (size_t i = 0; i < n; ++i) {
        memcpy(entries[i].bits, image.data + i * c, c * sizeof(float));
        entries[i].index = i;
    }
    chn_image_release(&image);
    qsort(entries, n, sizeof(palette_sort), compare_bits);
    size_t unique = 0;
    for (size_t i = 0; i < n; ++i) {
        if (unique && !memcmp(entries[i].bits, entries[unique - 1].bits, sizeof(entries[i].bits)))
            continue;
        entries[unique++] = entries[i];
    }
    for (size_t i = 0; i < unique; ++i) {
        float color[4];
        memcpy(color, entries[i].bits, sizeof(color));
        entries[i].key = total_order(palette_key(color, c));
    }
    qsort(entries, unique, sizeof(palette_sort), compare_key);
    for (size_t i = 0; i < unique; ++i) memcpy(colors + i * c, entries[i].bits, c * sizeof(float));
    free(entries);
    palette_object *self = PyObject_New(palette_object, type);
    if (!self) { free(colors); return NULL; }
    self->colors = colors;
    self->count = unique;
    self->channels = c;
    memset(self->plans, 0, sizeof(self->plans));
    return (PyObject *)self;
}

static void palette_dealloc(PyObject *self)
{
    palette_object *palette = (palette_object *)self;
    for (size_t i = 0; i < 5; ++i) cn_neighborhood_palette_free(palette->plans[i]);
    free(palette->colors);
    PyTypeObject *type = Py_TYPE(self);
    freefunc release = (freefunc)PyType_GetSlot(type, Py_tp_free);
    release(self);
    Py_DECREF(type);
}

static PyObject *palette_channels(PyObject *self, void *closure)
{
    (void)closure;
    return PyLong_FromSize_t(((palette_object *)self)->channels);
}

static PyObject *palette_colors(PyObject *self, void *closure)
{
    (void)closure;
    return PyLong_FromSize_t(((palette_object *)self)->count);
}

static PyGetSetDef palette_getset[] = {
    {"channels", palette_channels, NULL, "", NULL},
    {"colors", palette_colors, NULL, "", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

/* into_quantizer::<P>(): the palette as the image's pixel type (FromFlat: f32 takes 1
 * channel, Vec3A 1 or 3, Vec4 1, 3 or 4), then ColorPalette::new's non-empty assert. The
 * plan is built once per pixel type and is immutable. */
static const void *palette_plan(palette_object *palette, size_t channels)
{
    static const char *const expected[5] = {"", "[1]", "", "[1, 3]", "[1, 3, 4]"};
    size_t from = palette->channels;
    int accepted = from == 1 || from == channels || (channels == 4 && from == 3);
    if (!accepted) {
        chn_panic_format("Expected shape of palette to match.: ShapeMismatch { actual: %zu, "
            "expected: %s }", from, expected[channels]);
        return NULL;
    }
    if (!palette->count) {
        chn_panic("palette must contain at least one color");
        return NULL;
    }
    if (palette->plans[channels]) return palette->plans[channels];
    float *colors = malloc(palette->count * channels * sizeof(float));
    if (!colors) { PyErr_NoMemory(); return NULL; }
    int nan = 0;
    for (size_t i = 0; i < palette->count; ++i) {
        const float *source = palette->colors + i * from;
        float *target = colors + i * channels;
        for (size_t k = 0; k < channels; ++k) {
            target[k] = from == channels ? source[k] : k == 3 ? 1.0f : from == 1 ? source[0] : source[k];
            nan |= isnan(target[k]);
        }
    }
    /* ColorPalette::new puts 300 or more colours of 3 or 4 channels in rstar 0.11's R-tree,
     * whose bulk load unwraps partial_cmp: any NaN coordinate panics (aabb.rs:217). One
     * channel keeps the corrected search (upstream: "Point dimension too small"). Any
     * other nonfinite colour or pixel, which the R-tree's traversal decides, takes the
     * sub-300 linear rule (Consult 8 D-5, in the plan). */
    if (nan && channels > 1 && palette->count >= 300) {
        free(colors);
        chn_panic("called `Option::unwrap()` on a `None` value");
        return NULL;
    }
    void *plan = NULL;
    size_t bytes;
    cn_status status = cn_neighborhood_palette_create_ordered(colors, palette->count, channels, &plan, &bytes);
    free(colors);
    if (chn_status(status)) return NULL;
    palette->plans[channels] = plan;
    return plan;
}

/* ---------------------------------------------------------------------------------- */
/* Quant = Uniform(UniformQuantization) | Palette(PaletteQuantization) */

static int extract_quant(PyObject *value, PyObject **uniform, PyObject **palette)
{
    *uniform = *palette = NULL;
    if (PyObject_TypeCheck(value, chn_uniform_type)) { *uniform = value; return 0; }
    if (PyObject_TypeCheck(value, chn_palette_type)) { *palette = value; return 0; }
    PyObject *type = PyObject_GetAttrString((PyObject *)Py_TYPE(value), "__name__");
    if (!type) return -1;
    PyErr_Format(PyExc_TypeError,
        "argument 'quant': failed to extract enum Quant ('Uniform | Palette')\n"
        "- variant Uniform (Uniform): TypeError: failed to extract field Quant::Uniform.0, caused by "
        "TypeError: '%U' object cannot be converted to 'UniformQuantization'\n"
        "- variant Palette (Palette): TypeError: failed to extract field Quant::Palette.0, caused by "
        "TypeError: '%U' object cannot be converted to 'PaletteQuantization'", type, type);
    Py_DECREF(type);
    return -1;
}

/* A dithering call on a loaded image: the output array, then the kernel without the GIL.
   mode: 0 quantize, 1 ordered, 2 diffusion, 3 riemersma. */
typedef struct dither_call {
    uint32_t colors, history;
    int mode, algorithm;
    size_t map_size;
    float decay;
    const void *plan;
} dither_call;

static PyObject *run_dither(chn_image *image, const dither_call *call)
{
    float *out;
    PyObject *result = chn_new_array(image->height, image->width, image->channels, &out);
    if (!result || !chn_count(image)) return result;
    cn_status status;
    const float *src = image->data;
    size_t h = image->height, w = image->width, c = image->channels;
    Py_BEGIN_ALLOW_THREADS
    if (call->plan)
        status = cn_neighborhood_palette_apply(src, out, h, w, c, call->plan, call->mode,
            call->algorithm, call->history, call->decay, NULL);
    else if (call->mode == 3)
        status = cn_neighborhood_uniform_riemersma(src, out, h, w, c, call->colors,
            call->history, call->decay);
    else
        status = cn_neighborhood_dither(src, out, h, w, c, call->colors, call->mode,
            call->map_size, call->algorithm, NULL, 0, NULL);
    Py_END_ALLOW_THREADS
    if (chn_status(status)) Py_CLEAR(result);
    return result;
}

/* The palette path's common part: 1, 3 or 4 channels, then the quantizer. */
static int prepare(chn_image *image, PyObject *uniform, PyObject *palette, dither_call *call)
{
    size_t c = image->channels;
    if (c != 1 && c != 3 && c != 4) return chn_channel_error("1, 3, or 4", c);
    if (palette) {
        call->plan = palette_plan((palette_object *)palette, c);
        if (!call->plan) return -1;
    } else {
        call->colors = ((uniform_object *)uniform)->per_channel;
    }
    return chn_image_load(image);
}

PyObject *chn_quantize(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    static const char *const names[] = {"img", "quant"};
    PyObject *values[2], *uniform, *palette, *result = NULL;
    chn_image image;
    if (chn_parse(args, kwargs, "quantize", names, 2, 2, values) ||
        chn_image_extract(values[0], "img", &image)) return NULL;
    dither_call call = {0};
    if (!extract_quant(values[1], &uniform, &palette)) {
        if (uniform) {
            /* quantize_ndim: every sample, any channel count. */
            call.colors = ((uniform_object *)uniform)->per_channel;
            if (!chn_image_load(&image)) result = run_dither(&image, &call);
        } else if (!prepare(&image, NULL, palette, &call)) {
            result = run_dither(&image, &call);
        }
    }
    chn_image_release(&image);
    return result;
}

PyObject *chn_ordered_dither(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    static const char *const names[] = {"img", "quant", "map_size"};
    PyObject *values[3], *result = NULL;
    chn_image image;
    uint32_t map_size;
    if (chn_parse(args, kwargs, "ordered_dither", names, 3, 3, values) ||
        chn_image_extract(values[0], "img", &image)) return NULL;
    if (!PyObject_TypeCheck(values[1], chn_uniform_type)) {
        chn_type_error("quant", values[1], "UniformQuantization");
    } else if (!chn_u32(values[2], "map_size", &map_size)) {
        if (!map_size || (map_size & (map_size - 1))) {
            PyErr_SetString(PyExc_ValueError, "Argument 'map_size' must be a power of 2.");
        } else if (!chn_image_load(&image)) {
            dither_call call = {0};
            call.colors = ((uniform_object *)values[1])->per_channel;
            call.mode = 1;
            call.map_size = map_size;
            result = run_dither(&image, &call);
        }
    }
    chn_image_release(&image);
    return result;
}

PyObject *chn_error_diffusion_dither(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    static const char *const names[] = {"img", "quant", "algorithm"};
    PyObject *values[3], *uniform, *palette, *result = NULL;
    chn_image image;
    long algorithm;
    if (chn_parse(args, kwargs, "error_diffusion_dither", names, 3, 3, values) ||
        chn_image_extract(values[0], "img", &image)) return NULL;
    if (!extract_quant(values[1], &uniform, &palette) &&
        !chn_enum_value(values[2], chn_diffusion_type, "algorithm", "DiffusionAlgorithm", &algorithm)) {
        dither_call call = {0};
        call.mode = 2;
        call.algorithm = (int)algorithm;
        if (!prepare(&image, uniform, palette, &call)) result = run_dither(&image, &call);
    }
    chn_image_release(&image);
    return result;
}

PyObject *chn_riemersma_dither(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    static const char *const names[] = {"img", "quant", "history_length", "decay_ratio"};
    PyObject *values[4], *uniform, *palette, *result = NULL;
    chn_image image;
    uint32_t history;
    float decay;
    if (chn_parse(args, kwargs, "riemersma_dither", names, 4, 4, values) ||
        chn_image_extract(values[0], "img", &image)) return NULL;
    if (!extract_quant(values[1], &uniform, &palette) &&
        !chn_u32(values[2], "history_length", &history) && !chn_f32(values[3], "decay_ratio", &decay)) {
        dither_call call = {0};
        call.mode = 3;
        call.history = history;
        call.decay = decay;
        if (history < 2) {
            PyErr_SetString(PyExc_ValueError, "Argument 'history_length' must be at least 2.");
        } else if (!prepare(&image, uniform, palette, &call)) {
            /* riemersma_dither's assert, before any pixel (the kernel repeats it). */
            float base = expf(logf(decay) / ((float)history - 1.0f));
            if (!(0.0f < base && base < 1.0f))
                chn_panic("assertion failed: 0.0 < base && base < 1.0");
            else
                result = run_dither(&image, &call);
        }
    }
    chn_image_release(&image);
    return result;
}

/* ---------------------------------------------------------------------------------- */

int chn_dither_init(PyObject *module)
{
    PyType_Slot uniform_slots[] = {
        {Py_tp_new, (void *)uniform_new},
        {Py_tp_getset, uniform_getset},
        {Py_tp_doc, "UniformQuantization(colors_per_channel)\n--\n\n"},
        {0, NULL},
    };
    PyType_Spec uniform_spec = {CHN_TYPE_NAME("UniformQuantization"), sizeof(uniform_object), 0,
        Py_TPFLAGS_DEFAULT, uniform_slots};
    PyType_Slot palette_slots[] = {
        {Py_tp_new, (void *)palette_new},
        {Py_tp_dealloc, (void *)palette_dealloc},
        {Py_tp_getset, palette_getset},
        {Py_tp_doc, "PaletteQuantization(palette)\n--\n\n"},
        {0, NULL},
    };
    PyType_Spec palette_spec = {CHN_TYPE_NAME("PaletteQuantization"), sizeof(palette_object), 0,
        Py_TPFLAGS_DEFAULT, palette_slots};
    static const chn_enum_member algorithms[] = {
        {"FloydSteinberg", 0}, {"JarvisJudiceNinke", 1}, {"Stucki", 2}, {"Atkinson", 3},
        {"Burkes", 4}, {"Sierra", 5}, {"TwoRowSierra", 6}, {"SierraLite", 7},
    };
    chn_uniform_type = (PyTypeObject *)PyType_FromSpec(&uniform_spec);
    chn_palette_type = (PyTypeObject *)PyType_FromSpec(&palette_spec);
    chn_diffusion_type = chn_enum_type(CHN_TYPE_NAME("DiffusionAlgorithm"), algorithms,
        sizeof(algorithms) / sizeof(algorithms[0]));
    if (!chn_uniform_type || !chn_palette_type || !chn_diffusion_type) return -1;
    Py_INCREF((PyObject *)chn_uniform_type);
    Py_INCREF((PyObject *)chn_palette_type);
    Py_INCREF((PyObject *)chn_diffusion_type);
    if (PyModule_AddObject(module, "UniformQuantization", (PyObject *)chn_uniform_type) < 0 ||
        PyModule_AddObject(module, "PaletteQuantization", (PyObject *)chn_palette_type) < 0 ||
        PyModule_AddObject(module, "DiffusionAlgorithm", (PyObject *)chn_diffusion_type) < 0)
        return -1;
    return 0;
}
