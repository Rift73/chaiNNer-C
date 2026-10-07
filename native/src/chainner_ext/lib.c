/* Port of chainner_ext 0.3.10 crates/bindings/src/lib.rs, MIT OR Apache-2.0. */
/* The module: every name of the 0.3.10 .pyi in pyo3's registration order (__all__),
 * pyo3_runtime.PanicException, the _C_API capsule (regex.c), and lib.rs's own functions
 * (fill_alpha_*, binary_threshold, esdf, fast_gamma) over chainner_native's kernels.
 * It builds twice: chainner_ext.pyd and, with CHAINNER_EXT_HARNESS, chainner_ext_c.pyd. */
#include "ext.h"
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#ifdef CHAINNER_EXT_HARNESS
#define CHN_EXT_MODULE_NAME "chainner_ext_c"
#define CHN_EXT_MODULE_INIT PyInit_chainner_ext_c
#else
#define CHN_EXT_MODULE_NAME "chainner_ext"
#define CHN_EXT_MODULE_INIT PyInit_chainner_ext
#endif

PyObject *chn_panic_type;

void chn_panic(const char *message)
{
    PyErr_SetString(chn_panic_type, message);
}

PyObject *chn_panic_format(const char *format, ...)
{
    va_list arguments;
    va_start(arguments, format);
    PyObject *message = PyUnicode_FromFormatV(format, arguments);
    va_end(arguments);
    if (message) {
        PyErr_SetObject(chn_panic_type, message);
        Py_DECREF(message);
    }
    return NULL;
}

PyObject *chn_no_constructor(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    (void)type; (void)args; (void)kwargs;
    PyErr_SetString(PyExc_TypeError, "No constructor defined");
    return NULL;
}

/* ---------------------------------------------------------------------------------- */
/* pyo3 0.20's #[pyclass] enum: singleton class attributes, __int__, __repr__
 * "<Class>.<Variant>", == and != against the class and integers, no __hash__. */

typedef struct chn_enum_object {
    PyObject_HEAD
    long value;
    const char *name;
} chn_enum_object;

static PyObject *enum_repr(PyObject *self)
{
    chn_enum_object *member = (chn_enum_object *)self;
    PyObject *qualname = PyObject_GetAttrString((PyObject *)Py_TYPE(self), "__qualname__");
    if (!qualname) return NULL;
    PyObject *result = PyUnicode_FromFormat("%U.%s", qualname, member->name);
    Py_DECREF(qualname);
    return result;
}

static PyObject *enum_int(PyObject *self)
{
    return PyLong_FromLong(((chn_enum_object *)self)->value);
}

static PyObject *enum_richcompare(PyObject *self, PyObject *other, int op)
{
    if (op != Py_EQ && op != Py_NE) Py_RETURN_NOTIMPLEMENTED;
    long mine = ((chn_enum_object *)self)->value;
    long theirs;
    PyObject *index = PyNumber_Index(other);
    if (index) {
        Py_ssize_t number = PyLong_AsSsize_t(index);
        Py_DECREF(index);
        if (number == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            Py_RETURN_NOTIMPLEMENTED;
        }
        theirs = (long)number;
        if ((Py_ssize_t)theirs != number) return PyBool_FromLong(op == Py_NE);
    } else {
        PyErr_Clear();
        if (Py_TYPE(other) != Py_TYPE(self)) Py_RETURN_NOTIMPLEMENTED;
        theirs = ((chn_enum_object *)other)->value;
    }
    return PyBool_FromLong(op == Py_EQ ? mine == theirs : mine != theirs);
}

PyTypeObject *chn_enum_type(const char *qualified, const chn_enum_member *members, size_t count)
{
    PyType_Slot slots[] = {
        {Py_tp_new, (void *)chn_no_constructor},
        {Py_tp_repr, (void *)enum_repr},
        {Py_nb_int, (void *)enum_int},
        {Py_tp_richcompare, (void *)enum_richcompare},
        {Py_tp_hash, (void *)PyObject_HashNotImplemented},
        {0, NULL},
    };
    PyType_Spec spec = {qualified, sizeof(chn_enum_object), 0, Py_TPFLAGS_DEFAULT, slots};
    PyTypeObject *type = (PyTypeObject *)PyType_FromSpec(&spec);
    if (!type) return NULL;
    for (size_t i = 0; i < count; ++i) {
        chn_enum_object *member = PyObject_New(chn_enum_object, type);
        if (!member) { Py_DECREF(type); return NULL; }
        member->value = members[i].value;
        member->name = members[i].name;
        int status = PyObject_SetAttrString((PyObject *)type, members[i].name, (PyObject *)member);
        Py_DECREF(member);
        if (status < 0) { Py_DECREF(type); return NULL; }
    }
    return type;
}

int chn_enum_value(PyObject *value, PyTypeObject *type, const char *name, const char *target,
    long *out)
{
    int is_member = PyObject_TypeCheck(value, type);
    if (!is_member) return chn_type_error(name, value, target);
    *out = ((chn_enum_object *)value)->value;
    return 0;
}

/* ---------------------------------------------------------------------------------- */
/* fill_alpha: Image<Vec4> (1 -> [g, g, g, 1], 3 -> [r, g, b, 1], 4 as is). */

static int load_rgba(chn_image *image, float **rgba)
{
    *rgba = NULL;
    size_t c = image->channels;
    if (c != 1 && c != 3 && c != 4) return chn_shape_mismatch("1, 3, 4", c);
    if (chn_image_load(image)) return -1;
    if (c == 4) return 0;
    size_t pixels = image->height * image->width;
    *rgba = malloc(pixels ? pixels * 4 * sizeof(float) : sizeof(float));
    if (!*rgba) { PyErr_NoMemory(); return -1; }
    for (size_t p = 0; p < pixels; ++p) {
        const float *source = image->data + p * c;
        float *target = *rgba + p * 4;
        target[0] = source[0];
        target[1] = c == 1 ? source[0] : source[1];
        target[2] = c == 1 ? source[0] : source[2];
        target[3] = 1.0f;
    }
    return 0;
}

enum fill_mode { FILL_FRAGMENT, FILL_EXTEND, FILL_NEAREST };

static PyObject *fill_alpha(PyObject *args, PyObject *kwargs, enum fill_mode mode)
{
    static const char *const fragment[] = {"img", "threshold", "iterations", "fragment_count"};
    static const char *const extend[] = {"img", "threshold", "iterations"};
    static const char *const nearest[] = {"img", "threshold", "min_radius", "anti_aliasing"};
    const char *const *names = mode == FILL_FRAGMENT ? fragment : mode == FILL_EXTEND ? extend : nearest;
    const char *function = mode == FILL_FRAGMENT ? "fill_alpha_fragment_blur" :
        mode == FILL_EXTEND ? "fill_alpha_extend_color" : "fill_alpha_nearest_color";
    int count = mode == FILL_EXTEND ? 3 : 4;
    PyObject *values[4];
    if (chn_parse(args, kwargs, function, names, count, count, values)) return NULL;
    chn_image image;
    float threshold;
    uint32_t iterations = 0, extra = 0;
    int anti_aliasing = 0;
    if (chn_image_extract(values[0], "img", &image)) return NULL;
    if (chn_f32(values[1], "threshold", &threshold) ||
        chn_u32(values[2], names[2], &iterations) ||
        (mode == FILL_FRAGMENT && chn_u32(values[3], "fragment_count", &extra)) ||
        (mode == FILL_NEAREST && chn_bool(values[3], "anti_aliasing", &anti_aliasing))) {
        chn_image_release(&image);
        return NULL;
    }
    float *rgba;
    if (load_rgba(&image, &rgba)) { chn_image_release(&image); return NULL; }
    /* fragment_blur_alpha's asserts run on the first iteration, whatever the size. */
    if (mode == FILL_FRAGMENT && iterations && (extra < 1 || extra > 255)) {
        free(rgba); chn_image_release(&image);
        chn_panic(extra < 1 ? "assertion failed: count >= 1" : "assertion failed: count <= 255");
        return NULL;
    }
    float *out;
    PyObject *result = chn_new_array(image.height, image.width, 4, &out);
    if (result && image.height * image.width) {
        const float *source = rgba ? rgba : image.data;
        cn_status status;
        Py_BEGIN_ALLOW_THREADS
        if (mode == FILL_FRAGMENT)
            status = cn_alpha_fragment(source, out, image.height, image.width, threshold, iterations, extra);
        else if (mode == FILL_EXTEND)
            status = cn_alpha_extend(source, out, image.height, image.width, threshold, iterations);
        else
            status = cn_alpha_nearest(source, out, image.height, image.width, threshold, iterations, anti_aliasing);
        Py_END_ALLOW_THREADS
        if (chn_status(status)) Py_CLEAR(result);
    }
    free(rgba);
    chn_image_release(&image);
    return result;
}

static PyObject *fill_alpha_fragment_blur(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    return fill_alpha(args, kwargs, FILL_FRAGMENT);
}

static PyObject *fill_alpha_extend_color(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    return fill_alpha(args, kwargs, FILL_EXTEND);
}

static PyObject *fill_alpha_nearest_color(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    return fill_alpha(args, kwargs, FILL_NEAREST);
}

/* ---------------------------------------------------------------------------------- */

static PyObject *binary_threshold(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    static const char *const names[] = {"img", "threshold", "anti_aliasing", "extra_smoothness"};
    PyObject *values[4];
    if (chn_parse(args, kwargs, "binary_threshold", names, 4, 3, values)) return NULL;
    chn_image image;
    float threshold, smoothness = 0.0f;
    int anti_aliasing;
    if (chn_image_extract(values[0], "img", &image)) return NULL;
    if (chn_f32(values[1], "threshold", &threshold) ||
        chn_bool(values[2], "anti_aliasing", &anti_aliasing) ||
        (values[3] && values[3] != Py_None && chn_f32(values[3], "extra_smoothness", &smoothness)) ||
        chn_image_load(&image)) {
        chn_image_release(&image);
        return NULL;
    }
    float *out;
    PyObject *result = chn_new_array(image.height, image.width, image.channels, &out);
    size_t count = chn_count(&image);
    if (result && count) {
        cn_status status;
        Py_BEGIN_ALLOW_THREADS
        status = anti_aliasing ?
            cn_threshold_aa(image.data, out, image.height, image.width, image.channels, threshold, smoothness) :
            cn_threshold_hard(image.data, out, count, 0, threshold, 1.0f, 0);
        Py_END_ALLOW_THREADS
        if (chn_status(status)) Py_CLEAR(result);
    }
    chn_image_release(&image);
    return result;
}

static PyObject *esdf(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    static const char *const names[] = {"img", "radius", "cutoff", "pre_process", "post_process"};
    PyObject *values[5];
    if (chn_parse(args, kwargs, "esdf", names, 5, 5, values)) return NULL;
    chn_image image;
    float radius, cutoff;
    int pre, post;
    if (chn_image_extract(values[0], "img", &image)) return NULL;
    if (chn_f32(values[1], "radius", &radius) || chn_f32(values[2], "cutoff", &cutoff) ||
        chn_bool(values[3], "pre_process", &pre) || chn_bool(values[4], "post_process", &post)) {
        chn_image_release(&image);
        return NULL;
    }
    if (image.channels != 1) {
        chn_shape_mismatch("1", image.channels);
        chn_image_release(&image);
        return NULL;
    }
    if (chn_image_load(&image)) { chn_image_release(&image); return NULL; }
    if (!image.height || !image.width) {
        /* esdt1d reads xs[offset] of an empty stage. */
        chn_image_release(&image);
        chn_panic("index out of bounds: the len is 0 but the index is 0");
        return NULL;
    }
    float *out;
    PyObject *result = chn_new_array(image.height, image.width, 1, &out);
    if (result) {
        cn_status status;
        Py_BEGIN_ALLOW_THREADS
        status = cn_distance_esdf_ex(image.data, out, image.height, image.width, radius, cutoff, pre, post);
        Py_END_ALLOW_THREADS
        if (chn_status(status)) Py_CLEAR(result);
    }
    chn_image_release(&image);
    return result;
}

static PyObject *fast_gamma(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    static const char *const names[] = {"img", "gamma"};
    PyObject *values[2];
    if (chn_parse(args, kwargs, "fast_gamma", names, 2, 2, values)) return NULL;
    chn_image image;
    float gamma;
    if (chn_image_extract(values[0], "img", &image)) return NULL;
    if (chn_f32(values[1], "gamma", &gamma) || chn_image_load(&image)) {
        chn_image_release(&image);
        return NULL;
    }
    float *out;
    PyObject *result = chn_new_array(image.height, image.width, image.channels, &out);
    if (result && chn_count(&image)) {
        cn_status status;
        Py_BEGIN_ALLOW_THREADS
        status = cn_alpha_gamma(image.data, out, image.height * image.width, image.channels, gamma);
        Py_END_ALLOW_THREADS
        if (chn_status(status)) Py_CLEAR(result);
    }
    chn_image_release(&image);
    return result;
}

/* ---------------------------------------------------------------------------------- */

#define FILL_DOC "Fill the transparent pixels in the given image with nearby colors."
#define FUNCTION(name, signature, doc) \
    {#name, (PyCFunction)(void (*)(void))name, METH_VARARGS | METH_KEYWORDS, \
     #name signature "\n--\n\n" doc}
#define BOUND(name, function, signature) \
    {name, (PyCFunction)(void (*)(void))function, METH_VARARGS | METH_KEYWORDS, \
     name signature "\n--\n\n"}

static PyMethodDef module_functions[] = {
    BOUND("quantize", chn_quantize, "(img, quant)"),
    BOUND("error_diffusion_dither", chn_error_diffusion_dither, "(img, quant, algorithm)"),
    BOUND("ordered_dither", chn_ordered_dither, "(img, quant, map_size)"),
    BOUND("riemersma_dither", chn_riemersma_dither, "(img, quant, history_length, decay_ratio)"),
    BOUND("pixel_art_upscale", chn_pixel_art_upscale, "(img, algorithm, scale)"),
    BOUND("resize", chn_resize, "(img, new_size, filter, gamma_correction)"),
    FUNCTION(fill_alpha_fragment_blur, "(img, threshold, iterations, fragment_count)", FILL_DOC),
    FUNCTION(fill_alpha_extend_color, "(img, threshold, iterations)", FILL_DOC),
    FUNCTION(fill_alpha_nearest_color, "(img, threshold, min_radius, anti_aliasing)", FILL_DOC),
    FUNCTION(binary_threshold, "(img, threshold, anti_aliasing, extra_smoothness=None)", FILL_DOC),
    FUNCTION(esdf, "(img, radius, cutoff, pre_process, post_process)", FILL_DOC),
    BOUND("fast_gamma", fast_gamma, "(img, gamma)"),
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef chn_ext_module = {
    PyModuleDef_HEAD_INIT,
    CHN_EXT_MODULE_NAME,                    /* m_name */
    "A Python module implemented in Rust.", /* m_doc */
    -1,                                     /* m_size: single-phase, process-wide types */
    module_functions,                       /* m_methods */
    NULL,                                   /* m_slots */
    NULL,                                   /* m_traverse */
    NULL,                                   /* m_clear */
    NULL,                                   /* m_free */
};

/* pyo3's `m.add` order: classes and functions as lib.rs registers them. */
static const char *const all_names[] = {
    "RustRegex", "MatchGroup", "RegexMatch", "Clipboard", "DiffusionAlgorithm",
    "UniformQuantization", "PaletteQuantization", "quantize", "error_diffusion_dither",
    "ordered_dither", "riemersma_dither", "pixel_art_upscale", "ResizeFilter", "resize",
    "fill_alpha_fragment_blur", "fill_alpha_extend_color", "fill_alpha_nearest_color",
    "binary_threshold", "esdf", "fast_gamma",
};

#define PANIC_DOC "\nThe exception raised when Rust code called from Python panics.\n\n" \
    "Like SystemExit, this exception is derived from BaseException so that\n" \
    "it will typically propagate all the way through the stack and cause the\n" \
    "Python interpreter to exit.\n"

PyMODINIT_FUNC CHN_EXT_MODULE_INIT(void)
{
    if (!chn_panic_type) {
        chn_panic_type = PyErr_NewExceptionWithDoc("pyo3_runtime.PanicException", PANIC_DOC,
            PyExc_BaseException, NULL);
        if (!chn_panic_type) return NULL;
    }
    PyObject *module = PyModule_Create(&chn_ext_module);
    if (!module) return NULL;
    if (chn_regex_init(module) || chn_clipboard_init(module) || chn_dither_init(module) ||
        chn_resize_init(module)) {
        Py_DECREF(module);
        return NULL;
    }
    PyObject *all = PyList_New(0);
    if (!all) { Py_DECREF(module); return NULL; }
    for (size_t i = 0; i < sizeof(all_names) / sizeof(all_names[0]); ++i) {
        PyObject *name = PyUnicode_FromString(all_names[i]);
        if (!name || PyList_Append(all, name) < 0) {
            Py_XDECREF(name); Py_DECREF(all); Py_DECREF(module);
            return NULL;
        }
        Py_DECREF(name);
    }
    if (PyModule_AddObject(module, "__all__", all) < 0) {
        Py_DECREF(all); Py_DECREF(module);
        return NULL;
    }
    /* Not in __all__: the type the 0.3.10 module raises, for chaiNNer-C's own callers. */
    Py_INCREF(chn_panic_type);
    if (PyModule_AddObject(module, "PanicException", chn_panic_type) < 0) {
        Py_DECREF(chn_panic_type); Py_DECREF(module);
        return NULL;
    }
    return module;
}
