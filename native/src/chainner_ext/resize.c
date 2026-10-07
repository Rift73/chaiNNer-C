/* Port of chainner_ext 0.3.10 crates/bindings/src/resize.rs, MIT OR Apache-2.0. */
/* ResizeFilter and resize over resample_ops.c (nearest) and resample_filters.c (the
 * filtered two-pass resampler, with resize.rs's gamma and clipping rules). */
#include "ext.h"

PyTypeObject *chn_filter_type;

/* ResizeFilter's value -> cn_resample_filtered's filter id (nodes/impl/resize.py's
   ResizeFilter values); 0 is Nearest, which resamples through cn_resample_nearest. */
static const int filter_ids[12] = {0, 2, 3, 6, 1, 11, 7, 10, 4, 5, 8, 9};
#define FILTER_LINEAR 1

PyObject *chn_resize(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    static const char *const names[] = {"img", "new_size", "filter", "gamma_correction"};
    PyObject *values[4];
    chn_image image;
    uint32_t new_width = 0, new_height = 0;
    long filter = 0;
    int gamma = 0;
    if (chn_parse(args, kwargs, "resize", names, 4, 4, values) ||
        chn_image_extract(values[0], "img", &image)) return NULL;
    PyObject *size = values[1];
    int failed = 1;
    if (!PyTuple_Check(size)) {
        chn_type_error("new_size", size, "PyTuple");
    } else if (PyTuple_Size(size) != 2) {
        PyErr_Format(PyExc_ValueError, "expected tuple of length 2, but got tuple of length %zd",
            PyTuple_Size(size));
    } else if (!chn_u32(PyTuple_GetItem(size, 0), "new_size", &new_width) &&
               !chn_u32(PyTuple_GetItem(size, 1), "new_size", &new_height) &&
               !chn_enum_value(values[2], chn_filter_type, "filter", "ResizeFilter", &filter) &&
               !chn_bool(values[3], "gamma_correction", &gamma)) {
        failed = 0;
    }
    if (failed) { chn_image_release(&image); return NULL; }
    if (filter == 0) gamma = 0; /* no point in gamma correction without interpolation */
    size_t c = image.channels, h = image.height, w = image.width;
    if (c < 1 || c > 4) {
        chn_image_release(&image);
        chn_channel_error("1, 2, 3, or 4", c);
        return NULL;
    }
    int contiguous = chn_image_c_contiguous(&image);
    if (chn_image_load(&image)) { chn_image_release(&image); return NULL; }
    float *out;
    PyObject *result = NULL;
    if (new_width && new_height && (!h || !w)) {
        /* An empty source: scale's nearest_neighbor panics, the resize crate refuses. */
        if (filter == 0)
            chn_panic(w ? "index out of bounds: the len is 0 but the index is 0" :
                          "attempt to divide by zero");
        else
            PyErr_Format(PyExc_ValueError, "Not enough memory to allocate a %lux%lu image.",
                (unsigned long)new_width, (unsigned long)new_height);
        chn_image_release(&image);
        return NULL;
    }
    result = chn_new_array(new_height, new_width, c, &out);
    if (result && new_width && new_height) {
        /* Vec4 (glam's unordered clamp) only for a strided RGBA upscale of the
           non-gamma path, where resize.rs finds the copy worth it. */
        double factor = (double)new_width * (double)new_height / ((double)h * (double)w);
        int vector_clip = !gamma && c == 4 && !contiguous && factor >= 1.99;
        int id = filter_ids[filter];
        cn_status status;
        Py_BEGIN_ALLOW_THREADS
        status = filter == 0 ?
            cn_resample_nearest(image.data, out, h, w, c, new_height, new_width) :
            cn_resample_filtered(image.data, out, h, w, c, new_height, new_width, id, gamma, vector_clip);
        Py_END_ALLOW_THREADS
        if (chn_status(status)) Py_CLEAR(result);
    }
    chn_image_release(&image);
    return result;
}

int chn_resize_init(PyObject *module)
{
    static const chn_enum_member filters[] = {
        {"Nearest", 0}, {"Box", 8}, {"Linear", FILTER_LINEAR}, {"Hermite", 9}, {"CubicCatrom", 2},
        {"CubicMitchell", 3}, {"CubicBSpline", 6}, {"Hamming", 10}, {"Hann", 11}, {"Lanczos", 4},
        {"Lagrange", 7}, {"Gauss", 5},
    };
    chn_filter_type = chn_enum_type(CHN_TYPE_NAME("ResizeFilter"), filters,
        sizeof(filters) / sizeof(filters[0]));
    if (!chn_filter_type) return -1;
    Py_INCREF((PyObject *)chn_filter_type);
    return PyModule_AddObject(module, "ResizeFilter", (PyObject *)chn_filter_type);
}
