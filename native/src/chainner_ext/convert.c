/* Port of chainner_ext 0.3.10 crates/bindings/src/convert.rs, MIT OR Apache-2.0. */
/* PyImage over the buffer protocol (limited API, no NumPy C API), pyo3 0.20's argument
 * collection and FromPyObject conversions, and numpy.empty outputs. pyo3's
 * argument-conversion TypeErrors are matched by type; ValueErrors by message. */
#include "ext.h"
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

int chn_type_error(const char *name, PyObject *value, const char *target)
{
    PyObject *type = PyObject_GetAttrString((PyObject *)Py_TYPE(value), "__name__");
    if (!type) return -1;
    PyErr_Format(PyExc_TypeError, "argument '%s': '%U' object cannot be converted to '%s'",
        name, type, target);
    Py_DECREF(type);
    return -1;
}

/* pyo3's argument_extraction_error: a TypeError gains "argument '<name>': ". */
static int remap_type_error(const char *name)
{
    if (!PyErr_ExceptionMatches(PyExc_TypeError)) return -1;
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    PyObject *text = value ? PyObject_Str(value) : NULL;
    Py_XDECREF(type); Py_XDECREF(value); Py_XDECREF(traceback);
    if (!text) return -1;
    PyErr_Format(PyExc_TypeError, "argument '%s': %U", name, text);
    Py_DECREF(text);
    return -1;
}

int chn_parse(PyObject *args, PyObject *kwargs, const char *function,
    const char *const *names, int count, int required, PyObject **out)
{
    Py_ssize_t positional = PyTuple_Size(args);
    if (positional < 0) return -1;
    for (int i = 0; i < count; ++i) out[i] = NULL;
    if (positional > count) {
        PyErr_Format(PyExc_TypeError, "%s() takes %d positional arguments but %zd were given",
            function, count, positional);
        return -1;
    }
    for (Py_ssize_t i = 0; i < positional; ++i) out[i] = PyTuple_GetItem(args, i);
    if (kwargs) {
        PyObject *key, *value;
        Py_ssize_t position = 0;
        while (PyDict_Next(kwargs, &position, &key, &value)) {
            int found = -1;
            for (int i = 0; i < count && found < 0; ++i) {
                int equal = PyUnicode_CompareWithASCIIString(key, names[i]) == 0;
                if (PyErr_Occurred()) return -1;
                if (equal) found = i;
            }
            if (found < 0) {
                PyErr_Format(PyExc_TypeError, "%s() got an unexpected keyword argument '%S'",
                    function, key);
                return -1;
            }
            if (out[found]) {
                PyErr_Format(PyExc_TypeError, "%s() got multiple values for argument '%s'",
                    function, names[found]);
                return -1;
            }
            out[found] = value;
        }
    }
    for (int i = 0; i < required; ++i) {
        if (!out[i]) {
            PyErr_Format(PyExc_TypeError, "%s() missing 1 required positional argument: '%s'",
                function, names[i]);
            return -1;
        }
    }
    return 0;
}

int chn_f32(PyObject *value, const char *name, float *out)
{
    double number = PyFloat_AsDouble(value);
    if (number == -1.0 && PyErr_Occurred()) return remap_type_error(name);
    *out = (float)number; /* `as f32`: round to nearest, overflow to infinity */
    return 0;
}

int chn_u64(PyObject *value, const char *name, uint64_t *out)
{
    PyObject *index = PyNumber_Index(value);
    if (!index) return remap_type_error(name);
    unsigned long long number = PyLong_AsUnsignedLongLong(index);
    Py_DECREF(index);
    if (number == (unsigned long long)-1 && PyErr_Occurred()) return remap_type_error(name);
    *out = number;
    return 0;
}

int chn_u32(PyObject *value, const char *name, uint32_t *out)
{
    uint64_t number;
    if (chn_u64(value, name, &number)) return -1;
    if (number > UINT32_MAX) {
        PyErr_SetString(PyExc_OverflowError, "out of range integral type conversion attempted");
        return -1;
    }
    *out = (uint32_t)number;
    return 0;
}

int chn_bool(PyObject *value, const char *name, int *out)
{
    if (!PyBool_Check(value)) return chn_type_error(name, value, "PyBool");
    *out = value == Py_True;
    return 0;
}

const char *chn_utf8(PyObject *value, const char *name, Py_ssize_t *length)
{
    if (!PyUnicode_Check(value)) {
        chn_type_error(name, value, "PyString");
        return NULL;
    }
    return PyUnicode_AsUTF8AndSize(value, length);
}

/* ---------------------------------------------------------------------------------- */
/* numpy, imported on first use (as rust-numpy does) */

static PyObject *numpy_ndarray, *numpy_empty, *numpy_float32;

static int numpy_ready(void)
{
    if (numpy_empty) return 0;
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (!numpy) return -1;
    PyObject *ndarray = PyObject_GetAttrString(numpy, "ndarray");
    PyObject *empty = PyObject_GetAttrString(numpy, "empty");
    PyObject *float32 = PyObject_GetAttrString(numpy, "float32");
    Py_DECREF(numpy);
    if (!ndarray || !empty || !float32) {
        Py_XDECREF(ndarray); Py_XDECREF(empty); Py_XDECREF(float32);
        return -1;
    }
    numpy_ndarray = ndarray; numpy_float32 = float32; numpy_empty = empty;
    return 0;
}

PyObject *chn_new_array(size_t height, size_t width, size_t channels, float **data)
{
    if (numpy_ready()) return NULL;
    PyObject *shape = Py_BuildValue("(nnn)", (Py_ssize_t)height, (Py_ssize_t)width,
        (Py_ssize_t)channels);
    if (!shape) return NULL;
    PyObject *array = PyObject_CallFunctionObjArgs(numpy_empty, shape, numpy_float32, NULL);
    Py_DECREF(shape);
    if (!array) return NULL;
    Py_buffer view;
    if (PyObject_GetBuffer(array, &view, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) < 0) {
        Py_DECREF(array);
        return NULL;
    }
    *data = view.buf;
    /* The array owns its data; the exporter keeps it alive while the array lives. */
    PyBuffer_Release(&view);
    return array;
}

/* ---------------------------------------------------------------------------------- */
/* PyImage */

static int image_type_error(const char *name, PyObject *value, int ndim, const char *dtype)
{
    /* pyo3's derive(FromPyObject) message for enum PyImage { D2, D3 } (type compared). */
    PyObject *type = PyObject_GetAttrString((PyObject *)Py_TYPE(value), "__name__");
    if (!type) return -1;
    if (ndim < 0)
        PyErr_Format(PyExc_TypeError,
            "argument '%s': failed to extract enum PyImage ('D2 | D3')\n"
            "- variant D2 (D2): TypeError: failed to extract field PyImage::D2.0, caused by "
            "TypeError: '%U' object cannot be converted to 'PyArray<T, D>'\n"
            "- variant D3 (D3): TypeError: failed to extract field PyImage::D3.0, caused by "
            "TypeError: '%U' object cannot be converted to 'PyArray<T, D>'", name, type, type);
    else if (ndim == 2 || ndim == 3)
        PyErr_Format(PyExc_TypeError,
            "argument '%s': failed to extract enum PyImage ('D2 | D3')\n"
            "- variant D%d (D%d): TypeError: type mismatch:\n from=%s, to=float32",
            name, ndim, ndim, dtype);
    else
        PyErr_Format(PyExc_TypeError,
            "argument '%s': failed to extract enum PyImage ('D2 | D3')\n"
            "- dimensionality mismatch:\n from=%d", name, ndim);
    Py_DECREF(type);
    return -1;
}

int chn_image_extract(PyObject *value, const char *name, chn_image *image)
{
    memset(image, 0, sizeof(*image));
    if (numpy_ready()) return -1;
    int is_array = PyObject_IsInstance(value, numpy_ndarray);
    if (is_array < 0) return -1;
    if (!is_array) return image_type_error(name, value, -1, NULL);
    if (PyObject_GetBuffer(value, &image->view, PyBUF_RECORDS_RO) < 0) {
        /* An exporter refusing a strided read view: not a PyArray<f32> pyo3 accepts. */
        PyErr_Clear();
        return image_type_error(name, value, 0, NULL);
    }
    image->held = 1;
    int ndim = image->view.ndim;
    const char *format = image->view.format ? image->view.format : "B";
    int native_f32 = image->view.itemsize == 4 &&
        (!strcmp(format, "f") || !strcmp(format, "<f") || !strcmp(format, "=f") ||
         !strcmp(format, "@f"));
    if ((ndim != 2 && ndim != 3) || !native_f32) {
        char dtype[32];
        strncpy(dtype, format, sizeof(dtype) - 1);
        dtype[sizeof(dtype) - 1] = 0;
        chn_image_release(image);
        return image_type_error(name, value, ndim, dtype);
    }
    image->ndim = ndim;
    image->height = (size_t)image->view.shape[0];
    image->width = (size_t)image->view.shape[1];
    image->channels = ndim == 3 ? (size_t)image->view.shape[2] : 1;
    return 0;
}

int chn_image_c_contiguous(const chn_image *image)
{
    return PyBuffer_IsContiguous(&image->view, 'C');
}

size_t chn_count(const chn_image *image)
{
    return image->height * image->width * image->channels;
}

int chn_image_load(chn_image *image)
{
    if (image->data) return 0;
    size_t count = chn_count(image);
    if (chn_image_c_contiguous(image) && ((uintptr_t)image->view.buf % _Alignof(float)) == 0) {
        image->data = image->view.buf;
        return 0;
    }
    image->owned = malloc(count ? count * sizeof(float) : sizeof(float));
    if (!image->owned) { PyErr_NoMemory(); return -1; }
    if (count && PyBuffer_ToContiguous(image->owned, &image->view,
            (Py_ssize_t)(count * sizeof(float)), 'C') < 0) return -1;
    image->data = image->owned;
    return 0;
}

void chn_image_release(chn_image *image)
{
    if (image->held) PyBuffer_Release(&image->view);
    image->held = 0;
    free(image->owned);
    image->owned = NULL;
    image->data = NULL;
}

int chn_shape_mismatch(const char *expected, size_t channels)
{
    PyErr_Format(PyExc_ValueError,
        "Image does not have the right shape. Expected %s channel(s) but found %zu.",
        expected, channels);
    return -1;
}

int chn_channel_error(const char *expected, size_t channels)
{
    PyErr_Format(PyExc_ValueError,
        "Argument 'img' does not have the right shape. Expected %s channels but found %zu.",
        expected, channels);
    return -1;
}

int chn_status(cn_status status)
{
    switch (status) {
    case CN_OK: return 0;
    case CN_ALLOCATION_FAILED: PyErr_NoMemory(); return -1;
    case CN_SIZE_OVERFLOW:
        PyErr_SetString(PyExc_OverflowError, "Image size exceeds the C kernel addressable buffer range");
        return -1;
    default:
        PyErr_Format(PyExc_ValueError, "Invalid image dimensions or arguments passed to C kernels (%d)",
            (int)status);
        return -1;
    }
}
