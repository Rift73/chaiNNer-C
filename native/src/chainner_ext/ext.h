/* Port of chainner_ext 0.3.10 crates/bindings/src (convert.rs and pyo3 0.20's argument
 * extraction), MIT OR Apache-2.0. Shared declarations of the module's units. */
#ifndef CHAINNER_EXT_EXT_H
#define CHAINNER_EXT_EXT_H
#include <Python.h>
#include "chainner_ext_kernels.h"
#include <stddef.h>
#include <stdint.h>

/* pyo3 names a class without a module "builtins.<name>": __module__ is builtins. */
#define CHN_TYPE_NAME(name) "builtins." name

/* lib.c: pyo3_runtime.PanicException, raised with Rust's panic message. */
extern PyObject *chn_panic_type;
void chn_panic(const char *message);
PyObject *chn_panic_format(const char *format, ...);
/* A tp_new for classes without #[new]: TypeError "No constructor defined". */
PyObject *chn_no_constructor(PyTypeObject *type, PyObject *args, PyObject *kwargs);

/* convert.c: pyo3's argument collection. names[0..count) in order, the first `required`
 * mandatory; out[i] is a borrowed reference or NULL. */
int chn_parse(PyObject *args, PyObject *kwargs, const char *function,
    const char *const *names, int count, int required, PyObject **out);
/* FromPyObject conversions; TypeErrors carry "argument '<name>': ". */
int chn_f32(PyObject *value, const char *name, float *out);
int chn_u32(PyObject *value, const char *name, uint32_t *out);
int chn_u64(PyObject *value, const char *name, uint64_t *out);
int chn_bool(PyObject *value, const char *name, int *out);
const char *chn_utf8(PyObject *value, const char *name, Py_ssize_t *length);
int chn_type_error(const char *name, PyObject *value, const char *target);

/* PyImage (PyReadonlyArray2/3<f32>), viewed through the buffer protocol. */
typedef struct chn_image {
    Py_buffer view;
    int held;
    size_t height, width, channels;
    int ndim;
    const float *data; /* C-contiguous and aligned (view.buf or owned) */
    float *owned;
} chn_image;
int chn_image_extract(PyObject *value, const char *name, chn_image *image);
/* LoadImage: data C-contiguous (as_contiguous); a copy only when the view is not. */
int chn_image_load(chn_image *image);
int chn_image_c_contiguous(const chn_image *image);
void chn_image_release(chn_image *image);
/* "Image does not have the right shape. Expected 1, 3, 4 channel(s) but found N." */
int chn_shape_mismatch(const char *expected, size_t channels);
/* "Argument 'img' does not have the right shape. Expected <expected> channels but found N." */
int chn_channel_error(const char *expected, size_t channels);

/* A new float32 (h, w, c) numpy array (numpy.empty) and its writable data. */
PyObject *chn_new_array(size_t height, size_t width, size_t channels, float **data);
/* A kernel status as a Python error (0 when CN_OK). */
int chn_status(cn_status status);
size_t chn_count(const chn_image *image);

/* The module's classes (created at module init). */
extern PyTypeObject *chn_uniform_type, *chn_palette_type, *chn_diffusion_type,
    *chn_filter_type, *chn_clipboard_type, *chn_regex_type, *chn_match_type, *chn_group_type;

/* dither.c */
int chn_dither_init(PyObject *module);
PyObject *chn_quantize(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *chn_ordered_dither(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *chn_error_diffusion_dither(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *chn_riemersma_dither(PyObject *self, PyObject *args, PyObject *kwargs);
/* pixel_art.c */
PyObject *chn_pixel_art_upscale(PyObject *self, PyObject *args, PyObject *kwargs);
/* resize.c */
int chn_resize_init(PyObject *module);
PyObject *chn_resize(PyObject *self, PyObject *args, PyObject *kwargs);
/* clipboard.c */
int chn_clipboard_init(PyObject *module);
/* regex.c */
int chn_regex_init(PyObject *module);
/* The int-valued pyo3 enum classes (DiffusionAlgorithm, ResizeFilter). */
typedef struct chn_enum_member { const char *name; long value; } chn_enum_member;
PyTypeObject *chn_enum_type(const char *qualified, const chn_enum_member *members, size_t count);
int chn_enum_value(PyObject *value, PyTypeObject *type, const char *name, const char *target, long *out);
#endif
