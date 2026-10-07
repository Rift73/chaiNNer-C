/* Port of chainner_ext 0.3.10 crates/bindings/src/pixel_art.rs, MIT OR Apache-2.0. */
/* pixel_art_upscale over pixel_art_ops.cpp's eleven modes. */
#include "ext.h"
#include <string.h>

typedef struct pixel_art_mode { const char *algorithm; uint32_t scale; int mode; } pixel_art_mode;

static const pixel_art_mode modes[] = {
    {"adv_mame", 2, 0}, {"adv_mame", 3, 1}, {"adv_mame", 4, 2}, {"eagle", 2, 3}, {"eagle", 3, 4},
    {"super_eagle", 2, 5}, {"sai", 2, 6}, {"super_sai", 2, 7}, {"hqx", 2, 8}, {"hqx", 3, 9},
    {"hqx", 4, 10},
};

PyObject *chn_pixel_art_upscale(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    static const char *const names[] = {"img", "algorithm", "scale"};
    PyObject *values[3];
    chn_image image;
    Py_ssize_t length;
    uint32_t scale;
    if (chn_parse(args, kwargs, "pixel_art_upscale", names, 3, 3, values) ||
        chn_image_extract(values[0], "img", &image)) return NULL;
    const char *algorithm = chn_utf8(values[1], "algorithm", &length);
    if (!algorithm || chn_u32(values[2], "scale", &scale)) {
        chn_image_release(&image);
        return NULL;
    }
    size_t c = image.channels;
    if (c != 1 && c != 3 && c != 4) {
        chn_image_release(&image);
        chn_channel_error("1, 3, or 4", c);
        return NULL;
    }
    int mode = -1, known = 0;
    for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); ++i) {
        if (strlen(modes[i].algorithm) != (size_t)length ||
            memcmp(modes[i].algorithm, algorithm, (size_t)length)) continue;
        known = 1;
        if (modes[i].scale == scale) mode = modes[i].mode;
    }
    if (mode < 0) {
        chn_image_release(&image);
        if (known)
            PyErr_Format(PyExc_ValueError,
                "Scale %lu is not supported for pixel art upscaling algorithm '%U'.",
                (unsigned long)scale, values[1]);
        else
            PyErr_Format(PyExc_ValueError, "Unknown pixel art upscaling algorithm '%U'.", values[1]);
        return NULL;
    }
    if (chn_image_load(&image)) { chn_image_release(&image); return NULL; }
    float *out;
    PyObject *result = chn_new_array(image.height * scale, image.width * scale, c, &out);
    if (result && chn_count(&image)) {
        cn_status status;
        Py_BEGIN_ALLOW_THREADS
        status = cn_pixel_art_f32(image.data, out, image.height, image.width, c, mode);
        Py_END_ALLOW_THREADS
        if (chn_status(status)) Py_CLEAR(result);
    }
    chn_image_release(&image);
    return result;
}
