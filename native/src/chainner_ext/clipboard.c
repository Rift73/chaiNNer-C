/* Port of chainner_ext 0.3.10 crates/bindings/src/clipboard.rs, MIT OR Apache-2.0. */
/* Clipboard over Win32, with the pinned arboard 3.2 / clipboard-win sequence and texts,
 * its failure messages included (line numbers as upstream). Corrected (Consult 6 D-1): a
 * failed image transfer frees its memory with GlobalFree, where upstream leaks it and calls
 * DeleteObject on the HGLOBAL; the node path (utility_clipboard.cpp) does the same.
 * Pixels: clipboard_ops.c's float32 -> BGRA. */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#include "ext.h"
#include "clipboard_ops.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

PyTypeObject *chn_clipboard_type;

#ifdef _WIN32
#define PREFIX "Unknown error while interacting with the clipboard: "
#define TEXT_FAILURE PREFIX "Could not place the specified text to the clipboard"

typedef struct payload {
    unsigned char *bytes;
    size_t size;
} payload;

/* clipboard-win's ErrorCode display: "OS error <code>: <system message>". */
static void system_error(DWORD code, char *text, size_t capacity)
{
    wchar_t *buffer = NULL;
    DWORD count = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS, NULL, code, 0, (wchar_t *)&buffer, 0, NULL);
    while (count && buffer && (buffer[count - 1] == L'\r' || buffer[count - 1] == L'\n' ||
                               buffer[count - 1] == L' ')) --count;
    char message[256] = {0};
    if (count && buffer)
        WideCharToMultiByte(CP_UTF8, 0, buffer, (int)count, message, (int)sizeof(message) - 1, NULL, NULL);
    if (buffer) LocalFree(buffer);
    snprintf(text, capacity, message[0] ? "OS error %lu: %s" : "OS error %lu", (unsigned long)code, message);
}

/* The publication, without the GIL. Returns NULL or the failure's message (static or
   in `scratch`); an occupied clipboard is reported apart from the "Unknown error" ones. */
static const char *publish(const payload *data, int image, char *scratch, size_t capacity)
{
    int opened = 0;
    for (unsigned attempt = 0; attempt < 6; ++attempt) {
        if (OpenClipboard(NULL)) { opened = 1; break; }
        if (attempt < 5) Sleep(5);
    }
    if (!opened) return "The native clipboard is not accessible due to being held by an other party.";
    const char *failure = NULL;
    if (image && !EmptyClipboard()) {
        char error[256];
        system_error(GetLastError(), error, sizeof(error));
        snprintf(scratch, capacity, PREFIX "Failed to empty the clipboard. Got error code: %s", error);
        CloseClipboard();
        return scratch;
    }
    HGLOBAL memory = GlobalAlloc(GHND, data->size);
    if (!memory) {
        failure = image ? PREFIX "Could not allocate global memory object. GlobalAlloc returned null at line 86."
                        : TEXT_FAILURE;
    } else {
        void *pointer = GlobalLock(memory);
        if (!pointer) {
            GlobalFree(memory); /* corrected: upstream's image branch leaks it (Consult 6 D-1) */
            failure = image ? PREFIX "Could not lock the global memory object at line 94" : TEXT_FAILURE;
        } else {
            memcpy(pointer, data->bytes, data->size);
            GlobalUnlock(memory);
            if (!image) EmptyClipboard(); /* the text path ignores this result */
            if (!SetClipboardData(image ? CF_DIBV5 : CF_UNICODETEXT, memory)) {
                /* corrected: upstream's image branch calls DeleteObject on the HGLOBAL and
                 * leaks it (Consult 6 D-1) */
                GlobalFree(memory);
                failure = image ? PREFIX "Call to `SetClipboardData` returned NULL at line 131" : TEXT_FAILURE;
            }
        }
    }
    CloseClipboard();
    return failure;
}

static PyObject *finish(payload *data, int image)
{
    char scratch[512];
    const char *failure;
    Py_BEGIN_ALLOW_THREADS
    failure = publish(data, image, scratch, sizeof(scratch));
    Py_END_ALLOW_THREADS
    free(data->bytes);
    if (failure) {
        PyErr_SetString(PyExc_ValueError, failure);
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *clipboard_write_text(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    static const char *const names[] = {"text"};
    PyObject *values[1];
    Py_ssize_t length;
    if (chn_parse(args, kwargs, "Clipboard.write_text", names, 1, 1, values)) return NULL;
    const char *text = chn_utf8(values[0], "text", &length);
    if (!text) return NULL;
    if (length > INT_MAX) {
        PyErr_SetString(PyExc_ValueError, TEXT_FAILURE);
        return NULL;
    }
    int units = length ? MultiByteToWideChar(CP_UTF8, 0, text, (int)length, NULL, 0) : 0;
    if (!units && length) {
        PyErr_SetString(PyExc_ValueError, TEXT_FAILURE);
        return NULL;
    }
    payload data = {calloc((size_t)units + 1, sizeof(wchar_t)), ((size_t)units + 1) * sizeof(wchar_t)};
    if (!data.bytes) return PyErr_NoMemory();
    if (units) MultiByteToWideChar(CP_UTF8, 0, text, (int)length, (wchar_t *)data.bytes, units);
    return finish(&data, 0);
}

static PyObject *clipboard_write_image(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self;
    static const char *const names[] = {"image", "pixel_format"};
    PyObject *values[2];
    chn_image image;
    Py_ssize_t length;
    if (chn_parse(args, kwargs, "Clipboard.write_image", names, 2, 2, values) ||
        chn_image_extract(values[0], "image", &image)) return NULL;
    const char *format = chn_utf8(values[1], "pixel_format", &length);
    if (!format) { chn_image_release(&image); return NULL; }
    int rgb = length == 3 && !memcmp(format, "RGB", 3);
    if (!rgb && !(length == 3 && !memcmp(format, "BGR", 3))) {
        chn_image_release(&image);
        PyErr_Format(PyExc_ValueError, "Invalid pixel format: %U", values[1]);
        return NULL;
    }
    size_t h = image.height, w = image.width, c = image.channels;
    if (c != 1 && c != 3 && c != 4) {
        chn_image_release(&image);
        PyErr_Format(PyExc_ValueError, "Invalid number of channels: %zu", c);
        return NULL;
    }
    if (h > LONG_MAX || w > LONG_MAX || (w && h > UINT32_MAX / 4 / w)) {
        chn_image_release(&image);
        PyErr_SetString(PyExc_ValueError, "Clipboard image dimensions exceed the Windows DIBV5 limits");
        return NULL;
    }
    size_t bytes = w * h * 4;
    payload data = {malloc(sizeof(BITMAPV5HEADER) + bytes), sizeof(BITMAPV5HEADER) + bytes};
    if (!data.bytes) { chn_image_release(&image); return PyErr_NoMemory(); }
    BITMAPV5HEADER header;
    memset(&header, 0, sizeof(header));
    header.bV5Size = sizeof(header);
    header.bV5Width = (LONG)w;
    header.bV5Height = (LONG)h;
    header.bV5Planes = 1;
    header.bV5BitCount = 32;
    header.bV5Compression = BI_BITFIELDS;
    header.bV5SizeImage = (DWORD)bytes;
    header.bV5RedMask = 0x00ff0000;
    header.bV5GreenMask = 0x0000ff00;
    header.bV5BlueMask = 0x000000ff;
    header.bV5AlphaMask = 0xff000000;
    header.bV5CSType = LCS_sRGB;
    header.bV5Intent = LCS_GM_IMAGES;
    memcpy(data.bytes, &header, sizeof(header));
    const Py_buffer *view = &image.view;
    int status;
    Py_BEGIN_ALLOW_THREADS
    status = cn_clipboard_pixels_f32(view->buf, h, w, c, view->strides[0], view->strides[1],
        view->ndim == 3 ? view->strides[2] : 0, rgb, data.bytes + sizeof(header), bytes);
    Py_END_ALLOW_THREADS
    chn_image_release(&image);
    if (status) {
        free(data.bytes);
        PyErr_SetString(PyExc_ValueError, "Invalid or overflowing clipboard image buffer");
        return NULL;
    }
    return finish(&data, 1);
}

static PyObject *clipboard_create_instance(PyObject *type, PyObject *unused)
{
    (void)type; (void)unused;
    return PyType_GenericAlloc(chn_clipboard_type, 0);
}
#else
static PyObject *clipboard_write_text(PyObject *self, PyObject *args, PyObject *kwargs)
{
    (void)self; (void)args; (void)kwargs;
    PyErr_SetString(PyExc_ValueError, "Failed to create clipboard: unsupported platform");
    return NULL;
}

static PyObject *clipboard_write_image(PyObject *self, PyObject *args, PyObject *kwargs)
{
    return clipboard_write_text(self, args, kwargs);
}

static PyObject *clipboard_create_instance(PyObject *type, PyObject *unused)
{
    (void)type; (void)unused;
    PyErr_SetString(PyExc_ValueError, "Failed to create clipboard: unsupported platform");
    return NULL;
}
#endif

static PyMethodDef clipboard_methods[] = {
    {"write_text", (PyCFunction)(void (*)(void))clipboard_write_text, METH_VARARGS | METH_KEYWORDS,
     "write_text($self, text)\n--\n\n"},
    {"write_image", (PyCFunction)(void (*)(void))clipboard_write_image, METH_VARARGS | METH_KEYWORDS,
     "write_image($self, image, pixel_format)\n--\n\n"},
    {"create_instance", clipboard_create_instance, METH_NOARGS | METH_STATIC,
     "create_instance()\n--\n\n"},
    {NULL, NULL, 0, NULL},
};

int chn_clipboard_init(PyObject *module)
{
    PyType_Slot slots[] = {
        {Py_tp_new, (void *)chn_no_constructor},
        {Py_tp_methods, clipboard_methods},
        {0, NULL},
    };
    PyType_Spec spec = {CHN_TYPE_NAME("Clipboard"), 0, 0, Py_TPFLAGS_DEFAULT, slots};
    chn_clipboard_type = (PyTypeObject *)PyType_FromSpec(&spec);
    if (!chn_clipboard_type) return -1;
    Py_INCREF((PyObject *)chn_clipboard_type);
    return PyModule_AddObject(module, "Clipboard", (PyObject *)chn_clipboard_type);
}
