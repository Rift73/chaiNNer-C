/* The encoders' growing output buffer (GPL-3.0). */
#include "image_codecs.h"

#include <stdlib.h>
#include <string.h>

void cn_codec_output_reserve(cn_codec_output *out, size_t capacity) {
    if (out->failed || capacity <= out->capacity) return;
    uint8_t *grown = (uint8_t *)(out->allocate ? out->allocate(capacity) : malloc(capacity));
    if (!grown) {
        out->failed = 1;
        return;
    }
    if (out->size) memcpy(grown, out->data, out->size);
    if (out->release) out->release(out->data);
    else free(out->data);
    out->data = grown;
    out->capacity = capacity;
}

void cn_codec_output_append(cn_codec_output *out, const void *bytes, size_t count) {
    if (out->failed || !count) return;
    if (count > out->capacity - out->size) {
        size_t capacity = out->capacity ? out->capacity : 4096;
        while (capacity - out->size < count) {
            if (capacity > SIZE_MAX / 2) {
                out->failed = 1;
                return;
            }
            capacity *= 2;
        }
        cn_codec_output_reserve(out, capacity);
        if (out->failed) return;
    }
    memcpy(out->data + out->size, bytes, count);
    out->size += count;
}

void cn_codec_output_free(cn_codec_output *out) {
    if (out->release) out->release(out->data);
    else free(out->data);
    out->data = NULL;
    out->size = out->capacity = 0;
    out->failed = 0;
}
