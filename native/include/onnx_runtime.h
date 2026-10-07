#ifndef CHAINNER_ONNX_RUNTIME_H
#define CHAINNER_ONNX_RUNTIME_H
#include "chainner.h"
#include <wchar.h>

/* Public host adapter; this delegates inference to the installed ORT engine.
 * Status values are OrtErrorCode (0 success), not cn_status. Errors are copied
 * into caller-owned UTF-8 storage. Registry handles reject stale/double release.
 * Run copies numeric inputs into owned storage; outputs remain valid until the
 * result handle is released, independently of the caller's arrays/session.
 */
typedef struct cn_ort_input {
    const char *name;
    const void *data;
    size_t bytes;
    const int64_t *shape;
    size_t rank;
    int type;
} cn_ort_input;

#ifdef __cplusplus
extern "C" {
#endif
CN_EXPORT int cn_ort_create(const wchar_t *library_path, const void *model, size_t model_bytes,
    uint64_t *session, char *error, size_t error_capacity);
CN_EXPORT int cn_ort_metadata(uint64_t session, char *json, size_t capacity,
    size_t *required, char *error, size_t error_capacity);
CN_EXPORT int cn_ort_run(uint64_t session, const cn_ort_input *inputs, size_t input_count,
    const char *const *outputs, size_t output_count, uint64_t *result,
    char *error, size_t error_capacity);
CN_EXPORT int cn_ort_result_info(uint64_t result, size_t index, int *type, size_t *rank,
    int64_t *shape, size_t shape_capacity, size_t *bytes, char *error, size_t error_capacity);
CN_EXPORT int cn_ort_result_copy(uint64_t result, size_t index, void *data, size_t bytes,
    char *error, size_t error_capacity);
CN_EXPORT int cn_ort_release_session(uint64_t session, char *error, size_t error_capacity);
CN_EXPORT int cn_ort_release_result(uint64_t result, char *error, size_t error_capacity);
CN_EXPORT int cn_ort_live_handles(size_t *sessions, size_t *results);
#ifdef __cplusplus
}
#endif
#endif
