/* Public, checked typed-tensor ABI from tensor_complete_ops.cpp (ABI 2).
 * This consumer declaration does not change the established producer layout. */
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

struct cn_ncnn_tensor_view {
    void *data;
    size_t count, dimensions;
    const size_t *shape;
    const ptrdiff_t *strides;
    int type;
};
static_assert(std::is_standard_layout_v<cn_ncnn_tensor_view>);
static_assert(sizeof(void*) == 8);
static_assert(sizeof(cn_ncnn_tensor_view) == 48);
static_assert(offsetof(cn_ncnn_tensor_view, type) == 40);
static_assert(offsetof(cn_ncnn_tensor_view, strides) == 32);
#ifdef CN_TENSOR_IMPORTS
#define CN_NCNN_TENSOR_API __declspec(dllimport)
#else
#define CN_NCNN_TENSOR_API
#endif
extern "C" {
CN_NCNN_TENSOR_API int cn_tensor_scale_typed(const cn_ncnn_tensor_view*, const cn_ncnn_tensor_view*, double, int, int*) noexcept;
CN_NCNN_TENSOR_API int cn_tensor_add_typed(const cn_ncnn_tensor_view*, const cn_ncnn_tensor_view*, const cn_ncnn_tensor_view*, int*) noexcept;
CN_NCNN_TENSOR_API int cn_tensor_cast_typed(const cn_ncnn_tensor_view*, const cn_ncnn_tensor_view*, int*) noexcept;
}
