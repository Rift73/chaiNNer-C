/* Scoped ownership of the codecs' growing output buffer (GPL-3.0). */
#pragma once

#include "image_codecs.h"
#include "numpy_pool.hpp"

namespace cn::codecs {

struct Output {
    cn_codec_output value{};
    Output() = default;
    Output(const Output &) = delete;
    Output &operator=(const Output &) = delete;
    ~Output() { cn_codec_output_free(&value); }
};

/* An output over NumPy-pool blocks (numpy_pool.hpp), not zeroed, so a run's files reuse warm ones. */
struct PoolOutput : Output {
    PoolOutput() {
        value.allocate = cn_numpy_pool_allocate;
        value.release = cn_numpy_pool_free;
    }
};

} // namespace cn::codecs
