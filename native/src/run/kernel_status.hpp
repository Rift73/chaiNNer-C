/* A chainner_native kernel's status as the Python bridges raise it (nodes/impl/native.py check()). */
#pragma once

#include "run_engine.hpp"
#include "run_kernels.h"

#include <string>

namespace cn::run {

inline void check(cn_status status)
{
    if (status == CN_INVALID_ARGUMENT)
        throw NodeError("ValueError", "Invalid image dimensions or arguments passed to C kernels");
    if (status == CN_SIZE_OVERFLOW)
        throw NodeError("OverflowError", "Image size exceeds the C kernel addressable buffer range");
    if (status == CN_ALLOCATION_FAILED)
        throw NodeError("MemoryError", "Unable to allocate the native CPU worker pool or work item");
    if (status != CN_OK) throw NodeError("RuntimeError", "Unexpected C kernel status " + std::to_string(int(status)));
}

}  // namespace cn::run
