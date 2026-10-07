#ifndef CHAINNER_REPAIR_BUFFERS_HPP
#define CHAINNER_REPAIR_BUFFERS_HPP

#include "chainner.h"
#include <cstdint>
#include <climits>
#include <cmath>
#include <limits>

namespace cn_repair {
struct Region { uintptr_t begin, end; };
inline cn_status region(const void *pointer, size_t size, Region &out) noexcept {
    if (!pointer) return CN_INVALID_ARGUMENT;
    auto begin=reinterpret_cast<uintptr_t>(pointer);
    if (size>UINTPTR_MAX-begin) return CN_SIZE_OVERFLOW;
    out={begin,begin+size};
    return CN_OK;
}
inline bool overlap(Region a,Region b) noexcept { return a.begin<b.end && b.begin<a.end; }
inline cn_status dimensions(size_t height,size_t width,size_t channels,size_t &pixels) noexcept {
    if (!height || !width || !channels) return CN_INVALID_ARGUMENT;
    if (height>INT_MAX-2 || width>INT_MAX-2 || height>SIZE_MAX/width) return CN_SIZE_OVERFLOW;
    pixels=height*width;
    if (pixels>SIZE_MAX/channels || (height+2)>SIZE_MAX/(width+2)/sizeof(double)) return CN_SIZE_OVERFLOW;
    return CN_OK;
}
/* Windows cvRound uses nearest-even and INT_MIN for a non-representable result. */
inline int round_int(double value) noexcept {
    double rounded=std::nearbyint(value);
    if (!(rounded>=-2147483648.0 && rounded<2147483648.0)) return INT_MIN;
    return static_cast<int>(rounded);
}
inline int floor_int(double value) noexcept {
    int truncated=(!(value>=-2147483648.0 && value<2147483648.0)) ? INT_MIN : static_cast<int>(value);
    if (static_cast<double>(truncated)>value) return truncated==INT_MIN ? INT_MAX : truncated-1;
    return truncated;
}
} // namespace cn_repair
#endif
