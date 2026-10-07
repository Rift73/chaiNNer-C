/* Shared framework image arithmetic. Stages deliberately retain NumPy's
 * float32 rounding and floating-error boundaries. The sine polynomial is the
 * already attributed NumPy adaptation in numpy_trig_f32.h; the clip is
 * numpy_clip_f32.h's. */
#include "parallel.h"
#include "numpy_clip_f32.h"
#include "numpy_trig_f32.h"
#include <atomic>
#include <cfenv>
#include <cmath>
#include <limits>

namespace {
struct Range { uintptr_t begin, end; };
cn_status range(const void* pointer, size_t bytes, Range& output) noexcept {
    const auto value = reinterpret_cast<uintptr_t>(pointer);
    if (!pointer || value % alignof(float)) return CN_INVALID_ARGUMENT;
    if (value > UINTPTR_MAX - bytes) return CN_SIZE_OVERFLOW;
    output = {value, value + bytes};
    return CN_OK;
}
bool overlaps(Range a, Range b) noexcept {
    return a.begin < b.end && b.begin < a.end;
}
enum class Step { copy, clip, multiply, subtract, invert, divide, sine, difference, fill, ramp };
struct Job {
    const float* input;
    const float* second;
    float* output;
    float scalar;
    int fused;
    Step step;
    std::atomic<int> flags{0};
};
void apply(void* opaque, size_t begin, size_t end) noexcept {
    auto& job = *static_cast<Job*>(opaque);
    std::fenv_t previous;
    std::fegetenv(&previous);
    std::feclearexcept(FE_ALL_EXCEPT);
    for (size_t i = begin; i < end; ++i) {
        float value = 0;
        if (job.step != Step::fill && job.step != Step::ramp) value = job.input[i];
        switch (job.step) {
        case Step::copy: break;
        case Step::clip: value = cn_numpy_clip_f32(value, 0.f, 1.f); break;
        case Step::multiply: value *= job.scalar; break;
        case Step::subtract: value -= job.scalar; break;
        case Step::invert: value = job.scalar - value; break;
        case Step::divide: value /= job.scalar; break;
        case Step::sine:
            // The shared SIMD polynomial's range comparison must not signal on
            // quiet NaNs; preceding arithmetic already quiets signaling NaNs.
            // NumPy 2.5.3's FMA3 loop gives NPY_NANF for every NaN; the CRT sinf
            // keeps the input's.
            if (std::isnan(value)) value = job.fused ? std::numeric_limits<float>::quiet_NaN() : value;
            else value = cn_numpy_trig_f32(value, 0, job.fused);
            break;
        case Step::difference: value -= job.second[i]; break;
        case Step::fill: value = job.scalar; break;
        case Step::ramp: value = static_cast<float>(i) / job.scalar; break;
        }
        job.output[i] = value;
    }
    // np.clip reports no FP condition: NumPy 2.5.3 clears the status its min/max raise
    // on a NaN. The clip step reports none either, wherever the compiler places its
    // maxss/minss and NaN tests (it may run them before the NaN branch).
    const int state = job.step == Step::clip ? 0
        : std::fetestexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INVALID);
    const int flags = ((state & FE_DIVBYZERO) ? 1 : 0) | ((state & FE_OVERFLOW) ? 2 : 0)
        | ((state & FE_UNDERFLOW) ? 4 : 0) | ((state & FE_INVALID) ? 8 : 0);
    std::fesetenv(&previous);
    job.flags.fetch_or(flags, std::memory_order_relaxed);
}
}

/* mode: clip=0, sine=1, half-sine=2, alpha difference=3, fill=4,
 * ramp=5, sine ramp=6, half-sine ramp=7. Nine event slots are always required. */
extern "C" CN_EXPORT cn_status cn_framework_shared(
        const float* input, const float* second, float* output, size_t count,
        int mode, int fused, float scalar, int* events) noexcept {
    if (mode < 0 || mode > 7 || fused < 0 || fused > 1) return CN_INVALID_ARGUMENT;
    if (count > PTRDIFF_MAX / sizeof(float)) return CN_SIZE_OVERFLOW;
    Range a{}, b{}, destination{}, flags{};
    cn_status status = range(output, count * sizeof(float), destination);
    if (status != CN_OK) return status;
    status = range(events, 9 * sizeof(int), flags);
    if (status != CN_OK) return status;
    if (overlaps(destination, flags)) return CN_INVALID_ARGUMENT;
    if (mode <= 3) {
        status = range(input, count * sizeof(float), a);
        if (status != CN_OK) return status;
        if (overlaps(a, destination) || overlaps(a, flags)) return CN_INVALID_ARGUMENT;
    }
    if (mode == 3) {
        status = range(second, count * sizeof(float), b);
        if (status != CN_OK) return status;
        if (overlaps(b, destination) || overlaps(b, flags)) return CN_INVALID_ARGUMENT;
    }
    for (size_t index = 0; index < 9; ++index) events[index] = 0;
    Job job{input, second, output, scalar, fused, Step::copy};
    size_t phase = 0;
    auto run = [&](Step step, float operand) {
        job.step = step;
        job.scalar = operand;
        job.flags = 0;
        const auto result = cn_parallel_for(count, 65536, apply, &job);
        events[phase++] = job.flags.load(std::memory_order_relaxed);
        job.input = output;
        return result;
    };
    if (mode == 0) return run(Step::clip, 0);
    if (mode == 3) {
        status = run(Step::difference, 0);
        return status == CN_OK ? run(Step::invert, 1) : status;
    }
    if (mode == 4) return run(Step::fill, scalar);
    if (mode >= 5) {
        status = run(Step::ramp, scalar);
        if (status != CN_OK || mode == 5) return status;
    }
    if (mode == 2 || mode == 7) {
        status = run(Step::multiply, 2);
        if (status == CN_OK) status = run(Step::subtract, 0.5f);
        if (status == CN_OK) status = run(Step::clip, 0);
        if (status != CN_OK) return status;
    }
    constexpr float pi = 3.14159265358979323846f;
    status = run(Step::multiply, pi);
    if (status == CN_OK) status = run(Step::subtract, pi / 2);
    if (status == CN_OK) status = run(Step::sine, 0);
    if (status == CN_OK) status = run(Step::subtract, -1);
    if (status == CN_OK) status = run(Step::divide, 2);
    return status;
}
