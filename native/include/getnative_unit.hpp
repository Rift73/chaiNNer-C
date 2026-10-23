// Get Native Resolution: the function table each ISA unit of the vendored Vernier core exports
// (src/getnative_unit.cpp, built once per level with the core's namespace renamed) and the plain
// structs the C entries (src/getnative_ops.cpp) and the Python bridge share.
#pragma once
#include <atomic>
#include <cstddef>
#include <string>

extern "C" {
// Search options as Vernier's Options; kernels: 0 all 13 presets, 1 Bilinear, 2 Bicubic
// (Catmull-Rom with the free B/C refinement), 3 Lanczos 3.
struct cn_getnative_options {
    int min_height, max_height, min_width, max_width;
    int lines, refinement_lines, threads, refine, kernels;
};
// One analyzed plane's facts, as Vernier's Frame records them (UTF-8 text).
struct cn_getnative_frame {
    const char *source, *sample, *provenance, *light_domain, *color_assumptions;
    double requested_seconds;
    double weights[3];
    int keyframe_only;
};
}

// Text a session renders (render's `what`).
enum class GetnativeText { Report = 0, Parameters = 1, Summary = 2, Curves = 3 };
// The summary's batch facts (Vernier's BatchData counts, the sampling report and the error).
struct GetnativeBatch {
    int planned = 0, completed = 0, domains = 1;
    const char *sampling = "null", *error = "";
};

struct GetnativeUnit {
    void *(*create)(const cn_getnative_options &);
    void (*destroy)(void *);
    // plane: width x height doubles; cancel is checked as Vernier's.
    void (*add_frame)(void *, const double *plane, int width, int height, const cn_getnative_frame &,
                      std::atomic_bool *cancel);
    // Report: frame's full analysis JSON; Parameters: frame's candidate (-1: the recommended one);
    // Summary: the batch summary, a short record per analysis, the kernel decision and the selection;
    // Curves: frame's scan curves as CSV.
    std::string (*render)(void *, GetnativeText, int frame, int candidate, const GetnativeBatch &);
    // The best score per native size of a frame's horizontal (width) or vertical (height) scan;
    // returns the number of sizes and writes at most capacity of them.
    size_t (*curve)(void *, int frame, bool horizontal, int *sizes, double *scores, size_t capacity);
};

extern "C" const GetnativeUnit *cn_getnative_unit_base();
extern "C" const GetnativeUnit *cn_getnative_unit_avx2();
