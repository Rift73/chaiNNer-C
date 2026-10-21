/* native_filters.morphology for native entry points (Dilate, Erode, Edge Detection, ...). */
#pragma once

#include "run_engine.hpp"

#include <cstdint>

namespace cn::run {

/* native_filters.morphology(image, shape, radius, iterations, maximum=maximum), shape a cv2.MORPH_* value: in its
 * order, ValueError for an empty image ("Invalid image shape (h, w[, c])") or a negative radius or iteration count,
 * image itself for a zero one, ValueError for an unknown shape; then the ellipse or an image with a non-finite or
 * zero-class value through cn_morphology_complete with OpenCV's "morph" lanes, else cn_filter_morphology. The
 * result is raw (a caller's output enforce clamps it as the bodies' freeze_normalized(clamp=True) does). */
Image morphology(const Image &image, int64_t shape, int64_t radius, int64_t iterations, bool maximum);

}  // namespace cn::run
