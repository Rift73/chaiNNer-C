/* native_filters.morphology, line by line (image_morphology.hpp). */
#include "image_morphology.hpp"

#include "image_node_support.hpp"
#include "dispatch.hpp"

#include <string>

namespace cn::run {
namespace {

/* cv2.MORPH_RECT, cv2.MORPH_CROSS and cv2.MORPH_ELLIPSE. */
constexpr int64_t morph_rect = 0, morph_cross = 1, morph_ellipse = 2;

}  // namespace

Image morphology(const Image &image, int64_t shape, int64_t radius, int64_t iterations, bool maximum)
{
    if (image.height == 0 || image.width == 0) throw NodeError("ValueError", "Invalid image shape " + shape_text(image));
    if (radius < 0 || iterations < 0)
        throw NodeError("ValueError", "Morphology radius and iterations must be nonnegative");
    if (radius == 0 || iterations == 0) return image;
    if (shape != morph_rect && shape != morph_cross && shape != morph_ellipse)
        throw NodeError("ValueError", "Unsupported morphology shape");
    size_t height = size_t(image.height), width = size_t(image.width), channels = size_t(image.channels);
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    int exceptional = 0;
    if (shape != morph_ellipse) check(cn_image_exceptional(image.data, image.values(), &exceptional));
    if (shape == morph_ellipse || exceptional) {
        check(cn_morphology_complete(image.data, out.data(), height, width, channels, size_t(radius),
                                     size_t(iterations), int(shape), maximum, dispatch::opencv_lanes("morph")));
    } else {
        check(cn_filter_morphology(image.data, out.data(), height, width, channels, size_t(radius), size_t(iterations),
                                   shape == morph_cross, maximum));
    }
    return std::move(out).publish();
}

}  // namespace cn::run
