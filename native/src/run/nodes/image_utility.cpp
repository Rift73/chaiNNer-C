/* Native entry points of image_utility nodes.
 *
 * Pattern for a native node (copy this):
 * - Port a node whose Python body is our C kernels plus glue; replicate the body and its bridge
 *   (nodes/impl/native_*.py) line by line: the same early returns, the same checks in the same order, the same
 *   kernel calls with the same arguments, the same errors (NodeError{"<Python type>", "<str(e)>"}).
 * - Inputs arrive enforced, in schema order: an image is an Image (float32 HWC, rows contiguous, channels 1, 3 or 4;
 *   Python's 2-D grey is channels == 1), a dropdown is its option value (int64_t or std::string), a number int64_t
 *   or double as Python passes it, a toggle bool, a path Path. Read them with std::get<T>(inputs[i]).
 * - A result image: MutableImage::allocate(h, w, c), write every value, std::move(image).publish(). Returning an
 *   input Image unchanged (outputs.push_back(inputs[0])) is the Python body's `return img`: no copy.
 * - Image outputs need not be normalized: the engine enforces outputs as ImageOutput.enforce does.
 * - Kernel statuses go through check() (kernel_status.hpp), which raises what native.py check() raises.
 * - Declare each kernel you call in include/run_kernels.h under its file's heading, and include run_kernels.h in
 *   that kernel file. Register in the group function at the end; one CN_RUN_NODES(<group>) line per file in
 *   all_nodes.inc.
 * - Test: native/tests/test_run_nodes_<group>.py with native_differential.assert_native_matches over seeded
 *   images (1/3/4 channels, odd sizes, edge values), every dropdown option and the error cases. */
#include "../kernel_status.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace cn::run;

/* chainner:image:flip (image_utility/modification/flip.py, native_channels.flip). */
void flip(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    int64_t axis = std::get<int64_t>(inputs[1]);
    if (image.height == 0 || image.width == 0)
        throw NodeError("ValueError", "Expected a nonempty two- or three-dimensional image");
    if (axis == 2) {
        outputs.push_back(inputs[0]);
        return;
    }
    if (axis < -1 || axis > 1) throw NodeError("ValueError", "Invalid flip axis");
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    check(cn_channels_flip(image.data, out.data(), size_t(image.height), size_t(image.width), size_t(image.channels),
                           int(axis)));
    outputs.push_back(std::move(out).publish());
}

/* The repr of an image's NumPy shape: (h, w) for grey, else (h, w, c). */
std::string shape_text(const Image &image)
{
    std::string text = "(" + std::to_string(image.height) + ", " + std::to_string(image.width);
    return text + (image.channels == 1 ? ")" : ", " + std::to_string(image.channels) + ")");
}

/* A Blend Images layer: an image, or a Color (one pixel's channels, repeated over the other layer's size). */
struct Layer {
    const Image *image = nullptr;
    std::vector<float> color;
    int64_t height = 0, width = 0, channels = 0;
};

/* chainner:image:blend (image_utility/compositing/blend_images.py) through native_composite.dimensions and
 * canvas: the overlay's offset from the position, the canvas geometry (Python ints; an offset beyond int64 or a
 * canvas side beyond int32 declines to the Python body), cn_composite_canvas. */
void blend(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image *base_image = std::get_if<Image>(&inputs[0]);
    const Image *overlay_image = std::get_if<Image>(&inputs[1]);
    if (!base_image && !overlay_image) throw NodeError("ValueError", "At least one layer must be an image");
    const Image &reference = base_image ? *base_image : *overlay_image;
    Layer layers[2];
    for (size_t i = 0; i < 2; ++i) {
        Layer &layer = layers[i];
        if (const Image *image = std::get_if<Image>(&inputs[i])) {
            if (image->height == 0 || image->width == 0)
                throw NodeError("ValueError", "Blend images must be nonempty two- or three-dimensional arrays");
            layer.image = image;
            layer.height = image->height;
            layer.width = image->width;
            layer.channels = image->channels;
        } else {
            for (double value : std::get<Color>(inputs[i]).values) layer.color.push_back(float(value));
            layer.height = reference.height;
            layer.width = reference.width;
            layer.channels = int64_t(layer.color.size());
        }
    }
    const Layer &base = layers[0], &overlay = layers[1];
    int64_t dx = base.width - overlay.width, dy = base.height - overlay.height;

    const std::string &position = std::get<std::string>(inputs[3]);
    __int128 x, y;
    if (position == "percent_offset") {
        x = int64_t(std::nearbyint(double(dx * std::get<int64_t>(inputs[4])) / 100));
        y = int64_t(std::nearbyint(double(dy * std::get<int64_t>(inputs[5])) / 100));
    } else if (position == "pixel_offset") {
        const int64_t *px = std::get_if<int64_t>(&inputs[6]), *py = std::get_if<int64_t>(&inputs[7]);
        if (!px || !py) throw RunPython{};
        x = *px;
        y = *py;
    } else {
        /* BLEND_OVERLAY_X0_Y0_FACTORS: 0, 0.5 or 1 of the size difference, truncated toward zero (astype int). */
        auto factor = [](int64_t difference, bool half, bool whole) {
            return whole ? difference : half ? difference / 2 : 0;
        };
        bool top = position.starts_with("top"), bottom = position.starts_with("bottom");
        bool left = position.ends_with("left"), right = position.ends_with("right");
        x = factor(dx, !left && !right, right);
        y = factor(dy, !top && !bottom, bottom);
    }

    bool crop = std::get<bool>(inputs[8]) && base.image && overlay.image;
    const __int128 zero = 0;
    __int128 bh = base.height, bw = base.width, oh = overlay.height, ow = overlay.width;
    __int128 left = crop ? 0 : std::max(zero, -x), top = crop ? 0 : std::max(zero, -y);
    __int128 right = crop ? 0 : std::max(zero, x + ow - bw), bottom = crop ? 0 : std::max(zero, y + oh - bh);
    bool padded = left || top || right || bottom;
    if (std::max({left, top, right, bottom}) > INT32_MAX)
        throw NodeError("OverflowError", "Blend canvas border exceeds OpenCV's integer range");
    __int128 height = bh + top + bottom, width = bw + left + right;
    __int128 paste_top = crop ? std::max(zero, y) : y + top, paste_left = crop ? std::max(zero, x) : x + left;
    __int128 region_height = crop ? std::max(zero, std::min(y + oh, bh) - paste_top) : oh;
    __int128 region_width = crop ? std::max(zero, std::min(x + ow, bw) - paste_left) : ow;
    if (!region_height || !region_width) region_height = region_width = paste_top = paste_left = 0;
    bool offset = crop && region_height;
    __int128 overlay_top = offset ? std::max(zero, -y) : 0, overlay_left = offset ? std::max(zero, -x) : 0;
    int64_t canvas_channels = padded ? 4 : base.channels;
    int64_t channels = region_height ? std::max(canvas_channels, overlay.channels) : canvas_channels;
    if (height > INT32_MAX || width > INT32_MAX) throw RunPython{};

    canvas_layer kernel_layers[2];
    for (size_t i = 0; i < 2; ++i) {
        const Layer &layer = layers[i];
        kernel_layers[i] = {layer.image ? layer.image->data : layer.color.data(), size_t(layer.height),
                            size_t(layer.width), size_t(layer.channels), layer.image ? 0 : 1};
    }
    canvas_geometry geometry = {size_t(height),       size_t(width),         size_t(channels),
                                size_t(top),          size_t(left),          size_t(paste_top),
                                size_t(paste_left),   size_t(overlay_top),   size_t(overlay_left),
                                size_t(region_height), size_t(region_width), padded ? 1 : 0};
    MutableImage out = MutableImage::allocate(int32_t(height), int32_t(width), int32_t(channels));
    check(cn_composite_canvas(&kernel_layers[0], &kernel_layers[1], &geometry, out.data(),
                              int(std::get<int64_t>(inputs[2]))));
    outputs.push_back(std::move(out).publish());
}

/* Palette from Image's methods (PaletteExtractionMethod values) and MAX_COLORS. */
constexpr std::string_view extract_all = "all", extract_kmeans = "k-means";
constexpr int64_t max_colors = 4096;

/* The image's pixels as rows of channels (as_3d(img).reshape(-1, c), as dithering/palette.py makes them). */
size_t pixel_rows(const Image &image)
{
    return size_t(image.height) * size_t(image.width);
}

/* kmeans_palette: native_palette.kmeans on the pixel rows (more rows than colors here), as a one-row image. */
Image kmeans_palette(const Image &image, int64_t colors)
{
    MutableImage out = MutableImage::allocate(1, int32_t(colors), image.channels);
    cn_status status = cn_palette_kmeans(image.data, out.data(), pixel_rows(image), size_t(image.channels),
                                         size_t(colors));
    if (int(status) == 5)  // CN_KMEANS_NO_CONVERGENCE, beyond cn_status' enumerators
        throw NodeError("cv2.error", "kmeans: can't update cluster center (check input for huge or NaN values)");
    check(status);
    return std::move(out).publish();
}

/* median_cut_palette: native_palette.median_cut on the pixel rows, whose column's NumPy sum block is (0, 0) for
 * C-ordered rows; its planar-rows branch needs a transposed input, which engine images never are. */
Image median_cut_palette(const Image &image, int64_t colors)
{
    size_t rows = pixel_rows(image), capacity = std::min(size_t(colors), rows), written = 0;
    MutableImage out = MutableImage::allocate(1, int32_t(capacity), image.channels);
    cn_status status = cn_palette_median_cut(image.data, out.data(), rows, size_t(image.channels), size_t(colors), 0,
                                             0, 0, &written);
    if (int(status) == 6)  // an empty child bucket, beyond cn_status' enumerators
        throw NodeError("ValueError", "zero-size array to reduction operation minimum which has no identity");
    check(status);
    if (written == capacity) return std::move(out).publish();
    MutableImage head = MutableImage::allocate(1, int32_t(written), image.channels);
    std::copy_n(out.data(), written * size_t(image.channels), head.data());
    return std::move(head).publish();
}

/* chainner:image:palette_from_image (image_utility/miscellaneous/palette_from_image.py). */
void palette_from_image(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    const std::string &method = std::get<std::string>(inputs[1]);
    int64_t size = std::get<int64_t>(inputs[2]);
    size_t channels = size_t(image.channels);
    if (method != extract_all && 0 <= size && size <= max_colors) {
        if (image.height == 0 || image.width == 0) throw NodeError("ValueError", "Distinct colors require a nonempty image");
        int exceeds = 0;
        check(cn_image_distinct_exceeds(image.data, pixel_rows(image), channels, size_t(size), &exceeds));
        if (exceeds) {
            outputs.push_back(method == extract_kmeans ? kmeans_palette(image, size) : median_cut_palette(image, size));
            return;
        }
    }
    if (image.height == 0 || image.width == 0) throw NodeError("ValueError", "Invalid image shape " + shape_text(image));
    std::vector<float> distinct(image.values());
    size_t count = 0;
    check(cn_analysis_distinct(image.data, distinct.data(), pixel_rows(image), channels, &count));
    if (method == extract_all) {
        if (int64_t(count) > max_colors)
            throw NodeError("ValueError", "Image has " + std::to_string(count) +
                                              " distinct colors, but only palettes with at most 4096 colors are "
                                              "supported.");
        MutableImage out = MutableImage::allocate(1, int32_t(count), image.channels);
        std::copy_n(distinct.data(), count * channels, out.data());
        outputs.push_back(std::move(out).publish());
        return;
    }
    if (size >= int64_t(count)) {
        MutableImage out = MutableImage::allocate(1, int32_t(size), image.channels);
        check(cn_palette_pad_edge(distinct.data(), out.data(), count, channels, size_t(size)));
        outputs.push_back(std::move(out).publish());
        return;
    }
    outputs.push_back(method == extract_kmeans ? kmeans_palette(image, size) : median_cut_palette(image, size));
}

}  // namespace

CN_RUN_NODES_DECLARE(image_utility)
{
    registry.add("chainner:image:blend", blend);
    registry.add("chainner:image:flip", flip);
    registry.add("chainner:image:palette_from_image", palette_from_image);
}
