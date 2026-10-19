/* Native entry points of the image creation nodes (image/create_images; pattern: image_utility.cpp). */
#include "image_node_support.hpp"

#include <filesystem>
#include <limits>
#include <memory>

namespace {

using namespace cn::run;

/* A width or height input (NumberInput min=1); beyond what an Image holds declines to the Python body. */
int32_t side(const Value &value)
{
    int64_t size = integer(value);
    if (size > std::numeric_limits<int32_t>::max()) throw RunPython{};
    return int32_t(size);
}

/* native_generation.image_fill: mode 0 one colour, 1 a checkerboard of square-sized squares, 2 a gradient from the
 * first colour to the second (colours as float32). */
Value image_fill(int32_t width, int32_t height, const std::vector<double> &color, const std::vector<double> *second,
                 const float *gradient, int mode, size_t square)
{
    std::vector<float> a(color.begin(), color.end()), b;
    if (second) b.assign(second->begin(), second->end());
    MutableImage out = MutableImage::allocate(height, width, int32_t(a.size()));
    check(cn_image_fill(out.data(), size_t(height), size_t(width), a.size(), a.data(), second ? b.data() : nullptr,
                        gradient, mode, square));
    return std::move(out).publish();
}

/* Color.from_1x1_image(as_target_channels(color.to_1x1_image(), channels)): the colour as float32, widened by
 * cvt_color (GRAY2BGR, GRAY2BGRA, BGR2BGRA) when it has fewer channels, each value through Color's _norm. */
std::vector<double> widened(const Color &color, int32_t channels)
{
    int32_t own = int32_t(color.values.size());
    MutableImage pixel = MutableImage::allocate(1, 1, own);
    for (int32_t c = 0; c < own; ++c) pixel.data()[c] = float(color.values[size_t(c)]);
    Image image = std::move(pixel).publish();
    if (own != channels) image = cvt_color(image, 0, 0, channels);
    std::vector<double> values;
    for (int32_t c = 0; c < channels; ++c) {
        double v = double(image.data[c]);
        double low = 1 < v ? 1.0 : v;
        values.push_back(low > 0 ? low : 0.0);
    }
    return values;
}

/* chainner:image:create_color (create_images/create_color.py). */
void create_color(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Color &color = std::get<Color>(inputs[0]);
    outputs.push_back(image_fill(side(inputs[1]), side(inputs[2]), color.values, nullptr, nullptr, 0, 0));
}

/* chainner:image:create_checkerboard (create_images/create_checkerboard.py): both colours at the larger channel
 * count. */
void create_checkerboard(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    int32_t width = side(inputs[0]), height = side(inputs[1]);
    const Color &first = std::get<Color>(inputs[2]), &second = std::get<Color>(inputs[3]);
    int64_t square = integer(inputs[4]);
    int32_t channels = int32_t(std::max(first.values.size(), second.values.size()));
    std::vector<double> a = widened(first, channels), b = widened(second, channels);
    outputs.push_back(image_fill(width, height, a, &b, nullptr, 1, size_t(square)));
}

/* gradients.py through native_geometry.fill_gradient (every enforced input takes the native path: a float32 2-D
 * image, a float angle and an int width): kind 0 horizontal, 1 vertical, 2 diagonal, 3 radial, 4 conic. */
Image gradient(int32_t width, int32_t height, int kind, double a, double b)
{
    MutableImage out = MutableImage::allocate(height, width, 1);
    check(cn_gradient(out.data(), size_t(height), size_t(width), kind, a, b, 0));
    return std::move(out).publish();
}

/* chainner:image:create_gradient (create_images/create_gradient.py): GradientStyle values; the colour with fewer
 * channels is widened to the other's. */
void create_gradient(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    int32_t width = side(inputs[0]), height = side(inputs[1]);
    const Color *first = &std::get<Color>(inputs[2]), *second = &std::get<Color>(inputs[3]);
    const std::string &style = std::get<std::string>(inputs[5]);
    const double pi = 3.141592653589793;
    Image map;
    if (style == "Horizontal") {
        map = gradient(width, height, 0, 0, 1);
    } else if (style == "Vertical") {
        map = gradient(width, height, 1, 0, 1);
    } else if (style == "Diagonal") {
        map = gradient(width, height, 2, double(integer(inputs[6])) * pi / 180, double(integer(inputs[7])));
    } else if (style == "Radial") {
        map = gradient(width, height, 3, double(integer(inputs[8])) / 100, double(integer(inputs[9])) / 100);
    } else {
        map = gradient(width, height, 4, double(integer(inputs[10])) * pi / 180, 1);
    }
    if (std::get<bool>(inputs[4])) std::swap(first, second);
    std::vector<double> a = first->values, b = second->values;
    if (a.size() > b.size()) b = widened(*second, int32_t(a.size()));
    else if (b.size() > a.size()) a = widened(*first, int32_t(b.size()));
    outputs.push_back(image_fill(width, height, a, &b, map.data, 2, 0));
}

/* chainner:image:create_colorwheel (create_images/create_colorwheel.py): a conic gradient at rotation 0.0 through
 * native_color_complete.colorwheel (HSV to RGB, operation 14, on color_hsv's dispatched lanes; no IPP). */
void create_colorwheel(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    int32_t size = side(inputs[0]);
    Image map = gradient(size, size, 4, 0.0, 1);
    MutableImage out = MutableImage::allocate(size, size, 3);
    check(cn_color_convert_f32(map.data, out.data(), size_t(size), size_t(size), 1, 3, 14, 0,
                               int(dispatch::opencv_lanes("color_hsv")), 0));
    outputs.push_back(std::move(out).publish());
}

/* chainner:image:create_noise (create_images/create_noise.py): NoiseMethod values; blue noise through
 * noise_functions/blue.py's native path (native_blue.create at the default initial fraction 0.1, then normalize to
 * float32), the others through native_procedural.create, which returns float64 without fractal layers (normalized
 * here as the output enforce converts it). Seed.to_u32() outside [0, 2**32) (random.Random) declines to the Python
 * body. */
void create_noise(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    int32_t width = side(inputs[0]), height = side(inputs[1]);
    double brightness = number(inputs[5]) / 100;
    const Seed *seed = std::get_if<Seed>(&inputs[2]);
    if (!seed || seed->value < 0 || seed->value > 4294967295) throw RunPython{};
    const std::string &method = std::get<std::string>(inputs[3]);
    size_t count = size_t(height) * size_t(width);
    MutableImage out = MutableImage::allocate(height, width, 1);
    if (method == "Blue Noise") {
        double deviation = number(inputs[14]);
        std::vector<uint32_t> words = seed_words(inputs[2]);
        std::vector<int64_t> permutation(count);
        check(cn_seeded_permutation(words.data(), words.size(), permutation.data(), count));
        std::vector<int32_t> ranks(count);
        check(cn_blue_from_permutation(permutation.data(), ranks.data(), size_t(height), size_t(width), deviation,
                                       deviation, 0.1));
        check(cn_blue_normalize(ranks.data(), out.data(), count, 0));
        outputs.push_back(std::move(out).publish());
        return;
    }
    int index = method == "Value Noise" ? 0 : method == "Smooth Value Noise" ? 1 : 2;
    bool pink = std::get<std::string>(inputs[9]) == "Pink noise";
    int64_t layers = pink ? integer(inputs[10]) : 0;
    double scale_ratio = pink ? number(inputs[11]) : 1, brightness_ratio = pink ? number(inputs[12]) : 1;
    if (layers) {
        // Python's scalar overflow before entering C.
        python_pow(scale_ratio, double(layers - 1));
        python_pow(brightness_ratio, double(layers - 1));
    }
    auto run = [&](void *target) {
        check(cn_create_procedural(target, size_t(height), size_t(width), uint64_t(seed->value), index,
                                   number(inputs[4]), brightness, std::get<bool>(inputs[6]) ? 1 : 0,
                                   std::get<bool>(inputs[7]) ? 1 : 0, std::get<bool>(inputs[8]) ? 1 : 0, int(layers),
                                   scale_ratio, brightness_ratio, std::get<bool>(inputs[13]) ? 1 : 0));
    };
    if (layers) {
        run(out.data());
    } else {
        std::vector<double> values(count);
        run(values.data());
        check(cn_pixels_convert_checked(values.data(), out.data(), count, 1, 0, 1, nullptr));
    }
    outputs.push_back(std::move(out).publish());
}

/* Split Spritesheet's items (Generator.from_range): the sprite at each index (a NumPy slice of the sheet) and the
 * index; no source files. */
class SpriteItems final : public ItemSource {
public:
    SpriteItems(Value sheet, int64_t rows, int64_t columns, bool row_major)
        : sheet_(std::move(sheet)), rows_(rows), columns_(columns), row_major_(row_major)
    {
        const Image &image = std::get<Image>(sheet_);
        sprite_h_ = image.height / rows, sprite_w_ = image.width / columns;
    }
    int64_t expected_length() const noexcept override { return rows_ * columns_; }
    bool describe(int64_t &token) override
    {
        if (next_ >= rows_ * columns_) return false;
        token = next_++;
        return true;
    }
    void materialize(int64_t token, std::vector<Value> &item) const override
    {
        int64_t row = row_major_ ? token / columns_ : token % rows_;
        int64_t column = row_major_ ? token % columns_ : token / rows_;
        item.push_back(slice(sheet_, __int128(row) * sprite_h_, __int128(row + 1) * sprite_h_,
                             __int128(column) * sprite_w_, __int128(column + 1) * sprite_w_));
        item.push_back(token);
    }
    bool source_paths(std::vector<std::filesystem::path> &paths) const override
    {
        paths.clear();
        return true;
    }

private:
    Value sheet_;
    int64_t rows_, columns_, sprite_h_ = 0, sprite_w_ = 0, next_ = 0;
    bool row_major_;
};

/* chainner:image:split_spritesheet (batch_processing/split_spritesheet.py): OrderEnum 0 row major, 1 column major.
 * A sprite count beyond int64 declines to the Python body. */
std::unique_ptr<ItemSource> split_spritesheet(NodeContext &, std::span<const Value> inputs, std::vector<Value> &)
{
    const Image &sheet = std::get<Image>(inputs[0]);
    int64_t rows = integer(inputs[1]), columns = integer(inputs[2]);
    if (sheet.height % rows != 0)
        throw NodeError("AssertionError", "Height of sprite sheet must be a multiple of the number of rows");
    if (sheet.width % columns != 0)
        throw NodeError("AssertionError", "Width of sprite sheet must be a multiple of the number of columns");
    if (__int128(rows) * columns > std::numeric_limits<int64_t>::max()) throw RunPython{};
    return std::make_unique<SpriteItems>(inputs[0], rows, columns, std::get<int64_t>(inputs[3]) == 0);
}

}  // namespace

CN_RUN_NODES_DECLARE(image_generation)
{
    registry.add("chainner:image:create_checkerboard", create_checkerboard);
    registry.add("chainner:image:create_color", create_color);
    registry.add("chainner:image:create_colorwheel", create_colorwheel);
    registry.add("chainner:image:create_gradient", create_gradient);
    registry.add("chainner:image:create_noise", create_noise);
    registry.add_generator("chainner:image:split_spritesheet", split_spritesheet);
}
