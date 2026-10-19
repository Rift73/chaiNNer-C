/* Native entry points of utility value nodes (pattern: image_utility.cpp; a generator returns an ItemSource), with
 * the color category's Color forwarder. */
#include "run_engine.hpp"

#include <array>
#include <string_view>

namespace {

using namespace cn::run;

/* `return <input>`: Number, Text, Directory, Percent, Color and Execution Number (execution_ops.cpp's
 * execution_number returns its argument). Any value, a PyRef included, passes on unchanged. */
void forward(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    outputs.push_back(inputs[0]);
}

/* chainner:utility:pass_through (value/pass_through.py): the ten inputs as the ten outputs. */
void pass_through(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    outputs.assign(inputs.begin(), inputs.end());
}

/* chainner:utility:switch (value/switch.py): args[selection.value], an error when it is None. */
void switch_value(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Value &selected = inputs[1 + size_t(std::get<int64_t>(inputs[0]))];
    if (std::holds_alternative<std::monostate>(selected))
        throw NodeError("RuntimeError", "Invalid value selected. The selected value should have a connection.");
    outputs.push_back(selected);
}

/* chainner:utility:resolutions (value/resolutions.py): RESOLUTIONS by ResList value; Custom passes width, height. */
struct Resolution {
    std::string_view preset;
    int64_t width, height;
};

constexpr std::array<Resolution, 21> resolutions{{
    {"HD_1080", 1920, 1080},     {"NTSC MPEG2", 704, 480},    {"NTSC", 720, 486},
    {"NTSC_DV", 720, 480},       {"PAL", 720, 576},           {"PAL MPEG2", 702, 576},
    {"HD_720", 1280, 720},       {"UHD_4K", 3840, 2160},      {"UHD2_8K", 7680, 4320},
    {"DCI_2KFLAT", 1998, 1080},  {"DCI_2KSCOPE", 2048, 858},  {"DCI_2K1080", 2048, 1080},
    {"DCI_4K2160", 4096, 2160},  {"DCI_4KFLAT", 3996, 2160},  {"DCI_4KSCOPE", 4096, 1716},
    {"SQ256", 256, 256},         {"SQ512", 512, 512},         {"SQ1024", 1024, 1024},
    {"SQ2048", 2048, 2048},      {"SQ4096", 4096, 4096},      {"SQ8192", 8192, 8192},
}};

void resolution(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const std::string &preset = std::get<std::string>(inputs[0]);
    if (preset == "CUSTOM") {
        outputs.push_back(inputs[1]);
        outputs.push_back(inputs[2]);
        return;
    }
    for (const Resolution &entry : resolutions) {
        if (entry.preset == preset) {
            outputs.push_back(entry.width);
            outputs.push_back(entry.height);
            return;
        }
    }
    throw NodeError("ValueError", "'" + preset + "' is not a valid ResList");  // enforce admits ResList values only
}

/* Range's items: start + index for index in [0, count); numbers only, so no source paths. */
class RangeItems final : public ItemSource {
public:
    RangeItems(int64_t start, int64_t count) : start_(start), count_(count) {}
    int64_t expected_length() const noexcept override { return count_; }
    bool describe(int64_t &token) override
    {
        if (next_ >= count_) return false;
        token = next_++;
        return true;
    }
    void materialize(int64_t token, std::vector<Value> &item) const override { item.push_back(start_ + token); }
    bool source_paths(std::vector<std::filesystem::path> &paths) const override
    {
        paths.clear();
        return true;
    }

private:
    int64_t start_, count_, next_ = 0;
};

/* chainner:utility:range (utility/value/range.py, execution_ops.cpp range_node). Python's integers are unbounded;
 * bounds whose arithmetic leaves int64 raise OverflowError here (a range of 2^63 items never completes anyway). */
std::unique_ptr<ItemSource> range(NodeContext &, std::span<const Value> inputs, std::vector<Value> &)
{
    __int128 start = std::get<int64_t>(inputs[0]);
    __int128 end = std::get<int64_t>(inputs[2]);
    if (!std::get<bool>(inputs[1])) start += 1;
    if (std::get<bool>(inputs[3])) end += 1;
    __int128 count = end - start;
    if (count < 0) throw NodeError("AssertionError", "");
    constexpr __int128 low = INT64_MIN, high = INT64_MAX;
    if (start < low || start > high || count > high || start + count - 1 > high)
        throw NodeError("OverflowError", "Range bounds exceed the native 64-bit integer range");
    return std::make_unique<RangeItems>(int64_t(start), int64_t(count));
}

}  // namespace

CN_RUN_NODES_DECLARE(utility_value)
{
    registry.add_generator("chainner:utility:range", range);
    registry.add("chainner:utility:color", forward);
    registry.add("chainner:utility:directory", forward);
    registry.add("chainner:utility:execution_number", forward);
    registry.add("chainner:utility:number", forward);
    registry.add("chainner:utility:pass_through", pass_through);
    registry.add("chainner:utility:percent", forward);
    registry.add("chainner:utility:resolutions", resolution);
    registry.add("chainner:utility:switch", switch_value);
    registry.add("chainner:utility:text", forward);
}
