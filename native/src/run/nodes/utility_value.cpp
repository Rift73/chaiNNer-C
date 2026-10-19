/* Native entry points of utility value nodes (pattern: image_utility.cpp; a generator returns an ItemSource). */
#include "run_engine.hpp"

namespace {

using namespace cn::run;

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
}
