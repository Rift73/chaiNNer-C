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

}  // namespace

CN_RUN_NODES_DECLARE(image_utility)
{
    registry.add("chainner:image:flip", flip);
}
