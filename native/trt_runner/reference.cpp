#include "reference.h"

#include <vector>

namespace cn_trt {

void to_tensor(Source const& source, TileIn const& tile, void* tensor, Element element, int channels, bool clip) {
    int64_t const plane = static_cast<int64_t>(tile.width) * tile.height;
    for (int y = 0; y < tile.height; ++y) {
        int const sy = tile.y0 + pad_index(y, tile.real_h, tile.edge);
        for (int x = 0; x < tile.width; ++x) {
            float v[3];
            tensor_pixel(source, sy, tile.x0 + pad_index(x, tile.real_w, tile.edge), channels, clip, v);
            int64_t const index = static_cast<int64_t>(y) * tile.width + x;
            for (int c = 0; c < channels; ++c) store_element(tensor, element, c * plane + index, v[c]);
        }
    }
}

void accumulate(Planar const& tile, float* acc, int acc_w, int64_t acc_plane, int x0, int y0,
                float const* wy, int ny, float const* wx, int nx) {
    for (int c = 0; c < tile.channels; ++c)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                float const value = load_element(tile.data, tile.element, c * tile.plane + static_cast<int64_t>(y) * tile.pitch + x);
                float& total = acc[c * acc_plane + static_cast<int64_t>(y0 + y) * acc_w + x0 + x];
                total = total + value * (wy[y] * wx[x]);
            }
}

void store(Planar const& source, Result const& result, bool clip) {
    for (int y = 0; y < result.height; ++y)
        for (int x = 0; x < result.width; ++x) result_pixel(source, result, y, x, clip);
}

void reference_upscale(FrameJob const& job, HostPlane const* planes, HostPlane const* result, int input_channels,
                       int output_channels, Element input, Element output, Engine const& engine, bool clip) {
    int const out_w = job.width * job.scale, out_h = job.height * job.scale;
    std::vector<char> source_block(job.source.bytes), result_block(job.result.bytes);
    pack(planes, job.planes, job.width, job.height, job.format.sample, job.source, source_block.data());
    Source const source{source_block.data(), job.format.sample, job.planes, job.width, job.height, job.source.layout, job.codes};
    Result const target{result_block.data(), job.format.sample, job.result_planes, out_w, out_h, job.result.layout, job.codes};
    std::vector<float> acc(job.places.empty() ? 0 : static_cast<std::size_t>(output_channels) * out_w * out_h, 0.0f);
    std::vector<char> tensor, tile_out;
    for (std::size_t i = 0; i < job.tiles.size(); ++i) {
        TileIn const& tile = job.tiles[i];
        int const tile_out_w = tile.width * job.scale;
        int64_t const tile_out_plane = static_cast<int64_t>(tile_out_w) * tile.height * job.scale;
        tensor.assign(static_cast<std::size_t>(input_channels) * tile.width * tile.height * element_bytes(input), 0);
        tile_out.assign(static_cast<std::size_t>(output_channels * tile_out_plane) * element_bytes(output), 0);
        to_tensor(source, tile, tensor.data(), input, input_channels, clip);
        engine(tile, tensor.data(), tile_out.data());
        Planar const out{tile_out.data(), output, output_channels, tile_out_w, tile_out_plane};
        if (job.places.empty()) {
            store(out, target, clip);
        } else {
            Place const& place = job.places[i];
            accumulate(out, acc.data(), out_w, static_cast<int64_t>(out_w) * out_h, place.x0, place.y0,
                       job.weights.data() + place.wy, place.ny, job.weights.data() + place.wx, place.nx);
        }
    }
    if (!job.places.empty())
        store({acc.data(), Element::F32, output_channels, out_w, static_cast<int64_t>(out_w) * out_h}, target, clip);
    unpack(result_block.data(), job.result, result, job.result_planes, out_w, out_h, job.format.sample);
}

}  // namespace cn_trt
