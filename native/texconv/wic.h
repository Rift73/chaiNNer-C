// The DirectXTex steps that texconv.exe runs through WIC on Windows for some formats and filters (wic.cpp).
#pragma once

#include <DirectXTex.h>

namespace wic {

HRESULT Resize(const DirectX::Image* images, size_t count, const DirectX::TexMetadata& metadata, size_t width,
    size_t height, DirectX::TEX_FILTER_FLAGS filter, DirectX::ScratchImage& result);
HRESULT Convert(const DirectX::Image* images, size_t count, const DirectX::TexMetadata& metadata, DXGI_FORMAT format,
    DirectX::TEX_FILTER_FLAGS filter, float threshold, DirectX::ScratchImage& result);
HRESULT GenerateMipMaps(const DirectX::Image* images, size_t count, const DirectX::TexMetadata& metadata,
    DirectX::TEX_FILTER_FLAGS filter, size_t levels, DirectX::ScratchImage& result);

} // namespace wic
