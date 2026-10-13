"""Streaming must preserve full-image operators, including across band edges."""

import pytest
import torch

from nodes.impl.pytorch.vosr2_streaming import VaeStreamer
from vendor.vosr2.lightningdit import LightningDiT
from vendor.vosr2.qwenimage_vae2d import (
    AttentionBlock2D,
    AutoencoderKLQwenImage2D,
    Resample2D,
    ResidualBlock2D,
)


@pytest.mark.parametrize("shape", [(1, 4, 13, 17), (2, 4, 16, 12)])
@pytest.mark.parametrize("budget", [1, 18000])
@torch.inference_mode()
def test_streamed_vae_operators_match_full_image(shape, budget):
    stream = VaeStreamer(torch.device("cpu"), budget, lambda: None)
    x = torch.randn(shape)
    for channels in (4, 8):
        layer = ResidualBlock2D(4, channels).eval()
        torch.testing.assert_close(
            stream.residual(x, layer), layer(x), atol=2e-6, rtol=2e-5
        )
    for mode in ("upsample2d", "downsample2d"):
        layer = Resample2D(4, mode).eval()
        torch.testing.assert_close(
            stream.resample(x, layer), layer(x), atol=2e-6, rtol=2e-5
        )
    attention = AttentionBlock2D(4).eval()
    torch.testing.assert_close(
        stream.attention(x, attention), attention(x), atol=2e-6, rtol=2e-5
    )


@torch.inference_mode()
def test_streamed_full_vae_matches_direct_encode_decode():
    vae = AutoencoderKLQwenImage2D(
        base_dim=4, z_dim=2, dim_mult=(1, 2, 4, 4), num_res_blocks=1
    ).eval()
    stream = VaeStreamer(torch.device("cpu"), 10000, lambda: None)
    image = torch.randn(1, 3, 32, 48)
    expected = vae.encode(image).latent_dist.mode()
    actual = stream.encode(image, vae)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(
        stream.decode(expected, vae), vae.decode(expected).sample, atol=1e-5, rtol=1e-4
    )


@pytest.mark.parametrize("shape", [(8, 8), (12, 8), (4, 10)])
@torch.inference_mode()
def test_streamed_dit_keeps_global_attention_and_rectangular_geometry(shape):
    dit = LightningDiT(
        input_size=8,
        patch_size=2,
        in_channels=4,
        out_channels=2,
        hidden_size=32,
        depth=2,
        num_heads=4,
        use_rope=True,
        use_rmsnorm=True,
        use_qknorm=True,
        use_swiglu=True,
        z_dims=16,
    ).eval()
    x, t, z = torch.randn(1, 4, *shape), torch.ones(1), [torch.randn(1, 7, 16)]
    expected = dit.forward_flexible(x, t, z=z)
    actual = dit.forward_flexible(
        x, t, z=z, token_chunk=3, query_chunk=2, stream_blocks=True
    )
    assert actual.shape == (1, 2, *shape)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_streaming_checks_cancellation_inside_a_layer():
    calls = []

    def check():
        calls.append(True)
        if len(calls) == 2:
            raise InterruptedError("cancel")

    stream = VaeStreamer(torch.device("cpu"), 1, check)
    with pytest.raises(InterruptedError, match="cancel"):
        stream.conv(torch.randn(1, 3, 8, 8), torch.nn.Conv2d(3, 4, 3, padding=1))
    assert len(calls) == 2


def test_rectangular_unpatchify_preserves_pixel_positions():
    model = LightningDiT(
        input_size=8,
        patch_size=2,
        in_channels=4,
        out_channels=2,
        hidden_size=16,
        depth=1,
        num_heads=2,
    )
    image = torch.arange(2 * 6 * 10).reshape(1, 2, 6, 10).float()
    # Derive patch order by indexing pixels, independently of the model's einsum.
    patches = torch.stack(
        [
            image[:, :, y : y + 2, x : x + 2].permute(0, 2, 3, 1).flatten(1)
            for y in range(0, 6, 2)
            for x in range(0, 10, 2)
        ],
        dim=1,
    )
    torch.testing.assert_close(model.unpatchify(patches, (3, 5)), image, rtol=0, atol=0)
