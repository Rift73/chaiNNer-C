"""VOSR contracts that run on CPU without model downloads or ComfyUI."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from api import Aborted, Progress
from nodes.impl.pytorch.vosr2 import (
    BundlePaths,
    VosrBundle,
    VosrColorCorrection,
    VosrMemoryMode,
    color_correct,
    gaussian_weights,
    load_weights,
    prepare_image,
    read_configs,
    tile_positions,
)
from nodes.impl.pytorch.vosr2_context import upscale_scene_focus
from nodes.impl.pytorch.vosr2_official import FILES
from vendor.vosr2.dinov2 import DinoVisionTransformer
from vendor.vosr2.lightningdit import LightningDiT, MultiHeadCrossAttention
from vendor.vosr2.qwenimage_vae2d import AttentionBlock2D, AutoencoderKLQwenImage2D


@pytest.mark.parametrize("shape", [(1, 1), (1, 29), (17, 1), (15, 19), (32, 48)])
def test_conditioning_dimensions_channels_and_tiny_padding(shape):
    image = np.zeros((*shape, 3), dtype=np.float32)
    image[:, :, 0] = 1  # BGR blue must become RGB blue, never red.
    padded, reference = prepare_image(image, 3)
    assert reference.shape == (shape[0] * 3, shape[1] * 3, 3)
    assert padded.shape[-1] % 16 == padded.shape[-2] % 16 == 0
    np.testing.assert_array_equal(reference[0, 0], [0, 0, 1])
    assert torch.all(padded[:, 2] == 1)
    assert torch.all(padded[:, :2] == -1)


@pytest.mark.parametrize(
    "length,tile,overlap", [(16, 16, 8), (100, 32, 8), (160, 32, 0), (99, 24, 23)]
)
def test_tiles_cover_edges_without_duplicates(length, tile, overlap):
    positions = tile_positions(length, tile, overlap)
    coverage = np.zeros(length)
    for position in positions:
        coverage[position : position + tile] += 1
    assert coverage.min() > 0
    assert positions == sorted(set(positions))
    assert positions[0] == 0 and positions[-1] + tile == length


@pytest.mark.parametrize(
    "length,tile,overlap", [(0, 1, 0), (10, 12, 1), (10, 5, 5), (10, 5, -1)]
)
def test_invalid_tiles_are_rejected(length, tile, overlap):
    with pytest.raises(ValueError, match="Tiles require"):
        tile_positions(length, tile, overlap)


def test_window_weights_are_positive_and_symmetric():
    weights = gaussian_weights(32)
    assert weights.shape == (1, 1, 32, 32) and weights.min() > 0
    torch.testing.assert_close(weights, weights.flip(2, 3), rtol=0, atol=0)
    torch.testing.assert_close(weights, weights.transpose(2, 3), rtol=0, atol=0)


@pytest.mark.parametrize("color", list(VosrColorCorrection))
def test_color_correction_handles_single_pixel(color):
    result = color_correct(
        np.full((1, 1, 3), 0.25, dtype=np.float32),
        np.full((1, 1, 3), 0.75, dtype=np.float32),
        color,
    )
    assert np.isfinite(result).all()
    expected = 0.25 if color == VosrColorCorrection.NONE else 0.75
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_nonfinite_input_and_missing_bundle_fail_clearly(tmp_path):
    with pytest.raises(ValueError, match="non-finite"):
        prepare_image(np.full((4, 4, 3), np.nan, dtype=np.float32), 4)
    with pytest.raises(ValueError, match=r"Missing VOSR 2\.0 model file"):
        read_configs(BundlePaths(tmp_path))


def test_meta_loading_requires_every_checkpoint_key_without_random_initialization(
    tmp_path,
):
    state = {"weight": torch.eye(4), "bias": torch.arange(4).float()}
    path = tmp_path / "weights.safetensors"
    save_file(state, str(path))
    rng = torch.random.get_rng_state().clone()
    with torch.device("meta"):
        module = torch.nn.Linear(4, 4)
    load_weights(module, path, torch.float32)
    assert torch.equal(rng, torch.random.get_rng_state())
    torch.testing.assert_close(module(torch.ones(4)), 1 + state["bias"])
    save_file({"weight": torch.eye(4)}, str(path))
    with pytest.raises(RuntimeError, match="Missing key"):
        load_weights(module, path, torch.float32)


def test_vae_attention_matches_explicit_single_head_softmax():
    attention = AttentionBlock2D(8).eval()
    image = torch.randn(1, 8, 3, 4)
    normalized = attention.norm(image)
    query, key, value = (
        attention.to_qkv(normalized).reshape(1, 3, 8, 12).permute(0, 1, 3, 2).unbind(1)
    )
    probabilities = (query @ key.transpose(-1, -2) / np.sqrt(8)).softmax(-1)
    expected = (
        attention.proj((probabilities @ value).transpose(1, 2).reshape_as(image))
        + image
    )
    torch.testing.assert_close(attention(image), expected, atol=2e-6, rtol=2e-6)


def test_cross_attention_matches_explicit_multihead_softmax():
    attention = MultiHeadCrossAttention(16, 4, qk_norm=True).eval()
    image, conditioning = torch.randn(2, 5, 16), torch.randn(2, 7, 16)
    query = attention.q_norm(
        attention.q_linear(image).reshape(2, 5, 4, 4).transpose(1, 2)
    )
    key = attention.k_norm(
        attention.k_linear(conditioning).reshape(2, 7, 4, 4).transpose(1, 2)
    )
    value = attention.v_linear(conditioning).reshape(2, 7, 4, 4).transpose(1, 2)
    expected = attention.proj(
        (((query @ key.transpose(-1, -2) / 2).softmax(-1)) @ value)
        .transpose(1, 2)
        .reshape(2, 5, 16)
    )
    torch.testing.assert_close(
        attention(image, conditioning), expected, atol=2e-6, rtol=2e-6
    )


def test_dino_intermediate_is_raw_and_stops_before_unused_blocks():
    model = DinoVisionTransformer(
        img_size=28, patch_size=14, embed_dim=16, depth=3, num_heads=4
    ).eval()
    image = torch.randn(1, 3, 28, 42)
    with torch.no_grad():
        expected = model.blocks[1](model.blocks[0](model.prepare_tokens(image)))[:, 1:]
        actual = model.forward_intermediate_layer(image, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.allclose(actual, model.norm(expected))


def test_tiny_vae_shapes_and_dit_dynamic_rope_and_cancellation():
    vae = AutoencoderKLQwenImage2D(
        base_dim=4, z_dim=2, dim_mult=(1, 2, 4, 4), num_res_blocks=1
    ).eval()
    with torch.no_grad():
        latent = vae.encode(torch.rand(1, 3, 32, 48)).latent_dist.mode()
        assert latent.shape == (1, 2, 4, 6)
        assert vae.decode(latent).sample.shape == (1, 3, 32, 48)
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
    for size in (4, 8):
        result = dit.forward_flexible(
            torch.rand(1, 4, size, size), torch.ones(1), z=[torch.rand(1, 6, 16)]
        )
        assert result.shape == (1, 2, size, size)
        assert torch.isfinite(result).all()

    def cancel():
        raise Aborted()

    with pytest.raises(Aborted):
        dit.forward_flexible(
            torch.rand(1, 4, 8, 8),
            torch.ones(1),
            z=[torch.rand(1, 6, 16)],
            check_cancel=cancel,
        )


def test_scene_focus_offloads_the_bundle_when_cancelled():
    offloads = []

    def cancel():
        raise Aborted()

    bundle = cast(VosrBundle, SimpleNamespace(offload=lambda: offloads.append(True)))
    with pytest.raises(Aborted):
        upscale_scene_focus(
            bundle,
            torch.zeros(1, 3, 16, 16),
            np.zeros((16, 16, 3), dtype=np.float32),
            torch.device("cpu"),
            0,
            VosrMemoryMode.DEFAULT,
            VosrColorCorrection.NONE,
            Progress.noop_progress(),
            cancel,
        )
    assert offloads == [True]


def test_official_sizes_are_the_shared_download_pins():
    names = [local for _, local, _, _ in FILES]
    root = BundlePaths(Path("bundle"))
    assert names == [p.relative_to(root.root).as_posix() for p in root.files()]
    assert all(size > 0 and len(digest) == 64 for _, _, size, digest in FILES)


def test_a_damaged_configuration_names_its_file(tmp_path):
    paths = BundlePaths(tmp_path)
    for path in paths.files():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"ae_type": "qw', encoding="utf-8")
    with pytest.raises(ValueError, match=r"args\.json is damaged or incomplete"):
        read_configs(paths)
