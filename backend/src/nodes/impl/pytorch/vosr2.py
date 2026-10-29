"""Direct VOSR 2.0 inference. No ComfyUI server, hub code, or implicit downloads."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torch import nn

from api import Progress
from vendor.vosr2.dinov2 import build_dinov2_vitl14
from vendor.vosr2.lightningdit import LightningDiT
from vendor.vosr2.qwenimage_vae2d import AutoencoderKLQwenImage2D

from ...utils.utils import file_identity


class VosrColorCorrection(Enum):
    WAVELET = "wavelet"
    ADAIN = "adain"
    NONE = "none"


class VosrMemoryMode(Enum):
    DEFAULT = "default"
    LOW = "low"


class VosrBackend(Enum):
    PYTORCH = "pytorch"
    TENSORRT = "tensorrt"
    TENSORRT_FP8 = "tensorrt_fp8"
    MIGRAPHX = "migraphx"
    MIGRAPHX_FP8 = "migraphx_fp8"


TENSORRT_BACKENDS = (VosrBackend.TENSORRT, VosrBackend.TENSORRT_FP8)
# AMD GPUs on Linux (nodes/impl/migraphx/vosr2_migx.py)
MIGRAPHX_BACKENDS = (VosrBackend.MIGRAPHX, VosrBackend.MIGRAPHX_FP8)


@dataclass(frozen=True)
class MemoryPlan:
    """Storage profiles; callers scope them with vram_budget.

    A limit of None leaves the whole device available.
    """

    limit_bytes: int | None
    activation_bytes: int
    token_chunk: int
    stream_blocks: bool

    @staticmethod
    def for_mode(mode: VosrMemoryMode) -> MemoryPlan:
        return {
            VosrMemoryMode.DEFAULT: MemoryPlan(None, 1024 * 1024**2, 2048, False),
            VosrMemoryMode.LOW: MemoryPlan(7_000_000_000, 128 * 1024**2, 256, True),
        }[mode]


@dataclass(frozen=True)
class BundlePaths:
    root: Path

    @property
    def args(self) -> Path:
        return self.root / "args.json"

    @property
    def dit(self) -> Path:
        return self.root / "checkpoints/ema_model.safetensors"

    @property
    def vae_config(self) -> Path:
        return self.root / "Qwen-Image-vae-2d/config.json"

    @property
    def vae(self) -> Path:
        return self.root / "Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors"

    @property
    def dino(self) -> Path:
        return self.root / "dinov2_vitl14_pretrain.pth"

    def files(self) -> tuple[Path, ...]:
        return self.args, self.dit, self.vae_config, self.vae, self.dino

    def identity(self) -> tuple:
        return tuple(file_identity(path) for path in self.files())


def read_configs(paths: BundlePaths) -> tuple[dict, dict]:
    for path in paths.files():
        if not path.is_file():
            raise ValueError(
                f"Missing VOSR 2.0 model file: {path}. Run Upscale Image (VOSR) "
                "again to download it."
            )
    documents = []
    for path in (paths.args, paths.vae_config):
        try:
            documents.append(json.loads(path.read_text(encoding="utf-8")))
        except ValueError as error:  # not JSON or not UTF-8: incomplete or foreign
            raise ValueError(
                f"The VOSR 2.0 file {path} is damaged or incomplete ({error}). "
                "Delete it and run Upscale Image (VOSR) again to download it."
            ) from error
    args, config = documents
    if not isinstance(args, dict) or not isinstance(config, dict):
        raise ValueError("VOSR arguments and VAE config must be JSON objects.")
    expected = {
        "ae_type": "qwen",
        "dim": 1536,
        "depth": 36,
        "num_heads": 24,
        "patch_size": 2,
        "resolution": 512,
        "mlp_ratio": 4,
        "use_qknorm": True,
        "use_swiglu": True,
        "use_rope": True,
        "use_rmsnorm": True,
        "enc_type": "dinov2l",
        "enc_dim": 1024,
        "layer_dinov2b_list": [17],
        "encdim_ratio": 3,
        "dinov2_size": 448,
        "auxiliary_time_cond": False,
        "distill_type": "onestep",
    }
    for name, value in expected.items():
        if args.get(name) != value:
            raise ValueError(
                f"Unsupported VOSR 2.0 configuration: {name} must be {value!r}."
            )
    for name, value in {
        "base_dim": 96,
        "z_dim": 16,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_scales": [],
        "dropout": 0.0,
        "non_linearity": "silu",
    }.items():
        if config.get(name) != value:
            raise ValueError(
                f"Unsupported Qwen 2D VAE configuration: {name} must be {value!r}."
            )
    for name in ("latents_mean", "latents_std"):
        values = np.asarray(config.get(name), dtype=np.float32)
        if values.shape != (16,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid VAE {name}: expected 16 finite values.")
        if name == "latents_std" and (values <= 0).any():
            raise ValueError("VAE latent standard deviations must be positive.")
    return args, config


def load_weights(model: nn.Module, path: Path, dtype: torch.dtype) -> None:
    if path.suffix == ".safetensors":
        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError(f"Expected a flat tensor state dictionary: {path}")
    state = {key: value.to(dtype=dtype) for key, value in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    model.eval().requires_grad_(False)
    if any(tensor.is_meta for tensor in (*model.parameters(), *model.buffers())):
        raise ValueError(f"Checkpoint did not materialize every tensor: {path}")


class VosrBundle:
    def __init__(
        self,
        paths: BundlePaths,
        dtype: torch.dtype,
        progress: Progress,
        *,
        keep_cpu_weights: bool = False,
    ):
        _, config = read_configs(paths)
        self.dtype = dtype
        # No random initialization, global RNG consumption, or duplicate full-size weights.
        with torch.inference_mode(False), torch.device("meta"):
            self.dit = LightningDiT(
                input_size=64,
                patch_size=2,
                in_channels=32,
                out_channels=16,
                hidden_size=1536,
                depth=36,
                num_heads=24,
                mlp_ratio=4,
                use_qknorm=True,
                use_swiglu=True,
                use_rope=True,
                use_rmsnorm=True,
                z_dims=1024,
                encdim_ratio=3,
            )
            self.vae = AutoencoderKLQwenImage2D.from_config_dict(config)
            self.dino = build_dinov2_vitl14()
        models: list[tuple[nn.Module, Path, torch.dtype]] = [
            (self.vae, paths.vae, torch.float32),
            (self.dino, paths.dino, dtype),
            (self.dit, paths.dit, dtype),
        ]
        for model, path, weight_dtype in models:
            progress.suspend()
            load_weights(model, path, weight_dtype)
        self.mean = torch.tensor(config["latents_mean"]).view(1, 16, 1, 1)
        self.std = torch.tensor(config["latents_std"]).view(1, 16, 1, 1)
        self._cpu_weights = (
            {
                model: {
                    name: tensor.detach() for name, tensor in model.state_dict().items()
                }
                for model, _, _ in models
            }
            if keep_cpu_weights
            else {}
        )

    def offload_model(self, model: nn.Module) -> None:
        """Restore unchanged CPU masters instead of copying weights back from CUDA."""
        state = self._cpu_weights.get(model)
        if state is None:
            model.cpu()
        elif any(tensor.device.type != "cpu" for tensor in model.parameters()):
            model.load_state_dict(state, strict=True, assign=True)

    def offload(self) -> None:
        self.offload_model(self.vae)
        self.offload_model(self.dino)
        self.offload_model(self.dit)


def tile_positions(length: int, tile: int, overlap: int) -> list[int]:
    if not 0 <= overlap < tile <= length:
        raise ValueError("Tiles require 0 <= overlap < tile <= image dimension.")
    positions = list(range(0, length - tile + 1, tile - overlap))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def gaussian_weights(size: int) -> torch.Tensor:
    coordinate = (torch.arange(size, dtype=torch.float32) - (size - 1) / 2) / size
    weights = torch.exp(-coordinate.square() / 0.02)
    return (weights[:, None] * weights[None, :])[None, None]


def prepare_image(image: np.ndarray, scale: int) -> tuple[torch.Tensor, np.ndarray]:
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("VOSR 2.0 requires a non-empty RGB image.")
    if not np.isfinite(image).all():
        raise ValueError("VOSR 2.0 input contains non-finite values.")
    if scale not in range(1, 9):
        raise ValueError("VOSR 2.0 scale must be an integer from 1 to 8.")
    height, width = image.shape[:2]
    # chaiNNer is BGR; match the official CLI's PIL RGB bicubic 8-bit conditioning.
    rgb = np.rint(np.clip(image[:, :, ::-1], 0, 1) * 255).astype(np.uint8)
    resized = Image.fromarray(rgb).resize(
        (width * scale, height * scale), Image.Resampling.BICUBIC
    )
    rgb = np.asarray(resized).astype(np.float32) / 255
    # numpy reflect handles pads larger than tiny inputs (including length 1).
    padded = np.pad(
        rgb, ((0, -rgb.shape[0] % 16), (0, -rgb.shape[1] % 16), (0, 0)), mode="reflect"
    )
    tensor = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1)))[None]
    return tensor * 2 - 1, rgb


def finite(tensor: torch.Tensor, stage: str) -> torch.Tensor:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"VOSR 2.0 produced non-finite {stage}.")
    return tensor


def color_correct(
    result: np.ndarray, source: np.ndarray, mode: VosrColorCorrection
) -> np.ndarray:
    if mode == VosrColorCorrection.WAVELET:
        result = (
            result
            - cv2.GaussianBlur(result, (0, 0), 5)
            + cv2.GaussianBlur(source, (0, 0), 5)
        )
    elif mode == VosrColorCorrection.ADAIN:
        # Sample standard deviation, as upstream; singleton images have zero variance.
        correction = 1 if result.shape[0] * result.shape[1] > 1 else 0
        result_mean = result.mean(axis=(0, 1), keepdims=True)
        source_mean = source.mean(axis=(0, 1), keepdims=True)
        result_std = result.std(axis=(0, 1), ddof=correction, keepdims=True) + 1e-5
        source_std = source.std(axis=(0, 1), ddof=correction, keepdims=True) + 1e-5
        result = (result - result_mean) / result_std * source_std + source_mean
    return np.clip(result, 0, 1)
