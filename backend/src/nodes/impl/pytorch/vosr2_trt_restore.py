"""BF16 restoration with bounded outer tiles and seeded CPU noise."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.nn import functional

from api import Progress
from nodes.impl.tensorrt.tiling import ShapeBounds, tile_starts, tiled_upscale

from .vosr2 import (
    BundlePaths,
    VosrColorCorrection,
    color_correct,
    gaussian_weights,
    read_configs,
)
from .vosr2_scene import WindowPlan, face_feather, pad_image, scene_routing
from .vosr2_trt_execution import BoundedQueue
from .vosr2_trt_models import profiles
from .vosr2_trt_pipeline import FullSession


@dataclass(frozen=True)
class LatentStatistics:
    """Only the VAE normalization constants are needed by the full TRT path."""

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def load(cls, paths: BundlePaths) -> LatentStatistics:
        _, config = read_configs(paths)
        return cls(
            torch.tensor(config["latents_mean"]).view(1, 16, 1, 1),
            torch.tensor(config["latents_std"]).view(1, 16, 1, 1),
        )


def gpu_colour(
    session: FullSession, decoded: torch.Tensor, reference: np.ndarray
) -> np.ndarray:
    height, width = reference.shape[:2]
    rgb = decoded[:, :, :height, :width] * 0.5 + 0.5
    ref = torch.from_numpy(np.ascontiguousarray(reference)).to(session.device)
    ref = ref.permute(2, 0, 1).contiguous()[None]
    horizontal = session.constants.get("gaussian_horizontal")
    vertical = session.constants.get("gaussian_vertical")
    if horizontal is None or vertical is None:
        kernel = torch.from_numpy(cv2.getGaussianKernel(41, 5, cv2.CV_32F))
        horizontal = session.constant(
            "gaussian_horizontal",
            kernel.reshape(1, 1, 1, 41).expand(3, 1, 1, 41).contiguous(),
        )
        vertical = session.constant(
            "gaussian_vertical",
            kernel.reshape(1, 1, 41, 1).expand(3, 1, 41, 1).contiguous(),
        )

    def pad(value: torch.Tensor, axis: int) -> torch.Tensor:
        size = value.shape[axis]
        if size > 20:
            return functional.pad(
                value, (20, 20, 0, 0) if axis == 3 else (0, 0, 20, 20), mode="reflect"
            )
        # OpenCV's REFLECT_101 repeats on tiny references; torch reflect padding
        # requires the padding to be smaller than the axis being padded.
        indices = torch.arange(-20, size + 20, device=value.device)
        if size == 1:
            indices.zero_()
        else:
            indices = indices.remainder(2 * (size - 1))
            indices = torch.minimum(indices, 2 * (size - 1) - indices)
        return value.index_select(axis, indices)

    def blur(value: torch.Tensor) -> torch.Tensor:
        value = functional.conv2d(pad(value, 3), horizontal, groups=3)
        return functional.conv2d(pad(value, 2), vertical, groups=3)

    rgb = (rgb - blur(rgb) + blur(ref)).clamp_(0, 1)
    return rgb[0].flip(0).permute(1, 2, 0).contiguous().cpu().numpy()


class FiniteChecks:
    """Retain each stage's finite flag without a host synchronization per window."""

    def __init__(self) -> None:
        self.values: list[tuple[str, torch.Tensor]] = []

    def add(self, tensor: torch.Tensor, stage: str) -> torch.Tensor:
        self.values.append((stage, torch.isfinite(tensor).all()))
        return tensor

    def validate(self) -> None:
        flags = torch.stack([value for _, value in self.values]).cpu().tolist()
        for (stage, _), valid in zip(self.values, flags, strict=True):
            if not valid:
                raise ValueError(f"VOSR 2.0 produced non-finite {stage}.")


@torch.inference_mode()
def restore(
    statistics: LatentStatistics,
    image: torch.Tensor,
    reference: np.ndarray,
    seed: int,
    color: VosrColorCorrection,
    progress: Progress,
    check: Callable[[], None],
    session: FullSession,
    windows: WindowPlan | None = None,
) -> np.ndarray:
    """A video's `windows` keep each shot's window layout (outer tiles each their own)."""
    minimum, _, maximum = profiles("encoder_pre")["x"]
    if image.shape[-2] <= maximum[-2] and image.shape[-1] <= maximum[-1]:
        return _restore_tile(
            statistics, image, reference, seed, color, progress, check, session, windows
        )

    # Tile the already resized RGB reference: interpolation is still performed
    # once over the entire source. Engine profiles and within-tile restoration
    # stay unchanged; scene focus and VAE attention are local to each outer tile.
    bounds = ShapeBounds(*minimum[-2:], *maximum[-2:], alignment=16)
    overlap = 64  # Output pixels: 16 source pixels for a 4x upscale.
    count = len(tile_starts(reference.shape[0], bounds.max_h, overlap)) * len(
        tile_starts(reference.shape[1], bounds.max_w, overlap)
    )

    class Runner:
        def __init__(self) -> None:
            self.pending: deque[np.ndarray] = deque()
            self.done = 0

        def submit(self, tile: np.ndarray) -> None:
            check()
            self.pending.append(tile)

        def collect(self, height: int, width: int) -> np.ndarray:
            check()
            tile = self.pending.popleft()
            tensor = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1)))[
                None
            ]
            # Only one tile runs on CUDA at a time. Pending entries are CPU
            # views, and the same session retains weights and same-size buffers.
            result = _restore_tile(
                statistics,
                tensor * 2 - 1,
                tile,
                seed,
                color,
                progress.sub_progress(self.done / count, 1 / count),
                check,
                session,
                windows,
            )
            self.done += 1
            return result[:height, :width]

    return tiled_upscale(reference, 1, bounds, Runner(), overlap=overlap)


def _restore_tile(
    statistics: LatentStatistics,
    image: torch.Tensor,
    reference: np.ndarray,
    seed: int,
    color: VosrColorCorrection,
    progress: Progress,
    check: Callable[[], None],
    session: FullSession,
    windows: WindowPlan | None,
) -> np.ndarray:
    image = pad_image(image)
    session.begin_image(image.shape[-2], image.shape[-1], seed)
    device = session.device
    checks = FiniteChecks()
    with torch.cuda.stream(session.stream):
        check()
        session.activate(check)
        dino_image = image.to(device)
        latent = checks.add(session.vae("encoder", dino_image, check), "VAE encoding")
        # CPU-only window layout and routing overlap the already queued encoder work.
        layout, (ay, ax), strengths = scene_routing(image, windows)
        placed = layout.windows
        mean = session.constant("latent_mean", statistics.mean)
        std = session.constant("latent_std", statistics.std)
        latent = (latent - mean) / std
        noise = session.constants.get("noise")
        if noise is None:
            # Keep the established per-image CPU generator sequence, independent
            # of the default generator, CUDA scheduling and earlier frames.
            noise = torch.randn(
                latent.shape, generator=torch.Generator(device="cpu").manual_seed(seed)
            ).to(device)
            session.constants["noise"] = noise
        inputs = torch.cat((latent, noise), dim=1)
        extra = inputs[:, :, ay // 8 : ay // 8 + 64, ax // 8 : ax // 8 + 8]
        progress.set_progress(0.2)
        dino_mean = session.constant(
            "dino_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        dino_std = session.constant(
            "dino_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )
        height, width = latent.shape[-2:]
        velocity = torch.zeros_like(latent)
        weights = session.constant("blend_weights", gaussian_weights(64))
        feather = session.constant("face_feather", face_feather())
        # The grid's weight sum is built per image (the layout follows the content),
        # coverage checked on the CPU (no device sync).
        covered = np.zeros((height, width), dtype=bool)
        denominator = torch.zeros(
            (1, 1, height, width), dtype=torch.float32, device=device
        )
        for y, x in layout.grid:
            covered[y : y + 64, x : x + 64] = True
            denominator[:, :, y : y + 64, x : x + 64] += weights
        if not covered.all():
            raise RuntimeError("VOSR window grid did not cover every output pixel.")

        def predict(value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
            return checks.add(
                session.predict(value, condition).float(), "DiT prediction"
            )[:, :, :, :64]

        queue = BoundedQueue()
        faces = []
        for index, (y, x) in enumerate(placed):
            check()
            session.begin_window()
            source = dino_image[:, :, y * 8 : y * 8 + 512, x * 8 : x * 8 + 512]
            conditioning = functional.interpolate(
                source * 0.5 + 0.5, size=(448, 448), mode="bicubic", align_corners=False
            ).clamp_(0, 1)
            normalized = ((conditioning - dino_mean) / dino_std).float()
            condition = checks.add(session.condition(normalized), "DINO conditioning")
            crop = inputs[:, :, y : y + 64, x : x + 64]
            strength = strengths[index]
            if strength == 0:
                value = predict(crop, condition)
            elif strength == 1:
                value = predict(torch.cat((crop, extra), dim=3), condition)
            else:
                local = predict(crop, condition)
                check()
                value = local.lerp(
                    predict(torch.cat((crop, extra), dim=3), condition), strength
                )
            if index < len(layout.grid):
                velocity[:, :, y : y + 64, x : x + 64] += value * weights
            else:
                # Stage outputs are borrowed until the stage runs again.
                faces.append(value.clone())
            progress.set_progress(0.2 + 0.55 * (index + 1) / len(placed))
            queue.advance(check)
        velocity /= denominator
        for (y, x), value in zip(layout.faces, faces, strict=True):
            # Over the grid's blend, a face window takes the middle of its face.
            velocity[:, :, y : y + 64, x : x + 64].lerp_(value, feather)
        latent = noise - velocity
        check()
        decoded = checks.add(
            session.vae("decoder", latent * std + mean, check), "VAE decoding"
        )
        checks.validate()
        if color == VosrColorCorrection.WAVELET:
            result = gpu_colour(session, decoded, reference)
        else:
            height, width = reference.shape[:2]
            rgb = (
                decoded.cpu()[0, :, :height, :width].permute(1, 2, 0).numpy() * 0.5
                + 0.5
            )
            result = np.ascontiguousarray(
                color_correct(rgb, reference, color)[:, :, ::-1], dtype=np.float32
            )
        check()
        progress.set_progress(1)
        return result
