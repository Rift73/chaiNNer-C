"""Preserve local structure while supplying input-derived scene context.

Local predictions keep 512 px windows and their DINO conditioning. A second
prediction appends context outside each window; its contribution decreases
with local input detail. This remains an experimental sharpness heuristic,
not a learned defocus classifier or exact whole-image DiT inference.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch.nn import functional

from api import Progress

from .vosr2 import (
    MemoryPlan,
    VosrBundle,
    VosrColorCorrection,
    VosrMemoryMode,
    color_correct,
    finite,
    gaussian_weights,
)
from .vosr2_scene import WindowPlan, face_feather, pad_image, scene_routing
from .vosr2_streaming import VaeStreamer


@torch.inference_mode()
def context_denoise(
    bundle: VosrBundle,
    latent: torch.Tensor,
    image: torch.Tensor,
    device: torch.device,
    seed: int,
    plan: MemoryPlan,
    progress: Progress,
    check: Callable[[], None],
    optimized: bool = False,
    windows: WindowPlan | None = None,
) -> torch.Tensor:
    height, width = latent.shape[-2:]
    # Full 512px windows, placed by content with at least the original 64px overlap.
    # Replacing 128px and stepping by 320px made fine lines depend on weak
    # window-edge predictions, even when the replacement strip was disabled.
    layout, (ay, ax), strengths = scene_routing(image, windows)
    placed = layout.windows
    noise = torch.randn(
        latent.shape, generator=torch.Generator(device="cpu").manual_seed(seed)
    )
    inputs = torch.cat((latent, noise), dim=1)
    # The same seeded noise accompanies the same context in every window.
    extra = inputs[:, :, ay // 8 : ay // 8 + 64, ax // 8 : ax // 8 + 8]
    conditions = []
    try:
        bundle.dino.to(device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
        for index, (y, x) in enumerate(placed):
            check()
            crop = image[:, :, y * 8 : y * 8 + 512, x * 8 : x * 8 + 512]
            conditioning = (
                functional.interpolate(
                    crop * 0.5 + 0.5,
                    size=(448, 448),
                    mode="bicubic",
                    align_corners=False,
                )
                .clamp_(0, 1)
                .to(device)
            )
            normalized = ((conditioning - mean) / std).to(bundle.dtype)
            features = bundle.dino.forward_intermediate_layer(normalized, 17)
            conditions.append(
                finite(features.float().cpu(), "DINO conditioning").to(bundle.dtype)
            )
            progress.set_progress(0.2 + 0.1 * (index + 1) / len(placed))
        del features, conditioning, crop, mean, std
    finally:
        if optimized:
            bundle.offload_model(bundle.dino)
        else:
            bundle.dino.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    velocity = torch.zeros_like(latent)
    denominator = torch.zeros((1, 1, height, width), dtype=torch.float32)
    weights = gaussian_weights(64)
    faces = []
    try:
        check()
        if plan.stream_blocks:
            for name, module in bundle.dit.named_children():
                if name != "blocks":
                    module.to(device)
        else:
            bundle.dit.to(device)

        def predict(value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
            prediction = bundle.dit.forward_flexible(
                value.to(device=device, dtype=bundle.dtype),
                torch.ones(1, device=device, dtype=bundle.dtype),
                z=[condition],
                check_cancel=check,
                token_chunk=plan.token_chunk,
                query_chunk=1024,
                stream_blocks=plan.stream_blocks,
            )
            return finite(prediction.float().cpu(), "DiT prediction")[:, :, :, :64]

        for index, (y, x) in enumerate(placed):
            check()
            crop = inputs[:, :, y : y + 64, x : x + 64]
            strength = strengths[index]
            condition = conditions[index].to(device)
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
                denominator[:, :, y : y + 64, x : x + 64] += weights
            else:
                faces.append(value)
            progress.set_progress(0.3 + 0.45 * (index + 1) / len(placed))
    finally:
        if optimized:
            bundle.offload_model(bundle.dit)
        else:
            bundle.dit.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not bool(torch.all(denominator > 0)):
        raise RuntimeError("VOSR scene windows did not cover the output.")
    velocity /= denominator
    feather = face_feather()
    for (y, x), value in zip(layout.faces, faces, strict=True):
        # Over the grid's blend, a face window takes the middle of its face.
        velocity[:, :, y : y + 64, x : x + 64].lerp_(value, feather)
    return noise - velocity


@torch.inference_mode()
def upscale_scene_focus(
    bundle: VosrBundle,
    image: torch.Tensor,
    reference: np.ndarray,
    device: torch.device,
    seed: int,
    memory: VosrMemoryMode,
    color: VosrColorCorrection,
    progress: Progress,
    check: Callable[[], None],
    optimized: bool = False,
    windows: WindowPlan | None = None,
) -> np.ndarray:
    """Experimental scene preservation. Caller owns the scoped VRAM budget. A video's
    `windows` keep each shot's window layout."""
    plan = MemoryPlan.for_mode(memory)
    image = pad_image(image)

    def streamer() -> VaeStreamer:
        map_device = torch.device("cpu")
        if not optimized or device.type != "cuda":
            return VaeStreamer(device, plan.activation_bytes, check, map_device)
        # Account for live maps, residuals and band scratch before choosing storage.
        # The allocator cap remains authoritative; larger images use host maps.
        total = torch.cuda.get_device_properties(device).total_memory
        available = total * torch.cuda.get_per_process_memory_fraction(
            device
        ) - torch.cuda.memory_allocated(device)
        # A residual holds its shortcut, first-convolution map and output.
        # Deeper stages have more channels but fewer pixels; the final 96-channel
        # stage is the largest. Band scratch is separately reserved below.
        required = image.shape[-2] * image.shape[-1] * 96 * 4 * 3
        required += 2 * plan.activation_bytes
        if required <= available:
            map_device = device
        return VaeStreamer(device, plan.activation_bytes, check, map_device)

    try:
        check()
        bundle.vae.to(device)
        stream = streamer()
        latent = finite(
            stream.encode(image.to(stream.map_device), bundle.vae).cpu(),
            "VAE encoding",
        )
        latent = (latent - bundle.mean) / bundle.std
        if optimized:
            bundle.offload_model(bundle.vae)
        else:
            bundle.vae.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        progress.set_progress(0.2)
        latent = context_denoise(
            bundle,
            latent,
            image,
            device,
            seed,
            plan,
            progress,
            check,
            optimized,
            windows,
        )
        check()
        denormalized = latent * bundle.std + bundle.mean
        bundle.vae.to(device)
        stream = streamer()
        decoded = finite(
            stream.decode(denormalized.to(stream.map_device), bundle.vae).cpu(),
            "VAE decoding",
        )
        height, width = reference.shape[:2]
        rgb = decoded[0, :, :height, :width].permute(1, 2, 0).numpy() * 0.5 + 0.5
        rgb = color_correct(rgb, reference, color)
        check()
        progress.set_progress(1)
        return np.ascontiguousarray(rgb[:, :, ::-1], dtype=np.float32)
    finally:
        bundle.offload()
        if device.type == "cuda":
            torch.cuda.empty_cache()
