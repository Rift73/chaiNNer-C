"""Preserve local structure while supplying input-derived scene context.

Local predictions keep 512 px windows and their DINO conditioning. A second
prediction appends context outside each window; its contribution decreases
with local input detail. This remains an experimental sharpness heuristic,
not a learned defocus classifier or exact whole-image DiT inference.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple, cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from api import Progress
from nodes.impl import numpy_pool
from vendor.vosr2.lightningdit import LightningDiT, LightningDiTBlock

from .vosr2 import (
    FiniteChecks,
    MemoryPlan,
    VosrBundle,
    VosrColorCorrection,
    VosrMemoryMode,
    by_rows,
    color_correct,
    finite,
    gaussian_weights,
)
from .vosr2_memory import available_memory, unreserved_memory
from .vosr2_replay import Replays
from .vosr2_scene import Layout, WindowPlan, face_feather, pad_image, scene_routing
from .vosr2_streaming import VaeStreamer

# What a DitPredictor holds besides the weights: its 72 key tensors (226 MiB) and its graphs'
# tensors (up to 323 MiB, with Low memory's chunks).
PREDICTOR_BYTES = 800 * 1024**2


class WindowWork(NamedTuple):
    """A restoration's CPU work on its input, which needs no model."""

    layout: Layout
    anchor: tuple[int, int]
    strengths: list[float]
    noise: torch.Tensor
    # Each window's RGB (0 to 1) at DINO's size
    conditioning: list[torch.Tensor]


@torch.inference_mode()
def window_work(
    image: torch.Tensor, seed: int, windows: WindowPlan | None = None
) -> WindowWork:
    # Full 512px windows, placed by content with at least the original 64px overlap.
    # Replacing 128px and stepping by 320px made fine lines depend on weak
    # window-edge predictions, even when the replacement strip was disabled.
    layout, anchor, strengths = scene_routing(image, windows)
    noise = torch.randn(
        (1, 16, image.shape[-2] // 8, image.shape[-1] // 8),
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    return WindowWork(
        layout,
        anchor,
        strengths,
        noise,
        [
            functional.interpolate(
                image[:, :, y * 8 : y * 8 + 512, x * 8 : x * 8 + 512] * 0.5 + 0.5,
                size=(448, 448),
                mode="bicubic",
                align_corners=False,
            ).clamp_(0, 1)
            for y, x in layout.windows
        ],
    )


class Residents:
    """A GPU run's models stay on the device from stage to stage and from run to run
    (copying the DiT there takes 0.25 s). Whatever does not fit gives way."""

    def __init__(
        self,
        bundle: VosrBundle,
        device: torch.device,
        no_graphs: Path,
        alone: bool = False,
    ):
        self.bundle = bundle
        self.device = device
        # Low memory: one model on the device at a time, as a small budget needs
        self.alone = alone
        # Replays' mark that this GPU cannot capture CUDA graphs
        self.no_graphs = no_graphs
        # A model had to leave: this device has no room to keep them after the run
        self.tight = False

    def make_room(self, model: nn.Module, scratch: int) -> bool:
        """Whether `model` fits on the device with `scratch` more bytes free, the other
        models leaving if that takes it."""

        def needed() -> int:
            return scratch + sum(
                tensor.numel() * tensor.element_size()
                for tensor in model.parameters()
                if tensor.device.type == "cpu"
            )

        if self.alone or needed() > available_memory(self.device):
            for other in (self.bundle.vae, self.bundle.dino, self.bundle.dit):
                if other is not model and any(
                    tensor.device.type != "cpu" for tensor in other.parameters()
                ):
                    self.bundle.offload_model(other)
                    self.tight = True
            torch.cuda.empty_cache()
        return needed() <= available_memory(self.device)

    def place(self, model: nn.Module, scratch: int) -> None:
        """Put `model` on the device, with `scratch` more bytes free if make_room can."""
        self.make_room(model, scratch)
        model.to(self.device)

    def keep_blocks(self, scratch: int) -> Callable[[nn.Module], None]:
        """Put on the device as many of the DiT's blocks as leave `scratch` bytes free,
        in order, and give what takes a streamed block off it: the others' CPU masters."""
        kept = set()
        sizes = [
            sum(tensor.numel() * tensor.element_size() for tensor in block.parameters())
            for block in self.bundle.dit.blocks
        ]
        torch.cuda.empty_cache()
        # Two streamed blocks may be on the device beside the kept ones.
        room = available_memory(self.device) - scratch - 2 * max(sizes)
        for block, size in zip(self.bundle.dit.blocks, sizes, strict=True):
            room -= size
            if room < 0:
                break
            block.to(self.device)
            kept.add(block)

        def release(block: nn.Module) -> None:
            if block not in kept:
                self.bundle.release_block(block)

        return release

    def load_ahead(self, scratch: int) -> None:
        """Copy the models the restoration needs next to the device while the VAE works,
        when they fit beside `scratch` more bytes."""
        models = (self.bundle.dino, self.bundle.dit)
        needed = scratch + sum(
            tensor.numel() * tensor.element_size()
            for model in models
            for tensor in model.parameters()
            if tensor.device.type == "cpu"
        )
        if needed <= available_memory(self.device):
            for model in models:
                model.to(self.device)


def _dino_features(
    bundle: VosrBundle, device: torch.device, no_graphs: Path
) -> Replays:
    replays = bundle.replays.get(bundle.dino)
    if not isinstance(replays, Replays):
        dino = bundle.dino
        position = dino.interpolate_pos_encoding(
            torch.empty(1, 1025, 1024, device=device, dtype=bundle.dtype), 448, 448
        )
        replays = Replays(
            lambda x: dino.forward_intermediate_layer(x, 17, position),
            device,
            no_graphs,
        )
        bundle.replays[dino] = replays
    return replays


class DitPredictor:
    """The DiT with what follows from its time and a window's conditioning made once, as
    the engines keep it: a window's second prediction needs only the part after them."""

    def __init__(
        self,
        dit: LightningDiT,
        device: torch.device,
        dtype: torch.dtype,
        plan: MemoryPlan,
        no_graphs: Path,
    ):
        self.dit = dit
        self.plan = plan
        self.device = device
        c, c0 = dit.time_condition(torch.ones(1, device=device, dtype=dtype))
        self.modulations = [
            cast(LightningDiTBlock, block).modulation(c0) for block in dit.blocks
        ]
        self.final = dit.final_layer.modulation(c)
        self.ropes = {}
        # A window's keys and values: stable addresses for the prediction's graphs
        shape = (1, dit.num_heads, 1024, dit.hidden_size // dit.num_heads)
        self.keys = [
            (
                torch.empty(shape, device=device, dtype=dtype),
                torch.empty(shape, device=device, dtype=dtype),
            )
            for _ in dit.blocks
        ]
        self.fill = Replays(self._fill, device, no_graphs)
        self.predict = Replays(self._predict, device, no_graphs)

    def close(self) -> None:
        self.fill.close()
        self.predict.close()
        self.keys = []

    def _fill(self, condition: torch.Tensor) -> torch.Tensor:
        for pair, made in zip(self.keys, self.dit.condition_keys(condition)):
            for target, value in zip(pair, made):
                target.copy_(value)
        return condition

    def _predict(self, value: torch.Tensor) -> torch.Tensor:
        grid = (value.shape[-2] // 2, value.shape[-1] // 2)
        # Made in a shape's first, plain call: its graph holds the table.
        rope = self.ropes.get(grid)
        if rope is None:
            rope = self.dit.rope(grid, self.device, value.dtype)
            self.ropes[grid] = rope
        # Default memory's one token chunk has the bits of none.
        chunks = (self.plan.token_chunk, 1024) if self.plan.stream_blocks else ()
        return self.dit.forward_prepared(
            value, self.keys, rope, self.modulations, self.final, *chunks
        )


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
    work: WindowWork | None = None,
    residents: Residents | None = None,
) -> torch.Tensor:
    height, width = latent.shape[-2:]
    if work is None:
        work = window_work(image, seed, windows)
    layout, (ay, ax), strengths = work.layout, work.anchor, work.strengths
    placed = layout.windows
    noise = work.noise
    inputs = torch.cat((latent, noise), dim=1)
    # The same seeded noise accompanies the same context in every window.
    extra = inputs[:, :, ay // 8 : ay // 8 + 64, ax // 8 : ax // 8 + 8]
    conditions = []
    try:
        if residents is None:
            bundle.dino.to(device)
            dino = bundle.dino

            def plain_features(value: torch.Tensor) -> torch.Tensor:
                return dino.forward_intermediate_layer(value, 17)

            features_of = plain_features
        else:
            residents.place(bundle.dino, plan.activation_bytes)
            features_of = _dino_features(bundle, device, residents.no_graphs)
        checks = FiniteChecks()
        mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
        for index in range(len(placed)):
            check()
            conditioning = work.conditioning[index].to(device)
            normalized = ((conditioning - mean) / std).to(bundle.dtype)
            features = features_of(normalized)
            if residents is None:
                conditions.append(
                    finite(features.float().cpu(), "DINO conditioning").to(bundle.dtype)
                )
            elif plan.stream_blocks:
                conditions.append(checks.add(features, "DINO conditioning").cpu())
            else:
                # A replay's output is borrowed.
                conditions.append(checks.add(features, "DINO conditioning").clone())
            progress.set_progress(0.2 + 0.1 * (index + 1) / len(placed))
        if residents is not None:
            checks.validate()
        del features, conditioning, normalized, mean, std
    finally:
        if residents is None:
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
    predictor = None
    try:
        check()
        # Low memory streams the DiT's blocks only when the budget cannot hold it.
        if residents is not None and (
            residents.make_room(bundle.dit, PREDICTOR_BYTES) or not plan.stream_blocks
        ):
            bundle.dit.to(device)
            kept = bundle.replays.get(bundle.dit)
            prepared = (
                kept
                if isinstance(kept, DitPredictor)
                else DitPredictor(
                    bundle.dit, device, bundle.dtype, plan, residents.no_graphs
                )
            )
            bundle.replays[bundle.dit] = prepared
            predictor = prepared

            def predict(value: torch.Tensor) -> torch.Tensor:
                prediction = prepared.predict(value.to(dtype=bundle.dtype))
                return finite(prediction.float().cpu(), "DiT prediction")[:, :, :, :64]

        else:
            release_block = None
            if plan.stream_blocks:
                for name, module in bundle.dit.named_children():
                    if name != "blocks":
                        module.to(device)
                if residents is not None:
                    release_block = residents.keep_blocks(plan.activation_bytes)
            else:
                bundle.dit.to(device)

            def predict(value: torch.Tensor) -> torch.Tensor:
                prediction = bundle.dit.forward_flexible(
                    value.to(device=device, dtype=bundle.dtype),
                    torch.ones(1, device=device, dtype=bundle.dtype),
                    z=[condition],
                    check_cancel=check,
                    token_chunk=plan.token_chunk,
                    query_chunk=1024,
                    stream_blocks=plan.stream_blocks,
                    release_block=release_block,
                )
                return finite(prediction.float().cpu(), "DiT prediction")[:, :, :, :64]

        for index, (y, x) in enumerate(placed):
            check()
            crop = inputs[:, :, y : y + 64, x : x + 64]
            strength = strengths[index]
            condition = conditions[index].to(device)
            if predictor is not None:
                predictor.fill(condition)
            if strength == 0:
                value = predict(crop)
            elif strength == 1:
                value = predict(torch.cat((crop, extra), dim=3))
            else:
                local = predict(crop)
                check()
                value = local.lerp(predict(torch.cat((crop, extra), dim=3)), strength)
            if index < len(layout.grid):
                velocity[:, :, y : y + 64, x : x + 64] += value * weights
                denominator[:, :, y : y + 64, x : x + 64] += weights
            else:
                faces.append(value)
            progress.set_progress(0.3 + 0.45 * (index + 1) / len(placed))
    finally:
        if residents is None:
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
    retain: bool = False,
    no_graphs: Path | None = None,
) -> np.ndarray:
    """Experimental scene preservation. Caller owns the scoped VRAM budget. A video's
    `windows` keep each shot's window layout. A GPU run in Default memory keeps its models
    on the device from stage to stage, given `no_graphs` (Replays' mark); with `retain`,
    also after it, if they all had room."""
    plan = MemoryPlan.for_mode(memory)
    image = pad_image(image)
    residents = (
        Residents(bundle, device, no_graphs, alone=plan.stream_blocks)
        if no_graphs is not None and device.type == "cuda"
        else None
    )

    def streamer() -> VaeStreamer:
        map_device = torch.device("cpu")
        if not optimized or device.type != "cuda":
            return VaeStreamer(device, plan.activation_bytes, check, map_device)
        # Account for live maps, residuals and band scratch before choosing storage.
        # The allocator cap remains authoritative; larger images use host maps.
        # A residual holds its shortcut, first-convolution map and output.
        # Deeper stages have more channels but fewer pixels; the final 96-channel
        # stage is the largest. Band scratch is separately reserved below.
        required = image.shape[-2] * image.shape[-1] * 96 * 4 * 3
        required += 2 * plan.activation_bytes
        if residents is not None:
            residents.place(bundle.vae, required)
        else:
            bundle.vae.to(device)
        if required <= available_memory(device):
            map_device = device
        return VaeStreamer(
            device,
            plan.activation_bytes,
            check,
            map_device,
            None
            if map_device == device
            else lambda: unreserved_memory(device) - 2 * plan.activation_bytes,
        )

    finished = False
    try:
        check()
        with ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="vosr-windows",
            initializer=numpy_pool.install,
        ) as pool:
            # The CPU work on the input overlaps the VAE on the GPU.
            pending = pool.submit(window_work, image, seed, windows)
            if not optimized or device.type != "cuda":
                bundle.vae.to(device)
            stream = streamer()
            loading = None
            if (
                residents is not None
                and not plan.stream_blocks
                and stream.map_device == device
            ):
                # So far only the VAE and its bands' room are accounted for.
                loading = pool.submit(
                    residents.load_ahead,
                    image.shape[-2] * image.shape[-1] * 96 * 4 * 3
                    + 3 * plan.activation_bytes,
                )
            latent = finite(
                stream.encode(image.to(stream.map_device), bundle.vae).cpu(),
                "VAE encoding",
            )
            work = pending.result()
            if loading is not None:
                loading.result()
        latent = (latent - bundle.mean) / bundle.std
        if residents is None:
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
            work,
            residents,
        )
        del work
        check()
        denormalized = latent * bundle.std + bundle.mean
        if not optimized or device.type != "cuda":
            bundle.vae.to(device)
        stream = streamer()
        decoded = finite(
            stream.decode(denormalized.to(stream.map_device), bundle.vae),
            "VAE decoding",
        )
        height, width = reference.shape[:2]
        rgb = decoded[0, :, :height, :width] * 0.5 + 0.5
        rgb = rgb.permute(1, 2, 0).contiguous().cpu().numpy()
        rgb = color_correct(rgb, reference, color)
        check()
        progress.set_progress(1)
        # Low memory's budget would count what stays.
        finished = (
            retain
            and residents is not None
            and not residents.tight
            and not plan.stream_blocks
        )
        return by_rows(height, lambda top, bottom: rgb[top:bottom, :, ::-1])
    finally:
        if not finished:
            bundle.offload()
            if device.type == "cuda":
                torch.cuda.empty_cache()
