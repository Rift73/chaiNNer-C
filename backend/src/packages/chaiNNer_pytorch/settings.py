import functools
from dataclasses import dataclass

import torch
from sanic.log import logger

from api import DropdownSetting, NodeContext, NumberSetting, ToggleSetting
from gpu import amd, nvidia
from system import is_arm_mac

from . import package

# A ROCm build of PyTorch runs on AMD GPUs, through the same torch.cuda API.
rocm = torch.version.hip is not None

if not is_arm_mac:
    gpu_list = []
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_properties(i).name
        gpu_list.append(device_name)
    # On ROCm, the strongest AMD GPU: a Ryzen's small graphics often come first.
    default_gpu = amd.default_index if rocm and amd.default_index < len(gpu_list) else 0

    package.add_setting(
        DropdownSetting(
            label="GPU",
            key="gpu_index",
            description=(
                "Which GPU to use for PyTorch. This is only relevant if you have"
                " multiple GPUs."
            ),
            options=[{"label": x, "value": str(i)} for i, x in enumerate(gpu_list)],
            default=str(default_gpu),
        )
    )

package.add_setting(
    ToggleSetting(
        label="Use CPU Mode",
        key="use_cpu",
        description=(
            "Use CPU for PyTorch instead of GPU. This is much slower and not"
            " recommended."
        ),
        default=False,
    ),
)

should_fp16 = False
if rocm:
    should_fp16 = amd.is_available and amd.all_support_fp16
elif nvidia.is_available:
    should_fp16 = nvidia.all_support_fp16
else:
    should_fp16 = is_arm_mac

package.add_setting(
    ToggleSetting(
        label="Use FP16 Mode",
        key="use_fp16",
        description=(
            "Runs PyTorch in half-precision (FP16) mode for reduced RAM usage but falls"
            " back to full-precision (FP32) mode when PyTorch runs on the CPU."
            if is_arm_mac
            else (
                "Runs PyTorch in half-precision (FP16) mode for less VRAM usage. AMD"
                " GPUs from Vega (gfx9) on also get a speedup. It falls back to"
                " full-precision (FP32) mode when PyTorch runs on the CPU, where Use"
                " BF16 Mode (CPU) applies instead."
            )
            if rocm
            else (
                "Runs PyTorch in half-precision (FP16) mode for less VRAM usage. RTX"
                " GPUs also get a speedup. It falls back to full-precision (FP32)"
                " mode when PyTorch runs on the CPU, where Use BF16 Mode (CPU) applies"
                " instead."
            )
        ),
        default=should_fp16,
    ),
)

# AVX-512 BF16 (AMD Zen 4 and newer, Intel Xeon from Cooper Lake) or AMX (Intel Xeon
# from Sapphire Rapids). Elsewhere oneDNN emulates BF16 (1.5-2.4x slower than FP32) or
# PyTorch takes a generic path (6-10x slower). PyTorch's own checks (cpuinfo) are
# private: a torch without them leaves the setting unavailable.
cpu_has_bf16 = any(
    getattr(torch.cpu, check, lambda: False)()
    for check in ("_is_avx512_bf16_supported", "_is_amx_tile_supported")
)

package.add_setting(
    ToggleSetting(
        label="Use BF16 Mode (CPU)",
        key="use_cpu_bf16",
        description=(
            "Runs Upscale Image (PyTorch) in bfloat16 when PyTorch runs on the CPU (CPU"
            " Mode, or no GPU found): about 1.4x (Compact) to 3x (ESRGAN) faster on CPUs"
            " with AVX-512 BF16 or AMX, with a small loss of precision (PSNR 51 dB vs"
            " FP32 on ESRGAN): BF16 can shift a few pixels slightly versus FP32. It keeps"
            " about 8 bits of precision, so 16-bit images lose precision in the"
            " highlights (gradients can band after grading). Models without bfloat16"
            " support run in FP32. No effect on a GPU."
            + (
                ""
                if cpu_has_bf16
                else " This CPU has neither AVX-512 BF16 nor AMX, so the setting is"
                " unavailable."
            )
        ),
        default=False,
        disabled=not cpu_has_bf16,
    ),
)

package.add_setting(
    NumberSetting(
        label="Memory Budget Limit (GiB)",
        key="budget_limit",
        description="Maximum memory (VRAM if GPU, RAM if CPU) to use for PyTorch inference. 0 means no limit. Memory usage measurement is not completely accurate yet; you may need to significantly adjust this budget limit via trial-and-error if it's not having the effect you want.",
        default=0,
        min=0,
        max=1024**2,
    )
)

if nvidia.is_available or (rocm and amd.is_available):
    # A ROCm build's torch.cuda cache is HIP's.
    cache = "GPU memory cache (HIP's, on AMD GPUs)" if rocm else "CUDA cache"
    package.add_setting(
        ToggleSetting(
            label=f"Force {'GPU' if rocm else 'CUDA'} Cache Wipe (not recommended)",
            key="force_cache_wipe",
            description=f"Clears PyTorch's {cache} after each inference. This is NOT recommended, by us or PyTorch's developers, as it basically interferes with how PyTorch is intended to work and can significantly slow down inference time. Only enable this if you're experiencing issues with VRAM allocation.",
            default=False,
        )
    )


@functools.cache
def log_no_bf16() -> None:
    """BF16 Mode (CPU)'s fallback line, once per process: every PyTorch node builds its
    settings, and the disabled toggle cannot be turned off."""
    logger.info("Falling back to FP32 mode: this CPU has neither AVX-512 BF16 nor AMX.")


@dataclass(frozen=True)
class PyTorchSettings:
    use_cpu: bool
    use_fp16: bool
    gpu_index: int
    budget_limit: int
    force_cache_wipe: bool = False
    use_cpu_bf16: bool = False

    # Most CPUs have no FP16 arithmetic, so PyTorch's FP16 convolutions there take a
    # generic path hundreds of times slower than FP32. That holds in CPU mode and when
    # no GPU is found alike. BF16 Mode (CPU) stays on only where PyTorch runs on a CPU
    # with BF16 instructions (settings saved on another machine fall back here).
    def __post_init__(self):
        cpu = self.device.type == "cpu"
        if self.use_cpu_bf16 and (not cpu or not cpu_has_bf16):
            object.__setattr__(self, "use_cpu_bf16", False)
            if cpu:
                log_no_bf16()
        if self.use_fp16 and cpu:
            object.__setattr__(self, "use_fp16", False)
            logger.info(
                "FP16 is off on the CPU; Upscale Image runs BF16 Mode (CPU)."
                if self.use_cpu_bf16
                else "Falling back to FP32 mode on the CPU."
            )

    @property
    def device(self) -> torch.device:
        # CPU override
        if self.use_cpu:
            device = "cpu"
        # Check for Nvidia CUDA (AMD GPUs on a ROCm build)
        elif torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device = f"cuda:{self.gpu_index}"
        # Check for Apple MPS
        elif (
            hasattr(torch, "backends")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        ):  # type: ignore -- older pytorch versions dont support this technically
            device = "mps"
        # Check for DirectML
        elif hasattr(torch, "dml") and torch.dml.is_available():  # type: ignore
            device = "dml"
        else:
            device = "cpu"

        return torch.device(device)


def get_settings(context: NodeContext) -> PyTorchSettings:
    settings = context.settings

    return PyTorchSettings(
        use_cpu=settings.get_bool("use_cpu", False),
        use_fp16=settings.get_bool("use_fp16", False),
        gpu_index=settings.get_int("gpu_index", 0, parse_str=True),
        budget_limit=settings.get_int("budget_limit", 0, parse_str=True),
        force_cache_wipe=settings.get_bool("force_cache_wipe", False),
        use_cpu_bf16=settings.get_bool("use_cpu_bf16", False),
    )
