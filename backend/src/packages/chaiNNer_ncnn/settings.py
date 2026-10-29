from __future__ import annotations

from dataclasses import dataclass

from ncnn import ncnn

from api import DropdownSetting, NodeContext, NumberSetting, ToggleSetting
from api.settings import DropdownOption
from system import is_arm_mac, is_linux

from . import package

# The gpu_index that runs NCNN on the CPU, offered where Vulkan has only software devices
CPU_DEVICE = -1
# GpuInfo.type() of a CPU-type Vulkan device, such as Mesa's llvmpipe and lavapipe
CPU_DEVICE_TYPE = 3

use_gpu = ncnn.get_gpu_count() > 0
# NCNN's own CPU kernels beat a software Vulkan device, so on Linux, where Mesa installs
# one with no GPU behind it, they are the default; the software device stays selectable.
software_only = (
    is_linux
    and use_gpu
    and all(
        ncnn.get_gpu_info(i).type() == CPU_DEVICE_TYPE
        for i in range(ncnn.get_gpu_count())
    )
)

if not is_arm_mac and use_gpu:
    try:
        gpu_list = []
        for i in range(ncnn.get_gpu_count()):
            gpu_list.append(ncnn.get_gpu_info(i).device_name())

        options: list[DropdownOption] = [
            {"label": x, "value": str(i)} for i, x in enumerate(gpu_list)
        ]
        description = "Which GPU to use for NCNN. This is only relevant if you have multiple GPUs."
        if software_only:
            options.insert(0, {"label": "CPU", "value": str(CPU_DEVICE)})
            description = (
                "NCNN found only software Vulkan devices, which are slower than its"
                " own CPU kernels, so it runs on the CPU. Choose a device to use it anyway."
            )

        package.add_setting(
            DropdownSetting(
                label="GPU",
                key="gpu_index",
                description=description,
                options=options,
                default=str(CPU_DEVICE) if software_only else "0",
            )
        )
    except Exception:
        pass

default_net_opt = ncnn.Net().opt

# Disabling Winograd/SGEMM was only tested on the CPU, so these toggles are
# offered only when NCNN finds no Vulkan GPU (or only software ones). It should work
# fine regardless of CPU/GPU. Ditto for multithreading, except it only makes sense for CPU.
if not use_gpu or software_only:
    package.add_setting(
        ToggleSetting(
            label="Use Winograd",
            key="winograd",
            description="Enable Winograd convolution for NCNN. Typically faster but uses more memory.",
            default=default_net_opt.use_winograd_convolution,
        )
    )
    package.add_setting(
        ToggleSetting(
            label="Use SGEMM",
            key="sgemm",
            description="Enable SGEMM convolution for NCNN. Typically faster but uses more memory.",
            default=default_net_opt.use_sgemm_convolution,
        )
    )

    package.add_setting(
        NumberSetting(
            label="Thread Count",
            key="threads",
            description="Number of threads for NCNN. Only affects CPU mode.",
            default=default_net_opt.num_threads,
            min=1,
            max=default_net_opt.num_threads,
        )
    )
    package.add_setting(
        NumberSetting(
            label="Block Time",
            key="blocktime",
            description="Milliseconds for threads to busy-wait for more work before going to sleep. Higher values may be faster; lower values may decrease power consumption and CPU usage. Only affects CPU mode.",
            default=default_net_opt.openmp_blocktime,
            min=0,
            max=400,
        )
    )

package.add_setting(
    NumberSetting(
        label="Memory Budget Limit (GiB)",
        key="budget_limit",
        description="Maximum memory to use for NCNN inference. 0 means no limit. Memory usage measurement is not completely accurate yet; you may need to significantly adjust this budget limit via trial-and-error if it's not having the effect you want.",
        default=0,
        min=0,
        max=1024**2,
    )
)


@dataclass(frozen=True)
class NcnnSettings:
    gpu_index: int
    winograd: bool
    sgemm: bool
    threads: int
    blocktime: int
    budget_limit: int


def get_settings(context: NodeContext) -> NcnnSettings:
    settings = context.settings

    return NcnnSettings(
        gpu_index=settings.get_int(
            "gpu_index", CPU_DEVICE if software_only else 0, parse_str=True
        ),
        winograd=settings.get_bool(
            "winograd", default_net_opt.use_winograd_convolution
        ),
        sgemm=settings.get_bool("sgemm", default_net_opt.use_sgemm_convolution),
        threads=settings.get_int(
            "threads", default_net_opt.num_threads, parse_str=True
        ),
        blocktime=settings.get_int(
            "blocktime", default_net_opt.openmp_blocktime, parse_str=True
        ),
        budget_limit=settings.get_int("budget_limit", 0, parse_str=True),
    )
