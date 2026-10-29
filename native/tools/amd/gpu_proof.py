"""The AMD kit's GPU proof (native/AMD.md): PyTorch's ROCm build runs on each GPU and
matches the CPU, with only the GPU kernel wheels the dependency manager installed for
this machine; chaiNNer's GPU list (gpu.amd) against HIP's, and its VRAM gauge against
amd-smi; the kernel wheels against the GPUs; ONNX Runtime's providers, MIGraphX first;
ROCm's own tools; the Vulkan devices NCNN sees. Prints the report's section
(Markdown); a part that fails prints its error there and its traceback to stderr.

Runs in the test environment with backend/src on the path."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

failures = 0


def part(title: str, body: Callable[[], None]) -> None:
    global failures
    print(f"\n### {title}\n")
    try:
        body()
    except Exception as error:
        failures += 1
        print(f"ERROR: {type(error).__name__}: {error}")
        traceback.print_exc()


def tool(path: Path, *arguments: str) -> str:
    command = [str(path), *arguments]
    # ROCm's wheel ships some tools (amd-smi) as scripts for "/usr/bin/env python3",
    # which a bare image lacks: they run with this Python, which has ROCm's packages.
    with path.open("rb") as file:
        first_line = file.readline()
    if first_line.startswith(b"#!") and b"python" in first_line:
        command.insert(0, sys.executable)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return (result.stdout + result.stderr).strip()


def rocm_bin() -> Path:
    """The tools of ROCm's runtime wheel (rocm-sdk-core), as the dependency manager
    installed it."""
    spec = importlib.util.find_spec("_rocm_sdk_core")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("ROCm's runtime wheel (rocm-sdk-core) is not installed")
    return Path(spec.submodule_search_locations[0]) / "bin"


def pytorch() -> None:
    import torch

    wheels = sorted(
        f"{d.metadata['Name']} {d.version}"
        for d in importlib.metadata.distributions()
        if d.metadata["Name"].startswith(("rocm", "torch", "triton"))
    )
    print(f"- torch {torch.__version__}, HIP {torch.version.hip}")
    print(f"- ROCm wheels: {', '.join(wheels)}")
    print(
        f"- torch.cuda.is_available(): {torch.cuda.is_available()},"
        f" {torch.cuda.device_count()} device(s)"
    )
    if torch.version.hip is None:
        raise RuntimeError("the installed PyTorch is not a ROCm build")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch sees no AMD GPU")
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(2048, 2048, generator=generator)
    b = torch.randn(2048, 2048, generator=generator)
    expected = a @ b
    image = torch.rand(1, 3, 256, 256, generator=generator)
    weight = torch.randn(16, 3, 3, 3, generator=generator)
    expected_conv = torch.nn.functional.conv2d(image, weight, padding=1)
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        device = torch.device("cuda", index)
        start = time.perf_counter()
        got = (a.to(device) @ b.to(device)).cpu()
        conv = torch.nn.functional.conv2d(
            image.to(device).half(), weight.to(device).half(), padding=1
        ).float()
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        error = float((got - expected).abs().max() / expected.abs().max())
        conv_error = float((conv.cpu() - expected_conv).abs().max())
        print(
            f"- GPU {index}: {properties.name}, {properties.gcnArchName},"
            f" {properties.total_memory / 2**30:.1f} GiB,"
            f" warp size {getattr(properties, 'warp_size', '?')},"
            f" {properties.multi_processor_count} CUs; FP32 matmul 2048x2048 relative error"
            f" {error:.1e}, FP16 conv max error {conv_error:.1e} (first call {seconds:.2f} s)"
        )
        if error > 1e-3 or conv_error > 5e-2:
            raise RuntimeError(f"GPU {index} does not match the CPU")


def amd_smi_json(*arguments: str) -> list[dict]:
    """amd-smi's JSON output: a list of GPUs (ROCm 6), or that list under gpu_data."""
    output = json.loads(tool(rocm_bin() / "amd-smi", *arguments, "--json"))
    return output["gpu_data"] if isinstance(output, dict) else output


def amd_smi_vram() -> dict[str, tuple[float, float]]:
    """amd-smi's used and total VRAM in MiB by PCI address."""
    addresses: dict[int, str] = {}
    for entry in amd_smi_json("list"):
        addresses[entry["gpu"]] = entry["bdf"].lower()
    usage: dict[str, tuple[float, float]] = {}
    for entry in amd_smi_json("metric", "--mem-usage"):
        memory = entry["mem_usage"]

        def mib(field: dict) -> float:
            scale = {"B": 2**-20, "KB": 2**-10, "MB": 1, "GB": 2**10}[field["unit"]]
            return float(field["value"]) * scale

        usage[addresses[entry["gpu"]]] = (
            mib(memory["used_vram"]),
            mib(memory["total_vram"]),
        )
    return usage


def chainner_list() -> None:
    import torch

    import gpu

    print(
        f"- gpu.amd: {len(gpu.amd.devices)} device(s); torch: {torch.cuda.device_count()}"
    )
    try:
        smi = amd_smi_vram()
    except Exception as error:
        smi = {}
        print(
            f"- amd-smi's memory figures could not be read: {type(error).__name__}: {error}"
        )
        traceback.print_exc()
    print()
    print(
        "| HIP index | Name | gfx | PCI | FP16 | FP8 | chaiNNer VRAM used / total (MiB)"
        " | amd-smi used / total (MiB) | torch gfx |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for device in gpu.amd.devices:
        usage = device.get_current_vram_usage()
        address = device.sysfs.name.lower() if device.sysfs is not None else ""
        theirs = smi.get(address)
        torch_gfx = (
            torch.cuda.get_device_properties(device.index).gcnArchName
            if device.index < torch.cuda.device_count()
            else "none"
        )
        print(
            f"| {device.index} | {device.name} | {device.gfx} | {address} |"
            f" {device.supports_fp16} | {device.supports_fp8} |"
            f" {usage.used / 2**20:.0f} / {usage.total / 2**20:.0f} |"
            f" {f'{theirs[0]:.0f} / {theirs[1]:.0f}' if theirs else 'n/a'} | {torch_gfx} |"
        )


def kernel_wheels() -> None:
    """The dependency manager installs the GPU kernel wheels of this machine's GPUs, and
    no other."""
    from dependencies.rocm import gpu_architectures

    supported, unsupported = gpu_architectures()
    installed = sorted(
        d.metadata["Name"].removeprefix("rocm-sdk-device-")
        for d in importlib.metadata.distributions()
        if d.metadata["Name"].startswith("rocm-sdk-device-")
    )
    print(f"- GPUs ROCm has kernels for: {', '.join(supported) or 'none'}")
    if unsupported:
        print(f"- GPUs without ROCm kernels: {', '.join(unsupported)}")
    print(f"- Kernel wheels installed: {', '.join(installed) or 'none'}")
    if installed != supported:
        raise RuntimeError("the kernel wheels are not those of this machine's GPUs")


def onnx_runtime() -> None:
    import onnxruntime

    providers = onnxruntime.get_available_providers()
    print(f"- onnxruntime {onnxruntime.__version__}: {', '.join(providers)}")
    if providers[0] != "MIGraphXExecutionProvider":
        raise RuntimeError("ONNX Runtime's first provider is not MIGraphX")


def block(lines: list[str]) -> None:
    print("```")
    print("\n".join(lines))
    print("```")


def rocm_tools() -> None:
    folder = rocm_bin()
    keys = ("Marketing Name", "Name:", "Compute Unit", "Wavefront Size", "Max Waves")
    rocminfo = tool(folder / "rocminfo").splitlines()
    block(
        [
            tool(folder / "amd-smi", "version"),
            tool(folder / "amd-smi", "list"),
            *(line.strip() for line in rocminfo if line.strip().startswith(keys)),
        ]
    )


def vulkan() -> None:
    keys = ("deviceName", "deviceType", "driverName", "driverInfo", "apiVersion")
    summary = tool(Path("/usr/bin/vulkaninfo"), "--summary").splitlines()
    block([line.strip() for line in summary if line.strip().startswith(keys)])


def main() -> int:
    print("## GPU proof")
    part("PyTorch on ROCm", pytorch)
    part("chaiNNer's GPU list and VRAM gauge against HIP and amd-smi", chainner_list)
    part("ROCm's GPU kernel wheels", kernel_wheels)
    part("ONNX Runtime", onnx_runtime)
    part("ROCm's tools (from the rocm-sdk-core wheel)", rocm_tools)
    part("Vulkan devices (NCNN)", vulkan)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
