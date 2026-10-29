from __future__ import annotations

import copy
import gc
import os
import weakref
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from ncnn import ncnn
from spandrel import ImageModelDescriptor, ModelDescriptor, ModelLoader
from spandrel.architectures.Compact import Compact
from spandrel.architectures.DAT import DAT
from spandrel.architectures.DAT.__arch.DAT import Adaptive_Spatial_Attention
from spandrel.architectures.DRFT.__arch import legacy_drft_arch as legacy
from spandrel.architectures.DUAL.__arch.dual_arch import dual_xs
from spandrel.architectures.ESRGAN import ESRGAN
from spandrel.architectures.HAT import HAT
from spandrel.architectures.HAT.__arch.HAT import OCAB, WindowAttention
from spandrel.architectures.MixDehazeNet import MixDehazeNet

from api import NodeContext, SettingsParser
from nodes.impl.ncnn.model import NcnnModelWrapper
from nodes.impl.onnx.model import OnnxGeneric
from nodes.impl.onnx.onnx_to_ncnn import Onnx2NcnnConverter
from nodes.impl.onnx.utils import safely_optimize_onnx_model
from nodes.impl.pytorch import attention_export
from nodes.impl.pytorch.convert_to_onnx_impl import convert_to_onnx_impl
from packages.chaiNNer_pytorch.pytorch.io import load_model as load_model_module
from packages.chaiNNer_pytorch.pytorch.io.load_model import load_model
from packages.chaiNNer_pytorch.pytorch.utility import convert_to_onnx
from packages.chaiNNer_pytorch.pytorch.utility.convert_to_ncnn import (
    convert_to_ncnn_node,
)
from packages.chaiNNer_pytorch.pytorch.utility.convert_to_onnx import (
    DualShape,
    OnnxDataType,
    Opset,
    convert_to_onnx_node,
)
from packages.chaiNNer_pytorch.settings import PyTorchSettings

ARCHS: dict[str, Callable[[], torch.nn.Module]] = {
    "Compact": lambda: Compact(num_feat=16, num_conv=2, upscale=2),
    "ESRGAN": lambda: ESRGAN(num_filters=8, num_blocks=1, scale=4),
    "MixDehazeNet": lambda: MixDehazeNet(
        embed_dims=[12, 24, 48, 24, 12], depths=[1, 1, 1, 1, 1]
    ),
    # Heads of 6 channels, which the export pads to 8
    "HAT": lambda: HAT(
        embed_dim=12,
        depths=(2,),
        num_heads=(2,),
        window_size=8,
        compress_ratio=2,
        squeeze_factor=4,
        upsampler="pixelshuffle",
        upscale=2,
    ),
    # Three blocks: the third shifts its windows; heads of 6 channels, which the export
    # pads to 8 (24 channels: DAT's spatial interaction keeps a sixteenth of them)
    "DAT": lambda: DAT(
        embed_dim=24, depth=[3], num_heads=[4], upsampler="pixelshuffle", upscale=2
    ),
}
# The export traces a 32x32 image; 37x51 checks that height and width stay dynamic.
SIZES = [(32, 32), (37, 51)]
# MixDehazeNet needs at least 40x40: its graph must pad smaller sizes (also beyond
# reflect padding's limit) and must not crop larger ones.
MIN_40_SIZES = [(16, 30), (37, 51), (40, 40), (48, 56)]


def tiny_model(arch: str) -> ImageModelDescriptor:
    """A randomly initialised model, prepared as PyTorch's Load Model node does."""
    torch.manual_seed(0)
    model = ModelLoader().load_from_state_dict(ARCHS[arch]().state_dict())
    assert isinstance(model, ImageModelDescriptor)
    for _, v in model.model.named_parameters():
        v.requires_grad = False
    return model.eval()


def torch_output(model: ImageModelDescriptor, image: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(image)[None]).numpy()[0]


def images(sizes: list[tuple[int, int]]):
    rng = np.random.default_rng(0)
    return [rng.random((3, h, w), dtype=np.float32) for h, w in sizes]


@pytest.mark.parametrize(
    ("arch", "sizes"),
    [
        ("Compact", SIZES),
        ("ESRGAN", SIZES),
        ("MixDehazeNet", MIN_40_SIZES),
        ("HAT", SIZES),
        ("DAT", SIZES),
    ],
)
def test_convert_to_onnx_matches_pytorch(arch: str, sizes: list[tuple[int, int]]):
    model = tiny_model(arch)
    session = ort.InferenceSession(
        convert_to_onnx_impl(model, torch.device("cpu")),
        providers=["CPUExecutionProvider"],
    )
    for image in images(sizes):
        (actual,) = session.run(None, {"input": image[None]})
        np.testing.assert_allclose(
            np.asarray(actual)[0], torch_output(model, image), rtol=0, atol=1e-5
        )


@pytest.mark.parametrize("arch", ["HAT", "DAT"])
def test_the_attention_rewrite_works_on_a_copy(arch: str):
    model = tiny_model(arch)
    weights = {k: v.clone() for k, v in model.model.state_dict().items()}
    export = attention_export.for_export(model)
    assert export.model is not model.model
    state = model.model.state_dict()
    assert state.keys() == weights.keys()
    assert all(torch.equal(state[k], v) for k, v in weights.items())
    image = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(export(image), model(image), rtol=0, atol=1e-6)


def test_hat_heads_are_padded_to_a_multiple_of_8():
    export = attention_export.for_export(tiny_model("HAT"))
    attention = [
        m for m in export.model.modules() if isinstance(m, WindowAttention | OCAB)
    ]
    assert len(attention) == 3  # two blocks' window attention and the OCAB
    for module in attention:
        assert module.qkv.out_features == 3 * module.num_heads * 8
        assert module.proj.in_features == module.num_heads * 8


def test_dat_spatial_heads_are_padded_to_a_multiple_of_8():
    export = attention_export.for_export(tiny_model("DAT"))
    attention = [
        m for m in export.model.modules() if isinstance(m, Adaptive_Spatial_Attention)
    ]
    assert len(attention) == 2  # the first and third blocks; the second is channel
    for module in attention:
        assert module.qkv.out_features == 3 * module.num_heads * 8
        # DAT drops the padded channels itself before its output projection
        assert module.proj.in_features == 24


def test_convert_model_with_trainable_weights():
    # PyTorch's Interpolate Models node returns a freshly loaded model, whose weights
    # still require grad (Load Model turns that off).
    torch.manual_seed(0)
    model = ModelLoader().load_from_state_dict(ARCHS["Compact"]().state_dict())
    assert isinstance(model, ImageModelDescriptor)
    assert any(p.requires_grad for p in model.model.parameters())
    session = ort.InferenceSession(
        convert_to_onnx_impl(model, torch.device("cpu")),
        providers=["CPUExecutionProvider"],
    )
    (image,) = images([(37, 51)])
    (actual,) = session.run(None, {"input": image[None]})
    np.testing.assert_allclose(
        np.asarray(actual)[0], torch_output(model, image), rtol=0, atol=1e-5
    )


# MixDehazeNet's graph pads with shape operations, which NCNN does not support.
@pytest.mark.parametrize("arch", ["Compact", "ESRGAN"])
def test_convert_to_ncnn_matches_pytorch(arch: str, tmp_path: Path):
    # Runs native code, which GitHub's checks do not build.
    pytest.importorskip("nodes.impl._chainner_graph")
    # PyTorch's Convert To NCNN: an fp32 ONNX intermediate, then the ONNX converter.
    model = tiny_model(arch)
    onnx_bytes = convert_to_onnx_impl(model, torch.device("cpu"), False, "data")
    converted = Onnx2NcnnConverter(
        safely_optimize_onnx_model(onnx.load_model_from_string(onnx_bytes))
    ).convert(False, False)
    wrapper = NcnnModelWrapper(converted)
    assert wrapper.scale == model.scale

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    converted.write_bin(tmp_path / "model.bin")
    assert net.load_param_mem(converted.write_param()) == 0
    assert net.load_model(str(tmp_path / "model.bin")) == 0
    for image in images(SIZES):
        ex = net.create_extractor()
        assert ex.input(converted.layers[0].outputs[0], ncnn.Mat(image)) == 0
        ret, actual = ex.extract(converted.layers[-1].outputs[0])
        assert ret == 0
        np.testing.assert_allclose(
            np.array(actual), torch_output(model, image), rtol=0, atol=1e-4
        )


# What an export could change on the model it converts: dtype, device, mode, grad
# flags and weights.
Snapshot = tuple[torch.dtype, torch.device, bool, list[bool], dict[str, torch.Tensor]]


def snapshot(model: ImageModelDescriptor) -> Snapshot:
    return (
        model.dtype,
        model.device,
        model.model.training,
        [p.requires_grad for p in model.model.parameters()],
        {k: v.clone() for k, v in model.model.state_dict().items()},
    )


def assert_unchanged(model: ImageModelDescriptor, before: Snapshot):
    after = snapshot(model)
    assert after[:4] == before[:4]
    assert after[4].keys() == before[4].keys()
    assert all(torch.equal(after[4][k], v) for k, v in before[4].items())


@pytest.mark.parametrize("arch", ["Compact", "HAT"])
@pytest.mark.parametrize("loaded_half", [False, True])
@pytest.mark.parametrize("use_half", [False, True])
def test_converting_leaves_the_model_unchanged(
    arch: str, loaded_half: bool, use_half: bool
):
    # Load Model halves a model with PyTorch's FP16 on; Upscale Image and Save Model
    # share it with Convert To ONNX.
    model = tiny_model(arch)
    if loaded_half:
        model.half()
    before = snapshot(model)
    convert_to_onnx_impl(model, torch.device("cpu"), use_half)
    assert_unchanged(model, before)


def test_converting_leaves_trainable_weights_trainable():
    torch.manual_seed(0)
    model = ModelLoader().load_from_state_dict(ARCHS["Compact"]().state_dict())
    assert isinstance(model, ImageModelDescriptor)
    before = snapshot(model)
    convert_to_onnx_impl(model, torch.device("cpu"))
    assert_unchanged(model, before)


class Context:
    """The NodeContext members Convert To ONNX uses: PyTorch on the CPU."""

    def __init__(self, settings: dict[str, object]) -> None:
        self.settings = SettingsParser(settings)


CPU_CONTEXT: dict[str, object] = {"use_cpu": True}


def convert(
    model: ImageModelDescriptor,
    data_type: OnnxDataType = OnnxDataType.FP32,
    opset: Opset = Opset.OPSET_20,
    verify: bool = False,
    calibration: np.ndarray | None = None,
    dual_shape: DualShape = DualShape.DYNAMIC,
    settings: dict[str, object] = CPU_CONTEXT,
) -> tuple[OnnxGeneric, str, str]:
    return convert_to_onnx_node(
        cast(NodeContext, Context(settings)),
        model,
        data_type,
        opset,
        verify,
        calibration,
        dual_shape,
        512,
        512,
    )


def count_exports(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Counts Convert To ONNX's exports (one entry each) and stubs AutoCast, which
    returns its FP32 input (ModelOpt is not installed for the tests)."""
    exports: list[int] = []
    export_onnx = convert_to_onnx.export_onnx

    def counted(model: ModelDescriptor, opset_version: int) -> bytes:
        exports.append(opset_version)
        return export_onnx(model, opset_version=opset_version)

    monkeypatch.setattr(convert_to_onnx, "export_onnx", counted)
    monkeypatch.setattr(convert_to_onnx, "autocast", lambda model, low, image: model)
    return exports


def test_a_rerun_returns_the_kept_export(monkeypatch: pytest.MonkeyPatch):
    exports = count_exports(monkeypatch)
    model = tiny_model("Compact")
    first = convert(model)
    for _ in range(2):
        again = convert(model)
        assert again == first and again[0] is first[0]
    assert len(exports) == 1


def calibration_image(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((40, 48, 3), dtype=np.float32)


def test_each_export_input_has_its_own_export(monkeypatch: pytest.MonkeyPatch):
    exports = count_exports(monkeypatch)
    model = tiny_model("Compact")
    other = tiny_model("Compact")  # the same weights in another model
    runs = [
        lambda: convert(model),
        lambda: convert(model, opset=Opset.OPSET_17),
        lambda: convert(model, OnnxDataType.MIXED_FP16),
        lambda: convert(model, OnnxDataType.MIXED_BF16),
        lambda: convert(
            model, OnnxDataType.MIXED_FP16, calibration=calibration_image(1)
        ),
        lambda: convert(
            model, OnnxDataType.MIXED_FP16, calibration=calibration_image(2)
        ),
        lambda: convert(other),
    ]
    results = [run() for run in runs]
    assert len(exports) == len(runs)
    assert len({id(result[0]) for result in results}) == len(runs)
    # Each model keeps its latest export
    assert runs[-2]()[0] is results[-2][0]
    assert runs[-1]()[0] is results[-1][0]
    assert len(exports) == len(runs)


def test_a_changed_export_input_drops_the_superseded_export(
    monkeypatch: pytest.MonkeyPatch,
):
    # A kept export holds its ONNX sessions (GPU memory) while the model lives.
    count_exports(monkeypatch)
    model = tiny_model("Compact")
    first = weakref.ref(convert(model)[0])
    second = weakref.ref(convert(model, opset=Opset.OPSET_17)[0])
    gc.collect()
    assert first() is None
    model.half()
    convert(model)
    gc.collect()
    assert second() is None


def test_a_model_changed_in_place_is_exported_again(monkeypatch: pytest.MonkeyPatch):
    # Upscale Image converts the shared model to its precision in place.
    exports = count_exports(monkeypatch)
    model = tiny_model("Compact")
    first = convert(model)[0]
    model.half()
    assert convert(model)[0] is not first
    assert len(exports) == 2


def test_verify_checks_an_export_kept_without_it(monkeypatch: pytest.MonkeyPatch):
    exports = count_exports(monkeypatch)
    verified: list[torch.dtype] = []
    monkeypatch.setattr(
        convert_to_onnx,
        "verify_models",
        lambda model, onnx_model, fp16: verified.append(model.dtype),
    )
    model = tiny_model("Compact")
    model.half()
    convert(model)
    convert(model, verify=True)  # exported again and verified
    convert(model, verify=True)
    convert(model)
    assert len(exports) == 2
    # Verified against the export's copy, in the export's precision
    assert verified == [torch.float32]


class DualModel:
    """The members of a DUAL model Convert To ONNX uses."""

    def __init__(self) -> None:
        self.architecture = SimpleNamespace(id="DUAL")
        self.dtype = torch.float32
        self.model = torch.nn.Conv2d(3, 3, 1)
        self.tags: list[str] = []
        self.scale = 2


def test_a_dual_rerun_returns_the_kept_export(monkeypatch: pytest.MonkeyPatch):
    exports: list[tuple[int, int] | None] = []
    onnx_bytes = convert_to_onnx_impl(tiny_model("Compact"), torch.device("cpu"))

    def export(
        state: object,
        tags: object,
        scale: object,
        size: tuple[int, int] | None,
        context: object,
    ) -> bytes:
        exports.append(size)
        return onnx_bytes

    monkeypatch.setattr(convert_to_onnx.dual, "export_onnx", export)
    model = cast(ImageModelDescriptor, DualModel())
    dynamic = convert(model)
    assert convert(model)[0] is dynamic[0]
    fixed = convert(model, dual_shape=DualShape.FIXED)
    assert convert(model, dual_shape=DualShape.FIXED)[0] is fixed[0]
    assert exports == [None, (512, 512)]


class GpuSettings(PyTorchSettings):
    """PyTorch's settings as a GPU has them. The CPU stands in for the GPU here (CUDA
    is hidden), so FP16 skips the CPU's fallback to FP32."""

    def __post_init__(self) -> None:
        pass


# Load Model's settings with PyTorch's FP16 on and off.
FP16 = GpuSettings(use_cpu=False, use_fp16=True, gpu_index=0, budget_limit=0)
FP32 = PyTorchSettings(use_cpu=False, use_fp16=False, gpu_index=0, budget_limit=0)


def model_file(directory: Path, make: Callable[[], torch.nn.Module]) -> Path:
    torch.manual_seed(0)
    path = directory / "model.pth"
    torch.save(make().state_dict(), path)
    return path


def warnings_logged(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    logged: list[str] = []
    monkeypatch.setattr(
        load_model_module.logger,
        "warning",
        lambda message, *args: logged.append(message % args),
    )
    return logged


@pytest.mark.parametrize(
    "data_type", [OnnxDataType.FP32, OnnxDataType.MIXED_FP16, OnnxDataType.MIXED_BF16]
)
def test_an_export_of_a_model_loaded_in_fp16_carries_its_fp32_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data_type: OnnxDataType
):
    count_exports(monkeypatch)
    logged = warnings_logged(monkeypatch)
    path = model_file(tmp_path, ARCHS["Compact"])
    half = load_model(path, FP16)
    assert isinstance(half, ImageModelDescriptor) and half.dtype == torch.float16
    full = load_model(path, FP32)
    assert isinstance(full, ImageModelDescriptor) and full.dtype == torch.float32
    rounded = convert_to_onnx_impl(half, torch.device("cpu"), opset_version=20)
    before = snapshot(half)
    exported = convert(half, data_type)[0].bytes
    assert exported == convert(full, data_type)[0].bytes
    assert exported != rounded
    assert_unchanged(half, before)
    assert logged == []


def test_a_model_converted_back_to_fp32_in_place_still_exports_its_file_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Upscale Image converts the shared model to PyTorch's precision in place.
    count_exports(monkeypatch)
    path = model_file(tmp_path, ARCHS["Compact"])
    model = load_model(path, FP16)
    assert isinstance(model, ImageModelDescriptor)
    model.float()
    full = load_model(path, FP32)
    assert isinstance(full, ImageModelDescriptor)
    assert convert(model)[0].bytes == convert(full)[0].bytes


def test_an_fp16_export_does_not_read_the_file_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    reread: list[object] = []
    monkeypatch.setattr(load_model_module, "full_precision", reread.append)
    monkeypatch.setattr(convert_to_onnx, "get_settings", lambda _: FP16)
    path = model_file(tmp_path, ARCHS["Compact"])
    model = load_model(path, FP16)
    assert isinstance(model, ImageModelDescriptor)
    expected = convert_to_onnx_impl(
        model, torch.device("cpu"), use_half=True, opset_version=20
    )
    exported = convert(model, OnnxDataType.FP16)[0]
    assert exported.bytes == expected
    assert reread == []


def rewrite(path: Path) -> None:
    """Other weights in `path`, a second later (a rewrite within the file system's
    time resolution keeps the identity; the weights' comparison catches that)."""
    before = path.stat()
    torch.manual_seed(1)
    torch.save(ARCHS["Compact"]().state_dict(), path)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 10**9))


class UnreadableLoader:
    """spandrel's ModelLoader for a file it may not read (another program locked it)."""

    def __init__(self, device: torch.device) -> None:
        pass

    def load_from_file(self, path: Path) -> ModelDescriptor:
        raise PermissionError("Permission denied")


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("none", "does not come from Load Model"),
        ("rewritten", "changed after Load Model loaded it"),
        ("deleted", "changed after Load Model loaded it"),
        ("same identity", "no longer holds the weights Load Model loaded"),
        ("unreadable", "could not be read again (Permission denied)"),
    ],
)
def test_without_its_file_weights_the_fp16_weights_are_exported_and_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str, reason: str
):
    count_exports(monkeypatch)
    logged = warnings_logged(monkeypatch)
    if change == "none":
        model = tiny_model("Compact")  # as Interpolate Models returns one
        model.half()
    else:
        if change == "same identity":
            monkeypatch.setattr(load_model_module, "file_identity", lambda path: 0)
        path = model_file(tmp_path, ARCHS["Compact"])
        loaded = load_model(path, FP16)
        assert isinstance(loaded, ImageModelDescriptor)
        model = loaded
        if change == "deleted":
            path.unlink()
        elif change == "unreadable":
            monkeypatch.setattr(load_model_module, "ModelLoader", UnreadableLoader)
        else:
            rewrite(path)
    rounded = convert_to_onnx_impl(model, torch.device("cpu"), opset_version=20)
    assert convert(model)[0].bytes == rounded
    assert len(logged) == 1 and reason in logged[0]
    assert logged[0].startswith("Convert To ONNX: ")
    assert "FP16-rounded weights" in logged[0]


def test_a_dual_model_loaded_in_fp16_exports_its_fp32_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    states: list[dict[str, torch.Tensor]] = []
    onnx_bytes = convert_to_onnx_impl(tiny_model("Compact"), torch.device("cpu"))

    def export(
        state: dict[str, torch.Tensor],
        tags: object,
        scale: object,
        size: object,
        context: object,
    ) -> bytes:
        states.append(state)
        return onnx_bytes

    monkeypatch.setattr(convert_to_onnx.dual, "export_onnx", export)
    path = model_file(tmp_path, lambda: dual_xs(scale=1))
    half = load_model(path, FP16)
    full = load_model(path, FP32)
    assert isinstance(half, ImageModelDescriptor) and half.dtype == torch.float16
    assert isinstance(full, ImageModelDescriptor)
    convert(half)
    (state,) = states
    expected = full.model.state_dict()
    assert state.keys() == expected.keys()
    assert all(
        state[k].dtype == v.dtype and torch.equal(state[k], v)
        for k, v in expected.items()
    )


def tiny_legacy_drft() -> torch.nn.Module:
    """A small random legacy DRFT x2 with LayerNorm (DRFT's TensorRT route)."""
    return legacy.DRFT(
        embed_dim=48,
        depths=(6, 6),
        num_heads=(6, 6),
        window_size=16,
        overlap_ratio=0.5,
        upscale=2,
        num_feat=16,
        rank=8,
        drop_path_rate=0.0,
    )


def test_a_legacy_drft_loaded_in_fp16_exports_its_fp32_weights_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    states: list[dict[str, torch.Tensor]] = []
    devices: list[torch.device | None] = []
    onnx_bytes = convert_to_onnx_impl(tiny_model("Compact"), torch.device("cpu"))

    def export(model: ImageModelDescriptor, device: torch.device | None) -> bytes:
        states.append(model.model.state_dict())
        devices.append(device)
        return onnx_bytes

    monkeypatch.setattr(convert_to_onnx.drft_export, "export_onnx", export)
    path = model_file(tmp_path, tiny_legacy_drft)
    half = load_model(path, FP16)
    full = load_model(path, FP32)
    assert isinstance(half, ImageModelDescriptor) and half.dtype == torch.float16
    assert isinstance(full, ImageModelDescriptor)
    first = convert(half)
    assert first[1:] == ("mixed", "opset20")
    assert convert(half)[0] is first[0]
    (state,) = states
    expected = full.model.state_dict()
    assert state.keys() == expected.keys()
    assert all(
        state[k].dtype == v.dtype and torch.equal(state[k], v)
        for k, v in expected.items()
    )
    # traced where Load Model put the model (FP16 is slow on the CPU)
    assert devices == [half.device]


def to_ncnn(
    model: ImageModelDescriptor, is_fp16: int, directory: Path
) -> tuple[str, bytes]:
    """PyTorch's Convert To NCNN of `model`: its param and bin."""
    converted, _ = convert_to_ncnn_node(
        cast(NodeContext, Context(CPU_CONTEXT)), model, is_fp16
    )
    path = directory / "model.bin"
    converted.model.write_bin(path)
    return converted.model.write_param(), path.read_bytes()


@pytest.mark.parametrize("is_fp16", [0, 1], ids=["fp32", "fp16"])
def test_an_ncnn_conversion_of_a_model_loaded_in_fp16_uses_its_fp32_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, is_fp16: int
):
    # Runs native code, which GitHub's checks do not build.
    pytest.importorskip("nodes.impl._chainner_graph")
    logged = warnings_logged(monkeypatch)
    path = model_file(tmp_path, ARCHS["Compact"])
    half = load_model(path, FP16)
    full = load_model(path, FP32)
    assert isinstance(half, ImageModelDescriptor) and half.dtype == torch.float16
    assert isinstance(full, ImageModelDescriptor)
    converted = to_ncnn(half, is_fp16, tmp_path)
    assert converted == to_ncnn(full, is_fp16, tmp_path)
    assert logged == []
    # The FP16 weights alone (a model Load Model did not load) convert as before. An
    # FP16 conversion differs as well: the converter rounds FP32 weights to FP16 its
    # own way, not as PyTorch rounded them at load.
    rounded = to_ncnn(copy.deepcopy(half), is_fp16, tmp_path)
    assert rounded != converted
    assert len(logged) == 1 and logged[0].startswith("Convert To NCNN: ")
    assert "does not come from Load Model" in logged[0]
