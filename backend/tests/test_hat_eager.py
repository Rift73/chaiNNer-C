from __future__ import annotations

import pytest
import torch
from spandrel import ImageModelDescriptor, ModelLoader
from spandrel.architectures.Compact import Compact
from spandrel.architectures.HAT import HAT

from nodes.impl.pytorch import hat_export

CUDA = torch.device("cuda")


def model_of(net: torch.nn.Module) -> ImageModelDescriptor:
    torch.manual_seed(0)
    model = ModelLoader().load_from_state_dict(net.state_dict())
    assert isinstance(model, ImageModelDescriptor)
    return model.eval()


def tiny_hat() -> ImageModelDescriptor:
    # A 12-wide trunk, which the copy pads to 16
    torch.manual_seed(0)
    return model_of(
        HAT(
            embed_dim=12,
            depths=(2,),
            num_heads=(2,),
            window_size=8,
            compress_ratio=2,
            squeeze_factor=4,
            upsampler="pixelshuffle",
            upscale=2,
        )
    )


@pytest.fixture
def kernel_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The eager norm is available, as where spandrel's kernel can run."""
    padded_norm = pytest.importorskip(
        "spandrel.architectures.__arch_helpers.padded_norm"
    )
    monkeypatch.setattr(padded_norm, "PADDED_NORM", True)
    monkeypatch.setattr(torch.version, "hip", None)


@pytest.mark.usefixtures("kernel_runs")
def test_an_fp16_hat_on_an_nvidia_gpu_runs_a_padded_copy_made_once():
    model = tiny_hat()
    keys = set(model.model.state_dict())
    copied = hat_export.for_eager(model, CUDA, True, True)
    assert copied is not model and copied.model is not model.model
    assert copied.model.conv_first.out_channels == 16
    assert model.model.conv_first.out_channels == 12
    assert set(model.model.state_dict()) == keys
    assert hat_export.for_eager(model, CUDA, True, True) is copied
    assert not any(m.training for m in copied.model.modules())


@pytest.mark.usefixtures("kernel_runs")
def test_the_padded_copy_computes_the_model():
    # On the CPU the norm takes its PyTorch path: the same values as the kernel's.
    model = tiny_hat()
    copied = hat_export.for_eager(model, CUDA, True, True)
    image = torch.rand(1, 3, 16, 24)
    with torch.no_grad():
        torch.testing.assert_close(copied(image), model(image), atol=1e-5, rtol=0)


@pytest.mark.usefixtures("kernel_runs")
@pytest.mark.parametrize(
    ("device", "use_fp16", "hip"),
    [
        (torch.device("cpu"), True, None),
        (CUDA, False, None),
        (CUDA, True, "7.0"),
    ],
)
def test_other_devices_precisions_and_rocm_run_the_model(
    monkeypatch: pytest.MonkeyPatch,
    device: torch.device,
    use_fp16: bool,
    hip: str | None,
):
    monkeypatch.setattr(torch.version, "hip", hip)
    model = tiny_hat()
    assert hat_export.for_eager(model, device, use_fp16, True) is model


@pytest.mark.usefixtures("kernel_runs")
def test_other_architectures_run_as_they_are():
    model = model_of(
        Compact(num_in_ch=3, num_out_ch=3, num_feat=8, num_conv=2, upscale=2)
    )
    assert hat_export.for_eager(model, CUDA, True, True) is model


def test_without_the_kernel_the_model_runs_as_it_is(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(hat_export, "_eager_norm", lambda _arch: None)
    model = tiny_hat()
    assert hat_export.for_eager(model, CUDA, True, True) is model


@pytest.mark.usefixtures("kernel_runs")
def test_forwards_under_the_offloads_dispatch_mode_run_the_model():
    model = tiny_hat()
    assert hat_export.for_eager(model, CUDA, True, False) is model
    assert hat_export.for_eager(model, CUDA, True, True) is not model


def test_after_the_kernel_fails_the_next_image_runs_the_model(
    monkeypatch: pytest.MonkeyPatch,
):
    padded_norm = pytest.importorskip(
        "spandrel.architectures.__arch_helpers.padded_norm"
    )
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(padded_norm, "PADDED_NORM", True)
    model = tiny_hat()
    assert hat_export.for_eager(model, CUDA, True, True) is not model
    # What a Triton failure inside PaddedLayerNorm does
    monkeypatch.setattr(padded_norm, "PADDED_NORM", False)
    assert hat_export.for_eager(model, CUDA, True, True) is model


def test_on_the_gpu_the_padded_copy_runs_the_kernel_and_computes_the_model(
    monkeypatch: pytest.MonkeyPatch,
):
    padded_norm = pytest.importorskip(
        "spandrel.architectures.__arch_helpers.padded_norm"
    )
    if not (
        padded_norm.AVAILABLE
        and torch.cuda.is_available()
        and torch.version.hip is None
    ):
        pytest.skip("the kernel runs on NVIDIA GPUs with Triton")
    launches: list[int] = []
    launch = padded_norm.launch

    def counted(*args: object, **kwargs: object) -> torch.Tensor:
        launches.append(1)
        return launch(*args, **kwargs)

    monkeypatch.setattr(padded_norm, "launch", counted)
    model = tiny_hat()
    copied = hat_export.for_eager(model, CUDA, True, True)
    image = torch.rand(1, 3, 32, 48, device="cuda").half()
    with torch.no_grad():
        expected = model.model.cuda().half()(image).float()
        actual = copied.model.cuda().half()(image).float()
    assert launches
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=0)


def tiny_heart() -> ImageModelDescriptor:
    # A 30-wide trunk with a CAB middle of 10, which the copy pads to 32 and 16
    heart = pytest.importorskip("spandrel.architectures.HEART")
    torch.manual_seed(0)
    return model_of(
        heart.HEART(embed_dim=30, depths=(2,), num_heads=(2,), upscale=2, num_feat=16)
    )


@pytest.fixture
def iln_kernel_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """HEART's padded i-LN is available, as where spandrel's kernel can run."""
    iln = pytest.importorskip("spandrel.architectures.HEART.__arch.iln_triton")
    monkeypatch.setattr(iln, "ILN_KERNEL", True)
    monkeypatch.setattr(torch.version, "hip", None)


@pytest.mark.usefixtures("iln_kernel_runs")
def test_an_fp16_heart_on_an_nvidia_gpu_runs_a_padded_copy_made_once():
    model = tiny_heart()
    keys = set(model.model.state_dict())
    copied = hat_export.for_eager(model, CUDA, True, True)
    assert copied is not model and copied.model is not model.model
    assert copied.model.conv_first.out_channels == 32
    assert model.model.conv_first.out_channels == 30
    block = copied.model.layers[0].residual_group.blocks[0]
    assert block.conv_block.cab[0].out_channels == 16
    assert set(model.model.state_dict()) == keys
    assert hat_export.for_eager(model, CUDA, True, True) is copied
    assert not any(m.training for m in copied.model.modules())


@pytest.mark.usefixtures("iln_kernel_runs")
def test_the_padded_heart_computes_the_model_with_zeros_on_its_padding():
    # On the CPU the i-LN takes its PyTorch path: the same values as the kernel's.
    model = tiny_heart()
    copied = hat_export.for_eager(model, CUDA, True, True)
    padding: list[float] = []

    def record(_module: torch.nn.Module, _inputs: object, output: object) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        # the trunk: tokens (B, L, 32), channels last or channels first; not the
        # channel attention's gate, sigmoid(0) on the padding, which multiplies zeros
        if isinstance(tensor, torch.Tensor) and tensor.numel() >= 32 * 40 * 48:
            if tensor.shape[-1] == 32:
                padding.append(float(tensor[..., 30:].abs().max()))
            elif tensor.dim() == 4 and tensor.shape[1] == 32:
                padding.append(float(tensor[:, 30:].abs().max()))

    hooks = [m.register_forward_hook(record) for m in copied.model.modules()]
    image = torch.rand(1, 3, 40, 48)
    with torch.no_grad():
        torch.testing.assert_close(copied(image), model(image), atol=1e-5, rtol=0)
    for hook in hooks:
        hook.remove()
    assert padding
    assert max(padding) == 0


@pytest.mark.usefixtures("iln_kernel_runs")
def test_a_heart_export_is_not_padded():
    model = tiny_heart()
    model.half()
    assert hat_export.for_export(model) is model


def test_without_heart_s_kernel_the_heart_runs_as_it_is(
    monkeypatch: pytest.MonkeyPatch,
):
    iln = pytest.importorskip("spandrel.architectures.HEART.__arch.iln_triton")
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(iln, "ILN_KERNEL", False)
    model = tiny_heart()
    assert hat_export.for_eager(model, CUDA, True, True) is model
