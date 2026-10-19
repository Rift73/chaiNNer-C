"""Spandrel builds a model without random weight initialisation inside
skip_random_init, and the model is bit-identical: its state dict, every buffer
(persistent or not) and its output."""

from __future__ import annotations

import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from spandrel import MAIN_REGISTRY, ImageModelDescriptor, ModelDescriptor, ModelLoader
from spandrel.architectures.DAT import DAT
from spandrel.architectures.DRFT.__arch.drft_arch import drft_nano
from spandrel.architectures.DUAL.__arch.dual_arch import dual_xs
from spandrel.architectures.FDAT.__arch.fdat import FDAT
from spandrel.architectures.PLKSR.__arch.RealPLKSR import RealPLKSR
from spandrel.architectures.RealCUGAN import UpCunet2x
from spandrel.architectures.SPAN import SPAN
from spandrel.architectures.SwinIR import SwinIR
from spandrel_extra_arches import EXTRA_REGISTRY
from test_convert_to_onnx import ARCHS as CONVERT_ARCHS

from nodes.impl.pytorch.skip_init import skip_random_init
from packages.chaiNNer_pytorch.pytorch.io.load_model import load_model
from packages.chaiNNer_pytorch.settings import PyTorchSettings

MAIN_REGISTRY.add(*EXTRA_REGISTRY, ignore_duplicates=True)
BACKEND = Path(__file__).resolve().parents[1] / "src"

ARCHS: dict[str, Callable[[], torch.nn.Module]] = {
    **CONVERT_ARCHS,
    "SPAN": lambda: SPAN(num_in_ch=3, num_out_ch=3, feature_channels=16, upscale=2),
    "DRFT": lambda: drft_nano(scale=2),
    # DAT2's distinguishing settings, smaller
    "DAT2": lambda: DAT(
        embed_dim=24,
        depth=[2, 2],
        num_heads=[4, 4],
        expansion_factor=2,
        split_size=[8, 16],
        upsampler="pixelshuffle",
        upscale=2,
    ),
    "SwinIR": lambda: SwinIR(
        embed_dim=12,
        depths=[2],
        num_heads=[2],
        window_size=8,
        upscale=2,
        upsampler="pixelshuffle",
    ),
    "RealCUGAN": UpCunet2x,
    "RealPLKSR": lambda: RealPLKSR(dim=16, n_blocks=2, upscaling_factor=2),
    "FDAT": lambda: FDAT(
        scale=2, embed_dim=16, num_groups=1, depth_per_group=2, mid_dim=16
    ),
    "DUAL": lambda: dual_xs(scale=1),
}


def assert_identical(actual: ModelDescriptor, expected: ModelDescriptor):
    assert actual.architecture.id == expected.architecture.id
    state, reference = actual.model.state_dict(), expected.model.state_dict()
    assert state.keys() == reference.keys()
    for key, value in reference.items():
        assert state[key].dtype == value.dtype and torch.equal(state[key], value), key
    buffers = dict(actual.model.named_buffers(remove_duplicate=False))
    reference_buffers = dict(expected.model.named_buffers(remove_duplicate=False))
    assert buffers.keys() == reference_buffers.keys()
    for key, value in reference_buffers.items():
        assert buffers[key].dtype == value.dtype, key
        assert torch.equal(buffers[key], value), key
    assert isinstance(actual, ImageModelDescriptor)
    assert isinstance(expected, ImageModelDescriptor)
    image = torch.rand(1, expected.input_channels, 32, 32)
    with torch.no_grad():
        assert torch.equal(actual.eval()(image), expected.eval()(image))


@pytest.mark.parametrize("arch", ARCHS)
def test_a_model_built_without_random_init_is_identical(arch: str):
    torch.manual_seed(0)
    state = ARCHS[arch]().state_dict()
    expected = MAIN_REGISTRY.load(dict(state))
    with skip_random_init():
        actual = MAIN_REGISTRY.load(dict(state))
    assert_identical(actual, expected)


def test_load_model_builds_without_random_init(tmp_path: Path):
    torch.manual_seed(0)
    path = tmp_path / "hat.pth"
    torch.save(ARCHS["HAT"]().state_dict(), path)
    settings = PyTorchSettings(
        use_cpu=True, use_fp16=False, gpu_index=0, budget_limit=0
    )
    torch.manual_seed(1)
    expected = ModelLoader("cpu").load_from_file(path)
    after_init = torch.rand(8)
    torch.manual_seed(1)
    actual = load_model(path, settings)
    # The skipped initialisers drew no random numbers.
    assert not torch.equal(torch.rand(8), after_init)
    assert_identical(actual, expected)


def sentinel() -> torch.Tensor:
    return torch.full((4, 4), 7.0)


@pytest.mark.parametrize(
    "initialise",
    [
        torch.nn.init.uniform_,
        torch.nn.init.normal_,
        torch.nn.init.trunc_normal_,
        torch.nn.init.xavier_uniform_,
        torch.nn.init.xavier_normal_,
        torch.nn.init.kaiming_uniform_,
        torch.nn.init.kaiming_normal_,
        torch.nn.init.orthogonal_,
        lambda tensor: torch.nn.init.sparse_(tensor, 0.5),
    ],
)
def test_the_random_initialisers_are_skipped_inside_only(
    initialise: Callable[[torch.Tensor], torch.Tensor],
):
    with skip_random_init():
        tensor = sentinel()
        assert initialise(tensor) is tensor
        assert torch.equal(tensor, sentinel())
    assert not torch.equal(initialise(sentinel()), sentinel())


def test_the_deterministic_initialisers_still_run():
    with skip_random_init():
        assert torch.equal(torch.nn.init.zeros_(sentinel()), torch.zeros(4, 4))
        assert torch.equal(torch.nn.init.ones_(sentinel()), torch.ones(4, 4))
        assert torch.equal(
            torch.nn.init.constant_(sentinel(), 2.0), torch.full((4, 4), 2.0)
        )
        assert torch.equal(torch.nn.init.eye_(sentinel()), torch.eye(4))


def test_another_thread_initialises_while_one_skips():
    inside, done = threading.Event(), threading.Event()
    results: dict[str, torch.Tensor] = {}

    def skipping() -> None:
        with skip_random_init():
            inside.set()
            done.wait(timeout=60)
            results["skipping"] = torch.nn.init.kaiming_uniform_(sentinel())

    thread = threading.Thread(target=skipping)
    thread.start()
    assert inside.wait(timeout=60)
    results["other"] = torch.nn.init.kaiming_uniform_(sentinel())
    done.set()
    thread.join(timeout=60)
    assert torch.equal(results["skipping"], sentinel())
    assert not torch.equal(results["other"], sentinel())


def test_nested_blocks_restore_the_outer_one():
    with skip_random_init():
        with skip_random_init():
            pass
        assert torch.equal(torch.nn.init.normal_(sentinel()), sentinel())
    assert not torch.equal(torch.nn.init.normal_(sentinel()), sentinel())


def test_importing_the_module_again_replaces_its_wrappers():
    # vsview's Reload Script imports the PyTorch nodes again. Run apart: the test
    # process keeps one module.
    code = "\n".join(
        [
            "import importlib",
            "import sys",
            f"sys.path[:0] = [{str(BACKEND)!r}]",
            "import torch",
            "original = torch.nn.init.kaiming_uniform_",
            "from nodes.impl.pytorch import skip_init",
            "del sys.modules['nodes.impl.pytorch.skip_init']",
            "again = importlib.import_module('nodes.impl.pytorch.skip_init')",
            "assert again is not skip_init",
            "assert torch.nn.init.kaiming_uniform_.__wrapped__ is original",
            "tensor = torch.full((4, 4), 7.0)",
            "with again.skip_random_init():",
            "    torch.nn.init.kaiming_uniform_(tensor)",
            "assert bool((tensor == 7).all())",
            "print('ok', flush=True)",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "ok"
