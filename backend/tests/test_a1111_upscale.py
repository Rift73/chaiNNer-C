"""The Automatic1111 Upscale node against a stand-in for A1111's API (upstream chaiNNer
#2999 and #3226). Images travel as their (height, width), so no PNG coding is
involved."""

from __future__ import annotations

import inspect
from typing import Any, Callable

import numpy as np
import pytest

# The node's modules load the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from packages.chaiNNer_external.external_stable_diffusion.automatic1111 import (
    upscale as node,
)
from packages.chaiNNer_external.web_ui import (
    STABLE_DIFFUSION_EXTRA_SINGLE_IMAGE_PATH,
    STABLE_DIFFUSION_UPSCALERS_PATH,
    ExternalServiceHTTPError,
    UpscalerName,
)

ScaleBy = node.UpscalerMode.SCALE_BY
ScaleTo = node.UpscalerMode.SCALE_TO


# A1111's upscaler list before ScuNET GAN's first use, and after it, once the model
# is saved as ScuNET.pth (extensions-builtin/ScuNET/scripts/scunet_model.py).
UPSCALERS = ["None", "Lanczos", "Nearest", "ESRGAN_4x", "ScuNET GAN", "ScuNET PSNR"]
UPSCALERS_AFTER_SCUNET_USE = [
    "None",
    "Lanczos",
    "Nearest",
    "ESRGAN_4x",
    "ScuNET",
    "ScuNET PSNR",
]


class FakeA1111:
    """Sizes the result as A1111's /extra-single-image does: modules/upscaler.py
    (`int((w * scale) // 8 * 8)` from 1.4 on, `int(w * scale)` before) with
    scripts/postprocessing_upscale.py's scale-to ratio and crop canvas. Upscalers
    are looked up by their exact name, as scripts/postprocessing_upscale.py does."""

    def __init__(
        self,
        rounds: bool = True,
        returns: tuple[int, int] | None = None,
        upscalers: list[str] = UPSCALERS,
    ):
        self.rounds = rounds
        self.returns = returns
        self.upscalers = upscalers
        self.got: list[str] = []
        self.posted: list[dict[str, Any]] = []

    def get(self, path: str) -> list[dict[str, Any]]:
        assert path == STABLE_DIFFUSION_UPSCALERS_PATH
        self.got.append(path)
        return [{"name": name, "scale": 4} for name in self.upscalers]

    def post(self, path: str, json_data: dict[str, Any]) -> dict[str, Any]:
        assert path == STABLE_DIFFUSION_EXTRA_SINGLE_IMAGE_PATH
        self.posted.append(json_data)
        for key in ("upscaler_1", "upscaler_2"):
            if json_data[key] not in self.upscalers:
                raise ExternalServiceHTTPError(
                    f"could not find upscaler named {json_data[key]}"
                )
        ih, iw = json_data["image"]
        if json_data["resize_mode"] == 1:
            width, height = (
                json_data["upscaling_resize_w"],
                json_data["upscaling_resize_h"],
            )
            scale = max(width / iw, height / ih)
        else:
            scale = json_data["upscaling_resize"]
        if self.rounds:
            size = (int((ih * scale) // 8 * 8), int((iw * scale) // 8 * 8))
        else:
            size = (int(ih * scale), int(iw * scale))
        if json_data["resize_mode"] == 1 and json_data["upscaling_crop"]:
            size = (json_data["upscaling_resize_h"], json_data["upscaling_resize_w"])
        return {"image": self.returns or size}


InstallA1111 = Callable[..., FakeA1111]


@pytest.fixture
def a1111(monkeypatch: pytest.MonkeyPatch) -> InstallA1111:
    """Installs a FakeA1111 made with the given arguments and returns it."""

    def install(**kwargs: Any) -> FakeA1111:
        api = FakeA1111(**kwargs)
        monkeypatch.setattr(node, "get_api", lambda: api)
        monkeypatch.setattr(node, "encode_base64_image", lambda image: image.shape[:2])
        monkeypatch.setattr(
            node, "decode_base64_image", lambda hw: np.zeros((*hw, 3), np.float32)
        )
        return api

    return install


def upscale(
    in_w: int,
    in_h: int,
    mode: node.UpscalerMode,
    factor: float = 4.0,
    width: int = 512,
    height: int = 512,
    crop: bool = False,
    upscaler: UpscalerName = UpscalerName.LANCZOS,
    second: UpscalerName | None = None,
) -> tuple[int, int]:
    """The node's result for an in_w x in_h image, as (width, height)."""
    run = inspect.unwrap(node.upscale_node)  # without the node's output cache
    image = np.zeros((in_h, in_w, 3), np.float32)
    result = run(
        image,
        mode,
        factor,
        width,
        height,
        crop,
        upscaler,
        second is not None,
        second or upscaler,
        0.5,
    )
    return result.shape[1], result.shape[0]


def test_scale_to_without_crop_accepts_the_multiple_of_8(a1111: InstallA1111):
    a1111()
    # The report: 532x400 to 2720x2048 is a ratio of 5.12, i.e. 2723.84x2048.
    assert upscale(532, 400, ScaleTo, width=2720, height=2048) == (2720, 2048)


def test_scale_by_accepts_the_multiple_of_8(a1111: InstallA1111):
    a1111()
    assert upscale(533, 401, ScaleBy, factor=4.0) == (2128, 1600)
    assert upscale(532, 400, ScaleBy, factor=2.5) == (1328, 1000)


def test_scale_by_still_accepts_a1111_before_1_4(a1111: InstallA1111):
    a1111(rounds=False)
    assert upscale(533, 401, ScaleBy, factor=4.0) == (2132, 1604)


def test_scale_to_with_crop_is_exact(a1111: InstallA1111):
    a1111()
    assert upscale(533, 401, ScaleTo, width=2000, height=1000, crop=True) == (
        2000,
        1000,
    )


def test_wrong_size_names_the_expected_sizes(a1111: InstallA1111):
    a1111(returns=(100, 100))
    with pytest.raises(AssertionError) as error:
        upscale(533, 401, ScaleBy, factor=4.0)
    assert str(error.value).startswith(
        "Expected the returned image to be 2128x1600px or 2132x1604px but found"
        " 100x100px instead"
    )


# Upstream chaiNNer #3226: ScuNET GAN under the name A1111 lists it by.


def test_scunet_gan_is_sent_as_scunet_once_a1111_lists_it_so(a1111: InstallA1111):
    api = a1111(upscalers=UPSCALERS_AFTER_SCUNET_USE)
    gan = UpscalerName.SCUNET_GAN
    assert upscale(64, 48, ScaleBy, upscaler=gan, second=gan) == (256, 192)
    assert (api.posted[0]["upscaler_1"], api.posted[0]["upscaler_2"]) == (
        "ScuNET",
        "ScuNET",
    )


def test_scunet_gan_keeps_its_name_while_a1111_lists_it(a1111: InstallA1111):
    api = a1111()
    upscale(64, 48, ScaleBy, upscaler=UpscalerName.SCUNET_GAN)
    assert api.posted[0]["upscaler_1"] == "ScuNET GAN"


def test_other_upscalers_are_sent_without_reading_the_list(a1111: InstallA1111):
    api = a1111(upscalers=UPSCALERS_AFTER_SCUNET_USE)
    upscale(64, 48, ScaleBy, upscaler=UpscalerName.SCUNET_PSNR)
    assert api.posted[0]["upscaler_1"] == "ScuNET PSNR"
    assert api.posted[0]["upscaler_2"] == "None"
    assert api.got == []
