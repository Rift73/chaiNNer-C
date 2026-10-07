"""The tested-stack version constants and the OpenCV gates' fallback."""

from types import SimpleNamespace

import cv2
import numpy as np
import onnxruntime as ort
import pytest

from nodes.impl import (
    native_bilateral,
    native_color_complete,
    native_convolution,
    native_cv_resize,
    native_versions,
)


def capture_warnings(monkeypatch):
    messages = []
    stub = SimpleNamespace(warning=lambda *args: messages.append(args))
    monkeypatch.setattr(native_versions, "logger", stub)
    return messages


def test_constants_equal_installed_versions():
    assert (native_versions.CV2, native_versions.NUMPY, native_versions.ORT) == (
        cv2.__version__,
        np.__version__,
        ort.__version__,
    )


def test_channel_ceiling_equals_the_installed_opencv():
    # CV_CN_MAX channels convert to a 2-D Mat; one more becomes a 3-D Mat, which a
    # 2-D kernel refuses.
    ceiling = native_versions.CV_CN_MAX
    cv2.blur(np.zeros((5, 5, ceiling), np.float32), (3, 3))
    with pytest.raises(cv2.error, match="dims <= 2"):
        cv2.blur(np.zeros((5, 5, ceiling + 1), np.float32), (3, 3))


def test_gate_compares_exactly_and_logs_its_first_fallback_once(monkeypatch):
    messages = capture_warnings(monkeypatch)
    assert native_versions.cv2_is_tested("tested gate")
    monkeypatch.setattr(cv2, "__version__", native_versions.CV2 + ".1")
    for _ in range(3):
        assert not native_versions.cv2_is_tested("first gate")
    assert not native_versions.cv2_is_tested("second gate")
    assert [message[1:] for message in messages] == [
        ("first gate", native_versions.CV2 + ".1", native_versions.CV2),
        ("second gate", native_versions.CV2 + ".1", native_versions.CV2),
    ]


def test_each_gate_keeps_opencv_under_another_version(monkeypatch):
    messages = capture_warnings(monkeypatch)
    monkeypatch.setattr(cv2, "__version__", "4.14.0")
    image = np.random.default_rng(7).random((6, 7, 3), dtype=np.float32)
    for _ in range(2):
        np.testing.assert_array_equal(
            native_color_complete.cvt_color(image, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        )
        np.testing.assert_array_equal(
            native_cv_resize.resize(image, (3, 4), cv2.INTER_AREA),
            cv2.resize(image, (3, 4), interpolation=cv2.INTER_AREA),
        )
        np.testing.assert_array_equal(
            native_bilateral.bilateral(image, 1, 0.1, 1.0),
            cv2.bilateralFilter(image, 3, 0.1, 1.0, borderType=cv2.BORDER_REFLECT_101),
        )
        kernel = np.full((3, 3), 1 / 9, np.float32)
        padded = cv2.copyMakeBorder(
            image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0.0,)
        )
        np.testing.assert_array_equal(
            native_convolution.convolve(image, kernel, 1),
            cv2.filter2D(padded, -1, kernel),
        )
    assert [message[1] for message in messages] == [
        native_color_complete.__name__,
        native_cv_resize.__name__,
        native_bilateral.__name__,
        native_convolution.__name__,
    ]
