"""A declined native WebP encode must report OpenCV's failure before writing."""

from pathlib import Path

import numpy as np
import pytest

from nodes.impl import native_image_io


def test_failed_webp_encode_reports_format_and_size(tmp_path: Path):
    path = tmp_path / "too-wide.webp"
    image = np.zeros((1, 16384, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match=r"WEBP.*16384.*1"):
        native_image_io.cv_save_image(path, image, [])

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []
