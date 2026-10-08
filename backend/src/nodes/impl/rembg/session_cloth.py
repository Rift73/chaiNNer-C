from __future__ import annotations

from typing import cast

import numpy as np
from PIL import Image

from nodes.impl.native_framework_images import cloth_labels, cloth_masks
from nodes.utils.utils import get_h_w_c

from .session_base import BaseSession

pallete1 = [
    0,
    0,
    0,
    255,
    255,
    255,
    0,
    0,
    0,
    0,
    0,
    0,
]

pallete2 = [
    0,
    0,
    0,
    0,
    0,
    0,
    255,
    255,
    255,
    0,
    0,
    0,
]

pallete3 = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    255,
    255,
    255,
]


class ClothSession(BaseSession):
    def predict(self, img: np.ndarray) -> list[np.ndarray]:
        h, w, _ = get_h_w_c(img)
        ort_outs = self.run(img)

        pred = cloth_labels(cast(np.ndarray, ort_outs[0]))
        mask = Image.fromarray(pred, mode="L")
        mask = mask.resize((w, h), Image.Resampling.LANCZOS)

        return cloth_masks(np.asarray(mask))
