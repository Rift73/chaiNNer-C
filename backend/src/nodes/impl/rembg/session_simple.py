from __future__ import annotations

from typing import cast

import numpy as np

from nodes.impl.image_utils import normalize
from nodes.impl.native_framework_images import normalize_prediction
from nodes.impl.resize import ResizeFilter, resize
from nodes.utils.utils import get_h_w_c

from .session_base import BaseSession


class SimpleSession(BaseSession):
    def predict(self, img: np.ndarray) -> list[np.ndarray]:
        h, w, _ = get_h_w_c(img)
        ort_outs = self.run(img)

        pred = cast(np.ndarray, ort_outs[0])[:, 0, :, :]

        pred = normalize_prediction(pred)
        mask = normalize(np.squeeze(pred))
        mask = np.squeeze(resize(mask, (w, h), ResizeFilter.LANCZOS))

        return [mask]
