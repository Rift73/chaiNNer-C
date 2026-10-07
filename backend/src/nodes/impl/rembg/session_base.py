from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from nodes.impl.native_framework_images import normalize_model
from nodes.impl.resize import ResizeFilter, resize

if TYPE_CHECKING:
    # Annotation only: no new import at runtime.
    from nodes.impl.onnx.session import OnnxSession


class BaseSession(ABC):
    def __init__(
        self,
        inner_session: OnnxSession,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        size: tuple[int, int],
    ):
        self.inner_session = inner_session
        self.mean = mean
        self.std = std
        self.size = size

    def normalize(self, img: np.ndarray) -> dict[str, np.ndarray]:
        img = resize(img, self.size, ResizeFilter.LANCZOS)

        model_input_name = self.inner_session.get_inputs()[0].name
        return {model_input_name: normalize_model(img, self.mean, self.std)}

    @abstractmethod
    def predict(self, img: np.ndarray) -> list[np.ndarray]:
        pass
