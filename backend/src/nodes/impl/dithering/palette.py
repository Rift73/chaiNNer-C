import numpy as np

from ..image_utils import MAX_VALUES_BY_DTYPE, as_3d
from ..native_analysis import distinct_colors
from ..native_palette import kmeans, median_cut


def _as_float32(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.float32:
        return image
    max_value = MAX_VALUES_BY_DTYPE[image.dtype.name]
    return image.astype(np.float32) / max_value


def distinct_colors_palette(image: np.ndarray) -> np.ndarray:
    return distinct_colors(image)


def kmeans_palette(image: np.ndarray, num_colors: int) -> np.ndarray:
    image = as_3d(image)
    flat_image = _as_float32(image.reshape((-1, image.shape[2])))

    center = kmeans(flat_image, num_colors)
    return center.reshape((1, -1, image.shape[2]))


def median_cut_palette(image: np.ndarray, num_colors: int) -> np.ndarray:
    image = as_3d(image)
    flat_image = _as_float32(image.reshape((-1, image.shape[2])))

    return median_cut(flat_image, max(1, num_colors))
