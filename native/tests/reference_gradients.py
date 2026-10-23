import inspect
import math

import numpy as np


def horizontal_gradient(img: np.ndarray):
    x = np.arange(img.shape[1])
    p = x / (img.shape[1] - 1)
    img[:, :] = p.reshape((1, -1))


def vertical_gradient(img: np.ndarray):
    x = np.arange(img.shape[0])
    p = x / (img.shape[0] - 1)
    img[:, :] = p.reshape((-1, 1))


def diagonal_gradient(img: np.ndarray, angle: float, width: float):
    center = np.array([img.shape[0], img.shape[1]], dtype=np.float32) / 2
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

    start = center - direction * width / 2

    pixels = np.array(
        [[(r, c) for r in range(img.shape[0]) for c in range(img.shape[1])]]
    )
    projection = (pixels - start).dot(direction)
    p = np.clip((projection / width).ravel(), 0, 1)
    img[:] = p.reshape(img.shape)


def radial_gradient(
    img: np.ndarray, inner_radius_percent: float = 0, outer_radius_percent: float = 1
):
    inner_radius = inner_radius_percent * img.shape[1] / 2
    outer_radius = outer_radius_percent * img.shape[1] / 2

    center = np.array(img.shape[:2], dtype="float32") / 2
    pixels = np.array(
        [(r, c) for r in range(img.shape[0]) for c in range(img.shape[1])]
    )
    distance = np.sqrt(np.sum((pixels - center) ** 2, axis=1))
    p = (distance - inner_radius) / (outer_radius - inner_radius)
    img[:] = p.reshape(img.shape)


def conic_gradient(img: np.ndarray, rotation: float = 0):
    if rotation > np.pi:
        rotation -= 2 * np.pi
    if rotation < -np.pi:
        rotation += 2 * np.pi

    center = (np.array(img.shape[:2], dtype="float32") - 1) / 2
    pixels = np.array(
        [(r, c) for r in range(img.shape[0]) for c in range(img.shape[1])]
    )
    angles = np.arctan2(pixels[:, 0] - center[0], pixels[:, 1] - center[1]) + rotation
    angles[angles < 0] += 2 * np.pi
    p = angles / math.pi / 2
    img[:] = p.reshape(img.shape)


def judged(name: str):
    """chaiNNer-C's contract for the gradient `name` (ruling of 2026-10-23), not
    upstream's: the function above, except where it divides 0 by 0. A one-pixel axis
    is position 0; a zero-width diagonal or radial interval is a step, 0 before its
    edge and 1 at or after it. The radial position is clamped to 0..1, as the
    diagonal's is (2026-10-23)."""
    original = globals()[name]
    signature = inspect.signature(original)

    def gradient(img: np.ndarray, *args: float, **kwargs: float):
        with np.errstate(divide="ignore", invalid="ignore"):
            original(img, *args, **kwargs)
        undefined = np.isnan(img)
        if name in ("horizontal_gradient", "vertical_gradient"):
            img[undefined] = 0
        elif name == "diagonal_gradient":
            img[undefined] = 1
        elif name == "radial_gradient":
            radii = signature.bind(img, *args, **kwargs)
            radii.apply_defaults()
            if (
                radii.arguments["inner_radius_percent"]
                == radii.arguments["outer_radius_percent"]
            ):
                img[:] = undefined | (img > 0)
            else:
                np.clip(img, 0, 1, out=img)

    return gradient
