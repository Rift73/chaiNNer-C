from __future__ import annotations

import math
from math import ceil

import numpy as np

from nodes.groups import linked_inputs_group
from nodes.impl.native_box import kernel_2d, separable_box
from nodes.impl.native_buffers import freeze_normalized
from nodes.impl.native_convolution import box_blur
from nodes.impl.native_spectral_filter import filter2d
from nodes.properties.inputs import ImageInput, SliderInput
from nodes.properties.outputs import ImageOutput

from .. import blur_group


def get_kernel_1d(radius: float) -> np.ndarray:
    kernel = np.ones(ceil(radius) * 2 + 1, np.float32)

    d = radius % 1
    if d != 0:
        kernel[0] *= d
        kernel[-1] *= d

    # normalize
    kernel /= np.sum(kernel)

    return kernel


def get_kernel_2d(radius_x: float, radius_y: float) -> np.ndarray:
    return kernel_2d(radius_x, radius_y)


@blur_group.register(
    schema_id="chainner:image:blur",
    name="Box Blur",
    description="Apply box/average blur to an image.",
    icon="MdBlurOn",
    inputs=[
        ImageInput(),
        linked_inputs_group(
            SliderInput(
                "Radius X",
                min=0,
                max=1000,
                default=1,
                precision=1,
                step=1,
                slider_step=0.1,
                scale="log",
            ),
            SliderInput(
                "Radius Y",
                min=0,
                max=1000,
                default=1,
                precision=1,
                step=1,
                slider_step=0.1,
                scale="log",
            ),
        ),
    ],
    outputs=[ImageOutput(shape_as=0)],
)
def box_blur_node(
    img: np.ndarray,
    radius_x: float,
    radius_y: float,
) -> np.ndarray:
    if radius_x == 0 and radius_y == 0:
        return img

    # Every other return is a helper's fresh float32 array, clamped in place as the
    # output enforce would convert it, frozen and registered (freeze_normalized), so
    # the enforce can borrow it.

    # you can't tell the difference between a float and an integer when the radius is large enough
    radius_x = round(radius_x) if radius_x > 200 else radius_x
    radius_y = round(radius_y) if radius_y > 200 else radius_y

    # both radii are integers
    use_optimized_int = int(radius_x) == radius_x and int(radius_y) == radius_y

    if use_optimized_int:
        # we can use an optimized box blur implementation
        radius_x = round(radius_x)
        radius_y = round(radius_y)
        return freeze_normalized(box_blur(img, radius_x, radius_y), clamp=True)

    # cv2.blur is so much faster than the other methods, that it's worth manually separating the kernel.
    # the idea here is that we blur with cv2.blur in x or y if we can
    threshold = 15
    if radius_x >= threshold and int(radius_x) == radius_x:
        img = box_blur(img, int(radius_x), 0)
        radius_x = 1
    if radius_y >= threshold and int(radius_y) == radius_y:
        img = box_blur(img, 0, int(radius_y))
        radius_y = 1

    # Separable filter is faster for relatively small kernels, but after a certain size it becomes
    # slower than filter2D's DFT implementation. The exact cutoff depends on the hardware.
    avg_radius = math.sqrt(radius_x * radius_y)
    use_sep = avg_radius < 70

    if use_sep:
        return freeze_normalized(separable_box(img, radius_x, radius_y), clamp=True)
    else:
        return freeze_normalized(
            filter2d(img, get_kernel_2d(radius_x, radius_y)), clamp=True
        )
