from __future__ import annotations

import os
import sys
from enum import Enum

import numpy as np
from PIL import Image, ImageDraw

from nodes.groups import icon_set_group, menu_icon_row_group
from nodes.impl import native_text
from nodes.impl.caption import get_font_size
from nodes.impl.color.color import Color
from nodes.impl.font_cache import font_lease
from nodes.impl.image_utils import to_uint8
from nodes.impl.native_buffers import colorize_text
from nodes.properties.inputs import (
    Anchor,
    AnchorInput,
    BoolInput,
    ColorInput,
    EnumInput,
    NumberInput,
    TextInput,
)
from nodes.properties.outputs import ImageOutput

from .. import create_images_group

TEXT_AS_IMAGE_FONT_PATH = [
    ["Roboto/Roboto-Regular.ttf", "Roboto/Roboto-Italic.ttf"],
    ["Roboto/Roboto-Bold.ttf", "Roboto/Roboto-BoldItalic.ttf"],
]


class TextAlignment(Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


@create_images_group.register(
    schema_id="chainner:image:text_as_image",
    name="Text As Image",
    description="Create an image using any text.",
    icon="MdTextFields",
    inputs=[
        TextInput("Text", multiline=True, label_style="hidden"),
        menu_icon_row_group()(
            icon_set_group("Style")(
                BoolInput("Bold", default=False, icon="FaBold").with_id(1),
                BoolInput("Italic", default=False, icon="FaItalic").with_id(2),
            ),
            EnumInput(
                TextAlignment,
                label="Alignment",
                preferred_style="icons",
                icons={
                    TextAlignment.LEFT: "FaAlignLeft",
                    TextAlignment.CENTER: "FaAlignCenter",
                    TextAlignment.RIGHT: "FaAlignRight",
                },
                default=TextAlignment.CENTER,
            ).with_id(4),
        ),
        ColorInput(channels=[3], default=Color.bgr((0, 0, 0))).with_id(3),
        NumberInput("Width", min=1, max=None, default=500, unit="px").with_id(5),
        NumberInput("Height", min=1, max=None, default=100, unit="px").with_id(6),
        AnchorInput(label="Position", icon="MdTextFields").with_id(7),
    ],
    outputs=[
        ImageOutput(
            image_type="\n                Image {\n                    width: Input5,\n                    height: Input6,\n                }\n                ",
            channels=4,
            assume_normalized=True,
        )
    ],
)
def text_as_image_node(
    text: str,
    bold: bool,
    italic: bool,
    alignment: TextAlignment,
    color: Color,
    width: int,
    height: int,
    position: Anchor,
) -> np.ndarray:
    path = TEXT_AS_IMAGE_FONT_PATH[int(bold)][int(italic)]
    font_path = os.path.join(
        os.path.dirname(sys.modules["__main__"].__file__),  # type: ignore
        f"fonts/{path}",  # type: ignore
    )
    lines, max_line = native_text.scan(text)
    line_count = len(lines)
    with font_lease(font_path, 100) as font:
        w_ref, h_ref = (get_font_size(font, max_line)[0], get_font_size(font, "[§]")[1])
    font_size = native_text.fit(width, height, w_ref, h_ref, line_count)
    with font_lease(font_path, font_size) as font:
        w_text, h_text = get_font_size(font, max_line)
        h_text *= line_count
        ink = tuple(to_uint8(np.array(color.value)))
        pil_image = Image.new("L", (width, height))
        drawing = ImageDraw.Draw(pil_image)
        xy = native_text.anchor(
            width, height, w_text, h_text, list(Anchor).index(position)
        )
        native_text.draw_lines(drawing, xy, lines, font, alignment.value)
    img = colorize_text(np.array(pil_image), ink)
    return img
