from __future__ import annotations

# `x as x` imports: the native mirror reads these names in native/src/utility_scalar.cpp
import math as math
from enum import Enum

import numpy as np

from nodes.groups import if_enum_group
from nodes.impl.native_graph import graph
from nodes.properties.inputs import EnumInput, NumberInput
from nodes.properties.outputs import NumberOutput

from .. import math_group


class RoundOperation(Enum):
    FLOOR = "Round down"
    CEILING = "Round up"
    ROUND = "Round"


class RoundScale(Enum):
    UNIT = "Integer"
    MULTIPLE = "Multiple of..."
    POWER = "Power of..."


@math_group.register(
    schema_id="chainner:utility:math_round",
    name="Round",
    description="Round an input number",
    icon="MdCalculate",
    inputs=[
        NumberInput("Input", min=None, max=None, precision="unlimited", step=1),
        EnumInput(
            RoundOperation,
            "Operation",
            option_labels={k: k.value for k in RoundOperation},
        ),
        EnumInput(
            RoundScale,
            "To the nearest",
            option_labels={k: k.value for k in RoundScale},
        ),
        if_enum_group(2, RoundScale.MULTIPLE)(
            NumberInput(
                "Multiple",
                default=1,
                min=1e-100,
                max=None,
                precision="unlimited",
                step=1,
            )
        ),
        if_enum_group(2, RoundScale.POWER)(
            NumberInput(
                "Power",
                default=2,
                min=np.nextafter(1.0, np.inf),
                max=None,
                precision="unlimited",
                step=1,
            )
        ),
    ],
    outputs=[
        NumberOutput(
            "Result",
            output_type="""
                let x = Input0;
                let m = Input3;
                let p = Input4;

                match Input2 {
                    RoundScale::Unit => match Input1 {
                        RoundOperation::Floor => floor(x),
                        RoundOperation::Ceiling => ceil(x),
                        RoundOperation::Round => round(x),
                    },
                    RoundScale::Multiple => match Input1 {
                        RoundOperation::Floor => floor(x/m) * m,
                        RoundOperation::Ceiling => ceil(x/m) * m,
                        RoundOperation::Round => round(x/m) * m,
                    },
                    RoundScale::Power => match Input1 {
                        RoundOperation::Floor => p ** floor(number::log(x)/number::log(p)),
                        RoundOperation::Ceiling => p ** ceil(number::log(x)/number::log(p)),
                        RoundOperation::Round => p ** round(number::log(x)/number::log(p)),
                    },
                }
                """,
        )
    ],
)
def round_node(
    a: float,
    operation: RoundOperation,
    scale: RoundScale,
    m: float,
    p: float,
) -> int | float:
    return graph().utility_round(globals(), a, operation, scale, m, p)
