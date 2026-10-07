from __future__ import annotations

from enum import Enum

from api import BaseInput, Collector, IteratorInputInfo
from nodes.impl.native_graph import graph
from nodes.properties.inputs import EnumInput
from nodes.properties.outputs import NumberOutput

from .. import math_group


class AnyNumberInput(BaseInput):
    def __init__(self, label: str):
        super().__init__(
            input_type="number",
            label=label,
            kind="generic",
            has_handle=True,
            associated_type=float,
        )


class Operation(Enum):
    SUM = "sum"
    PRODUCT = "prod"
    MAXIMUM = "max"
    MINIMUM = "min"

    @property
    def neutral(self) -> float:
        return graph().execution_accumulate_neutral(globals(), self)

    def reduce(self, a: float, b: float) -> float:
        return graph().execution_accumulate_reduce(globals(), self, a, b)


@math_group.register(
    schema_id="chainner:utility:accumulate",
    name="Accumulate",
    description="Calculates a single number of a sequence of number.",
    icon="MdCalculate",
    inputs=[
        AnyNumberInput("Numbers"),
        EnumInput(Operation),
    ],
    iterator_inputs=IteratorInputInfo(inputs=0),
    outputs=[
        NumberOutput(
            "Result",
            output_type="""
                let x = Input0;
                let op = Input1;
                let length = int(0..); // TODO: iterator sequence length

                let neutral = match op {
                    Operation::Sum     => 0,
                    Operation::Product => 1,
                    Operation::Maximum => -inf,
                    Operation::Minimum => inf,
                };

                match length {
                    0 => neutral,
                    uint => match op {
                        Operation::Sum     => x * length,
                        Operation::Product => x ** length,
                        Operation::Maximum => x,
                        Operation::Minimum => x,
                    }
                }
                """,
        )
    ],
    kind="collector",
)
def accumulate_node(_: None, operation: Operation) -> Collector[float, float]:
    return graph().execution_accumulate(globals(), operation)
