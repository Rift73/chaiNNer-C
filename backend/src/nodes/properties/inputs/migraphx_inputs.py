import navi
from api import BaseInput

from ...impl.migraphx.model import MIGraphXProgram


class MIGraphXProgramInput(BaseInput):
    """Input for a compiled MIGraphX program."""

    def __init__(
        self,
        label: str = "MIGraphX Program",
        input_type: navi.ExpressionJson = "MIGraphXProgram",
    ):
        super().__init__(input_type, label)
        self.associated_type = MIGraphXProgram
