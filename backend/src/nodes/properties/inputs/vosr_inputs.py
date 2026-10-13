from api import BaseInput

from ...impl.pytorch.vosr2 import BundlePaths


class VosrModelInput(BaseInput):
    """Input for a loaded VOSR model folder"""

    def __init__(self, label: str = "VOSR Model"):
        super().__init__("VosrModel", label)
        self.associated_type = BundlePaths
