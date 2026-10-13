from api import BaseOutput

from ...impl.pytorch.vosr2 import BundlePaths


class VosrModelOutput(BaseOutput[BundlePaths]):
    def __init__(self, label: str = "VOSR Model"):
        super().__init__("VosrModel", label, associated_type=BundlePaths)
