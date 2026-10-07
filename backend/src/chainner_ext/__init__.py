import os

if os.name == "nt":
    # chainner_ext.pyd binds the kernels of chainner_native.dll, which ships with the
    # node bridges in nodes/impl, outside the extension's own (searched) directory.
    os.add_dll_directory(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nodes", "impl"
        )
    )

from . import chainner_ext
from .chainner_ext import *

del os

__doc__ = chainner_ext.__doc__
if hasattr(chainner_ext, "__all__"):
    __all__ = chainner_ext.__all__
