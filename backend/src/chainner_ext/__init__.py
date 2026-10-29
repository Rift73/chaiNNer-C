import os

if os.name == "nt":
    # chainner_ext.pyd binds the kernels of chainner_native.dll, which ships with the
    # node bridges in nodes/impl, outside the extension's own (searched) directory.
    os.add_dll_directory(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nodes", "impl"
        )
    )
else:
    # chainner_ext.so is linked to chainner_native.so, which ships with the node
    # bridges in nodes/impl, outside the extension's own directory. The loader takes a
    # library already loaded under the name the extension asks for, so loading it by
    # its path first satisfies the dependency.
    import ctypes

    ctypes.CDLL(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "nodes",
            "impl",
            "chainner_native.so",
        )
    )
    del ctypes

from . import chainner_ext
from .chainner_ext import *

del os

__doc__ = chainner_ext.__doc__
if hasattr(chainner_ext, "__all__"):
    __all__ = chainner_ext.__all__
