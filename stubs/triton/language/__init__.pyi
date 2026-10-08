# Triton's kernel language: kernel code is checked by Triton, not by type stubs.

from typing import Any

def __getattr__(name: str) -> Any: ...
