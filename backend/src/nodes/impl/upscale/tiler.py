from abc import ABC, abstractmethod

from ...utils.utils import Size
from ..native_graph import graph


class Tiler(ABC):
    @abstractmethod
    def allow_smaller_tile_size(self) -> bool:
        return graph().tiling_installed_tiler_Tiler_allow_smaller_tile_size(
            globals(), self
        )

    @abstractmethod
    def starting_tile_size(self, width: int, height: int, channels: int) -> Size:
        return graph().tiling_installed_tiler_Tiler_starting_tile_size(
            globals(), self, width, height, channels
        )

    def split(self, tile_size: Size) -> Size:
        return graph().tiling_installed_tiler_Tiler_split(globals(), self, tile_size)


class NoTiling(Tiler):
    def allow_smaller_tile_size(self) -> bool:
        return graph().tiling_installed_tiler_NoTiling_allow_smaller_tile_size(
            globals(), self
        )

    def starting_tile_size(self, width: int, height: int, channels: int) -> Size:
        return graph().tiling_installed_tiler_NoTiling_starting_tile_size(
            globals(), self, width, height, channels
        )

    def split(self, tile_size: Size) -> Size:
        return graph().tiling_installed_tiler_NoTiling_split(globals(), self, tile_size)


class MaxTileSize(Tiler):
    def __init__(self, tile_size: int = 2**31) -> None:
        graph().tiling_installed_tiler_MaxTileSize_init(globals(), self, tile_size)

    def allow_smaller_tile_size(self) -> bool:
        return graph().tiling_installed_tiler_MaxTileSize_allow_smaller_tile_size(
            globals(), self
        )

    def starting_tile_size(self, width: int, height: int, channels: int) -> Size:
        return graph().tiling_installed_tiler_MaxTileSize_starting_tile_size(
            globals(), self, width, height, channels
        )


class ExactTileSize(Tiler):
    def __init__(self, exact_size: Size) -> None:
        graph().tiling_installed_tiler_ExactTileSize_init(globals(), self, exact_size)

    def allow_smaller_tile_size(self) -> bool:
        return graph().tiling_installed_tiler_ExactTileSize_allow_smaller_tile_size(
            globals(), self
        )

    def starting_tile_size(self, width: int, height: int, channels: int) -> Size:
        return graph().tiling_installed_tiler_ExactTileSize_starting_tile_size(
            globals(), self, width, height, channels
        )

    def split(self, tile_size: Size) -> Size:
        return graph().tiling_installed_tiler_ExactTileSize_split(
            globals(), self, tile_size
        )
