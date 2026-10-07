from __future__ import annotations

from api import NodeGroup

from .. import tensorrt_category

io_group: NodeGroup | None = None
processing_group: NodeGroup | None = None
utility_group: NodeGroup | None = None

if tensorrt_category is not None:
    io_group = tensorrt_category.add_node_group("Input & Output")
    processing_group = tensorrt_category.add_node_group("Processing")
    utility_group = tensorrt_category.add_node_group("Utility")
