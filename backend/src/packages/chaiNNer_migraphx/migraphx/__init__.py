from __future__ import annotations

from api import NodeGroup

from .. import migraphx_category, migraphx_loaded

io_group: NodeGroup | None = None
processing_group: NodeGroup | None = None
utility_group: NodeGroup | None = None

if migraphx_category is not None and migraphx_loaded:
    io_group = migraphx_category.add_node_group("Input & Output")
    processing_group = migraphx_category.add_node_group("Processing")
    utility_group = migraphx_category.add_node_group("Utility")
