from __future__ import annotations

from nodes.impl.vapoursynth.catalog import storage_dir

from .. import vapoursynth_category
from ..chains import remove_old_videos

clips_group = vapoursynth_category.add_node_group("Clips")
filters_group = vapoursynth_category.add_node_group("Filters")
functions_group = vapoursynth_category.add_node_group("Functions")

# Once per start: the temporary videos of Frames to Clip nodes that no run replaces.
remove_old_videos(storage_dir())
