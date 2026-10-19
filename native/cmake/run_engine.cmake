# The native run engine (src/run) and its native node entry points (src/run/nodes, one file per group listed in
# src/run/nodes/all_nodes.inc), inside _chainner_graph.pyd beside the units that call chainner_native.
file(GLOB chainner_c_run_nodes CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/run/nodes/*.cpp")
target_sources(_chainner_graph PRIVATE src/run/image.cpp src/run/registry.cpp src/run/python.cpp
    src/run/executor.cpp
    ${chainner_c_run_nodes})
