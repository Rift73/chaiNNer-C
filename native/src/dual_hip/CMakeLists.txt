# dual_migx_ops: DUAL's MIGraphX custom ops (dual_migx_ops.cpp), beside nodes/impl/migraphx/dual.py. Host code
# only: the kernels are Triton's, compiled at Build Program time and carried by the program's nodes. Linux only,
# as MIGraphX.
if(NOT WIN32)
    chainner_c_hip_module(dual_migx_ops migraphx CXX dual_migx_ops.cpp)
endif()
