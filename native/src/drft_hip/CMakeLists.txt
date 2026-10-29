# drft_migx_ops: legacy DRFT's MIGraphX ops (drft_migx_ops.cpp over the HIP ports of src/drft's TensorRT plugin
# kernels), beside the DRFT route in nodes/impl/migraphx. Device maths contract a * b + c into fma as nvcc's default
# --fmad=true does for the plugins, so the kernels sum as theirs do.
chainner_c_hip_module(drft_migx_ops migraphx
    HIP drft_hip_kernels.hip
    CXX drft_migx_ops.cpp
    HIP_OPTIONS -ffp-contract=fast)
