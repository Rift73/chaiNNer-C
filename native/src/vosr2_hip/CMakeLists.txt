# vosr2_migx_ops: VOSR's MIGraphX custom op chainner_vosr2_qkv (ops.cpp), the counterpart of the VosrQkvPack TensorRT
# plugin, with src/vosr2/qkv.cu's kernel in HIP (qkv.hip), beside its loader in nodes/impl/migraphx.
chainner_c_hip_module(vosr2_migx_ops migraphx HIP qkv.hip CXX ops.cpp)
