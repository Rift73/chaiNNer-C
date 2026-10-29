# The TensorRT runner (yuv-planes DESIGN, node 5): one static library on nvinfer 11 and cudart,
# linked into _chainner_trt.pyd (the TensorRT Upscale node's binding) and, when
# CHAINNER_C_VAPOURSYNTH_INCLUDE names VapourSynth's API 4 headers, chainner_trt_vs.dll (a
# VapourSynth plugin), both beside nodes/impl/tensorrt. Configured with vosr2_qkv.dll, from the
# same TensorRT and CUDA roots. kernels.cu is compiled by nvcc as qkv.cu is (the exception
# of 2026-10-13), with the same exact device math, for every GPU generation TensorRT 11 builds
# for plus compute_120 PTX; the rest uses clang-cl with every option of the parent directory.
# On Linux: _chainner_trt.so and chainner_trt_vs.so, kernels.cu's host compiler is the parent
# directory's g++, and TensorRT is opened at run time instead of delay-loaded (runner.cpp).
if(WIN32)
    set(chainner_c_nvinfer_plugin_library
        "${CHAINNER_C_TENSORRT_ROOT}/lib/nvinfer_plugin_${chainner_c_tensorrt_major}.lib")
    if(NOT EXISTS "${chainner_c_nvinfer_plugin_library}")
        message(FATAL_ERROR "The TensorRT runner: ${chainner_c_nvinfer_plugin_library} is missing")
    endif()
endif()
set(chainner_c_trt_kernels "${CMAKE_CURRENT_BINARY_DIR}/trt_runner_kernels${CMAKE_CXX_OUTPUT_EXTENSION}")
add_custom_command(OUTPUT "${chainner_c_trt_kernels}"
    COMMAND ${chainner_c_nvcc_command} -c "${CMAKE_CURRENT_SOURCE_DIR}/kernels.cu" -o "${chainner_c_trt_kernels}"
        -ccbin "${chainner_c_nvcc_host}" -O3 -std=c++17
        -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80
        -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89
        -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100
        "-gencode=arch=compute_120,code=[sm_120,compute_120]"
        --fmad=false --prec-div=true --prec-sqrt=true -Werror all-warnings
        -Xcompiler "${chainner_c_nvcc_host_strict}"
    DEPENDS kernels.cu kernels.h convert.h
    COMMENT "Compiling trt_runner/kernels.cu with nvcc"
    VERBATIM)
set_source_files_properties("${chainner_c_trt_kernels}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
add_library(chainner_trt_core STATIC plan.cpp frame.cpp reference.cpp runner.cpp "${chainner_c_trt_kernels}")
target_include_directories(chainner_trt_core SYSTEM PUBLIC
    "${CHAINNER_C_TENSORRT_ROOT}/include" "${CHAINNER_C_CUDA_ROOT}/include")
target_include_directories(chainner_trt_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
if(WIN32)
    # CUDA 13's cudart.lib is the static runtime. TensorRT's two DLLs are delay-loaded (runner.cpp,
    # load_tensorrt): the binaries load, and their CPU functions run, without TensorRT.
    target_link_libraries(chainner_trt_core PUBLIC "${chainner_c_nvinfer_library}"
        "${chainner_c_nvinfer_plugin_library}" "${chainner_c_cudart_library}" delayimp)
    target_link_options(chainner_trt_core INTERFACE
        "/DELAYLOAD:nvinfer_${chainner_c_tensorrt_major}.dll" "/DELAYLOAD:nvinfer_plugin_${chainner_c_tensorrt_major}.dll")
else()
    # TensorRT's two libraries are opened, not linked (runner.cpp, load_tensorrt); CUDA's static
    # runtime needs glibc's libdl and librt.
    target_link_libraries(chainner_trt_core PUBLIC "${chainner_c_cudart_library}" ${CMAKE_DL_LIBS} rt)
endif()
set(chainner_c_trt_targets chainner_trt_core _chainner_trt)

add_library(_chainner_trt MODULE python_module.cpp)
target_include_directories(_chainner_trt SYSTEM PRIVATE
    "${chainner_c_python_include}" "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/pybind11/include")
if(WIN32)
    target_link_libraries(_chainner_trt PRIVATE chainner_trt_core "${chainner_c_python_library}")
else()
    target_link_libraries(_chainner_trt PRIVATE chainner_trt_core)
endif()
set_target_properties(_chainner_trt PROPERTIES SUFFIX "${chainner_c_module_suffix}")

set(CHAINNER_C_VAPOURSYNTH_INCLUDE "" CACHE PATH
    "VapourSynth's API 4 headers (the directory of VapourSynth4.h) for chainner_trt_vs.dll; empty: not built")
if(CHAINNER_C_VAPOURSYNTH_INCLUDE)
    if(NOT EXISTS "${CHAINNER_C_VAPOURSYNTH_INCLUDE}/VapourSynth4.h")
        message(FATAL_ERROR "chainner_trt_vs.dll: ${CHAINNER_C_VAPOURSYNTH_INCLUDE}/VapourSynth4.h is missing")
    endif()
    add_library(chainner_trt_vs MODULE vapoursynth_filter.cpp)
    target_include_directories(chainner_trt_vs SYSTEM PRIVATE "${CHAINNER_C_VAPOURSYNTH_INCLUDE}")
    target_link_libraries(chainner_trt_vs PRIVATE chainner_trt_core)
    set_target_properties(chainner_trt_vs PROPERTIES SUFFIX "${CMAKE_SHARED_MODULE_SUFFIX}")
    list(APPEND chainner_c_trt_targets chainner_trt_vs)
else()
    message(STATUS "TensorRT runner's VapourSynth plugin (chainner_trt_vs.dll) skipped: set "
        "CHAINNER_C_VAPOURSYNTH_INCLUDE to the directory of VapourSynth4.h to build it")
endif()

foreach(target ${chainner_c_trt_targets})
    set_target_properties(${target} PROPERTIES PREFIX "" CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO LIBRARY_OUTPUT_DIRECTORY "${CHAINNER_C_OUTPUT_DIR}/tensorrt")
    target_compile_options(${target} PRIVATE /W4 /WX /EHsc)
    if(CHAINNER_C_PDB)
        target_compile_options(${target} PRIVATE /Zi)
    endif()
endforeach()
if(CHAINNER_C_PDB)
    foreach(target _chainner_trt chainner_trt_vs)
        if(TARGET ${target})
            target_link_options(${target} PRIVATE /DEBUG /OPT:REF /OPT:ICF
                "/MAP:${CHAINNER_C_OUTPUT_DIR}/tensorrt/${target}.map")
        endif()
    endforeach()
endif()
