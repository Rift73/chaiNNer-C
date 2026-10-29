# The MIGraphX runner (cmake/rocm.cmake adds this directory when CHAINNER_C_MIGRAPHX_ROOT is set): _chainner_migraphx.so
# beside nodes/impl/migraphx, on MIGraphX 2.16 (ROCm 7.14's, built from source: AMD ships no binary) and HIP. It reuses
# the TensorRT runner's CPU parts (trt_runner's plan, frame, reference and convert.h, unchanged, compiled again here)
# with their kernels in HIP (kernels.hip), and MIGraphX's C++ API, whose ONNX parser it extends with custom ops
# (onnx_model.h). It needs MIGraphX's and HIP's libraries by their sonames, as the TensorRT parts need libnvinfer.so.11:
# the Python side makes them loadable before importing it. chainner_example_migx_ops, a custom-op library for the
# tests (tests/example_ops.hip), stays in the build directory. MIGraphX's install (ROCm 7.14's layout): the public and
# generated headers in include, the C++ API's in lib/migraphx/include, the libraries in lib/migraphx/lib; on Windows
# every header in include and the import libraries in lib (_chainner_migraphx.pyd needs MIGraphX's DLLs by name).
set(chainner_c_migraphx_major 2)
set(chainner_c_migraphx_minor 16)
set(chainner_c_migraphx_version_header "${CHAINNER_C_MIGRAPHX_ROOT}/include/migraphx/version.h")
if(WIN32)
    set(chainner_c_migraphx_includes "${CHAINNER_C_MIGRAPHX_ROOT}/include")
    set(chainner_c_migraphx_libraries "${CHAINNER_C_MIGRAPHX_ROOT}/lib/migraphx.lib"
        "${CHAINNER_C_MIGRAPHX_ROOT}/lib/migraphx_onnx.lib")
else()
    set(chainner_c_migraphx_includes "${CHAINNER_C_MIGRAPHX_ROOT}/include"
        "${CHAINNER_C_MIGRAPHX_ROOT}/lib/migraphx/include")
    set(chainner_c_migraphx_libraries "${CHAINNER_C_MIGRAPHX_ROOT}/lib/migraphx/lib/libmigraphx.so"
        "${CHAINNER_C_MIGRAPHX_ROOT}/lib/migraphx/lib/libmigraphx_onnx.so")
endif()
list(GET chainner_c_migraphx_includes -1 chainner_c_migraphx_api)
foreach(chainner_c_file "${chainner_c_migraphx_version_header}" "${chainner_c_migraphx_api}/migraphx/onnx.hpp"
        ${chainner_c_migraphx_libraries})
    if(NOT EXISTS "${chainner_c_file}")
        message(FATAL_ERROR "MIGraphX at CHAINNER_C_MIGRAPHX_ROOT: ${chainner_c_file} is missing")
    endif()
endforeach()
file(READ "${chainner_c_migraphx_version_header}" chainner_c_migraphx_text)
set(chainner_c_migraphx_version "")
foreach(chainner_c_part MAJOR MINOR PATCH)
    string(REGEX MATCH "#define MIGRAPHX_VERSION_${chainner_c_part} ([0-9]+)" chainner_c_match
        "${chainner_c_migraphx_text}")
    list(APPEND chainner_c_migraphx_version "${CMAKE_MATCH_1}")
endforeach()
list(JOIN chainner_c_migraphx_version "." chainner_c_migraphx_version)
if(NOT chainner_c_migraphx_version MATCHES "^${chainner_c_migraphx_major}\\.${chainner_c_migraphx_minor}\\.[0-9]+$")
    message(FATAL_ERROR "The MIGraphX runner builds with MIGraphX ${chainner_c_migraphx_major}."
        "${chainner_c_migraphx_minor} only, not '${chainner_c_migraphx_version}' (${chainner_c_migraphx_version_header})")
endif()
message(STATUS "MIGraphX runner (_chainner_migraphx): MIGraphX ${chainner_c_migraphx_version}")

if(WIN32)
    # MIGraphX's C++ API hands the standard library's objects and exceptions across its DLLs' boundary, so the modules
    # here use MIGraphX's C runtime, the release DLL one, where the parent directory links the static one.
    set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
endif()
chainner_c_hip_module(_chainner_migraphx migraphx
    HIP kernels.hip
    CXX python_module.cpp ops.cpp onnx_model.cpp program.cpp runner.cpp
        ../trt_runner/plan.cpp ../trt_runner/frame.cpp ../trt_runner/reference.cpp
    INCLUDE ../trt_runner)
target_include_directories(_chainner_migraphx SYSTEM PRIVATE ${chainner_c_migraphx_includes}
    "${chainner_c_python_include}" "${PROJECT_SOURCE_DIR}/third_party/pybind11/include")
target_link_libraries(_chainner_migraphx PRIVATE ${chainner_c_migraphx_libraries})
# C++17, MIGraphX's own: its headers name members `module`, which C++20 reads as a module declaration.
set_target_properties(_chainner_migraphx PROPERTIES SUFFIX "${chainner_c_module_suffix}" CXX_STANDARD 17)
if(WIN32)
    # Windows: the interpreter's import library, as the other modules, and windows.h without min and max (ops.cpp).
    target_link_libraries(_chainner_migraphx PRIVATE "${chainner_c_python_library}")
    target_compile_definitions(_chainner_migraphx PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    # MIGraphX's exceptions unwind through this module, so it links libstdc++ and libgcc shared, as MIGraphX does (two
    # unwinders cannot pass one exception), where the parent directory links them statically.
    get_target_property(chainner_c_options _chainner_migraphx LINK_OPTIONS)
    list(REMOVE_ITEM chainner_c_options "$<$<LINK_LANGUAGE:CXX>:-static-libstdc++>" -static-libgcc)
    set_target_properties(_chainner_migraphx PROPERTIES LINK_OPTIONS "${chainner_c_options}")
endif()

chainner_c_hip_module(chainner_example_migx_ops migraphx HIP tests/example_ops.hip)
set_target_properties(chainner_example_migx_ops PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
if(WIN32)
    # Its CPU op converts halves through compiler-rt's builtins (__extendhfsf2, __truncsfhf2), which the MSVC
    # libraries lack: the toolchain's own.
    string(REGEX MATCH "^[0-9]+" chainner_c_clang_major "${CMAKE_CXX_COMPILER_VERSION}")
    target_link_libraries(chainner_example_migx_ops PRIVATE
        "${chainner_c_llvm}/../lib/clang/${chainner_c_clang_major}/lib/windows/clang_rt.builtins-x86_64.lib")
endif()
