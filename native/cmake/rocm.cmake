# AMD GPUs: the MIGraphX runner (migraphx_runner/CMakeLists.txt) and the custom-op libraries of src/<model>_hip
# (drft, dual, vosr2), from the ROCm SDK at CHAINNER_C_ROCM_ROOT, else skipped with one status line. Optional and
# separate: CMakeLists.txt includes this file before its TensorRT part, and nothing here touches another target. HIP
# sources are compiled by the SDK's clang++ through custom commands, as nvcc compiles the .cu files: device code for
# every architecture of CHAINNER_C_HIP_ARCHITECTURES (every target ROCm 7.14 ships for the system), host code with the
# parent directory's intent (no contraction unless a target asks, hidden visibility, the stack protector); the modules
# link with the parent toolchain. Windows (Build.ps1 -RocmRoot, -MigraphxRoot): the same modules as .pyd and .dll
# except dual's, whose kernels come from Triton, which has no Windows build for AMD GPUs; the SDK's clang++ compiles
# against the MSVC toolset and Windows SDK that toolchain.cmake pins, for the C runtime of the target's directory
# (CMAKE_MSVC_RUNTIME_LIBRARY). Without a root, Windows declares none of this file's cache entries, so such a build
# configures as before the AMD parts existed.
if(WIN32 AND NOT CHAINNER_C_ROCM_ROOT AND NOT CHAINNER_C_MIGRAPHX_ROOT)
    message(STATUS "AMD GPU parts (MIGraphX runner, HIP custom-op libraries) skipped: pass Build.ps1 -RocmRoot "
        "(and -MigraphxRoot for the runner) to build them")
    return()
endif()
set(chainner_c_hip_major 7)
set(chainner_c_hip_minor 14)
set(CHAINNER_C_ROCM_ROOT "" CACHE PATH
    "ROCm ${chainner_c_hip_major}.${chainner_c_hip_minor} SDK root (include/hip, lib, lib/llvm/bin/clang++) for the MIGraphX runner and the HIP custom-op libraries; empty: not built")
set(CHAINNER_C_MIGRAPHX_ROOT "" CACHE PATH
    "MIGraphX root (include/migraphx, lib) for the MIGraphX runner (_chainner_migraphx); empty: not built")
if(WIN32)
    # ROCm 7.14's Windows targets: no gfx942, gfx950 or gfx1250 there.
    set(chainner_c_hip_architectures
        "gfx908;gfx90a;gfx1010;gfx1011;gfx1012;gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201")
else()
    set(chainner_c_hip_architectures
        "gfx908;gfx90a;gfx942;gfx950;gfx1010;gfx1011;gfx1012;gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201;gfx1250")
endif()
set(CHAINNER_C_HIP_ARCHITECTURES "${chainner_c_hip_architectures}"
    CACHE STRING "The AMD GPU architectures HIP device code is compiled for (default: every target of ROCm 7.14)")
if(NOT CHAINNER_C_ROCM_ROOT)
    if(CHAINNER_C_MIGRAPHX_ROOT)
        message(FATAL_ERROR "The MIGraphX runner needs CHAINNER_C_ROCM_ROOT too "
            "(CHAINNER_C_MIGRAPHX_ROOT=${CHAINNER_C_MIGRAPHX_ROOT})")
    endif()
    message(STATUS "AMD GPU parts (MIGraphX runner, HIP custom-op libraries) skipped: set CHAINNER_C_ROCM_ROOT "
        "(and CHAINNER_C_MIGRAPHX_ROOT for the runner) to build them")
    return()
endif()

set(chainner_c_hip_version_header "${CHAINNER_C_ROCM_ROOT}/include/hip/hip_version.h")
set(chainner_c_hip_clang "${CHAINNER_C_ROCM_ROOT}/lib/llvm/bin/clang++${CMAKE_EXECUTABLE_SUFFIX}")
if(WIN32)
    # The import library: the modules need amdhip64_7.dll by name, as nvinfer_11.dll. The device libraries sit in the
    # SDK's LLVM directory on Windows.
    set(chainner_c_hip_library "${CHAINNER_C_ROCM_ROOT}/lib/amdhip64.lib")
    set(chainner_c_hip_device_libraries "${CHAINNER_C_ROCM_ROOT}/lib/llvm/amdgcn/bitcode")
else()
    # Linked by its file, so the modules need libamdhip64.so.7 by its soname, as libnvinfer.so.11.
    set(chainner_c_hip_library "${CHAINNER_C_ROCM_ROOT}/lib/libamdhip64.so")
    set(chainner_c_hip_device_libraries "")
endif()
foreach(chainner_c_file "${chainner_c_hip_version_header}" "${chainner_c_hip_clang}" "${chainner_c_hip_library}"
        ${chainner_c_hip_device_libraries})
    if(NOT EXISTS "${chainner_c_file}")
        message(FATAL_ERROR "The ROCm SDK at CHAINNER_C_ROCM_ROOT: ${chainner_c_file} is missing")
    endif()
endforeach()
file(READ "${chainner_c_hip_version_header}" chainner_c_hip_text)
set(chainner_c_hip_version "")
foreach(chainner_c_part MAJOR MINOR PATCH)
    string(REGEX MATCH "#define HIP_VERSION_${chainner_c_part} ([0-9]+)" chainner_c_match "${chainner_c_hip_text}")
    list(APPEND chainner_c_hip_version "${CMAKE_MATCH_1}")
endforeach()
list(JOIN chainner_c_hip_version "." chainner_c_hip_version)
if(NOT chainner_c_hip_version MATCHES "^${chainner_c_hip_major}\\.${chainner_c_hip_minor}\\.[0-9]+$")
    message(FATAL_ERROR "The AMD GPU parts build with ROCm ${chainner_c_hip_major}.${chainner_c_hip_minor} only, "
        "not HIP '${chainner_c_hip_version}' (${chainner_c_hip_version_header})")
endif()
message(STATUS "AMD GPU parts: HIP ${chainner_c_hip_version} for ${CHAINNER_C_HIP_ARCHITECTURES}")

# Device compiles: every architecture, -O3, C++20, contraction off (a target's HIP_OPTIONS may turn it on), IEEE
# division and square root, warnings as errors, position-independent host code with hidden visibility and the stack
# protector, and a dependency file for Ninja.
set(chainner_c_hip_options -x hip --rocm-path=${CHAINNER_C_ROCM_ROOT} -O3 -std=c++20 -fno-gpu-rdc
    -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -Wall -Wextra -Werror
    "-ffile-prefix-map=${PROJECT_SOURCE_DIR}=." -fPIC -fvisibility=hidden -fvisibility-inlines-hidden
    -fstack-protector-strong)
foreach(chainner_c_architecture IN LISTS CHAINNER_C_HIP_ARCHITECTURES)
    list(APPEND chainner_c_hip_options --offload-arch=${chainner_c_architecture})
endforeach()
# Host code against the C and C++ libraries the parent toolchain compiles with: its clang++'s configuration file when
# it has one (the release image's glibc 2.28 sysroot and GCC Toolset 14, native/linux/Dockerfile).
if(EXISTS "${CMAKE_CXX_COMPILER}.cfg")
    list(APPEND chainner_c_hip_options "--config=${CMAKE_CXX_COMPILER}.cfg")
endif()
if(WIN32)
    # Windows: no position-independent code (a DLL is relocated, as clang-cl's are), the SDK's device libraries, the
    # MSVC toolset and Windows SDK of toolchain.cmake, and objects without a time stamp (clang-cl's /Brepro).
    list(REMOVE_ITEM chainner_c_hip_options -fPIC)
    list(APPEND chainner_c_hip_options "--rocm-device-lib-path=${chainner_c_hip_device_libraries}"
        -Xmicrosoft-visualc-tools-root "${chainner_c_vc_tools}" -Xmicrosoft-windows-sdk-root "${chainner_c_sdk}"
        -Xmicrosoft-windows-sdk-version "${chainner_c_sdk_version}" -mno-incremental-linker-compatible)
endif()

# chainner_c_hip_module(<target> <output subdirectory of CHAINNER_C_OUTPUT_DIR>
#     [HIP <sources with device code>] [CXX <host sources>] [HIP_OPTIONS <device-compile options>]
#     [INCLUDE <directories>])
# A MODULE with the HIP runtime linked: its HIP sources compiled by the SDK's clang++ for every architecture, its CXX
# sources by the parent toolchain with every option of the parent directory, /W4 /WX, the SDK's headers and
# __HIP_PLATFORM_AMD__; both see migraphx_runner (custom_op.h) and the INCLUDE directories. On Windows both use the C
# runtime of the calling directory's CMAKE_MSVC_RUNTIME_LIBRARY: the static one unless it names the DLL, the debug one
# in Debug builds where it names it.
function(chainner_c_hip_module target output_subdirectory)
    cmake_parse_arguments(PARSE_ARGV 2 arg "" "" "HIP;CXX;HIP_OPTIONS;INCLUDE")
    set(runtime "")
    if(WIN32 AND CMAKE_MSVC_RUNTIME_LIBRARY MATCHES "DLL")
        set(runtime -fms-runtime-lib=dll)
    elseif(WIN32)
        set(runtime -fms-runtime-lib=static)
    endif()
    if(WIN32 AND CMAKE_MSVC_RUNTIME_LIBRARY MATCHES "Debug")
        string(APPEND runtime "$<$<CONFIG:Debug>:_dbg>")
    endif()
    set(include_directories "${PROJECT_SOURCE_DIR}/migraphx_runner")
    foreach(directory IN LISTS arg_INCLUDE)
        cmake_path(ABSOLUTE_PATH directory BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" NORMALIZE)
        list(APPEND include_directories "${directory}")
    endforeach()
    list(TRANSFORM include_directories PREPEND "-I" OUTPUT_VARIABLE includes)
    set(objects "")
    foreach(source IN LISTS arg_HIP)
        cmake_path(ABSOLUTE_PATH source BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" NORMALIZE OUTPUT_VARIABLE path)
        cmake_path(GET path STEM stem)
        set(object "${CMAKE_CURRENT_BINARY_DIR}/${target}_${stem}${CMAKE_CXX_OUTPUT_EXTENSION}")
        add_custom_command(OUTPUT "${object}"
            COMMAND "${chainner_c_hip_clang}" ${chainner_c_hip_options} ${runtime} ${includes} ${arg_HIP_OPTIONS}
                -c "${path}" -o "${object}" -MD -MF "${object}.d"
            DEPENDS "${path}"
            DEPFILE "${object}.d"
            COMMENT "Compiling ${source} for HIP"
            VERBATIM)
        list(APPEND objects "${object}")
    endforeach()
    set_source_files_properties(${objects} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    add_library(${target} MODULE ${arg_CXX} ${objects})
    target_include_directories(${target} SYSTEM PRIVATE "${CHAINNER_C_ROCM_ROOT}/include")
    target_include_directories(${target} PRIVATE ${include_directories})
    target_compile_definitions(${target} PRIVATE __HIP_PLATFORM_AMD__)
    target_link_libraries(${target} PRIVATE "${chainner_c_hip_library}")
    set_target_properties(${target} PROPERTIES PREFIX "" SUFFIX "${CMAKE_SHARED_MODULE_SUFFIX}" LINKER_LANGUAGE CXX
        CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO SKIP_BUILD_RPATH YES
        LIBRARY_OUTPUT_DIRECTORY "${CHAINNER_C_OUTPUT_DIR}/${output_subdirectory}")
    target_compile_options(${target} PRIVATE /W4 /WX /EHsc)
    if(NOT WIN32)
        # lld-link keeps only the imports a DLL uses.
        target_link_options(${target} PRIVATE -Wl,--as-needed)
    endif()
    if(CHAINNER_C_PDB)
        target_compile_options(${target} PRIVATE /Zi)
        target_link_options(${target} PRIVATE /DEBUG /OPT:REF /OPT:ICF
            "/MAP:${CHAINNER_C_OUTPUT_DIR}/${output_subdirectory}/${target}.map")
    endif()
endfunction()

# DUAL's ops run kernels Triton compiles, and Triton has no Windows build for AMD GPUs: no dual_migx_ops.dll.
set(chainner_c_hip_models drft dual vosr2)
if(WIN32)
    list(REMOVE_ITEM chainner_c_hip_models dual)
endif()
foreach(chainner_c_model IN LISTS chainner_c_hip_models)
    if(EXISTS "${PROJECT_SOURCE_DIR}/src/${chainner_c_model}_hip/CMakeLists.txt")
        add_subdirectory("${PROJECT_SOURCE_DIR}/src/${chainner_c_model}_hip")
    endif()
endforeach()
if(CHAINNER_C_MIGRAPHX_ROOT)
    add_subdirectory("${PROJECT_SOURCE_DIR}/migraphx_runner")
else()
    message(STATUS "MIGraphX runner (_chainner_migraphx) skipped: set CHAINNER_C_MIGRAPHX_ROOT to build it")
endif()
