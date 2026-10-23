cmake_minimum_required(VERSION 3.20)
project(chainner_native VERSION 1.0.0 LANGUAGES C CXX)

# The one toolchain (decision 2026-10-06): clang-cl of LLVM 23.1.2 with lld-link and
# llvm-lib under Ninja, against the Build Tools' MSVC libraries and the Windows SDK, all
# pinned by Build.ps1, which configures every build directory.
foreach(chainner_c_language C CXX)
    if(NOT CMAKE_${chainner_c_language}_COMPILER_ID STREQUAL "Clang"
        OR NOT CMAKE_${chainner_c_language}_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"
        OR NOT CMAKE_${chainner_c_language}_COMPILER_VERSION VERSION_EQUAL 23.1.2)
        message(FATAL_ERROR "chaiNNer-C builds with clang-cl 23.1.2 only, not "
            "${CMAKE_${chainner_c_language}_COMPILER_ID} ${CMAKE_${chainner_c_language}_COMPILER_VERSION} "
            "(${CMAKE_${chainner_c_language}_COMPILER_FRONTEND_VARIANT}); configure through native/Build.ps1")
    endif()
endforeach()
# Every target:
# - baseline x86-64 code (clang-cl's default, SSE2), so the binaries run on any x86-64 CPU;
#   only the ISA units below and the vendored libraries' own guarded paths go beyond it,
#   behind runtime checks;
# - ThinLTO;
# - /fp:strict: no contraction, and strict FP exceptions and rounding, since the kernels'
#   flag windows mirror NumPy's np.seterr reporting and the pool carries MXCSR (Consult 9 P2);
# - dynamic denormals: kernels run under DAZ/FTZ too, so LLVM must not fold them as IEEE;
# - /Brepro on compile and link: no timestamps, so the same sources give the same binaries;
# - this directory mapped to "." for __FILE__ and for the ?A0x<hash> that clang's MSVC
#   mangling gives an anonymous namespace (a hash of the main file's path, reaching RTTI
#   names), so the binaries do not depend on where the checkout lies.
add_compile_options(-flto=thin /fp:strict /clang:-fdenormal-fp-math=dynamic /Brepro
    "/clang:-fmacro-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=.")
add_link_options(/Brepro)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
# A local-machine option, never for releases: -march=<value> on every target (for example
# icelake-server on the development PC). The binaries then run only on CPUs with that ISA.
set(CHAINNER_C_MARCH "" CACHE STRING
    "Local builds only, never releases: -march=<value> for every target (default: none, any x86-64 CPU)")
if(CHAINNER_C_MARCH)
    add_compile_options(-march=${CHAINNER_C_MARCH})
endif()

set(CHAINNER_C_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../backend/src/nodes/impl" CACHE PATH
    "Output directory of chainner_native.dll and _chainner_graph.pyd")
# Review and sampling builds: PDBs and linker maps, never over the gate and package binaries.
option(CHAINNER_C_PDB "Build with PDBs and linker maps (/Zi, /DEBUG, /MAP) outside backend/src" OFF)
# Fuzzing builds (tests/fuzz/README.md): every target with AddressSanitizer and libFuzzer's coverage, without
# ThinLTO, plus the fuzzers; never over the gate and package binaries.
option(CHAINNER_C_FUZZ "Build with AddressSanitizer and the libFuzzer harnesses of tests/fuzz outside backend/src" OFF)
if(CHAINNER_C_FUZZ)
    string(REGEX MATCH "^[0-9]+" chainner_c_clang_major "${CMAKE_C_COMPILER_VERSION}")
    set(chainner_c_clang_rt "${chainner_c_llvm}/../lib/clang/${chainner_c_clang_major}/lib/windows")
    # libFuzzer's runtime is built without the STL's container annotations, so they are off here too (lld-link's
    # /failifmismatch refuses the mix).
    add_compile_options(-fno-lto /Zi -fsanitize=address -fsanitize=fuzzer-no-link /D_DISABLE_STRING_ANNOTATION
        /D_DISABLE_VECTOR_ANNOTATION)
    # CMake links with lld-link itself: the runtime clang-cl's driver would add for /MT.
    add_link_options(/DEBUG /INCREMENTAL:NO "${chainner_c_clang_rt}/clang_rt.asan_dynamic-x86_64.lib"
        "/WHOLEARCHIVE:${chainner_c_clang_rt}/clang_rt.asan_static_runtime_thunk-x86_64.lib")
endif()
# chainner_ext.pyd's directory. A PDB or fuzzing build places it under CHAINNER_C_OUTPUT_DIR by default.
if(CHAINNER_C_PDB OR CHAINNER_C_FUZZ)
    set(chainner_c_ext_output_default "${CHAINNER_C_OUTPUT_DIR}/chainner_ext")
else()
    set(chainner_c_ext_output_default "${CMAKE_CURRENT_SOURCE_DIR}/../backend/src/chainner_ext")
endif()
set(CHAINNER_C_EXT_OUTPUT_DIR "${chainner_c_ext_output_default}" CACHE PATH
    "Output directory of chainner_ext.pyd")
if(CHAINNER_C_PDB OR CHAINNER_C_FUZZ)
    # A relative output directory is resolved against the build directory, as the
    # generator resolves the targets' output directories.
    cmake_path(SET chainner_c_backend NORMALIZE "${CMAKE_CURRENT_SOURCE_DIR}/../backend/src")
    string(TOLOWER "${chainner_c_backend}" chainner_c_backend)
    foreach(chainner_c_dir_var CHAINNER_C_OUTPUT_DIR CHAINNER_C_EXT_OUTPUT_DIR)
        cmake_path(SET chainner_c_output NORMALIZE "${${chainner_c_dir_var}}")
        cmake_path(ABSOLUTE_PATH chainner_c_output BASE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}" NORMALIZE)
        string(TOLOWER "${chainner_c_output}" chainner_c_output)
        cmake_path(IS_PREFIX chainner_c_backend "${chainner_c_output}" NORMALIZE chainner_c_inside)
        if(chainner_c_inside)
            message(FATAL_ERROR "CHAINNER_C_PDB and CHAINNER_C_FUZZ need ${chainner_c_dir_var} outside backend/src "
                "(${chainner_c_dir_var}=${${chainner_c_dir_var}})")
        endif()
    endforeach()
endif()

# The regex engine of chainner_ext (regex 1.8.4 and regex-syntax 0.7.2 in C; file map and
# owners in src/regex/README.md) and its JSON-lines probe. CHAINNER_C_REGEX_ONLY configures
# these two targets alone, without the CPython runtime, for the regex lanes' own build
# directories.
option(CHAINNER_C_REGEX_ONLY "Configure only chainner_regex and regex_probe" OFF)
add_library(chainner_regex STATIC
    src/regex/parse.c src/regex/ast.c src/regex/error.c src/regex/translate.c src/regex/hir.c
    src/regex/interval.c src/regex/unicode.c src/regex/utf8.c
    src/regex/unicode_tables/age.c src/regex/unicode_tables/case_folding_simple.c
    src/regex/unicode_tables/general_category.c src/regex/unicode_tables/grapheme_cluster_break.c
    src/regex/unicode_tables/perl_decimal.c src/regex/unicode_tables/perl_space.c
    src/regex/unicode_tables/perl_word.c src/regex/unicode_tables/property_bool.c
    src/regex/unicode_tables/property_names.c src/regex/unicode_tables/property_values.c
    src/regex/unicode_tables/script.c src/regex/unicode_tables/script_extension.c
    src/regex/unicode_tables/sentence_break.c src/regex/unicode_tables/word_break.c
    src/regex/compile.c src/regex/prog.c src/regex/input.c src/regex/pikevm.c src/regex/backtrack.c
    src/regex/exec.c src/regex/iter.c src/regex/prefilter.c src/regex/position.c src/regex/dfa.c
    src/regex/hir_literal.c)
target_include_directories(chainner_regex PUBLIC include)
add_executable(regex_probe src/regex/probe.c)
target_link_libraries(regex_probe PRIVATE chainner_regex)
foreach(target chainner_regex regex_probe)
    set_target_properties(${target} PROPERTIES C_STANDARD 17 C_STANDARD_REQUIRED YES C_EXTENSIONS NO)
    target_compile_options(${target} PRIVATE /W4 /WX)
    if(CHAINNER_C_PDB)
        target_compile_options(${target} PRIVATE /Zi)
    endif()
endforeach()
if(CHAINNER_C_PDB)
    target_link_options(regex_probe PRIVATE /DEBUG /OPT:REF /OPT:ICF)
endif()
if(CHAINNER_C_REGEX_ONLY)
    return()
endif()

add_library(chainner_native SHARED
    src/version.c src/crt_math.c src/parallel.c src/isa.c src/adjustments.c src/blend.c src/gradients.c src/normals.c
    src/generation.c src/channels.c src/color_ops.c src/analysis_ops.c
    src/buffers.c src/transparency_color.c src/filter_ops.c src/layout_ops.c src/palette_ops.c
    src/tensor_ops.c src/resample_ops.c src/neighborhood_ops.c src/transfer_ops.c
    src/wavelet_ops.c src/alpha_gamma_ops.c src/convolution_ops.c src/kmeans_ops.c
    src/resample_filters.c src/alpha_complete_ops.c src/rotate_ops.c src/morphology_ops.c
    src/box_complete_ops.c src/distance_complete_ops.c src/pocketfft_kernel.c src/fft_ops.c
    src/blue_noise_ops.c src/noise_complete_ops.c src/spectral_filter_ops.c
    src/threshold_complete_ops.c src/transfer_complete_ops.c src/lens_complete_ops.c
    src/procedural_complete_ops.c src/color_complete_ops.c src/cas_complete_ops.c
    src/gaussian_complete_ops.c src/matting_complete_ops.c src/cv_resize_ops.c
    src/text_layout_ops.c src/normal_kernel_ops.c src/composite_complete_ops.c
    src/statistics_ops.c src/unsharp_complete_ops.c src/tensor_complete_ops.cpp
    src/canny_ops.cpp src/inpaint_ops.cpp src/pixel_art_ops.cpp
    src/bilateral_ops.c src/denoise_ops.cpp src/onnx_runtime.cpp src/framework_image_ops.cpp src/framework_shared_ops.cpp
    src/clipboard_ops.c src/image_setup_ops.c src/scratch.c src/planar_ops.c src/descale_ops.c
    src/yuv_planes_ops.c src/plane_resample_ops.c src/dpid_ops.c src/png_encode.c src/linear_light_ops.c
    src/getnative_ops.cpp
    # ISA units (SP4 spec 4.3): export-free, one file per kernel family and level, compiled
    # for that level only and reached through a C entry's ISA dispatch, which runs a unit
    # only when the CPU has its level (src/isa.c) or CHAINNER_C_ISA caps it lower.
    src/separable_avx2.c src/convolution_avx2.c src/convert_avx2.c src/lens_avx2.c
    src/morphology_avx2.c src/palette_avx2.c src/dither_avx2.c src/blend_avx2.c
    src/resample_avx2.c src/spectral_avx2.c src/color_avx2.c src/planar_avx2.c
    src/yuv_planes_avx2.c src/plane_resample_avx2.c src/png_encode_avx2.c src/alpha_gamma_avx2.c
    src/neighborhood_avx2.c src/statistics_avx2.c src/unsharp_avx2.c
    src/wavelet_avx2.c src/cast_half_avx2.c src/linear_light_avx2.c src/separable_avx512.c src/convolution_avx512.c)
# The levels' instruction sets, exactly what src/isa.c checks: AVX2 is AVX2, FMA, BMI1 and
# BMI2; AVX-512 adds F, DQ, CD, BW and VL.
set(chainner_c_avx2_options /clang:-mavx2 /clang:-mfma /clang:-mbmi /clang:-mbmi2)
set_source_files_properties(src/separable_avx2.c src/convolution_avx2.c src/convert_avx2.c
    src/lens_avx2.c src/morphology_avx2.c src/palette_avx2.c src/dither_avx2.c
    src/blend_avx2.c src/resample_avx2.c src/spectral_avx2.c src/color_avx2.c
    src/planar_avx2.c src/yuv_planes_avx2.c src/plane_resample_avx2.c src/png_encode_avx2.c src/alpha_gamma_avx2.c
    src/neighborhood_avx2.c src/statistics_avx2.c src/unsharp_avx2.c src/wavelet_avx2.c src/cast_half_avx2.c
    src/linear_light_avx2.c
    PROPERTIES COMPILE_OPTIONS "${chainner_c_avx2_options}")
set_source_files_properties(src/separable_avx512.c src/convolution_avx512.c
    PROPERTIES COMPILE_OPTIONS "${chainner_c_avx2_options};/clang:-mavx512f;/clang:-mavx512dq;/clang:-mavx512cd;/clang:-mavx512bw;/clang:-mavx512vl")
# Byte-for-byte vendor source of vapoursynth-descale Release 12 (MIT): its solver core and AVX2
# unit, without the VapourSynth and AviSynth plugin layers. src/descale_ops.c picks the unit from
# chaiNNer's ISA level (CHAINNER_C_ISA caps it), never through descale's own CPUID path.
add_library(chainner_descale STATIC third_party/descale/src/descale.c
    third_party/descale/src/x86/cpuinfo_x86.c third_party/descale/src/x86/descale_avx2.c)
target_include_directories(chainner_descale SYSTEM PUBLIC third_party/descale/include)
target_include_directories(chainner_descale PRIVATE third_party/descale/src)
target_compile_definitions(chainner_descale PRIVATE DESCALE_X86)
set_target_properties(chainner_descale PROPERTIES C_STANDARD 17 C_STANDARD_REQUIRED YES)
target_compile_options(chainner_descale PRIVATE /W3)
set_source_files_properties(third_party/descale/src/x86/descale_avx2.c
    PROPERTIES COMPILE_OPTIONS "${chainner_c_avx2_options}")
target_link_libraries(chainner_native PRIVATE chainner_descale)
# Vernier 0.2's analysis core (MIT; third_party/vernier/PROVENANCE.md) for Get Native Resolution,
# built once per ISA level with the session (src/getnative_unit.cpp) and the kernel decision
# (src/getnative_kernel.cpp). Each build renames the core's namespace on the command line, so both
# link side by side; src/getnative_ops.cpp picks one per session by cn_isa_current(). Both use
# Vernier's floating-point model (the compiler's default: /fp:strict's reason, NumPy's flag windows,
# does not apply to them) with contraction off, so they give bit-identical results. The baseline
# library links first, so the standard library's inline functions both compile resolve to its
# baseline copies.
foreach(level base avx2)
    set(target chainner_vernier_${level})
    add_library(${target} STATIC third_party/vernier/src/resample.cpp third_party/vernier/src/analyze.cpp
        third_party/vernier/src/spectrum.cpp src/getnative_unit.cpp src/getnative_kernel.cpp)
    target_include_directories(${target} SYSTEM PRIVATE third_party/vernier/include)
    target_include_directories(${target} PRIVATE include src)
    target_compile_definitions(${target} PRIVATE native=cn_vernier_${level}
        CN_GETNATIVE_UNIT=cn_getnative_unit_${level})
    set_target_properties(${target} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
    get_target_property(options ${target} COMPILE_OPTIONS)
    list(REMOVE_ITEM options /fp:strict)
    set_target_properties(${target} PROPERTIES COMPILE_OPTIONS "${options}")
    target_compile_options(${target} PRIVATE /EHsc /clang:-ffp-contract=off)
    target_link_libraries(chainner_native PRIVATE ${target})
endforeach()
target_compile_options(chainner_vernier_avx2 PRIVATE ${chainner_c_avx2_options})
set_source_files_properties(third_party/vernier/src/resample.cpp third_party/vernier/src/analyze.cpp
    third_party/vernier/src/spectrum.cpp PROPERTIES COMPILE_OPTIONS /W3)
set_source_files_properties(src/getnative_unit.cpp src/getnative_kernel.cpp PROPERTIES COMPILE_OPTIONS "/W4;/WX")
# The default PNG encoder (src/png_encode.c) compresses with the vendored libdeflate (chainner_deflate,
# cmake/codecs.cmake), patched for its band pieces (third_party/libdeflate/PROVENANCE.md).
target_link_libraries(chainner_native PRIVATE chainner_deflate)
target_include_directories(chainner_native PRIVATE include)
set_target_properties(chainner_native PROPERTIES
    C_STANDARD 17 C_STANDARD_REQUIRED YES C_EXTENSIONS NO
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO
    PREFIX "" RUNTIME_OUTPUT_DIRECTORY "${CHAINNER_C_OUTPUT_DIR}")
target_compile_options(chainner_native PRIVATE /W4 /WX $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
if(CHAINNER_C_PDB)
    target_compile_options(chainner_native PRIVATE /Zi)
    target_link_options(chainner_native PRIVATE /DEBUG /OPT:REF /OPT:ICF
        "/MAP:${CHAINNER_C_OUTPUT_DIR}/chainner_native.map")
endif()

# Keep the provisioned runtime's interpreter and public object/container API. This module
# owns graph algorithms; it does not execute copied Python algorithm source. The runtime
# (native/tools/provision_runtime.ps1) reports its own headers, import library and NumPy
# headers (np.get_include()).
set(CHAINNER_C_PYTHON_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/runtime/cpython-3.14.8" CACHE PATH
    "CPython runtime (python.exe, include, libs, NumPy) that _chainner_graph builds against")
execute_process(COMMAND "${CHAINNER_C_PYTHON_ROOT}/python.exe" -I -B -c
    "import sysconfig, numpy; print(sysconfig.get_path('include'), sysconfig.get_config_var('py_version_nodot'), numpy.get_include(), sep='|', end='')"
    RESULT_VARIABLE chainner_c_python_result OUTPUT_VARIABLE chainner_c_python ERROR_VARIABLE chainner_c_python_error)
if(NOT chainner_c_python_result EQUAL 0)
    message(FATAL_ERROR "${CHAINNER_C_PYTHON_ROOT}/python.exe did not report its headers, version and NumPy "
        "headers (${chainner_c_python_result}): ${chainner_c_python_error}")
endif()
string(REPLACE "|" ";" chainner_c_python "${chainner_c_python}")
file(TO_CMAKE_PATH "${chainner_c_python}" chainner_c_python)
list(GET chainner_c_python 0 chainner_c_python_include)
list(GET chainner_c_python 1 chainner_c_python_version)
list(GET chainner_c_python 2 chainner_c_numpy_include)
set(chainner_c_python_library "${CHAINNER_C_PYTHON_ROOT}/libs/python${chainner_c_python_version}.lib")
if(NOT EXISTS "${chainner_c_python_library}")
    message(FATAL_ERROR "The runtime's import library ${chainner_c_python_library} is missing")
endif()
add_library(_chainner_graph MODULE src/graph_module.cpp src/onnx_graph.cpp src/ncnn_graph.cpp src/onnx_converter.cpp src/onnx_tensorproto.cpp src/onnx_model_application.cpp src/tiling_control.cpp
    src/utility_scalar.cpp src/utility_random.cpp src/utility_text.cpp src/image_io.cpp
    src/execution_ops.cpp src/file_sequence.cpp src/video_io.cpp src/utility_clipboard.cpp
    src/numpy_pool.cpp)
target_compile_definitions(_chainner_graph PRIVATE CN_TENSOR_IMPORTS)
# The runtime's NumPy headers serve numpy_pool.cpp, the one unit that includes them.
target_include_directories(_chainner_graph SYSTEM PRIVATE
    "${chainner_c_python_include}" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11/include"
    "${chainner_c_numpy_include}")
target_include_directories(_chainner_graph PRIVATE include)
target_link_libraries(_chainner_graph PRIVATE "${chainner_c_python_library}" chainner_native user32)
set_target_properties(_chainner_graph PROPERTIES PREFIX "" SUFFIX ".pyd"
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO
    LIBRARY_OUTPUT_DIRECTORY "${CHAINNER_C_OUTPUT_DIR}")
target_compile_options(_chainner_graph PRIVATE /W4 /WX /EHsc)
if(CHAINNER_C_PDB)
    target_compile_options(_chainner_graph PRIVATE /Zi)
    target_link_options(_chainner_graph PRIVATE /DEBUG /OPT:REF /OPT:ICF
        "/MAP:${CHAINNER_C_OUTPUT_DIR}/_chainner_graph.map")
endif()
include(cmake/run_engine.cmake)
include(cmake/codecs.cmake)

# chainner_ext.pyd: the drop-in for chainner_ext 0.3.10 (docs/superpowers/specs/
# 2026-10-06-chainner-ext-c-design.md). Limited API 3.11 against the runtime's python3.lib,
# so one binary loads on CPython 3.11 and 3.14. chainner_ext_c.pyd is the conformance
# harness's copy from the same sources (PyInit_chainner_ext_c), in <build>/harness.
set(chainner_c_python3_library "${CHAINNER_C_PYTHON_ROOT}/libs/python3.lib")
if(NOT EXISTS "${chainner_c_python3_library}")
    message(FATAL_ERROR "The runtime's stable-ABI import library ${chainner_c_python3_library} is missing")
endif()
set(chainner_ext_sources src/chainner_ext/lib.c src/chainner_ext/regex.c src/chainner_ext/clipboard.c
    src/chainner_ext/convert.c src/chainner_ext/dither.c src/chainner_ext/pixel_art.c
    src/chainner_ext/resize.c)
add_library(chainner_ext MODULE ${chainner_ext_sources})
add_library(chainner_ext_c MODULE ${chainner_ext_sources})
target_compile_definitions(chainner_ext_c PRIVATE CHAINNER_EXT_HARNESS)
set_target_properties(chainner_ext PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CHAINNER_C_EXT_OUTPUT_DIR}")
set_target_properties(chainner_ext_c PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/harness")
foreach(target chainner_ext chainner_ext_c)
    target_compile_definitions(${target} PRIVATE Py_LIMITED_API=0x030B0000 CN_TENSOR_IMPORTS)
    target_include_directories(${target} SYSTEM PRIVATE "${chainner_c_python_include}")
    target_include_directories(${target} PRIVATE include)
    target_link_libraries(${target} PRIVATE "${chainner_c_python3_library}" chainner_native chainner_regex user32)
    set_target_properties(${target} PROPERTIES PREFIX "" SUFFIX ".pyd"
        C_STANDARD 17 C_STANDARD_REQUIRED YES C_EXTENSIONS NO)
    target_compile_options(${target} PRIVATE /W4 /WX)
    if(CHAINNER_C_PDB)
        target_compile_options(${target} PRIVATE /Zi)
        target_link_options(${target} PRIVATE /DEBUG /OPT:REF /OPT:ICF
            "/MAP:$<TARGET_FILE_DIR:${target}>/${target}.map")
    endif()
endforeach()
if(CHAINNER_C_FUZZ)
    add_subdirectory(tests/fuzz)
endif()

# vosr2_qkv.dll: the VosrQkvPack TensorRT plugin of VOSR's TensorRT path (src/vosr2), beside its
# loader in nodes/impl/pytorch. Optional: built when both roots below are given (Build.ps1 -Define,
# on any configure), else skipped with one status line. qkv.cu alone is compiled by nvcc with the
# pinned toolset's cl.exe as its host compiler (approved exception, 2026-10-13); the plugin
# interface and the link use clang-cl and lld-link with every option above. nvcc keeps exact device
# math (no contraction, IEEE division and square root) and emits SASS for sm_80, sm_86, sm_89 and
# sm_120 plus compute_120 PTX (BF16 needs sm_80 or newer). Configuration refuses any other TensorRT
# or CUDA major than these two pins.
set(chainner_c_tensorrt_major 11)
set(chainner_c_cuda_major 13)
set(CHAINNER_C_TENSORRT_ROOT "" CACHE PATH
    "TensorRT ${chainner_c_tensorrt_major} SDK root (include, lib) for vosr2_qkv.dll, drft_trt_plugins.dll and the TensorRT runner; empty: not built")
set(CHAINNER_C_CUDA_ROOT "" CACHE PATH
    "CUDA ${chainner_c_cuda_major} toolkit root (bin/nvcc.exe, include, lib/x64) for vosr2_qkv.dll, drft_trt_plugins.dll and the TensorRT runner; empty: not built")
if(NOT CHAINNER_C_TENSORRT_ROOT AND NOT CHAINNER_C_CUDA_ROOT)
    message(STATUS "VOSR and DRFT TensorRT plugins (vosr2_qkv.dll, drft_trt_plugins.dll) and TensorRT runner "
        "(trt_runner) skipped: set "
        "CHAINNER_C_TENSORRT_ROOT and CHAINNER_C_CUDA_ROOT to build them")
    return()
endif()
if(NOT CHAINNER_C_TENSORRT_ROOT OR NOT CHAINNER_C_CUDA_ROOT)
    message(FATAL_ERROR "vosr2_qkv.dll needs both CHAINNER_C_TENSORRT_ROOT and CHAINNER_C_CUDA_ROOT "
        "(CHAINNER_C_TENSORRT_ROOT=${CHAINNER_C_TENSORRT_ROOT}, CHAINNER_C_CUDA_ROOT=${CHAINNER_C_CUDA_ROOT})")
endif()
set(chainner_c_tensorrt_header "${CHAINNER_C_TENSORRT_ROOT}/include/NvInferVersion.h")
set(chainner_c_nvcc "${CHAINNER_C_CUDA_ROOT}/bin/nvcc.exe")
# The two version sources first, then the files that depend on the versions.
foreach(chainner_c_file "${chainner_c_tensorrt_header}" "${chainner_c_nvcc}")
    if(NOT EXISTS "${chainner_c_file}")
        message(FATAL_ERROR "vosr2_qkv.dll: ${chainner_c_file} is missing")
    endif()
endforeach()
file(READ "${chainner_c_tensorrt_header}" chainner_c_tensorrt_text)
set(chainner_c_tensorrt_version "")
foreach(chainner_c_part MAJOR MINOR PATCH BUILD)
    string(REGEX MATCH "#define TRT_${chainner_c_part}_ENTERPRISE ([0-9]+)" chainner_c_match
        "${chainner_c_tensorrt_text}")
    list(APPEND chainner_c_tensorrt_version "${CMAKE_MATCH_1}")
endforeach()
list(JOIN chainner_c_tensorrt_version "." chainner_c_tensorrt_version)
if(NOT chainner_c_tensorrt_version MATCHES "^${chainner_c_tensorrt_major}\\.[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR "vosr2_qkv.dll builds with TensorRT ${chainner_c_tensorrt_major} only, not "
        "'${chainner_c_tensorrt_version}' (${chainner_c_tensorrt_header})")
endif()
execute_process(COMMAND "${chainner_c_nvcc}" --version
    RESULT_VARIABLE chainner_c_nvcc_result OUTPUT_VARIABLE chainner_c_nvcc_output ERROR_VARIABLE chainner_c_nvcc_output)
string(REGEX MATCH "release [0-9]+\\.[0-9]+, V([0-9.]+)" chainner_c_match "${chainner_c_nvcc_output}")
set(chainner_c_cuda_version "${CMAKE_MATCH_1}")
if(NOT chainner_c_nvcc_result EQUAL 0 OR NOT chainner_c_cuda_version MATCHES "^${chainner_c_cuda_major}\\.")
    message(FATAL_ERROR "vosr2_qkv.dll builds with CUDA ${chainner_c_cuda_major} only, not "
        "'${chainner_c_cuda_version}' (${chainner_c_nvcc} --version, ${chainner_c_nvcc_result}): "
        "${chainner_c_nvcc_output}")
endif()
set(chainner_c_nvinfer_library "${CHAINNER_C_TENSORRT_ROOT}/lib/nvinfer_${chainner_c_tensorrt_major}.lib")
set(chainner_c_cudart_library "${CHAINNER_C_CUDA_ROOT}/lib/x64/cudart.lib")
set(chainner_c_nvcc_host "${chainner_c_vc_tools}/bin/Hostx64/x64/cl.exe")
foreach(chainner_c_file "${chainner_c_nvinfer_library}" "${chainner_c_cudart_library}" "${chainner_c_nvcc_host}")
    if(NOT EXISTS "${chainner_c_file}")
        message(FATAL_ERROR "vosr2_qkv.dll: ${chainner_c_file} is missing")
    endif()
endforeach()
message(STATUS "VOSR TensorRT plugin (vosr2_qkv.dll): TensorRT ${chainner_c_tensorrt_version}, "
    "CUDA ${chainner_c_cuda_version}")
# cl.exe finds the toolset's and the SDK's headers through INCLUDE, as in a Developer prompt.
set(chainner_c_nvcc_include "${chainner_c_vc_tools}/include"
    "${chainner_c_sdk}/Include/${chainner_c_sdk_version}/ucrt"
    "${chainner_c_sdk}/Include/${chainner_c_sdk_version}/shared"
    "${chainner_c_sdk}/Include/${chainner_c_sdk_version}/um")
list(JOIN chainner_c_nvcc_include "$<SEMICOLON>" chainner_c_nvcc_include)
set(chainner_c_qkv_object "${CMAKE_CURRENT_BINARY_DIR}/vosr2_qkv_cuda.obj")
add_custom_command(OUTPUT "${chainner_c_qkv_object}"
    COMMAND "${CMAKE_COMMAND}" -E env "INCLUDE=${chainner_c_nvcc_include}"
        "${chainner_c_nvcc}" -c "${CMAKE_CURRENT_SOURCE_DIR}/src/vosr2/qkv.cu" -o "${chainner_c_qkv_object}"
        -ccbin "${chainner_c_nvcc_host}" -O3 -std=c++17
        -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86
        -gencode=arch=compute_89,code=sm_89 "-gencode=arch=compute_120,code=[sm_120,compute_120]"
        --fmad=false --prec-div=true --prec-sqrt=true -Werror all-warnings
        -Xcompiler "$<IF:$<CONFIG:Debug>,/MTd,/MT>,/W4,/WX,/fp:strict,/Brepro"
    DEPENDS src/vosr2/qkv.cu
    COMMENT "Compiling src/vosr2/qkv.cu with nvcc"
    VERBATIM)
set_source_files_properties("${chainner_c_qkv_object}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
add_library(vosr2_qkv MODULE src/vosr2/qkv_plugin.cpp "${chainner_c_qkv_object}")
target_include_directories(vosr2_qkv SYSTEM PRIVATE
    "${CHAINNER_C_TENSORRT_ROOT}/include" "${CHAINNER_C_CUDA_ROOT}/include")
# CUDA 13's cudart.lib is the static runtime (it loads the driver itself): the DLL imports only
# nvinfer_11.dll and KERNEL32.dll.
target_link_libraries(vosr2_qkv PRIVATE "${chainner_c_nvinfer_library}" "${chainner_c_cudart_library}")
set_target_properties(vosr2_qkv PROPERTIES PREFIX "" SUFFIX ".dll"
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO
    LIBRARY_OUTPUT_DIRECTORY "${CHAINNER_C_OUTPUT_DIR}/pytorch")
target_compile_options(vosr2_qkv PRIVATE /W4 /WX /EHsc)
if(CHAINNER_C_PDB)
    target_compile_options(vosr2_qkv PRIVATE /Zi)
    target_link_options(vosr2_qkv PRIVATE /DEBUG /OPT:REF /OPT:ICF
        "/MAP:${CHAINNER_C_OUTPUT_DIR}/pytorch/vosr2_qkv.map")
endif()

# drft_trt_plugins.dll: the TensorRT plugins of legacy DRFT's engines (src/drft, the
# LegacyBarrierV1 library, compile definitions as his), from the same two roots, beside the DRFT
# loader in nodes/impl/tensorrt. Engines embed it, so it is never loaded to run one. The four .cu
# files (kernels and their plugin classes) are compiled by nvcc as qkv.cu is, except two points:
# - device math: nvcc's defaults (contraction on, IEEE division and square root), the numerics
#   Codex measured;
# - /W4 without /WX or -Werror: two warning classes remain, C4211 in nvcc's generated stubs of
#   the kernels in anonymous namespaces, and EDG's #997 on DrftFinalAffineCrop_TRT, the one
#   plugin on TensorRT's deprecated IPluginV2DynamicExt (its legacy overloads are private there).
# DrftAliasedIdentityBarrierPlugin.cpp and the link use clang-cl and lld-link with /W4 /WX.
set(chainner_c_drft_definitions -DDRFT_ENABLE_OCAB_QKV_PACK_PLUGIN=1
    -DDRFT_ENABLE_OCAB_QKV_PACK_TOKEN_MAJOR=1 -DDRFT_ENABLE_RELATIVE_BIAS_FP32=1
    -DDRFT_ENABLE_IDENTITY_BARRIER_PLUGIN=1 -DDRFT_ENABLE_FINAL_AFFINE_CROP_PLUGIN=1
    -DDRFT_ENABLE_WINDOW_NEURAL_BIAS_PLUGIN=1 -DNOMINMAX -DWIN32_LEAN_AND_MEAN)
set(chainner_c_drft_objects "")
foreach(chainner_c_drft_source DrftOCABAttentionSafePlugin DrftOCABQKVPackPlugin
        DrftFinalAffineCropPlugin DrftWindowNeuralBiasPlugin)
    set(chainner_c_drft_object "${CMAKE_CURRENT_BINARY_DIR}/drft_${chainner_c_drft_source}.obj")
    add_custom_command(OUTPUT "${chainner_c_drft_object}"
        COMMAND "${CMAKE_COMMAND}" -E env "INCLUDE=${chainner_c_nvcc_include}"
            "${chainner_c_nvcc}" -c "${CMAKE_CURRENT_SOURCE_DIR}/src/drft/${chainner_c_drft_source}.cu"
            -o "${chainner_c_drft_object}" -ccbin "${chainner_c_nvcc_host}" -O3 -std=c++17
            "-I${CHAINNER_C_TENSORRT_ROOT}/include" ${chainner_c_drft_definitions}
            -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86
            -gencode=arch=compute_89,code=sm_89 "-gencode=arch=compute_120,code=[sm_120,compute_120]"
            --fmad=true --prec-div=true --prec-sqrt=true
            -Xcompiler "$<IF:$<CONFIG:Debug>,/MTd,/MT>,/W4,/EHsc,/Zc:preprocessor,/fp:strict,/Brepro"
        DEPENDS "src/drft/${chainner_c_drft_source}.cu"
        COMMENT "Compiling src/drft/${chainner_c_drft_source}.cu with nvcc"
        VERBATIM)
    list(APPEND chainner_c_drft_objects "${chainner_c_drft_object}")
endforeach()
set_source_files_properties(${chainner_c_drft_objects} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
add_library(drft_trt_plugins MODULE src/drft/DrftAliasedIdentityBarrierPlugin.cpp ${chainner_c_drft_objects})
target_include_directories(drft_trt_plugins SYSTEM PRIVATE
    "${CHAINNER_C_TENSORRT_ROOT}/include" "${CHAINNER_C_CUDA_ROOT}/include")
target_compile_definitions(drft_trt_plugins PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(drft_trt_plugins PRIVATE "${chainner_c_nvinfer_library}" "${chainner_c_cudart_library}")
set_target_properties(drft_trt_plugins PROPERTIES PREFIX "" SUFFIX ".dll"
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO
    LIBRARY_OUTPUT_DIRECTORY "${CHAINNER_C_OUTPUT_DIR}/tensorrt")
target_compile_options(drft_trt_plugins PRIVATE /W4 /WX /EHsc)
if(CHAINNER_C_PDB)
    target_compile_options(drft_trt_plugins PRIVATE /Zi)
    target_link_options(drft_trt_plugins PRIVATE /DEBUG /OPT:REF /OPT:ICF
        "/MAP:${CHAINNER_C_OUTPUT_DIR}/tensorrt/drft_trt_plugins.map")
endif()

# The TensorRT runner (trt_runner/CMakeLists.txt), from the same two roots.
add_subdirectory(trt_runner)
