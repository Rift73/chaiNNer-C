cmake_minimum_required(VERSION 3.20)
project(chainner_native VERSION 1.0.0 LANGUAGES C CXX)

# The one toolchain (owner directive 2026-10-06): clang-cl of LLVM 23.1.2 with lld-link and
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
# Every target, vendored fpng included:
# - baseline x86-64 code (clang-cl's default, SSE2), so the binaries run on any x86-64 CPU;
#   only the ISA units below and fpng's guarded paths go beyond it, behind runtime checks;
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
# icelake-server on the owner's PC). The binaries then run only on CPUs with that ISA.
set(CHAINNER_C_MARCH "" CACHE STRING
    "Local builds only, never releases: -march=<value> for every target (default: none, any x86-64 CPU)")
if(CHAINNER_C_MARCH)
    add_compile_options(-march=${CHAINNER_C_MARCH})
endif()

set(CHAINNER_C_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../backend/src/nodes/impl" CACHE PATH
    "Output directory of chainner_native.dll and _chainner_graph.pyd")
# Review and sampling builds: PDBs and linker maps, never over the gate and package binaries.
option(CHAINNER_C_PDB "Build with PDBs and linker maps (/Zi, /DEBUG, /MAP) outside backend/src" OFF)
# chainner_ext.pyd's directory. A PDB build places it under CHAINNER_C_OUTPUT_DIR by default.
if(CHAINNER_C_PDB)
    set(chainner_c_ext_output_default "${CHAINNER_C_OUTPUT_DIR}/chainner_ext")
else()
    set(chainner_c_ext_output_default "${CMAKE_CURRENT_SOURCE_DIR}/../backend/src/chainner_ext")
endif()
set(CHAINNER_C_EXT_OUTPUT_DIR "${chainner_c_ext_output_default}" CACHE PATH
    "Output directory of chainner_ext.pyd")
if(CHAINNER_C_PDB)
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
            message(FATAL_ERROR "CHAINNER_C_PDB needs ${chainner_c_dir_var} outside backend/src "
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
    src/clipboard_ops.c src/image_setup_ops.c src/scratch.c
    # ISA units (SP4 spec 4.3): export-free, one file per kernel family and level, compiled
    # for that level only and reached through a C entry's ISA dispatch, which runs a unit
    # only when the CPU has its level (src/isa.c) or CHAINNER_C_ISA caps it lower.
    src/separable_avx2.c src/convolution_avx2.c src/convert_avx2.c src/lens_avx2.c
    src/morphology_avx2.c src/palette_avx2.c src/dither_avx2.c src/blend_avx2.c
    src/resample_avx2.c src/spectral_avx2.c src/color_avx2.c
    src/separable_avx512.c src/convolution_avx512.c)
# The levels' instruction sets, exactly what src/isa.c checks: AVX2 is AVX2, FMA, BMI1 and
# BMI2; AVX-512 adds F, DQ, CD, BW and VL.
set(chainner_c_avx2_options /clang:-mavx2 /clang:-mfma /clang:-mbmi /clang:-mbmi2)
set_source_files_properties(src/separable_avx2.c src/convolution_avx2.c src/convert_avx2.c
    src/lens_avx2.c src/morphology_avx2.c src/palette_avx2.c src/dither_avx2.c
    src/blend_avx2.c src/resample_avx2.c src/spectral_avx2.c src/color_avx2.c
    PROPERTIES COMPILE_OPTIONS "${chainner_c_avx2_options}")
set_source_files_properties(src/separable_avx512.c src/convolution_avx512.c
    PROPERTIES COMPILE_OPTIONS "${chainner_c_avx2_options};/clang:-mavx512f;/clang:-mavx512dq;/clang:-mavx512cd;/clang:-mavx512bw;/clang:-mavx512vl")
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
# Byte-for-byte vendor source used by the user's Imagetools resize build.
# Keep vendor compiler settings separate from strict first-party diagnostics.
add_library(chainner_fpng STATIC third_party/fpng/fpng.cpp)
target_include_directories(chainner_fpng SYSTEM PUBLIC third_party/fpng)
target_compile_definitions(chainner_fpng PUBLIC FPNG_NO_STDIO)
set_target_properties(chainner_fpng PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
# fpng's SSE4.1 and PCLMUL intrinsics need these targets under clang; fpng runs those paths
# only after its own CPUID check (fpng_init).
target_compile_options(chainner_fpng PRIVATE /W3 /EHsc /clang:-msse4.1 /clang:-mpclmul)
if(CHAINNER_C_PDB)
    # Symbols for fpng's static functions in _chainner_graph.pdb (sampling).
    target_compile_options(chainner_fpng PRIVATE /Zi)
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
target_link_libraries(_chainner_graph PRIVATE "${chainner_c_python_library}" chainner_native chainner_fpng user32)
set_target_properties(_chainner_graph PROPERTIES PREFIX "" SUFFIX ".pyd"
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO
    LIBRARY_OUTPUT_DIRECTORY "${CHAINNER_C_OUTPUT_DIR}")
target_compile_options(_chainner_graph PRIVATE /W4 /WX /EHsc)
if(CHAINNER_C_PDB)
    target_compile_options(_chainner_graph PRIVATE /Zi)
    target_link_options(_chainner_graph PRIVATE /DEBUG /OPT:REF /OPT:ICF
        "/MAP:${CHAINNER_C_OUTPUT_DIR}/_chainner_graph.map")
endif()

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
