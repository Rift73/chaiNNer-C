# libFuzzer harnesses over chaiNNer-C's readers of outside input (README.md here), configured by CHAINNER_C_FUZZ, which
# gives every target of the build AddressSanitizer and libFuzzer's coverage (../../CMakeLists.txt). libFuzzer's
# runtime is linked as clang-cl's driver would add it, since CMake links with lld-link itself.

# Load Image's decoders: image_decode.cpp over the codecs, with parallel.c (OpenEXR's chunk loop) compiled in, so
# coverage reaches libFuzzer from one module instead of through chainner_native.dll.
add_executable(fuzz_image_decode fuzz_image_decode.cpp ../../src/image_decode.cpp ../../src/parallel.c)
target_link_libraries(fuzz_image_decode PRIVATE chainner_codecs)
# The regex engine of chainner_ext and Text nodes.
add_executable(fuzz_regex fuzz_regex.c)
target_link_libraries(fuzz_regex PRIVATE chainner_regex)
foreach(target fuzz_image_decode fuzz_regex)
    set_target_properties(${target} PROPERTIES C_STANDARD 17 C_STANDARD_REQUIRED YES C_EXTENSIONS NO
        CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO RUNTIME_OUTPUT_DIRECTORY "${CHAINNER_C_OUTPUT_DIR}")
    target_include_directories(${target} PRIVATE ../../include)
    target_compile_options(${target} PRIVATE /W4 /WX $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
    target_link_options(${target} PRIVATE "/WHOLEARCHIVE:${chainner_c_clang_rt}/clang_rt.fuzzer-x86_64.lib")
endforeach()
# A throw from inside a catch handler, as OpenJPH reports its decode errors, faults when the handler is instrumented
# (a breakpoint, a false stack overflow or an access violation in it; the Release build throws and catches it), which
# would end every OpenEXR run at its first damaged HTJ2K tile (test_exr_decode.py's test_damaged_htj2k_tile_raises):
# OpenJPH, a vendored library, is built without AddressSanitizer and libFuzzer's coverage.
target_compile_options(chainner_openjph PRIVATE -fno-sanitize=address,fuzzer-no-link)
# UndefinedBehaviorSanitizer on chaiNNer-C's own code the harnesses run (the vendored libraries are left as they are),
# stopping at the first report; its handlers are in the AddressSanitizer runtime. Unsigned wrapping is defined (the
# regex engine relies on it) and so is IEEE float division by zero under /fp:strict, so neither is checked.
foreach(target chainner_codecs chainner_regex fuzz_image_decode fuzz_regex)
    target_compile_options(${target} PRIVATE
        -fsanitize=signed-integer-overflow,shift,integer-divide-by-zero,float-cast-overflow,bool,enum,bounds,null,return,unreachable,vla-bound,pointer-overflow,nonnull-attribute
        -fno-sanitize-recover=all)
endforeach()
