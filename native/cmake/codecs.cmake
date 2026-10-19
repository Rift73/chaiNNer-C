# The image codecs of Load Image, Load Images and Save Image (lane I of the instant-exec design): the libraries
# today's Pillow 12.3.0 and OpenCV 5.0.0 decode and encode with, at their versions and settings, so decoded pixels and
# saved bytes stay what they are. Each library is byte-for-byte upstream source in third_party/<library> (LICENSE,
# PROVENANCE.md with the tarball's SHA-256), built from explicit source lists with every option of CMakeLists.txt;
# a library's chainner/ folder holds the configuration headers its own build would generate for these settings.
# chainner_codecs (src/codecs) is chaiNNer-C's code over them, linked into _chainner_graph.

# libjpeg-turbo 3.1.4.1 with the libjpeg v8 API (Pillow's WITH_JPEG8 build), arithmetic coding, and its x86-64 SIMD
# (SSE2 and AVX2 behind its own CPUID check; JSIMD_FORCENONE / JSIMD_FORCESSE2 cap it), assembled by NASM 3.01.
set(chainner_c_jpeg "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libjpeg-turbo")
set(chainner_c_jpeg_sources
    jcapimin.c wrapper/jcapistd-8.c wrapper/jcapistd-12.c wrapper/jcapistd-16.c wrapper/jccoefct-8.c
    wrapper/jccoefct-12.c wrapper/jccolor-8.c wrapper/jccolor-12.c wrapper/jccolor-16.c wrapper/jcdctmgr-8.c
    wrapper/jcdctmgr-12.c wrapper/jcdiffct-8.c wrapper/jcdiffct-12.c wrapper/jcdiffct-16.c jchuff.c jcicc.c
    jcinit.c jclhuff.c wrapper/jclossls-8.c wrapper/jclossls-12.c wrapper/jclossls-16.c wrapper/jcmainct-8.c
    wrapper/jcmainct-12.c wrapper/jcmainct-16.c jcmarker.c jcmaster.c jcomapi.c jcparam.c jcphuff.c
    wrapper/jcprepct-8.c wrapper/jcprepct-12.c wrapper/jcprepct-16.c wrapper/jcsample-8.c wrapper/jcsample-12.c
    wrapper/jcsample-16.c jctrans.c jdapimin.c wrapper/jdapistd-8.c wrapper/jdapistd-12.c wrapper/jdapistd-16.c
    jdatadst.c jdatasrc.c wrapper/jdcoefct-8.c wrapper/jdcoefct-12.c wrapper/jdcolor-8.c wrapper/jdcolor-12.c
    wrapper/jdcolor-16.c wrapper/jddctmgr-8.c wrapper/jddctmgr-12.c wrapper/jddiffct-8.c wrapper/jddiffct-12.c
    wrapper/jddiffct-16.c jdhuff.c jdicc.c jdinput.c jdlhuff.c wrapper/jdlossls-8.c wrapper/jdlossls-12.c
    wrapper/jdlossls-16.c wrapper/jdmainct-8.c wrapper/jdmainct-12.c wrapper/jdmainct-16.c jdmarker.c jdmaster.c
    wrapper/jdmerge-8.c wrapper/jdmerge-12.c jdphuff.c wrapper/jdpostct-8.c wrapper/jdpostct-12.c
    wrapper/jdpostct-16.c wrapper/jdsample-8.c wrapper/jdsample-12.c wrapper/jdsample-16.c jdtrans.c jerror.c
    jfdctflt.c wrapper/jfdctfst-8.c wrapper/jfdctfst-12.c wrapper/jfdctint-8.c wrapper/jfdctint-12.c
    wrapper/jidctflt-8.c wrapper/jidctflt-12.c wrapper/jidctfst-8.c wrapper/jidctfst-12.c wrapper/jidctint-8.c
    wrapper/jidctint-12.c wrapper/jidctred-8.c wrapper/jidctred-12.c jmemmgr.c jmemnobs.c jpeg_nbits.c
    wrapper/jquant1-8.c wrapper/jquant1-12.c wrapper/jquant2-8.c wrapper/jquant2-12.c wrapper/jutils-8.c
    wrapper/jutils-12.c wrapper/jutils-16.c jaricom.c jcarith.c jdarith.c)
list(TRANSFORM chainner_c_jpeg_sources PREPEND "${chainner_c_jpeg}/src/")
set(chainner_c_jpeg_simd
    jsimdcpu jfdctflt-sse jccolor-sse2 jcgray-sse2 jchuff-sse2 jcphuff-sse2 jcsample-sse2 jdcolor-sse2
    jdmerge-sse2 jdsample-sse2 jfdctfst-sse2 jfdctint-sse2 jidctflt-sse2 jidctfst-sse2 jidctint-sse2
    jidctred-sse2 jquantf-sse2 jquanti-sse2 jccolor-avx2 jcgray-avx2 jcsample-avx2 jdcolor-avx2 jdmerge-avx2
    jdsample-avx2 jfdctint-avx2 jidctint-avx2 jquanti-avx2)
# NASM is pinned like the compilers: version 3.01, found on PATH or given as CHAINNER_C_NASM. The objects are
# custom commands rather than an ASM_NASM language, which would receive the compile options above.
find_program(CHAINNER_C_NASM nasm DOC "NASM 3.01 (nasm.exe), which assembles libjpeg-turbo's x86-64 SIMD")
if(NOT CHAINNER_C_NASM)
    message(FATAL_ERROR "NASM 3.01 is needed for libjpeg-turbo's SIMD: put nasm.exe on PATH or pass "
        "CHAINNER_C_NASM=<nasm.exe> (Build.ps1 -Define)")
endif()
execute_process(COMMAND "${CHAINNER_C_NASM}" -v OUTPUT_VARIABLE chainner_c_nasm_version RESULT_VARIABLE chainner_c_nasm_result)
if(NOT chainner_c_nasm_result EQUAL 0 OR NOT chainner_c_nasm_version MATCHES "^NASM version 3\\.01 ")
    message(FATAL_ERROR "chaiNNer-C assembles with NASM 3.01 only, not '${chainner_c_nasm_version}' (${CHAINNER_C_NASM})")
endif()
file(GLOB chainner_c_jpeg_simd_includes "${chainner_c_jpeg}/simd/nasm/*.inc" "${chainner_c_jpeg}/simd/x86_64/*ext-*.asm")
set(chainner_c_jpeg_objects "")
foreach(chainner_c_unit IN LISTS chainner_c_jpeg_simd)
    set(chainner_c_object "${CMAKE_CURRENT_BINARY_DIR}/jpeg-simd/${chainner_c_unit}.obj")
    add_custom_command(OUTPUT "${chainner_c_object}"
        COMMAND "${CHAINNER_C_NASM}" -fwin64 -DWIN64 -D__x86_64__ --reproducible
            "-I${chainner_c_jpeg}/simd/nasm/" "-I${chainner_c_jpeg}/simd/x86_64/"
            -o "${chainner_c_object}" "${chainner_c_jpeg}/simd/x86_64/${chainner_c_unit}.asm"
        DEPENDS "${chainner_c_jpeg}/simd/x86_64/${chainner_c_unit}.asm" ${chainner_c_jpeg_simd_includes}
        COMMENT "Assembling libjpeg-turbo simd/x86_64/${chainner_c_unit}.asm"
        VERBATIM)
    list(APPEND chainner_c_jpeg_objects "${chainner_c_object}")
endforeach()
set_source_files_properties(${chainner_c_jpeg_objects} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
add_library(chainner_jpeg STATIC ${chainner_c_jpeg_sources} "${chainner_c_jpeg}/simd/x86_64/jsimd.c"
    ${chainner_c_jpeg_objects})
target_include_directories(chainner_jpeg SYSTEM PUBLIC "${chainner_c_jpeg}/chainner" "${chainner_c_jpeg}/src")
target_compile_definitions(chainner_jpeg PRIVATE _CRT_NONSTDC_NO_WARNINGS)

# zlib 1.3.2, classic (OpenCV's PNG and TIFF encoders deflate with it; Pillow's zlib-ng gives other streams).
set(chainner_c_zlib "${CMAKE_CURRENT_SOURCE_DIR}/third_party/zlib")
add_library(chainner_zlib STATIC "${chainner_c_zlib}/adler32.c" "${chainner_c_zlib}/compress.c"
    "${chainner_c_zlib}/crc32.c" "${chainner_c_zlib}/deflate.c" "${chainner_c_zlib}/inffast.c"
    "${chainner_c_zlib}/inflate.c" "${chainner_c_zlib}/inftrees.c" "${chainner_c_zlib}/trees.c"
    "${chainner_c_zlib}/uncompr.c" "${chainner_c_zlib}/zutil.c")
target_include_directories(chainner_zlib SYSTEM PUBLIC "${chainner_c_zlib}")
target_compile_definitions(chainner_zlib PRIVATE _CRT_SECURE_NO_DEPRECATE _CRT_NONSTDC_NO_DEPRECATE)

# libpng 1.6.57 configured as OpenCV 5.0.0 builds it: OpenCV's pnglibconf.h and the Intel SSE2 filters.
set(chainner_c_png "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libpng")
add_library(chainner_png STATIC "${chainner_c_png}/png.c" "${chainner_c_png}/pngerror.c" "${chainner_c_png}/pngget.c"
    "${chainner_c_png}/pngmem.c" "${chainner_c_png}/pngpread.c" "${chainner_c_png}/pngread.c"
    "${chainner_c_png}/pngrio.c" "${chainner_c_png}/pngrtran.c" "${chainner_c_png}/pngrutil.c"
    "${chainner_c_png}/pngset.c" "${chainner_c_png}/pngtrans.c" "${chainner_c_png}/pngwio.c"
    "${chainner_c_png}/pngwrite.c" "${chainner_c_png}/pngwtran.c" "${chainner_c_png}/pngwutil.c"
    "${chainner_c_png}/intel/intel_init.c" "${chainner_c_png}/intel/filter_sse2_intrinsics.c")
target_include_directories(chainner_png SYSTEM PUBLIC "${chainner_c_png}/chainner" "${chainner_c_png}")
target_compile_definitions(chainner_png PRIVATE PNG_INTEL_SSE_OPT=1 _CRT_SECURE_NO_DEPRECATE)
target_link_libraries(chainner_png PUBLIC chainner_zlib)

# libwebp 1.6.0 with sharpyuv, demux and mux, as OpenCV 5.0 and Pillow 12.3 build it: every x86 unit with libwebp's
# runtime dispatch (SSE2 baseline; the SSE4.1 and AVX2 units compiled for their level, run after its CPUID check)
# and WEBP_USE_THREAD.
set(chainner_c_webp "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libwebp")
set(chainner_c_webp_sources
    sharpyuv/sharpyuv.c sharpyuv/sharpyuv_cpu.c sharpyuv/sharpyuv_csp.c sharpyuv/sharpyuv_dsp.c
    sharpyuv/sharpyuv_gamma.c sharpyuv/sharpyuv_sse2.c src/dec/alpha_dec.c src/dec/buffer_dec.c src/dec/frame_dec.c
    src/dec/idec_dec.c src/dec/io_dec.c src/dec/quant_dec.c src/dec/tree_dec.c src/dec/vp8_dec.c src/dec/vp8l_dec.c
    src/dec/webp_dec.c src/demux/anim_decode.c src/demux/demux.c src/dsp/alpha_processing.c
    src/dsp/alpha_processing_sse2.c src/dsp/alpha_processing_sse41.c src/dsp/cost.c src/dsp/cost_sse2.c src/dsp/cpu.c
    src/dsp/dec.c src/dsp/dec_clip_tables.c src/dsp/dec_sse2.c src/dsp/dec_sse41.c src/dsp/enc.c src/dsp/enc_sse2.c
    src/dsp/enc_sse41.c src/dsp/filters.c src/dsp/filters_sse2.c src/dsp/lossless.c src/dsp/lossless_avx2.c
    src/dsp/lossless_enc.c src/dsp/lossless_enc_avx2.c src/dsp/lossless_enc_sse2.c src/dsp/lossless_enc_sse41.c
    src/dsp/lossless_sse2.c src/dsp/lossless_sse41.c src/dsp/rescaler.c src/dsp/rescaler_sse2.c src/dsp/ssim.c
    src/dsp/ssim_sse2.c src/dsp/upsampling.c src/dsp/upsampling_sse2.c src/dsp/upsampling_sse41.c src/dsp/yuv.c
    src/dsp/yuv_sse2.c src/dsp/yuv_sse41.c src/enc/alpha_enc.c src/enc/analysis_enc.c
    src/enc/backward_references_cost_enc.c src/enc/backward_references_enc.c src/enc/config_enc.c src/enc/cost_enc.c
    src/enc/filter_enc.c src/enc/frame_enc.c src/enc/histogram_enc.c src/enc/iterator_enc.c
    src/enc/near_lossless_enc.c src/enc/picture_csp_enc.c src/enc/picture_enc.c src/enc/picture_psnr_enc.c
    src/enc/picture_rescale_enc.c src/enc/picture_tools_enc.c src/enc/predictor_enc.c src/enc/quant_enc.c
    src/enc/syntax_enc.c src/enc/token_enc.c src/enc/tree_enc.c src/enc/vp8l_enc.c src/enc/webp_enc.c
    src/mux/muxedit.c src/mux/muxinternal.c src/mux/muxread.c src/utils/bit_reader_utils.c
    src/utils/bit_writer_utils.c src/utils/color_cache_utils.c src/utils/filters_utils.c
    src/utils/huffman_encode_utils.c src/utils/huffman_utils.c src/utils/palette.c src/utils/quant_levels_dec_utils.c
    src/utils/quant_levels_utils.c src/utils/random_utils.c src/utils/rescaler_utils.c src/utils/thread_utils.c
    src/utils/utils.c)
list(TRANSFORM chainner_c_webp_sources PREPEND "${chainner_c_webp}/")
add_library(chainner_webp STATIC ${chainner_c_webp_sources})
target_include_directories(chainner_webp SYSTEM PUBLIC "${chainner_c_webp}/src")
target_include_directories(chainner_webp SYSTEM PRIVATE "${chainner_c_webp}")
target_compile_definitions(chainner_webp PRIVATE WEBP_USE_THREAD)
foreach(chainner_c_unit IN LISTS chainner_c_webp_sources)
    if(chainner_c_unit MATCHES "_sse41[.]c$")
        set_source_files_properties("${chainner_c_unit}" PROPERTIES COMPILE_OPTIONS "/clang:-msse4.1")
    elseif(chainner_c_unit MATCHES "_avx2[.]c$")
        set_source_files_properties("${chainner_c_unit}" PROPERTIES COMPILE_OPTIONS "/clang:-mavx2")
    endif()
endforeach()

foreach(chainner_c_target chainner_jpeg chainner_zlib chainner_png chainner_webp)
    set_target_properties(${chainner_c_target} PROPERTIES C_STANDARD 17 C_STANDARD_REQUIRED YES)
    target_compile_options(${chainner_c_target} PRIVATE /W3)
    if(CHAINNER_C_PDB)
        target_compile_options(${chainner_c_target} PRIVATE /Zi)
    endif()
endforeach()

# chaiNNer-C's codec layer: the decoders' and encoders' calls into the libraries with today's settings.
add_library(chainner_codecs STATIC src/codecs/codec_output.c src/codecs/jpeg_codec.c src/codecs/png_codec.c
    src/codecs/webp_codec.c src/codecs/bmp_codec.c src/codecs/pxm_codec.c)
target_include_directories(chainner_codecs PUBLIC include)
target_link_libraries(chainner_codecs PRIVATE chainner_jpeg chainner_png chainner_webp)
set_target_properties(chainner_codecs PROPERTIES C_STANDARD 17 C_STANDARD_REQUIRED YES C_EXTENSIONS NO)
target_compile_options(chainner_codecs PRIVATE /W4 /WX)
if(CHAINNER_C_PDB)
    target_compile_options(chainner_codecs PRIVATE /Zi)
endif()
target_link_libraries(_chainner_graph PRIVATE chainner_codecs)
target_sources(_chainner_graph PRIVATE src/image_decode.cpp src/image_encode.cpp src/image_preview.cpp)
target_sources(_chainner_graph PRIVATE src/atomic_file.cpp src/atomic_file_python.cpp)

# libtiff 4.7.1 configured as OpenCV 5.0.0 builds it (third_party/libtiff/chainner): its codec set with zlib and
# 8-bit libjpeg-turbo, Win32 I/O, no strip chopping; and chaiNNer-C's TIFF codec over it.
set(chainner_c_tiff "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtiff")
set(chainner_c_tiff_sources
    tif_aux.c tif_close.c tif_codec.c tif_color.c tif_compress.c tif_dir.c tif_dirinfo.c tif_dirread.c
    tif_dirwrite.c tif_dumpmode.c tif_error.c tif_extension.c tif_fax3.c tif_fax3sm.c tif_flush.c tif_getimage.c
    tif_hash_set.c tif_jpeg.c tif_luv.c tif_lzw.c tif_next.c tif_open.c tif_packbits.c tif_predict.c tif_print.c
    tif_read.c tif_strip.c tif_swab.c tif_thunder.c tif_tile.c tif_version.c tif_warning.c tif_win32.c tif_write.c
    tif_zip.c)
list(TRANSFORM chainner_c_tiff_sources PREPEND "${chainner_c_tiff}/")
add_library(chainner_tiff STATIC ${chainner_c_tiff_sources})
target_include_directories(chainner_tiff SYSTEM PUBLIC "${chainner_c_tiff}/chainner" "${chainner_c_tiff}")
target_compile_definitions(chainner_tiff PRIVATE _CRT_SECURE_NO_DEPRECATE _CRT_NONSTDC_NO_DEPRECATE)
target_link_libraries(chainner_tiff PUBLIC chainner_zlib PRIVATE chainner_jpeg)
set_target_properties(chainner_tiff PROPERTIES C_STANDARD 17 C_STANDARD_REQUIRED YES)
target_compile_options(chainner_tiff PRIVATE /W3)
if(CHAINNER_C_PDB)
    target_compile_options(chainner_tiff PRIVATE /Zi)
endif()
target_sources(chainner_codecs PRIVATE src/codecs/tiff_codec.c)
target_link_libraries(chainner_codecs PRIVATE chainner_tiff)

# OpenEXR's C library OpenEXRCore 3.5.2 for Load Image's .exr (the runtime's OpenCV has no OpenEXR) with what it
# builds against in its own release: libdeflate 1.26 (ZIP, ZIPS and the DWA and PXR24 stages), zstd 1.5.7 (ZSTD),
# OpenJPH 0.32.0 (HTJ2K) and Imath 3.2.3's half.h, each as upstream's CMake builds it for a static Windows x64
# library; every SIMD path behind its library's own CPUID check. third_party/*/PROVENANCE.md says what is whose.
set(chainner_c_deflate "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libdeflate")
add_library(chainner_deflate STATIC "${chainner_c_deflate}/lib/adler32.c" "${chainner_c_deflate}/lib/crc32.c"
    "${chainner_c_deflate}/lib/deflate_compress.c" "${chainner_c_deflate}/lib/deflate_decompress.c"
    "${chainner_c_deflate}/lib/gzip_compress.c" "${chainner_c_deflate}/lib/gzip_decompress.c"
    "${chainner_c_deflate}/lib/utils.c" "${chainner_c_deflate}/lib/x86/cpu_features.c"
    "${chainner_c_deflate}/lib/zlib_compress.c" "${chainner_c_deflate}/lib/zlib_decompress.c")
target_include_directories(chainner_deflate SYSTEM PUBLIC "${chainner_c_deflate}")

# zstd's library sources as OpenEXR 3.5.2 compiles them into OpenEXRCore (no legacy formats, no dictionary builder,
# single-threaded); ZSTD_DISABLE_ASM as on OpenEXR's MSVC path.
set(chainner_c_zstd "${CMAKE_CURRENT_SOURCE_DIR}/third_party/zstd/lib")
set(chainner_c_zstd_sources
    common/debug.c common/entropy_common.c common/error_private.c common/fse_decompress.c common/pool.c
    common/threading.c common/xxhash.c common/zstd_common.c compress/fse_compress.c compress/hist.c
    compress/huf_compress.c compress/zstd_compress.c compress/zstd_compress_literals.c
    compress/zstd_compress_sequences.c compress/zstd_compress_superblock.c compress/zstd_double_fast.c
    compress/zstd_fast.c compress/zstd_lazy.c compress/zstd_ldm.c compress/zstd_opt.c compress/zstd_preSplit.c
    compress/zstdmt_compress.c decompress/huf_decompress.c decompress/zstd_ddict.c decompress/zstd_decompress.c
    decompress/zstd_decompress_block.c)
list(TRANSFORM chainner_c_zstd_sources PREPEND "${chainner_c_zstd}/")
add_library(chainner_zstd STATIC ${chainner_c_zstd_sources})
target_include_directories(chainner_zstd SYSTEM PUBLIC "${chainner_c_zstd}")
target_compile_definitions(chainner_zstd PRIVATE ZSTD_DISABLE_ASM)

# OpenJPH 0.32.0's core with its x86-64 kernels, each unit compiled for its level as OpenJPH's CMake does for clang.
# (unit|flags, flags separated by commas)
set(chainner_c_ojph "${CMAKE_CURRENT_SOURCE_DIR}/third_party/openjph/src/core")
set(chainner_c_ojph_sources
    codestream/ojph_codeblock.cpp codestream/ojph_codeblock_fun.cpp codestream/ojph_codestream.cpp
    codestream/ojph_codestream_gen.cpp codestream/ojph_codestream_local.cpp codestream/ojph_params.cpp
    codestream/ojph_precinct.cpp codestream/ojph_resolution.cpp codestream/ojph_subband.cpp codestream/ojph_tile.cpp
    codestream/ojph_tile_comp.cpp coding/ojph_block_common.cpp coding/ojph_block_decoder32.cpp
    coding/ojph_block_decoder64.cpp coding/ojph_block_encoder.cpp others/ojph_arch.cpp others/ojph_file.cpp
    others/ojph_mem.cpp others/ojph_mem_c.c others/ojph_message.cpp transform/ojph_colour.cpp
    transform/ojph_transform.cpp)
set(chainner_c_ojph_simd
    "codestream/ojph_codestream_sse.cpp|-msse" "codestream/ojph_codestream_sse2.cpp|-msse2"
    "codestream/ojph_codestream_avx.cpp|-mavx" "codestream/ojph_codestream_avx2.cpp|-mavx2"
    "coding/ojph_block_decoder_ssse3.cpp|-mssse3" "coding/ojph_block_decoder_avx2.cpp|-mavx2"
    "coding/ojph_block_encoder_avx2.cpp|-mavx2" "coding/ojph_block_encoder_avx512.cpp|-mavx512f,-mavx512cd"
    "transform/ojph_colour_sse.cpp|-msse" "transform/ojph_colour_sse2.cpp|-msse2" "transform/ojph_colour_avx.cpp|-mavx"
    "transform/ojph_colour_avx2.cpp|-mavx2" "transform/ojph_transform_sse.cpp|-msse"
    "transform/ojph_transform_sse2.cpp|-msse2" "transform/ojph_transform_avx.cpp|-mavx"
    "transform/ojph_transform_avx2.cpp|-mavx2" "transform/ojph_transform_avx512.cpp|-mavx512f,-mavx512cd")
list(TRANSFORM chainner_c_ojph_sources PREPEND "${chainner_c_ojph}/")
add_library(chainner_openjph STATIC ${chainner_c_ojph_sources})
foreach(chainner_c_entry IN LISTS chainner_c_ojph_simd)
    string(REPLACE "|" ";" chainner_c_entry "${chainner_c_entry}")
    string(REPLACE "," ";" chainner_c_entry "${chainner_c_entry}")
    list(POP_FRONT chainner_c_entry chainner_c_unit)
    list(TRANSFORM chainner_c_entry PREPEND "/clang:")
    target_sources(chainner_openjph PRIVATE "${chainner_c_ojph}/${chainner_c_unit}")
    set_source_files_properties("${chainner_c_ojph}/${chainner_c_unit}" PROPERTIES COMPILE_OPTIONS "${chainner_c_entry}")
endforeach()
target_include_directories(chainner_openjph SYSTEM PUBLIC "${chainner_c_ojph}" PRIVATE "${chainner_c_ojph}/openjph")
target_compile_definitions(chainner_openjph PRIVATE _CRT_SECURE_NO_WARNINGS)

# OpenEXRCore 3.5.2, static and not embedded (OpenEXR's default), threading on; third_party/openexr/chainner holds
# the OpenEXRConfig.h and IlmThreadConfig.h its CMake writes, third_party/imath/chainner Imath's ImathConfig.h
# (IMATH_HALF_USE_LOOKUP_TABLE off: half.h converts without Imath's library).
set(chainner_c_exr "${CMAKE_CURRENT_SOURCE_DIR}/third_party/openexr")
set(chainner_c_exr_sources
    internal_rle.c internal_zip.c internal_pxr24.c internal_b44.c internal_b44_table.c internal_b44_table_init.c
    internal_piz.c internal_ht.cpp internal_ht_common.cpp internal_dwa.c internal_dwa_table.c
    internal_dwa_table_init.c internal_huf.c internal_zstd.c internal_zstd_delta.c internal_zstd_shuffle.c
    attributes.c string.c string_vector.c float_vector.c channel_list.c opaque.c bytes.c preview.c base.c context.c
    memory.c internal_structs.c part.c part_attr.c std_attr.c color_metadata.c parse_header.c write_header.c chunk.c
    coding.c compression.c decoding.c encoding.c pack.c unpack.c validation.c debug.c)
list(TRANSFORM chainner_c_exr_sources PREPEND "${chainner_c_exr}/OpenEXRCore/")
add_library(chainner_openexr STATIC ${chainner_c_exr_sources})
target_include_directories(chainner_openexr SYSTEM PUBLIC "${chainner_c_exr}/OpenEXRCore" "${chainner_c_exr}/chainner"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/imath/chainner" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/imath/chainner/Imath"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/imath/Imath")
target_link_libraries(chainner_openexr PRIVATE chainner_deflate chainner_zstd chainner_openjph)

foreach(chainner_c_target chainner_deflate chainner_zstd chainner_openjph chainner_openexr)
    set_target_properties(${chainner_c_target} PROPERTIES C_STANDARD 17 C_STANDARD_REQUIRED YES CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES)
    target_compile_options(${chainner_c_target} PRIVATE /W3 $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
    if(CHAINNER_C_PDB)
        target_compile_options(${chainner_c_target} PRIVATE /Zi)
    endif()
endforeach()
target_sources(chainner_codecs PRIVATE src/codecs/exr_codec.c)
target_link_libraries(chainner_codecs PRIVATE chainner_openexr chainner_native)
