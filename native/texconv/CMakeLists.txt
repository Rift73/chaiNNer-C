# texconv for Linux (texconv.cpp, wic.cpp): the DDS converter that nodes/impl/dds/texconv.py runs there, as
# Windows runs Microsoft's texconv.exe 2026.5.8.1 (backend/src/texconv). It is built on DirectXTex may2026, the
# release that texconv.exe comes from, with DirectXMath and DirectX-Headers (all three MIT) and the .NET runtime's
# sal.h (MIT; DirectXMath's annotations outside Windows, the file vcpkg installs). The sources are fetched at
# configure time into the build directory, each pinned by its SHA-256; the binary and its notices
# (texconv.LICENSE.txt: the license texts of the four) land in CHAINNER_C_TEXCONV_OUTPUT_DIR.
set(CHAINNER_C_TEXCONV_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../backend/src/texconv" CACHE PATH
    "Output directory of the Linux texconv and its notices")
set(chainner_c_texconv_sources "${CMAKE_CURRENT_BINARY_DIR}/sources")
# name|URL|SHA-256|the archive's top directory, or the file's name
set(chainner_c_texconv_downloads
    "DirectXTex|https://github.com/microsoft/DirectXTex/archive/refs/tags/may2026.tar.gz|204ae68a66fb15d87867d836efbac9ae5aa799ca2a1867024d96a8168fa3651f|DirectXTex-may2026"
    "DirectXMath|https://github.com/microsoft/DirectXMath/archive/refs/tags/apr2025.tar.gz|8ac7f060d7fcc971d77c5a9f3ff98548ecf119c144a7c2dd9d21f4b66ba94cd4|DirectXMath-apr2025"
    "DirectX-Headers|https://github.com/microsoft/DirectX-Headers/archive/refs/tags/v1.619.5.tar.gz|24a0b7d8079a2dbbc90753c0d8bc812040d052acce2302e69a97c5d873b313b8|DirectX-Headers-1.619.5"
    "sal|https://raw.githubusercontent.com/dotnet/runtime/v9.0.2/src/coreclr/pal/inc/rt/sal.h|e26ee611231a859b5a800eb2e1606e7ca03b8e3981d7c98cf1ca20c89b658b68|sal.h"
    "dotnet|https://raw.githubusercontent.com/dotnet/runtime/v9.0.2/LICENSE.TXT|cfc21f5e8bd655ae997eec916138b707b1d290b83272c02a95c9f821b8c87310|LICENSE.TXT")
foreach(chainner_c_download IN LISTS chainner_c_texconv_downloads)
    string(REPLACE "|" ";" chainner_c_download "${chainner_c_download}")
    list(GET chainner_c_download 0 chainner_c_name)
    list(GET chainner_c_download 1 chainner_c_url)
    list(GET chainner_c_download 2 chainner_c_sha256)
    list(GET chainner_c_download 3 chainner_c_entry)
    set(chainner_c_file "${chainner_c_texconv_sources}/${chainner_c_sha256}")
    set(chainner_c_actual "")
    if(EXISTS "${chainner_c_file}")
        file(SHA256 "${chainner_c_file}" chainner_c_actual)
    endif()
    set(chainner_c_texconv_${chainner_c_name} "${chainner_c_texconv_sources}/${chainner_c_name}")
    if(NOT chainner_c_actual STREQUAL chainner_c_sha256)
        message(STATUS "texconv: fetching ${chainner_c_name} (${chainner_c_url})")
        file(REMOVE_RECURSE "${chainner_c_texconv_${chainner_c_name}}")
        file(DOWNLOAD "${chainner_c_url}" "${chainner_c_file}" EXPECTED_HASH SHA256=${chainner_c_sha256}
            STATUS chainner_c_status TLS_VERIFY ON)
        list(GET chainner_c_status 0 chainner_c_code)
        if(NOT chainner_c_code EQUAL 0)
            file(REMOVE "${chainner_c_file}")
            message(FATAL_ERROR "texconv: fetching ${chainner_c_url} failed: ${chainner_c_status}")
        endif()
    endif()
    if(NOT EXISTS "${chainner_c_texconv_${chainner_c_name}}")
        if(chainner_c_url MATCHES "\\.tar\\.gz$")
            file(ARCHIVE_EXTRACT INPUT "${chainner_c_file}" DESTINATION "${chainner_c_texconv_sources}/extract")
            file(RENAME "${chainner_c_texconv_sources}/extract/${chainner_c_entry}" "${chainner_c_texconv_${chainner_c_name}}")
            file(REMOVE_RECURSE "${chainner_c_texconv_sources}/extract")
        else()
            configure_file("${chainner_c_file}" "${chainner_c_texconv_${chainner_c_name}}/${chainner_c_entry}" COPYONLY)
        endif()
    endif()
endforeach()

# DirectXTex's library sources that the DDS steps use: no WIC, no Direct3D, no OpenMP (texconv.cpp spreads block
# compression over threads itself). C++17, DirectXTex's own minimum; its warnings are its own (-w).
set(chainner_c_directxtex_sources BC BC6HBC7 DirectXTexCompress DirectXTexConvert DirectXTexDDS DirectXTexImage
    DirectXTexMipmaps DirectXTexMisc DirectXTexPMAlpha DirectXTexResize DirectXTexUtil)
list(TRANSFORM chainner_c_directxtex_sources PREPEND "${chainner_c_texconv_DirectXTex}/DirectXTex/")
list(TRANSFORM chainner_c_directxtex_sources APPEND ".cpp")
# BC4BC5.cpp with one change: FindClosestUNORM takes its 8 values from bc4_unorm_gradient.h, texconv.exe's
# arithmetic for them (BC4 and BC5 UNORM blocks are otherwise not texconv.exe's at exact ties).
file(READ "${chainner_c_texconv_DirectXTex}/DirectXTex/BC4BC5.cpp" chainner_c_bc4)
set(chainner_c_bc4_call "rGradient[i] = pBC->DecodeFromIndex(i);")
string(FIND "${chainner_c_bc4}" "${chainner_c_bc4_call}" chainner_c_at)
string(FIND "${chainner_c_bc4}" "void FindClosestUNORM(" chainner_c_unorm)
string(FIND "${chainner_c_bc4}" "void FindClosestSNORM(" chainner_c_snorm)
if(chainner_c_at LESS chainner_c_unorm OR chainner_c_at GREATER chainner_c_snorm OR chainner_c_unorm LESS 0)
    message(FATAL_ERROR "texconv: BC4BC5.cpp's FindClosestUNORM is not the may2026 one")
endif()
string(SUBSTRING "${chainner_c_bc4}" 0 ${chainner_c_at} chainner_c_before)
string(LENGTH "${chainner_c_bc4_call}" chainner_c_length)
math(EXPR chainner_c_at "${chainner_c_at} + ${chainner_c_length}")
string(SUBSTRING "${chainner_c_bc4}" ${chainner_c_at} -1 chainner_c_after)
# Written beside, then copied only when it changes, so a configure does not rebuild it.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/BC4BC5.cpp.new" "#include \"bc4_unorm_gradient.h\"\n${chainner_c_before}"
    "rGradient[i] = texconv_unorm_gradient(pBC->red_0, pBC->red_1, i);${chainner_c_after}")
configure_file("${CMAKE_CURRENT_BINARY_DIR}/BC4BC5.cpp.new" "${CMAKE_CURRENT_BINARY_DIR}/BC4BC5.cpp" COPYONLY)
add_library(chainner_directxtex STATIC ${chainner_c_directxtex_sources} "${CMAKE_CURRENT_BINARY_DIR}/BC4BC5.cpp")
target_include_directories(chainner_directxtex SYSTEM PUBLIC "${chainner_c_texconv_DirectXTex}/DirectXTex"
    "${chainner_c_texconv_DirectXMath}/Inc" "${chainner_c_texconv_DirectX-Headers}/include"
    "${chainner_c_texconv_DirectX-Headers}/include/wsl/stubs" "${chainner_c_texconv_sal}")
target_include_directories(chainner_directxtex PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(chainner_directxtex PUBLIC USING_DIRECTX_HEADERS)
# texconv.exe's DirectXTex is built with MSVC's /fp:fast, which turns a division by a constant into a multiplication
# by its reciprocal (x / 255.0f is x * (1 / 255.0f)): -freciprocal-math in place of /fp:strict, without contraction,
# gives its BC1, BC2, BC3, BC6H and BC7 blocks and its conversions.
target_compile_options(chainner_directxtex PRIVATE -w -ffp-model=precise -fno-rounding-math -ffp-exception-behavior=ignore
    -freciprocal-math -ffp-contract=off)
set_target_properties(chainner_directxtex PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)

add_executable(texconv texconv.cpp wic.cpp)
target_link_libraries(texconv PRIVATE chainner_directxtex)
set_target_properties(texconv PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO
    RUNTIME_OUTPUT_DIRECTORY "${CHAINNER_C_TEXCONV_OUTPUT_DIR}")
target_compile_options(texconv PRIVATE /W4 /WX)

# The notices beside the binary: the license texts of what it is built from.
set(chainner_c_texconv_notices "Third-party notices for texconv (Linux)\n\n"
    "texconv is chaiNNer-C's DDS converter for Linux (native/texconv), built on DirectXTex may2026, DirectXMath "
    "apr2025 and DirectX-Headers v1.619.5 (Microsoft) with sal.h of the .NET runtime 9.0.2; its conversion steps "
    "follow DirectXTex's Texconv/texconv.cpp.\n")
foreach(chainner_c_notice "DirectXTex|${chainner_c_texconv_DirectXTex}/LICENSE"
        "DirectXMath|${chainner_c_texconv_DirectXMath}/LICENSE"
        "DirectX-Headers|${chainner_c_texconv_DirectX-Headers}/LICENSE"
        "sal.h (.NET runtime)|${chainner_c_texconv_dotnet}/LICENSE.TXT")
    string(REPLACE "|" ";" chainner_c_notice "${chainner_c_notice}")
    list(GET chainner_c_notice 0 chainner_c_title)
    list(GET chainner_c_notice 1 chainner_c_license)
    file(READ "${chainner_c_license}" chainner_c_text)
    string(APPEND chainner_c_texconv_notices "\n${chainner_c_title}\n\n${chainner_c_text}")
endforeach()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/texconv.LICENSE.txt" "${chainner_c_texconv_notices}")
add_custom_command(TARGET texconv POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_BINARY_DIR}/texconv.LICENSE.txt"
        "${CHAINNER_C_TEXCONV_OUTPUT_DIR}/texconv.LICENSE.txt"
    VERBATIM)
