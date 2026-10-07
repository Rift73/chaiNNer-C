# chaiNNer-C's one toolchain (owner directive 2026-10-06), pinned: clang-cl, lld-link and
# llvm-lib of LLVM 23.1.2, with the Build Tools' MSVC 14.44.35207 libraries and the Windows
# SDK 10.0.26100.0 as clang-cl's headers and libraries (no Developer environment needed).
# Build.ps1 passes this file. CMakeLists.txt refuses any other clang-cl, and this file any other
# toolset or SDK version (Consult 14 D-30); package_port.py records the three in the manifest.
set(chainner_c_msvc_toolset_version "14.44.35207")
set(chainner_c_windows_sdk_version "10.0.26100.0")
# The four roots are cache variables whose defaults are these paths. Another machine passes its
# own on a build directory's first configure (Build.ps1 -Define chainner_c_llvm=..., etc.): any
# location, but the toolset directory is the pinned version's and the SDK version the pinned one.
set(chainner_c_llvm "C:/Executables/clang+llvm-23.1.2-x86_64-pc-windows-msvc/bin" CACHE PATH
    "LLVM 23.1.2's bin directory: clang-cl, lld-link, llvm-lib and llvm-rc")
set(chainner_c_vc_tools
    "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/${chainner_c_msvc_toolset_version}"
    CACHE PATH "The MSVC toolset directory: clang-cl's /vctoolsdir")
set(chainner_c_sdk "C:/Program Files (x86)/Windows Kits/10" CACHE PATH "The Windows SDK root: clang-cl's /winsdkdir")
set(chainner_c_sdk_version "${chainner_c_windows_sdk_version}" CACHE STRING
    "The Windows SDK version: clang-cl's /winsdkversion")
string(REGEX REPLACE "[/\\\\]+$" "" chainner_c_vc_tools_name "${chainner_c_vc_tools}")
get_filename_component(chainner_c_vc_tools_name "${chainner_c_vc_tools_name}" NAME)
if(NOT chainner_c_vc_tools_name STREQUAL chainner_c_msvc_toolset_version)
    message(FATAL_ERROR "chaiNNer-C builds with the MSVC ${chainner_c_msvc_toolset_version} toolset only, not "
        "${chainner_c_vc_tools_name} (chainner_c_vc_tools=${chainner_c_vc_tools})")
endif()
if(NOT chainner_c_sdk_version STREQUAL chainner_c_windows_sdk_version)
    message(FATAL_ERROR "chaiNNer-C builds with the Windows SDK ${chainner_c_windows_sdk_version} only, not "
        "${chainner_c_sdk_version} (chainner_c_sdk_version)")
endif()

set(CMAKE_C_COMPILER "${chainner_c_llvm}/clang-cl.exe")
set(CMAKE_CXX_COMPILER "${chainner_c_llvm}/clang-cl.exe")
set(CMAKE_LINKER "${chainner_c_llvm}/lld-link.exe")
set(CMAKE_AR "${chainner_c_llvm}/llvm-lib.exe")
set(CMAKE_RC_COMPILER "${chainner_c_llvm}/llvm-rc.exe")
set(CMAKE_MT "${chainner_c_sdk}/bin/${chainner_c_sdk_version}/x64/mt.exe")
set(chainner_c_compile_sysroot
    "/vctoolsdir \"${chainner_c_vc_tools}\" /winsdkdir \"${chainner_c_sdk}\" /winsdkversion ${chainner_c_sdk_version}")
set(chainner_c_link_sysroot
    "\"/vctoolsdir:${chainner_c_vc_tools}\" \"/winsdkdir:${chainner_c_sdk}\" /winsdkversion:${chainner_c_sdk_version}")
set(CMAKE_C_FLAGS_INIT "${chainner_c_compile_sysroot}")
set(CMAKE_CXX_FLAGS_INIT "${chainner_c_compile_sysroot}")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${chainner_c_link_sysroot}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${chainner_c_link_sysroot}")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "${chainner_c_link_sysroot}")
