# chaiNNer-C's Linux toolchain, the counterpart of toolchain.cmake: clang, clang++, ld.lld, llvm-ar and llvm-ranlib
# of LLVM 23.1.2's official Linux release, against the build machine's glibc and its GCC installation's libstdc++
# and libgcc (linked statically, cmake/linux.cmake). build_linux.sh passes this file; CMakeLists.txt refuses any
# other clang.
# The root is a cache variable whose default is the development layout; another machine passes its own on a build
# directory's first configure (build_linux.sh -D chainner_c_llvm=...).
set(chainner_c_llvm "$ENV{HOME}/chainner-linux/toolchains/llvm-23.1.2/bin" CACHE PATH
    "LLVM 23.1.2's bin directory: clang, clang++, ld.lld, llvm-ar and llvm-ranlib")

set(CMAKE_C_COMPILER "${chainner_c_llvm}/clang")
set(CMAKE_CXX_COMPILER "${chainner_c_llvm}/clang++")
set(CMAKE_AR "${chainner_c_llvm}/llvm-ar")
set(CMAKE_RANLIB "${chainner_c_llvm}/llvm-ranlib")
# The driver takes ld.lld from its own directory.
set(CMAKE_EXE_LINKER_FLAGS_INIT "-fuse-ld=lld")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-fuse-ld=lld")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "-fuse-ld=lld")
# The Windows Release build's /O2 (CMake's own Release default for this driver is -O3).
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "Flags used by the C compiler during RELEASE builds.")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "Flags used by the CXX compiler during RELEASE builds.")
