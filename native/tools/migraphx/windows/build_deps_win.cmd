@echo off
rem MIGraphX 2.16.1's third-party dependencies for Windows at the versions its requirements.txt pins, built the way
rem AMDMIGraphX's Windows GPU CI builds them (rbuild.ini [gh-win-gpu]): clang-cl, /std:c++17, the DLL C runtime
rem (/MD), Ninja, Release. Composable kernel is not built (MIGraphX turns it off on Windows); pybind11 is not built
rem (no Python module, as [gh-win-gpu]). Sources: the archives of build_deps.sh (same URLs, same SHA-256), each unpacked
rem into ROOT\deps-src\NAME (ROOT is the work directory, settings_win.cmd): NAME is rocm-cmake, abseil, protobuf, json,
rem msgpack, eigen or sqlite, holding the archive's own top folder (abseil-cpp-20250512.0 and so on).
rem rocMLIR is a separate step (build_rocmlir_win.cmd).
rem Building with clang-cl rather than MSVC keeps abseil's MSVC-only "-ignore:4221" link option out of its exported
rem targets; MIGraphX links with clang++'s GNU-style driver, which rejects it.
setlocal
call "%~dp0env_win.cmd" || exit /b 1
set "SRC=%ROOT%\deps-src"
set "BLD=%ROOT%\deps-build"
set "CXXFLAGS=/std:c++17"
if not exist "%BLD%" mkdir "%BLD%"
set COMMON=-G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_INSTALL_PREFIX=%DEPS%" "-DCMAKE_PREFIX_PATH=%DEPS%" "-DCMAKE_C_COMPILER=%LLVMBIN%\clang-cl.exe" "-DCMAKE_CXX_COMPILER=%LLVMBIN%\clang-cl.exe" -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL
"%LLVMBIN%\clang-cl.exe" --version | findstr /C:"clang version"

set "X="
call :cmbuild rocm-cmake "%SRC%\rocm-cmake\rocm-cmake-1d4652ae2ec0e44a67a7c415dd7e51c88a6aa68d" || exit /b 1
set X=-DABSL_ENABLE_INSTALL=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DABSL_MSVC_STATIC_RUNTIME=Off -DABSL_PROPAGATE_CXX_STD=ON
call :cmbuild abseil "%SRC%\abseil\abseil-cpp-20250512.0" || exit /b 1
set X=-DCMAKE_POSITION_INDEPENDENT_CODE=On -Dprotobuf_BUILD_TESTS=Off -Dprotobuf_BUILD_SHARED_LIBS=Off -Dprotobuf_MSVC_STATIC_RUNTIME=Off -Dprotobuf_BUILD_LIBPROTOC=On -Dprotobuf_BUILD_PROTOC_BINARIES=On -Dprotobuf_BUILD_PROTOBUF_BINARIES=On -Dprotobuf_ABSL_PROVIDER=package -Dprotobuf_WITH_ZLIB=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5
call :cmbuild protobuf "%SRC%\protobuf\protobuf-30.0" || exit /b 1
set X=-DJSON_BuildTests=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5
call :cmbuild json "%SRC%\json\json-3.8.0" || exit /b 1
set X=-DMSGPACK_BUILD_TESTS=Off -DMSGPACK_BUILD_EXAMPLES=Off -DMSGPACK_ENABLE_SHARED=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5
call :cmbuild msgpack "%SRC%\msgpack\msgpack-c-cpp-3.3.0" || exit /b 1
set X=-DBUILD_TESTING=Off -DEIGEN_BUILD_DOC=Off -DEIGEN_BUILD_LAPACK=Off -DEIGEN_BUILD_BLAS=Off
call :cmbuild eigen "%SRC%\eigen\eigen-5.0.1" || exit /b 1

rem SQLite 3.50.4 amalgamation as a static library (cget's sqlite3 recipe builds the same amalgamation).
if exist "%BLD%\sqlite.done" goto :sqlite_done
if not exist "%BLD%\sqlite" mkdir "%BLD%\sqlite"
if not exist "%DEPS%\include" mkdir "%DEPS%\include"
if not exist "%DEPS%\lib" mkdir "%DEPS%\lib"
"%LLVMBIN%\clang-cl.exe" /nologo /O2 /MD -DSQLITE_THREADSAFE=1 -DSQLITE_OMIT_LOAD_EXTENSION /c "%SRC%\sqlite\sqlite-amalgamation-3500400\sqlite3.c" "/Fo%BLD%\sqlite\sqlite3.obj" || exit /b 1
"%LLVMBIN%\llvm-lib.exe" /nologo "/OUT:%DEPS%\lib\sqlite3.lib" "%BLD%\sqlite\sqlite3.obj" || exit /b 1
copy /Y "%SRC%\sqlite\sqlite-amalgamation-3500400\sqlite3.h" "%DEPS%\include\" >nul
copy /Y "%SRC%\sqlite\sqlite-amalgamation-3500400\sqlite3ext.h" "%DEPS%\include\" >nul
type nul > "%BLD%\sqlite.done"
echo [%DATE% %TIME%] sqlite: done
:sqlite_done
echo [%DATE% %TIME%] DEPS DONE
exit /b 0

rem cmbuild <name> <source dir>; extra CMake arguments in %X%.
:cmbuild
if exist "%BLD%\%~1.done" (
  echo [%DATE% %TIME%] %~1: already built
  exit /b 0
)
echo [%DATE% %TIME%] %~1: configure
cmake -S "%~2" -B "%BLD%\%~1" %COMMON% %X% > "%BLD%\%~1.configure.log" 2>&1 || (
  powershell -NoProfile -Command "Get-Content -Tail 40 '%BLD%\%~1.configure.log'"
  exit /b 1
)
echo [%DATE% %TIME%] %~1: build
cmake --build "%BLD%\%~1" -j %MIGX_JOBS% > "%BLD%\%~1.build.log" 2>&1 || (
  powershell -NoProfile -Command "Get-Content -Tail 60 '%BLD%\%~1.build.log'"
  exit /b 1
)
cmake --install "%BLD%\%~1" > "%BLD%\%~1.install.log" 2>&1 || (
  powershell -NoProfile -Command "Get-Content -Tail 40 '%BLD%\%~1.install.log'"
  exit /b 1
)
type nul > "%BLD%\%~1.done"
echo [%DATE% %TIME%] %~1: done
exit /b 0
