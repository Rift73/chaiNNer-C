@echo off
rem rocMLIR 518955ca (MIGraphX 2.16.1's requirements.txt pin) for Windows as the static fat rockCompiler library,
rem built as AMDMIGraphX's Windows GPU CI builds it: clang-cl, /std:c++17, /MD, Ninja, Release,
rem -DBUILD_FAT_LIBROCKCOMPILER=On -DLLVM_INCLUDE_TESTS=Off -DLLVM_DISABLE_ASSEMBLY_FILES=On -DLLVM_ENABLE_DIA_SDK=Off.
rem LLVM's optional zlib/zstd/libxml2 links are off, as in the Linux build. The ROCm root is on the prefix path so
rem rocMLIR embeds the SDK's AMD device libraries, as on Linux. Installs into the dependency prefix. Its archive
rem (build_deps.sh's rocmlir, same URL and SHA-256) is unpacked into ROOT\deps-src\rocmlir; ROOT is the work directory
rem (settings_win.cmd). CMake uses the Python interpreter MIGX_PYTHON only to configure.
setlocal
call "%~dp0env_win.cmd" || exit /b 1
set "SRC=%ROOT%\deps-src\rocmlir\rocMLIR-518955cab3ec2a53cbb03216cb55ac6ab47ee1da"
set "B=%ROOT%\deps-build\rocmlir"
set "CXXFLAGS=/std:c++17"
set "PY=%MIGX_PYTHON%"
if exist "%ROOT%\deps-build\rocmlir.done" (
  echo rocmlir: already built
  exit /b 0
)
echo [%DATE% %TIME%] rocmlir: configure
cmake -S "%SRC%" -B "%B%" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_INSTALL_PREFIX=%DEPS%" "-DCMAKE_PREFIX_PATH=%DEPS%;%ROCM_PATH%" "-DCMAKE_C_COMPILER=%LLVMBIN%\clang-cl.exe" "-DCMAKE_CXX_COMPILER=%LLVMBIN%\clang-cl.exe" -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL -DBUILD_FAT_LIBROCKCOMPILER=On -DLLVM_INCLUDE_TESTS=Off -DLLVM_DISABLE_ASSEMBLY_FILES=On -DLLVM_ENABLE_DIA_SDK=Off -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_LIBXML2=OFF -DROCMLIR_PARALLEL_LINK_JOBS=2 "-DPython3_EXECUTABLE=%PY%" > "%B%.configure.log" 2>&1 || (
  powershell -NoProfile -Command "Get-Content -Tail 60 '%B%.configure.log'"
  exit /b 1
)
echo [%DATE% %TIME%] rocmlir: build
cmake --build "%B%" -j %MIGX_JOBS% > "%B%.build.log" 2>&1 || (
  powershell -NoProfile -Command "Select-String -Path '%B%.build.log' -Pattern 'error|FAILED' | Select-Object -First 40; Get-Content -Tail 40 '%B%.build.log'"
  exit /b 1
)
cmake --install "%B%" > "%B%.install.log" 2>&1 || (
  powershell -NoProfile -Command "Get-Content -Tail 40 '%B%.install.log'"
  exit /b 1
)
type nul > "%ROOT%\deps-build\rocmlir.done"
echo [%DATE% %TIME%] rocmlir: done
exit /b 0
