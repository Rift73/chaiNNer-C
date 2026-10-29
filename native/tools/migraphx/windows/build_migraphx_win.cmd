@echo off
rem MIGraphX 2.16.1 (AMDMIGraphX release/rocm-rel-7.14 at 2487b688, "MIGraphX 2.16.1 for ROCm 7.14.1") for Windows
rem against the ROCm 7.14.1 Windows SDK, with the settings of AMDMIGraphX's Windows GPU CI (rbuild.ini [gh-win-gpu]:
rem GPU target with MIOpen, rocBLAS, hipBLASLt; composable kernel off, as MIGraphX has it on Windows; no Python module;
rem DLL C runtime; ROCm's clang++) and rocMLIR (build_rocmlir_win.cmd), except BUILD_DEV off, as in a release build
rem (and the Linux build): hipRTC compiles the JIT kernels at run time (BUILD_DEV would call a ROCm clang the user
rem does not have) and the inline namespace carries the version. Device code for all 22 ROCm 7.14.1 Windows targets.
rem Source: AMDMIGraphX at that commit in ROOT\src\AMDMIGraphX (fetch_sources.sh); ROOT is the work directory
rem (settings_win.cmd).
rem Output: %ROOT%\install (bin\*.dll, bin\migraphx-hiprtc-driver.exe, lib\*.lib, include\, lib\cmake\migraphx\).
setlocal
call "%~dp0env_win.cmd" || exit /b 1
rem No ROCM_PATH here: with it set, clang looks for the device libraries in %ROCM_PATH%\amdgcn\bitcode, which the
rem Windows wheels lack (Linux has an amdgcn symlink there); without it clang finds them beside itself
rem (lib\llvm\amdgcn\bitcode). HIP_PATH still names the SDK.
rem The offload-arch check run after linking migraphx_device uses the SDK's LLVM tools: those of the VS Build Tools
rem (first on PATH after vcvars64) predate gfx1153 and report it missing.
set "ROCM_PATH="
set "SRC=%ROOT%\src\AMDMIGraphX"
set "B=%ROOT%\build"
set "I=%ROOT%\install"
set "TARGETS=gfx908;gfx90a;gfx1010;gfx1011;gfx1012;gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201"
git -C "%SRC%" rev-parse HEAD | findstr /B 2487b688c453b9007ac69ff1d79ad6cf9b509901 >nul || (
  echo wrong MIGraphX commit
  exit /b 1
)
echo [%DATE% %TIME%] MIGraphX configure, targets %TARGETS%
cmake -S "%SRC%" -B "%B%" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_INSTALL_PREFIX=%I%" "-DCMAKE_PREFIX_PATH=%DEPS%;%HIP_PATH%" "-DCMAKE_C_COMPILER=%LLVMBIN%\clang.exe" "-DCMAKE_CXX_COMPILER=%LLVMBIN%\clang++.exe" -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL "-DGPU_TARGETS=%TARGETS%" -DBUILD_TESTING=OFF -DBUILD_DEV=Off -DMIGRAPHX_ENABLE_GPU=On -DMIGRAPHX_ENABLE_CPU=Off -DMIGRAPHX_ENABLE_FPGA=Off -DMIGRAPHX_ENABLE_PYTHON=Off -DMIGRAPHX_USE_MIOPEN=On -DMIGRAPHX_USE_ROCBLAS=On -DMIGRAPHX_USE_HIPBLASLT=On -DMIGRAPHX_USE_COMPOSABLEKERNEL=Off -DMIGRAPHX_ENABLE_MLIR=On "-DCLANG_OFFLOAD_BUNDLER=%LLVMBIN%\clang-offload-bundler.exe" "-DLLVM_OBJCOPY=%LLVMBIN%\llvm-objcopy.exe" "-DLLVM_READOBJ=%LLVMBIN%\llvm-readobj.exe" "-DLLVM_OBJDUMP=%LLVMBIN%\llvm-objdump.exe" -DCMAKE_JOB_POOLS=link_pool=2 -DCMAKE_JOB_POOL_LINK=link_pool > "%B%.configure.log" 2>&1 || (
  powershell -NoProfile -Command "Get-Content -Tail 60 '%B%.configure.log'"
  exit /b 1
)
powershell -NoProfile -Command "Select-String -Path '%B%.configure.log' -Pattern 'Enable gpu|rocMLIR|hipRTC|HIP Clang|GPU_TARGETS|MIOpen|hipBLASLt|rocBLAS|Eigen|composable' | Select-Object -First 30 | ForEach-Object { $_.Line }"
echo [%DATE% %TIME%] MIGraphX build
cmake --build "%B%" -j %MIGX_JOBS% > "%B%.build.log" 2>&1 || (
  powershell -NoProfile -Command "Select-String -Path '%B%.build.log' -Pattern 'error|FAILED|undefined' | Select-Object -First 40 | ForEach-Object { $_.Line }; Get-Content -Tail 30 '%B%.build.log'"
  exit /b 1
)
cmake --install "%B%" > "%B%.install.log" 2>&1 || (
  powershell -NoProfile -Command "Get-Content -Tail 40 '%B%.install.log'"
  exit /b 1
)
echo [%DATE% %TIME%] MIGRAPHX DONE
dir /b "%I%\bin"
exit /b 0
