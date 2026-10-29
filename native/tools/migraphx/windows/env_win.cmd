@rem Build environment for MIGraphX 2.16.1 on Windows: VS 2022 Build Tools (x64), its CMake 3.31.6 and Ninja 1.12.1,
@rem the ROCm 7.14.1 SDK (pip wheels, devel expanded) and its LLVM (AMD clang 23: clang-cl for the third-party
@rem dependencies and rocMLIR, as AMDMIGraphX's own Windows GPU CI does; clang++ for MIGraphX itself).
@rem Settings (work directory, tool locations, jobs): settings_win.cmd.
call "%~dp0settings_win.cmd" || exit /b 1
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "ROCM_PATH=%MIGX_ROCM%"
set "HIP_PATH=%ROCM_PATH%"
set "PATH=%ROCM_PATH%\bin;%PATH%"
set "LLVMBIN=%ROCM_PATH%\lib\llvm\bin"
set "PATH=%VSROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%VSROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
set "DEPS=%ROOT%\deps"
set "CMAKE_PREFIX_PATH="
