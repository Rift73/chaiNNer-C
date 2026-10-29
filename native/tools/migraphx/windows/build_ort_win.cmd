@echo off
rem ONNX Runtime 1.30.0 Python wheel (cp314, win_amd64) with the DirectML and MIGraphX execution providers (plus CPU).
rem Source: the v1.30.0 tag (f2c39fe) with its submodules in ROOT\ort-src (fetch_sources.sh; ROOT is the work directory,
rem settings_win.cmd) plus the two MIGraphX provider fixes of the Linux wheel, in order: ort-migraphx-cache-name.patch,
rem ort-migraphx-shape-programs.patch, then ort-migraphx-windows-build.patch (build system only: the provider as C++17
rem on MSVC, as migraphx.hpp does not compile as C++20; no empty HIP SDK DLL copy). The files the patches touch are
rem restored from the tag and the patches applied again on every run, as build_ort.sh does.
rem Flags: those of the 1.30.0+trt11.4 Windows wheel (MSVC 14.44, Ninja, --enable_lto
rem --use_binskim_compliant_compile_flags, warnings not fatal, link pool of 1) with the CUDA/TensorRT
rem options replaced by --use_dml (DirectML 1.15.4 NuGet, as ORT pins it) and --use_migraphx against MIGraphX 2.16.1
rem (ROOT\install); unit tests not built (as the Linux MIGraphX wheel). MIGX_JOBS parallel jobs.
rem Only HIP_PATH points at the ROCm SDK: neither its bin folder (it has a flatc.exe of flatbuffers 25, ORT needs 23)
rem nor its root goes on PATH or CMAKE_PREFIX_PATH; the MIGraphX provider's CMake finds HIP through HIP_PATH.
rem setup.py names the distribution onnxruntime-migraphx. MIGX_PYTHON runs the build; ORT's build.py installs its
rem requirements (requirements\pybind) into that interpreter with pip.
setlocal
call "%~dp0settings_win.cmd" || exit /b 1
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "HIP_PATH=%MIGX_ROCM%"
set "ROCM_PATH=%HIP_PATH%"
set "CMAKE_PREFIX_PATH="
set "S=%ROOT%\ort-src"
set "PY=%MIGX_PYTHON%"
set "RECIPE=%~dp0.."
set "CMAKE_BIN=%VSROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
set "NINJA_BIN=%VSROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
set "PATH=%CMAKE_BIN%;%NINJA_BIN%;%PATH%"
if not exist "%PY%" echo MIGX_PYTHON %PY% does not exist & exit /b 1
git -C "%S%" rev-parse HEAD | findstr /B f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7 >nul || (
  echo wrong ONNX Runtime commit
  exit /b 1
)
git -C "%S%" checkout HEAD -- onnxruntime/core/providers/migraphx cmake/onnxruntime_providers_migraphx.cmake cmake/onnxruntime_python.cmake || exit /b 1
for %%P in (ort-migraphx-cache-name ort-migraphx-shape-programs ort-migraphx-windows-build) do git -C "%S%" apply "%RECIPE%\%%P.patch" || exit /b 1
git -C "%S%" diff --stat
echo [%DATE% %TIME%] build start
"%PY%" "%S%\tools\ci_build\build.py" ^
  --build_dir "%ROOT%\ort-build" --config Release --update --build ^
  --skip_submodule_sync --skip_tests --parallel %MIGX_JOBS% ^
  --cmake_generator Ninja --cmake_path "%CMAKE_BIN%\cmake.exe" ^
  --enable_pybind --build_wheel --enable_lto --use_binskim_compliant_compile_flags ^
  --use_dml --use_migraphx --migraphx_home "%ROOT%\install" ^
  --compile_no_warning_as_error ^
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF "CMAKE_JOB_POOLS=link_pool=1" "CMAKE_JOB_POOL_LINK=link_pool"
if errorlevel 1 (
  echo [%DATE% %TIME%] build FAILED
  exit /b 1
)
echo [%DATE% %TIME%] ORT DONE
dir /b "%ROOT%\ort-build\Release\dist"
exit /b 0
