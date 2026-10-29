@rem Settings of the Windows MIGraphX 2.16.1 and ONNX Runtime 1.30.0 build, called by the other scripts of this folder.
@rem Each is an environment variable; set it before running a script to replace its default.
@rem   MIGX_WORK    work directory, no parentheses in the path (default USERPROFILE\chainner-migraphx): src\AMDMIGraphX,
@rem                ort-src, deps-src\NAME\, the build trees and logs, deps (the dependencies' install prefix), install
@rem                (MIGraphX's install prefix, the -MigraphxRoot of Build.ps1) and dist (the wheel)
@rem   MIGX_JOBS    parallel build jobs (default 12)
@rem   MIGX_VS      Visual Studio 2022 Build Tools (default C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools)
@rem   MIGX_ROCM    ROCm 7.14.1 SDK root, the expanded _rocm_sdk_devel of the rocm-sdk-devel wheel (required); its
@rem                parent holds the _rocm_sdk_core and _rocm_sdk_libraries folders
@rem   MIGX_PYTHON  CPython 3.14 with pip and wheel, for rocMLIR's configuration and the ONNX Runtime build (default
@rem                MIGX_WORK\venv-build\Scripts\python.exe)
if not defined MIGX_WORK set "MIGX_WORK=%USERPROFILE%\chainner-migraphx"
if not defined MIGX_JOBS set "MIGX_JOBS=12"
if not defined MIGX_VS set "MIGX_VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
if not defined MIGX_ROCM (
  echo MIGX_ROCM must name the ROCm 7.14.1 SDK root ^(the rocm-sdk-devel wheel's _rocm_sdk_devel^)
  exit /b 1
)
if not defined MIGX_PYTHON set "MIGX_PYTHON=%MIGX_WORK%\venv-build\Scripts\python.exe"
echo "%MIGX_WORK%" | findstr /R /C:"[()]" >nul
if not errorlevel 1 (
  echo MIGX_WORK must not contain parentheses
  exit /b 1
)
set "ROOT=%MIGX_WORK%"
set "VSROOT=%MIGX_VS%"
exit /b 0
