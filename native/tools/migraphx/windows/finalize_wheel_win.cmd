@echo off
rem The wheel build_ort_win.cmd built, made final by finalize_wheel_win.py: the +rocm7.14.dml version, MIGraphX's DLLs
rem and the Visual C++ runtime in onnxruntime\capi, the third-party notices, and a check of every binary's imports
rem against the wheel, Windows and the ROCm pip packages. Output: ROOT\dist (ROOT is the work directory,
rem settings_win.cmd), then its SHA-256. MIGX_PYTHON needs the wheel package.
setlocal
call "%~dp0settings_win.cmd" || exit /b 1
for %%I in ("%MIGX_ROCM%\..") do set "SITE=%%~fI"
"%MIGX_PYTHON%" -I "%~dp0finalize_wheel_win.py" --vs-root "%VSROOT%" "%ROOT%\ort-build\Release\dist" "%ROOT%\dist" "%ROOT%\install" "%ROOT%" "%SITE%" || exit /b 1
powershell -NoProfile -Command "Get-FileHash -Algorithm SHA256 -Path '%ROOT%\dist\*.whl' | ForEach-Object { $_.Hash.ToLower() + '  ' + $_.Path }"
exit /b 0
