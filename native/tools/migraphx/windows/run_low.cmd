@echo off
rem run_low.cmd <script.cmd> <log>: run a build script at BelowNormal priority (children inherit it), output to <log>.
start "" /B /WAIT /BELOWNORMAL cmd /c ""%~1" > "%~2" 2>&1"
exit /b %ERRORLEVEL%
