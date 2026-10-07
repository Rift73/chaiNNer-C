<#
.SYNOPSIS
Configures and builds chaiNNer-C's native binaries: Ninja (by default the Build Tools' copy) with the
toolchain that toolchain.cmake pins (clang-cl 23.1.2, lld-link, llvm-lib).

.PARAMETER BuildDirectory
The CMake build directory (default native\build).

.PARAMETER Define
CMake cache entries, NAME=VALUE each: CHAINNER_C_PYTHON_ROOT=<runtime>, CHAINNER_C_OUTPUT_DIR=<dir>,
CHAINNER_C_PDB=ON, CHAINNER_C_REGEX_ONLY=ON or, for local builds only, CHAINNER_C_MARCH=<cpu> (for example
icelake-server; never for releases); on a directory's first configure, toolchain.cmake's roots
chainner_c_llvm, chainner_c_vc_tools, chainner_c_sdk and chainner_c_sdk_version (defaults in that file).

.PARAMETER Ninja
ninja.exe (default: the Visual Studio 2022 Build Tools' copy); it must exist.
#>
param(
    [ValidateSet('Debug', 'Release')][string]$Configuration = 'Release',
    [string]$BuildDirectory = (Join-Path $PSScriptRoot 'build'),
    [string[]]$Define = @(),
    [string]$Ninja = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Ninja -PathType Leaf)) { throw "Ninja not found: $Ninja" }
$arguments = @('-S', $PSScriptRoot, '-B', $BuildDirectory, '-G', 'Ninja', "-DCMAKE_MAKE_PROGRAM=$Ninja",
    "-DCMAKE_BUILD_TYPE=$Configuration") + @($Define | ForEach-Object { "-D$_" })
# CMake reads the toolchain at a directory's first configure only (and warns if passed again).
if (-not (Test-Path -LiteralPath (Join-Path $BuildDirectory 'CMakeCache.txt'))) {
    $arguments += "-DCMAKE_TOOLCHAIN_FILE=$(Join-Path $PSScriptRoot 'toolchain.cmake')"
}
& cmake @arguments
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed ($LASTEXITCODE)" }
& cmake --build $BuildDirectory
if ($LASTEXITCODE -ne 0) { throw "Native build failed ($LASTEXITCODE)" }
