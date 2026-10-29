<#
.SYNOPSIS
Configures and builds chaiNNer-C's native binaries: Ninja (by default the Build Tools' copy) with the
toolchain that toolchain.cmake pins (clang-cl 23.1.2, lld-link, llvm-lib).

.PARAMETER BuildDirectory
The CMake build directory (default native\build).

.PARAMETER Define
CMake cache entries, NAME=VALUE each: CHAINNER_C_PYTHON_ROOT=<runtime>, CHAINNER_C_OUTPUT_DIR=<dir>,
CHAINNER_C_PDB=ON, CHAINNER_C_FUZZ=ON (tests\fuzz\README.md), CHAINNER_C_REGEX_ONLY=ON or, for local builds
only, CHAINNER_C_MARCH=<cpu> (for example icelake-server; never for releases); on a directory's first configure,
toolchain.cmake's roots
chainner_c_llvm, chainner_c_vc_tools, chainner_c_sdk and chainner_c_sdk_version (defaults in that file);
CHAINNER_C_TENSORRT_ROOT=<TensorRT 11 SDK> with CHAINNER_C_CUDA_ROOT=<CUDA 13 toolkit> also builds VOSR's
TensorRT plugin vosr2_qkv.dll and the TensorRT runner _chainner_trt.pyd (on any configure; without them both are
skipped), and CHAINNER_C_VAPOURSYNTH_INCLUDE=<directory of VapourSynth4.h> the runner's VapourSynth plugin
chainner_trt_vs.dll.

.PARAMETER Ninja
ninja.exe (default: the Visual Studio 2022 Build Tools' copy); it must exist.

.PARAMETER RocmRoot
AMD GPUs (optional): the ROCm 7.14 SDK root for Windows (the rocm-sdk-devel wheel's expanded _rocm_sdk_devel, with
include\hip, lib\amdhip64.lib and lib\llvm\bin\clang++.exe). It builds the HIP custom-op libraries beside
nodes\impl\migraphx (drft_migx_ops.dll; cmake\rocm.cmake). Without it and -MigraphxRoot the configuration has no AMD
parts at all; a build directory keeps the roots of its earlier configures.

.PARAMETER MigraphxRoot
AMD GPUs (optional, with -RocmRoot): the MIGraphX 2.16 install for Windows that chaiNNer-C's ONNX Runtime MIGraphX
wheel bundles. It also builds the MIGraphX runner nodes\impl\migraphx\_chainner_migraphx.pyd.
#>
param(
    [ValidateSet('Debug', 'Release')][string]$Configuration = 'Release',
    [string]$BuildDirectory = (Join-Path $PSScriptRoot 'build'),
    [string[]]$Define = @(),
    [string]$Ninja = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe',
    [string]$RocmRoot = '',
    [string]$MigraphxRoot = ''
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Ninja -PathType Leaf)) { throw "Ninja not found: $Ninja" }
# The AMD roots as CMake paths, each passed only when given.
$amd = @()
if ($RocmRoot) { $amd += "CHAINNER_C_ROCM_ROOT=$([IO.Path]::GetFullPath($RocmRoot).Replace('\', '/'))" }
if ($MigraphxRoot) { $amd += "CHAINNER_C_MIGRAPHX_ROOT=$([IO.Path]::GetFullPath($MigraphxRoot).Replace('\', '/'))" }
$arguments = @('-S', $PSScriptRoot, '-B', $BuildDirectory, '-G', 'Ninja', "-DCMAKE_MAKE_PROGRAM=$Ninja",
    "-DCMAKE_BUILD_TYPE=$Configuration") + @(($Define + $amd) | ForEach-Object { "-D$_" })
# CMake reads the toolchain at a directory's first configure only (and warns if passed again).
if (-not (Test-Path -LiteralPath (Join-Path $BuildDirectory 'CMakeCache.txt'))) {
    $arguments += "-DCMAKE_TOOLCHAIN_FILE=$(Join-Path $PSScriptRoot 'toolchain.cmake')"
}
& cmake @arguments
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed ($LASTEXITCODE)" }
& cmake --build $BuildDirectory
if ($LASTEXITCODE -ne 0) { throw "Native build failed ($LASTEXITCODE)" }
