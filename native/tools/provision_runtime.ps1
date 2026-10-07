<#
Provisions the Python runtime chaiNNer-C builds, tests and packages with: python-build-standalone CPython into
native\runtime\cpython-<version> (git-ignored), then the stack the tracked lock native\python-stack.lock.txt records.
By default the lock is the input. Its header names the interpreter, the build tag, the archive's SHA-256 and pip's
version; its pins are installed as they stand, without dependency resolution, each from one source (torch and
torchvision from the PyTorch CUDA index, chainner-pip from the backend's bundled wheel, the rest from PyPI). The run
fails unless pip check is clean, pip's version is the header's and pip freeze equals the pins line for line. The lock
is not written.
-Relock is the deliberate upgrade step, followed by the full acceptance: the archive checked against the release's
SHA256SUMS, the backend's dependencies at their latest versions (the 2026-10-06 Python-stack upgrade, Fable's ruling),
then the freeze and a new lock (the freeze under a header with the interpreter, the archive's SHA-256 and pip's version).
-BuildOnly provisions what native\Build.ps1 needs and nothing else: the lock's CPython, downloaded and checked as
above, with only the lock's numpy and pybind11 pins (no torch, no pip check or freeze comparison). The full run on the
same runtime later installs the rest.
Usage: powershell -NoProfile -File native\tools\provision_runtime.ps1 [-Relock | -BuildOnly] [-Version 3.14.8] [-Tag 20261003]
-Version and -Tag default to the lock's; without -Relock any other value is refused. -Relock needs no readable lock
header (it writes a new one), only -Version and -Tag when the lock cannot supply them.
The archive's SHA-256 is recorded beside the runtime it was extracted into (cpython-<version>.archive-sha256), and an
existing runtime extracted from another archive, or with no record, is refused: remove it to extract this one.
#>
param([string]$Version, [string]$Tag, [switch]$Relock, [switch]$BuildOnly)

$ErrorActionPreference = 'Stop'
if ($Relock -and $BuildOnly) { throw '-BuildOnly provisions from the lock; it cannot be combined with -Relock' }
# Isolated from user state: no user site-packages, no pip user configuration and no PIP_* variables, so
# every requirement is installed into (and frozen from) the runtime itself.
$env:PYTHONNOUSERSITE = '1'
$Pip = @('-s', '-m', 'pip', '--isolated')
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Runtime = Join-Path $Root "native\runtime"
$LockPath = Join-Path $Root 'native\python-stack.lock.txt'
$TorchIndex = 'https://download.pytorch.org/whl/cu132'
# UTF-8 without a BOM (Set-Content -Encoding utf8 writes one in Windows PowerShell 5.1).
$Utf8 = New-Object Text.UTF8Encoding $false

# The lock's header fields, matched as package_manifest.LOCK_FIELDS matches them, and its pins without their notes.
$LockFields = [ordered]@{
    interpreter = '^# Interpreter: CPython (\d+\.\d+\.\d+) .*$'
    build_tag = '^# Build tag: python-build-standalone (\d+)$'
    archive_sha256 = '^# Archive SHA-256: ([0-9a-f]{64})$'
    pip = '^# pip: (\S+) .*$'
}
# Without -Relock the lock is the input and must be whole; -Relock writes a new one, so a missing lock or header field
# fails it only when -Version or -Tag is absent too.
$LockLines = @(if (Test-Path -LiteralPath $LockPath) { [IO.File]::ReadAllLines($LockPath, $Utf8) })
$Header = @{}
foreach ($LockLine in $LockLines) {
    foreach ($Field in $LockFields.Keys) {
        if ($LockLine -cmatch $LockFields[$Field]) { $Header[$Field] = $Matches[1] }
    }
}
$Pins = @($LockLines | Where-Object { $_ -and -not $_.StartsWith('#') } | ForEach-Object { ($_ -split '  # ')[0] })
if (-not $Relock) {
    $Missing = @($LockFields.Keys | Where-Object { -not $Header.ContainsKey($_) })
    if ($Missing.Count) { throw "the lock $LockPath lacks the header fields $($Missing -join ', ')" }
    if ($Version -and $Version -cne $Header.interpreter) {
        throw "-Version $Version is not the lock's CPython $($Header.interpreter); another interpreter is an upgrade (-Relock)"
    }
    if ($Tag -and $Tag -cne $Header.build_tag) {
        throw "-Tag $Tag is not the lock's build tag $($Header.build_tag); another build is an upgrade (-Relock)"
    }
}
if (-not $Version) { $Version = $Header.interpreter }
if (-not $Tag) { $Tag = $Header.build_tag }
if (-not $Version -or -not $Tag) {
    throw "the lock $LockPath names no interpreter or build tag; -Relock needs -Version and -Tag"
}

$Target = Join-Path $Runtime "cpython-$Version"
$Download = Join-Path $Runtime 'download'
$Name = "cpython-$Version+$Tag-x86_64-pc-windows-msvc-install_only.tar.gz"
$Base = "https://github.com/astral-sh/python-build-standalone/releases/download/$Tag"

New-Item -ItemType Directory -Force -Path $Download | Out-Null
$Python = Join-Path $Target 'python.exe'
# The archive is kept and checked on every run: against the lock's SHA-256, or with -Relock against the release's
# SHA256SUMS (the new lock then records it).
$Archive = Join-Path $Download $Name
if (-not (Test-Path -LiteralPath $Archive)) {
    Invoke-WebRequest -Uri "$Base/$([uri]::EscapeDataString($Name))" -OutFile $Archive -UseBasicParsing
}
if ($Relock) {
    $SumsFile = Join-Path $Download "SHA256SUMS-$Tag"
    if (-not (Test-Path -LiteralPath $SumsFile)) {
        Invoke-WebRequest -Uri "$Base/SHA256SUMS" -OutFile $SumsFile -UseBasicParsing
    }
    $Line = Select-String -LiteralPath $SumsFile -Pattern ("  " + [regex]::Escape($Name) + '$') | Select-Object -First 1
    $Expected = if ($Line) { ($Line.Line -split '\s+')[0] } else { '' }
    if (-not $Expected) { throw "no SHA-256 for $Name in the release's SHA256SUMS" }
} else {
    $Expected = $Header.archive_sha256
}
$ArchiveSha = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLower()
if ($ArchiveSha -ne $Expected.ToLower()) {
    # A partial or corrupt download would fail every later run the same way.
    Remove-Item -LiteralPath $Archive
    throw "SHA-256 mismatch for ${Name}: expected '$Expected', got '$ArchiveSha'; the download was removed"
}
"verified $Name $ArchiveSha ($((Get-Item -LiteralPath $Archive).Length) bytes)"

# The runtime's directory names its version, not its build: the archive it was extracted from is recorded beside it
# (outside it, so the packaged runtime is unchanged), and an existing runtime is used only if that is this archive.
$ExtractedFrom = Join-Path $Runtime "cpython-$Version.archive-sha256"
if (Test-Path -LiteralPath $Python) {
    $Recorded = if (Test-Path -LiteralPath $ExtractedFrom) { [IO.File]::ReadAllText($ExtractedFrom, $Utf8).Trim() } else { '' }
    if (-not $Recorded) {
        throw "the runtime $Target has no record of its archive ($ExtractedFrom); remove it to extract $Name"
    }
    if ($Recorded -ne $Expected) {
        throw "the runtime $Target was extracted from the archive $Recorded, not $Name ($Expected); remove it to extract this one"
    }
    "runtime $Target exists, extracted from $Name; skipping its extraction (pip still installs into it)"
} else {
$Stage = Join-Path $Runtime "stage-$Version"
# A failed run's stage is never extracted over.
if (Test-Path -LiteralPath $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
# Windows' own tar: a GNU tar earlier on PATH (Git, MSYS2) reads C:\... as a remote host.
& (Join-Path $env:SystemRoot 'System32\tar.exe') -xzf $Archive -C $Stage
if ($LASTEXITCODE -ne 0) { throw 'tar failed' }
# Recorded before the move: a move that fails leaves no runtime, and the next run extracts again.
[IO.File]::WriteAllText($ExtractedFrom, "$ArchiveSha`n", $Utf8)
Move-Item -LiteralPath (Join-Path $Stage 'python') -Destination $Target
Remove-Item -LiteralPath $Stage -Recurse
}
if (-not (Test-Path -LiteralPath (Join-Path $Target "libs\python$($Version.Split('.')[0])$($Version.Split('.')[1]).lib"))) { throw 'the runtime has no import library (libs\python3XX.lib)' }
& $Python -c "import sys; print('runtime', sys.version)"
if ($BuildOnly) {
    # The build's inputs only: NumPy's headers and pybind11, at the lock's pins, from PyPI.
    $BuildPins = @($Pins -cmatch '^(numpy|pybind11)==')
    if ($BuildPins.Count -ne 2) { throw "the lock $LockPath does not pin numpy and pybind11 once each" }
    & $Python @Pip install --no-deps @BuildPins
    if ($LASTEXITCODE -ne 0) { throw 'numpy and pybind11 install failed' }
    "provisioned $Target for the build only ($($BuildPins -join ', '))"
    return
}

# chainner-pip from the backend's bundled wheel, which the host would otherwise install into the package at first
# launch (Consult 6 P2). Resolved by name from that directory only, so the freeze pins its version, not a file path.
$Wheels = Join-Path $Root 'backend\src\dependencies\whls\chainner-pip'
if ($Relock) {
    & $Python @Pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
    # The backend's dependencies at their latest versions, CUDA torch from the PyTorch index.
    & $Python @Pip install torch==2.14.1 torchvision==0.29.1 --index-url $TorchIndex
    if ($LASTEXITCODE -ne 0) { throw 'torch install failed' }
    $Packages = @(
        'numpy==2.5.3', 'opencv-python==5.0.0.93', 'Pillow==12.3.0', 'scipy==1.18.1', 'numba==0.68.0', 'llvmlite==0.50.0',
        'PyMatting==1.1.16', 'pillow-avif-plugin', 'ffmpeg-python', 'requests', 'wcmatch',
        'google-re2',  # oracle/reference only (upstream's onnx/load.py imports re2)
        'Sanic-Cors==2.2.0',  # oracle/reference only (upstream's server imports sanic_cors)
        'spandrel==0.4.2', 'spandrel_extra_arches', 'facexlib', 'einops', 'safetensors',
        'onnxruntime-gpu==1.30.0', 'onnx', 'onnxoptimizer', 'protobuf', 'ncnn==1.0.20260526',
        'sanic==25.12.1', 'aiofiles', 'html5tagger', 'sanic-routing', 'tracerite', 'websockets', 'typing_extensions',
        'nvidia-ml-py', 'psutil', 'aiohttp',
        'pynvml',  # oracle/reference only (upstream's server declares it; nvidia-ml-py provides the module)
        'pybind11==3.1.0', 'pytest', 'pytest-asyncio', 'pytest-cov'
    )
    & $Python @Pip install @Packages
    if ($LASTEXITCODE -ne 0) { throw 'package install failed' }
    & $Python @Pip install --no-index --find-links $Wheels chainner-pip
    if ($LASTEXITCODE -ne 0) { throw 'chainner-pip install failed' }
} else {
    # The lock is a whole freeze, so every pin is installed as it stands (--no-deps), each from its one source.
    & $Python @Pip install "pip==$($Header.pip)"
    if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }
    $FromTorchIndex = '^(torch|torchvision)=='
    $FromWheel = '^chainner-pip=='
    & $Python @Pip install --no-deps --index-url $TorchIndex @($Pins -cmatch $FromTorchIndex)
    if ($LASTEXITCODE -ne 0) { throw 'torch install failed' }
    # The rest from PyPI: the lock less those lines (pip reads the notes after a pin as comments).
    $PypiPins = Join-Path $Runtime "pypi-$Version.txt"
    [IO.File]::WriteAllLines($PypiPins, [string[]]@($LockLines | Where-Object { $_ -cnotmatch $FromTorchIndex -and $_ -cnotmatch $FromWheel }), $Utf8)
    & $Python @Pip install --no-deps -r $PypiPins
    if ($LASTEXITCODE -ne 0) { throw 'package install failed' }
    & $Python @Pip install --no-deps --no-index --find-links $Wheels @($Pins -cmatch $FromWheel)
    if ($LASTEXITCODE -ne 0) { throw 'chainner-pip install failed' }
    & $Python @Pip check
    if ($LASTEXITCODE -ne 0) { throw 'pip check failed' }
}
& $Python -s -m chainner_pip --version
if ($LASTEXITCODE -ne 0) { throw 'chainner-pip does not run' }
$PipVersion = ((& $Python @Pip --version) -split ' ')[1]
if ($LASTEXITCODE -ne 0) { throw 'pip --version failed' }
$Freeze = @(& $Python @Pip freeze)
if ($LASTEXITCODE -ne 0) { throw 'pip freeze failed' }
$FreezePath = Join-Path $Runtime "freeze-$Version.txt"
[IO.File]::WriteAllLines($FreezePath, [string[]]$Freeze, $Utf8)
if ($Relock) {
    # The lock: the freeze under a header package_port.py reads (python_stack); a pin's note follows it on its line.
    $Notes = @{
        'google-re2' = "oracle/reference only (upstream's onnx/load.py imports re2)"
        'Sanic-Cors' = "oracle/reference only (upstream's server imports sanic_cors)"
        'pynvml' = "oracle/reference only (upstream's server declares it; nvidia-ml-py provides the module)"
    }
    $Lock = @(
        "# The tested set, not a ceiling: the stack chaiNNer-C is tested on. The backend's Dependency",
        '# pins are set at these versions and act as floors; newer versions are not excluded.',
        "# Interpreter: CPython $Version (standard build, not free-threaded).",
        "# Build tag: python-build-standalone $Tag",
        "# Archive: $Name",
        "# Archive SHA-256: $ArchiveSha",
        "# pip: $PipVersion (installed; pip freeze does not list it)",
        '# Packages: pip freeze of that runtime, provisioned by native/tools/provision_runtime.ps1 (torch and',
        '# torchvision from https://download.pytorch.org/whl/cu132, chainner-pip from the bundled wheel in',
        '# backend/src/dependencies/whls/chainner-pip/, the rest from PyPI).'
    )
    foreach ($Pin in $Freeze) {
        $Note = $Notes[($Pin -split '==')[0]]
        $Lock += if ($Note) { "$Pin  # $Note" } else { $Pin }
    }
    [IO.File]::WriteAllLines($LockPath, [string[]]$Lock, $Utf8)
} elseif ($PipVersion -cne $Header.pip -or ($Freeze -join "`n") -cne ($Pins -join "`n")) {
    # The runtime is the lock only if pip's version and the freeze, line for line, are the lock's.
    $Differences = @()
    if ($PipVersion -cne $Header.pip) { $Differences += "pip $PipVersion, the lock's $($Header.pip)" }
    $Differences += @($Pins | Where-Object { $Freeze -cnotcontains $_ } | ForEach-Object { "lock only: $_" })
    $Differences += @($Freeze | Where-Object { $Pins -cnotcontains $_ } | ForEach-Object { "freeze only: $_" })
    if ($Differences.Count -eq 0) { $Differences += 'the freeze lists the same pins in another order' }
    throw "the runtime is not the lock ($FreezePath):`n$($Differences -join "`n")"
}
"provisioned $Target"
