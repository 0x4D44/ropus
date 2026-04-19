<#
.SYNOPSIS
  Build foo_ropus.dll (Release) and package it as foo_ropus.fb2k-component.

.DESCRIPTION
  Runs CMake Release configure + build (which cascades into `cargo build
  -p ropus-fb2k` for the Rust staticlib, then links everything into
  foo_ropus.dll), stages the DLL under build\stage\x64\, and zips it to
  a `.fb2k-component` archive at build\foo_ropus.fb2k-component.

  Layout per HLD §6.3:
      foo_ropus.fb2k-component  (zip archive)
      └── x64/
          └── foo_ropus.dll

  fb2k 2.0+ expects 64-bit components under the x64/ folder; 32-bit bits
  would sit at the zip root but we don't ship those.

  Idempotent: re-run safely. Pass -Clean to wipe build\ first.

.EXAMPLE
  pwsh -File fb2k-ropus\build.ps1
  pwsh -File fb2k-ropus\build.ps1 -Clean
#>

[CmdletBinding()]
param(
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'

$ScriptDir = $PSScriptRoot
Push-Location $ScriptDir

try {
    # Locate CMake. Prefer VS-bundled (matches the SDK's .vcxproj toolchain);
    # fall back to PATH.
    $CMake = $null
    foreach ($p in @(
        'C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe',
        'C:\Program Files\Microsoft Visual Studio\17\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe',
        'C:\Program Files\Microsoft Visual Studio\17\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe',
        'C:\Program Files\Microsoft Visual Studio\17\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
    )) {
        if (Test-Path $p) { $CMake = $p; break }
    }
    if (-not $CMake) {
        $cmd = Get-Command cmake.exe -ErrorAction SilentlyContinue
        if ($cmd) { $CMake = $cmd.Source }
    }
    if (-not $CMake) {
        Write-Error 'cmake.exe not found. Install Visual Studio with the C++ workload, or put CMake on PATH.'
        exit 1
    }
    Write-Host "[build] cmake    : $CMake"

    # Detect VS generator. Prefer 2026 (newest first), fall back to 2022.
    $HelpText = & $CMake --help
    $Generator = $null
    foreach ($g in @('Visual Studio 18 2026', 'Visual Studio 17 2022')) {
        if ($HelpText -match [regex]::Escape($g)) { $Generator = $g; break }
    }
    if (-not $Generator) {
        Write-Error "Neither 'Visual Studio 18 2026' nor 'Visual Studio 17 2022' generator is available via $CMake. Install VS 2022 or 2026 with the C++ workload."
        exit 1
    }
    Write-Host "[build] generator: $Generator"

    $BuildDir = Join-Path $ScriptDir 'build'
    if ($Clean -and (Test-Path $BuildDir)) {
        Write-Host "[build] -Clean   : removing $BuildDir"
        Remove-Item $BuildDir -Recurse -Force
    }
    if (-not (Test-Path $BuildDir)) {
        New-Item -ItemType Directory -Path $BuildDir | Out-Null
    }

    # Configure + build. The CMake ALL-target drives cargo for the Rust
    # staticlib as a dependency of the C++ link step (see CMakeLists.txt).
    Push-Location $BuildDir
    try {
        & $CMake -G $Generator -A x64 ..
        if ($LASTEXITCODE -ne 0) { throw "cmake configure failed ($LASTEXITCODE)" }
        & $CMake --build . --config Release
        if ($LASTEXITCODE -ne 0) { throw "cmake --build failed ($LASTEXITCODE)" }
    } finally {
        Pop-Location
    }

    $Dll = Join-Path $BuildDir 'Release\foo_ropus.dll'
    if (-not (Test-Path $Dll)) {
        throw "build finished but $Dll was not produced. Inspect build\ for diagnostics."
    }
    $dllBytes = (Get-Item $Dll).Length
    Write-Host "[build] built    : $Dll ($dllBytes bytes)"

    # Stage for zip: only x64\foo_ropus.dll ships.
    $Stage = Join-Path $BuildDir 'stage'
    if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
    $StageX64 = Join-Path $Stage 'x64'
    New-Item -ItemType Directory -Path $StageX64 | Out-Null
    Copy-Item -Path $Dll -Destination (Join-Path $StageX64 'foo_ropus.dll')

    # Pack as .fb2k-component. Compress-Archive writes .zip; fb2k accepts
    # any extension for install, but .fb2k-component is the user-facing
    # convention. We zip the x64\ dir so it appears at the archive root.
    $Component = Join-Path $BuildDir 'foo_ropus.fb2k-component'
    if (Test-Path $Component) { Remove-Item $Component -Force }
    $TempZip = "$Component.zip"
    if (Test-Path $TempZip) { Remove-Item $TempZip -Force }
    Compress-Archive -Path $StageX64 -DestinationPath $TempZip -Force
    Rename-Item -Path $TempZip -NewName (Split-Path $Component -Leaf)

    $zipBytes = (Get-Item $Component).Length
    Write-Host ''
    Write-Host "[build] packaged : $Component ($zipBytes bytes)"
    Write-Host '[build] install  : double-click the .fb2k-component, OR drag it into'
    Write-Host "[build]            foobar2000's Preferences -> Components page."
} finally {
    Pop-Location
}
