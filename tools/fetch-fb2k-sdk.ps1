<#
.SYNOPSIS
  Fetch foobar2000 SDK 2025-03-07 and unpack it into fb2k-ropus\sdk\.

.DESCRIPTION
  Downloads SDK-2025-03-07.7z from foobar2000.org, verifies Content-Length
  matches what was observed at pin time (2026-04-19), extracts with 7-Zip,
  and spot-checks the result so the caller doesn't silently move on with
  an empty sdk\ directory.

  Idempotent: if sdk\ is already populated, exits 0 with a note. Pass
  -Force to wipe and reinstall. The script prints the SHA256 of every
  successful download so a future run can pin it if desired.

  foobar2000 SDK is not redistributable, which is why we fetch at
  provisioning time rather than vendor into git. The HLD that drives this
  is wrk_docs\2026.04.18 - HLD - foobar2000 opus decoder component.md
  (§5.1, §8 "SDK version").

.EXAMPLE
  pwsh -File tools\fetch-fb2k-sdk.ps1
  pwsh -File tools\fetch-fb2k-sdk.ps1 -Force
#>

[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$SdkUrl      = 'https://www.foobar2000.org/downloads/SDK-2025-03-07.7z'
$ExpectedLen = 765947  # bytes, from HEAD response 2026-04-19

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$SdkDir   = Join-Path $RepoRoot 'fb2k-ropus\sdk'
$TempFile = Join-Path $env:TEMP "SDK-2025-03-07-$([Guid]::NewGuid().ToString('N')).7z"

Write-Host "[fetch-fb2k-sdk] workspace  : $RepoRoot"
Write-Host "[fetch-fb2k-sdk] target     : $SdkDir"

# Already populated? Treat as success unless -Force.
if ((Test-Path $SdkDir) -and (Get-ChildItem -Path $SdkDir -ErrorAction SilentlyContinue)) {
    if (-not $Force) {
        Write-Host "[fetch-fb2k-sdk] SDK already unpacked. Re-run with -Force to reinstall."
        exit 0
    }
    Write-Host "[fetch-fb2k-sdk] -Force: wiping existing sdk\ contents"
    Remove-Item -Path (Join-Path $SdkDir '*') -Recurse -Force
}
New-Item -ItemType Directory -Path $SdkDir -Force | Out-Null

# Locate 7-Zip. Default install dirs first, PATH second.
$SevenZ = $null
foreach ($p in @('C:\Program Files\7-Zip\7z.exe', 'C:\Program Files (x86)\7-Zip\7z.exe')) {
    if (Test-Path $p) { $SevenZ = $p; break }
}
if (-not $SevenZ) {
    $cmd = Get-Command 7z.exe -ErrorAction SilentlyContinue
    if ($cmd) { $SevenZ = $cmd.Source }
}
if (-not $SevenZ) {
    Write-Error "7-Zip not found. Install it (e.g. 'winget install 7zip.7zip') and retry."
    exit 3
}
Write-Host "[fetch-fb2k-sdk] 7-Zip      : $SevenZ"

try {
    Write-Host "[fetch-fb2k-sdk] downloading: $SdkUrl"
    & curl.exe --fail --silent --show-error --location --output $TempFile $SdkUrl
    if ($LASTEXITCODE -ne 0) { throw "curl.exe failed (exit $LASTEXITCODE)" }

    $actualLen = (Get-Item $TempFile).Length
    if ($actualLen -ne $ExpectedLen) {
        throw "Size mismatch: expected $ExpectedLen bytes, got $actualLen. The file on foobar2000.org may have been replaced; inspect and, if it's a legitimate new SDK, update the ExpectedLen constant at the top of this script."
    }

    $sha = (Get-FileHash $TempFile -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Host "[fetch-fb2k-sdk] size       : $actualLen bytes"
    Write-Host "[fetch-fb2k-sdk] sha256     : $sha"

    Write-Host "[fetch-fb2k-sdk] extracting to $SdkDir"
    & $SevenZ x -bso0 -bsp0 "-o$SdkDir" $TempFile
    if ($LASTEXITCODE -ne 0) { throw "7-Zip extraction failed (exit $LASTEXITCODE)" }
}
finally {
    if (Test-Path $TempFile) { Remove-Item $TempFile -Force -ErrorAction SilentlyContinue }
}

# Spot-check: confirm the archive produced the top-level dirs we'll link
# against in fb2k-ropus\CMakeLists.txt. If these are missing, the SDK
# layout has drifted and the M4 build will fail in a harder-to-diagnose
# spot.
$wantDirs = @('foobar2000', 'pfc')
$missing  = @($wantDirs | Where-Object { -not (Test-Path (Join-Path $SdkDir $_)) })
if ($missing.Count -gt 0) {
    Write-Warning "Post-extract sanity check: missing expected subdir(s): $($missing -join ', ')"
    Write-Warning "Inspect $SdkDir manually — the SDK layout may have changed since 2025-03-07."
    exit 4
}

Write-Host "[fetch-fb2k-sdk] OK. Ready for M4 (C++ component shell)."
