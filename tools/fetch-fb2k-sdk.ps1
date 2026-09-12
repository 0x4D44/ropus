<#
.SYNOPSIS
  Fetch foobar2000 SDK 2025-03-07 and unpack it into foo_ropus\sdk\.

.DESCRIPTION
  Downloads SDK-2025-03-07.7z from foobar2000.org, verifies its pinned
  Content-Length and SHA-256, extracts with 7-Zip into a temporary staging
  directory, and spot-checks the result before replacing sdk\.

  Idempotent: if sdk\ is already populated, exits 0 with a note. Pass
  -Force to replace it after the new archive has been verified.

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

$SdkUrl        = 'https://www.foobar2000.org/downloads/SDK-2025-03-07.7z'
$ExpectedLen   = 765947  # bytes, from HEAD response 2026-04-19
$ExpectedSha256 = 'ccda3c5840e66e0e28a7e4fe36407c4e78581aa30c40c362a188fcbaae799a3e'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$SdkDir   = Join-Path $RepoRoot 'foo_ropus\sdk'
$InstallId = [Guid]::NewGuid().ToString('N')
$TempFile  = Join-Path $env:TEMP "SDK-2025-03-07-$InstallId.7z"
$StagingDir = Join-Path $env:TEMP "SDK-2025-03-07-$InstallId-staging"

Write-Host "[fetch-fb2k-sdk] workspace  : $RepoRoot"
Write-Host "[fetch-fb2k-sdk] target     : $SdkDir"

# Already populated? Treat as success unless -Force.
if ((Test-Path $SdkDir) -and (Get-ChildItem -Path $SdkDir -ErrorAction SilentlyContinue)) {
    if (-not $Force) {
        Write-Host "[fetch-fb2k-sdk] SDK already unpacked. Re-run with -Force to reinstall."
        exit 0
    }
}

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

New-Item -ItemType Directory -Path $StagingDir -Force | Out-Null

try {
    Write-Host "[fetch-fb2k-sdk] downloading: $SdkUrl"
    & curl.exe --fail --silent --show-error --location --output $TempFile $SdkUrl
    if ($LASTEXITCODE -ne 0) { throw "curl.exe failed (exit $LASTEXITCODE)" }

    $actualLen = (Get-Item $TempFile).Length
    if ($actualLen -ne $ExpectedLen) {
        throw "Size mismatch: expected $ExpectedLen bytes, got $actualLen. The file on foobar2000.org may have been replaced; inspect and, if it's a legitimate new SDK, update the size and SHA-256 pins at the top of this script."
    }

    $sha = (Get-FileHash -LiteralPath $TempFile -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($sha -ne $ExpectedSha256) {
        throw "SHA-256 mismatch: expected $ExpectedSha256, got $sha. Aborting without touching $SdkDir."
    }
    Write-Host "[fetch-fb2k-sdk] size       : $actualLen bytes"
    Write-Host "[fetch-fb2k-sdk] sha256     : $sha"

    Write-Host "[fetch-fb2k-sdk] extracting to staging directory"
    & $SevenZ x -bso0 -bsp0 "-o$StagingDir" $TempFile
    if ($LASTEXITCODE -ne 0) { throw "7-Zip extraction failed (exit $LASTEXITCODE)" }

    # Verify the extracted layout before touching an existing SDK.
    $wantDirs = @('foobar2000', 'pfc')
    $missing  = @($wantDirs | Where-Object { -not (Test-Path (Join-Path $StagingDir $_)) })
    if ($missing.Count -gt 0) {
        Write-Warning "Post-extract sanity check: missing expected subdir(s): $($missing -join ', ')"
        Write-Warning "Inspect the staged SDK manually — the SDK layout may have changed since 2025-03-07."
        exit 4
    }

    if (Test-Path $SdkDir) {
        Write-Host "[fetch-fb2k-sdk] -Force: replacing existing sdk\ contents"
        Remove-Item -LiteralPath $SdkDir -Recurse -Force
    }
    Move-Item -LiteralPath $StagingDir -Destination $SdkDir
}
finally {
    if (Test-Path -LiteralPath $TempFile) { Remove-Item -LiteralPath $TempFile -Force -ErrorAction SilentlyContinue }
    if (Test-Path -LiteralPath $StagingDir) { Remove-Item -LiteralPath $StagingDir -Recurse -Force -ErrorAction SilentlyContinue }
}

Write-Host "[fetch-fb2k-sdk] OK. Ready for M4 (C++ component shell)."
