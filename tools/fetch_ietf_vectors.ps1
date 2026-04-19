# Fetch the IETF RFC 6716 / RFC 8251 Opus test vectors.
#
# Downloads the canonical archive from opus-codec.org, verifies its
# SHA-256, and extracts it to `tests/vectors/ietf/` at the repo root.
# Safe to re-run: if the target directory already contains the expected
# vectors, we skip the download. If SHA-256 verification fails we abort
# without touching the destination.
#
# Used by the `conformance` crate's `ietf_vectors` test. That test panics
# with a helpful message if this script hasn't been run.

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$Url = 'https://opus-codec.org/static/testvectors/opus_testvectors-rfc8251.tar.gz'
$ExpectedSize = 74624664L
$ExpectedSha256 = '6b26a22f9ba87b2b836906a9bb7afec5f8e54d49553b1200382520ee6fedfa55'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir '..')).Path

if (-not (Test-Path (Join-Path $repoRoot 'Cargo.toml'))) {
    Write-Error "Could not locate repo root (no Cargo.toml at $repoRoot)"
    exit 1
}

$destDir = Join-Path $repoRoot 'tests/vectors/ietf'
$tmpDir = Join-Path ([System.IO.Path]::GetTempPath()) ("ropus_ietf_" + [System.IO.Path]::GetRandomFileName())
$archive = Join-Path $tmpDir 'opus_testvectors-rfc8251.tar.gz'

# If already extracted, skip.
if (Test-Path (Join-Path $destDir 'testvector12m.dec')) {
    Write-Host "IETF vectors already present at $destDir; nothing to do."
    exit 0
}

New-Item -ItemType Directory -Force -Path $destDir | Out-Null
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

try {
    Write-Host "Downloading IETF vectors from $Url..."
    # Force TLS 1.2 on older PowerShells; on pwsh 7+ this is a no-op.
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $Url -OutFile $archive -UseBasicParsing

    $actualSize = (Get-Item $archive).Length
    if ($actualSize -ne $ExpectedSize) {
        Write-Error "Downloaded size $actualSize != expected $ExpectedSize (archive: $archive)"
        exit 2
    }

    Write-Host "Verifying SHA-256..."
    $actualSha = (Get-FileHash -Path $archive -Algorithm SHA256).Hash.ToLower()
    if ($actualSha -ne $ExpectedSha256) {
        Write-Error "SHA-256 mismatch`n  expected: $ExpectedSha256`n  actual:   $actualSha`n  archive:  $archive`nAborting without touching $destDir."
        exit 3
    }

    Write-Host "Extracting to $destDir..."
    # Use tar which ships with Windows 10+ (bsdtar). Equivalent to the .sh
    # script's `tar -xzf`.
    tar -xzf $archive -C $tmpDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "tar extraction failed with exit code $LASTEXITCODE"
        exit 4
    }

    # Locate the extracted directory (upstream uses `opus_newvectors/`).
    $extractedDir = $null
    foreach ($candidate in @('opus_newvectors', 'opus_testvectors')) {
        $path = Join-Path $tmpDir $candidate
        if (Test-Path $path) {
            $extractedDir = $path
            break
        }
    }
    if (-not $extractedDir) {
        Write-Error 'Tarball layout unexpected: no opus_newvectors/ or opus_testvectors/ subdir'
        exit 4
    }

    # Move files up so they live at ietf/testvectorNN.{bit,dec}.
    Move-Item -Path (Join-Path $extractedDir '*') -Destination $destDir -Force

    # Sanity-check.
    $missing = 0
    foreach ($n in '01','02','03','04','05','06','07','08','09','10','11','12') {
        foreach ($suffix in 'bit','dec') {
            $f = Join-Path $destDir "testvector$n.$suffix"
            if (-not (Test-Path $f)) {
                Write-Warning "Missing $f"
                $missing++
            }
        }
    }
    if ($missing -gt 0) {
        Write-Error "$missing expected vector files missing after extraction"
        exit 5
    }

    Write-Host "IETF vectors installed in $destDir."
}
finally {
    # Always clean up the tmp dir on both success and failure.
    if (Test-Path $tmpDir) {
        Remove-Item -Recurse -Force $tmpDir
    }
}
