# Windows-native cargo-fuzz runner.
#
# This is the PowerShell counterpart to fuzz_run.sh. The full-test Rust
# runner supervises this script, so it does not depend on GNU timeout, env,
# bash, or Git Bash being installed on the machine.

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Arguments = @($args)

function Show-Usage {
    @'
Usage:
  tools/fuzz_run.ps1 --sanity
  tools/fuzz_run.ps1 --list
  tools/fuzz_run.ps1 --target fuzz_decode --duration 300

Options:
  --sanity / --check-crashes  Build targets and replay committed crashes.
  --list                     List manifest targets and corpus counts.
  --target NAME              Select one target; may be repeated.
  --duration SECONDS         LibFuzzer time per target (default: 600).
  --jobs COUNT               LibFuzzer workers (default: 1).
  --max-len BYTES            LibFuzzer input limit (default: 16384).
  --no-diff                  Rejected; select Rust-only safety targets with --target.
'@ | Write-Output
}

function Get-FuzzTargets {
    param([string]$Manifest)

    if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
        throw "cannot read fuzz manifest: $Manifest"
    }

    $targets = New-Object 'System.Collections.Generic.List[string]'
    $inBin = $false
    $seenName = $false
    foreach ($line in Get-Content -LiteralPath $Manifest) {
        if ($line -match '^\s*\[\[bin\]\]\s*(#.*)?$') {
            if ($inBin -and -not $seenName) {
                throw "missing name in [[bin]] entry"
            }
            $inBin = $true
            $seenName = $false
            continue
        }
        if ($line -match '^\s*\[') {
            if ($inBin -and -not $seenName) {
                throw "missing name in [[bin]] entry"
            }
            $inBin = $false
            $seenName = $false
            continue
        }
        if ($inBin -and $line -match '^\s*name\s*=\s*"([^"]+)"') {
            if ($seenName) {
                throw "duplicate name in [[bin]] entry"
            }
            [void]$targets.Add($Matches[1])
            $seenName = $true
        }
    }
    if ($inBin -and -not $seenName) {
        throw "missing name in [[bin]] entry"
    }
    return $targets.ToArray()
}

function Get-FileCount {
    param([string]$Path, [string]$Filter = '*')

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        return 0
    }
    return @(
        Get-ChildItem -LiteralPath $Path -File -Recurse -Filter $Filter -ErrorAction SilentlyContinue
    ).Count
}

function Get-LastOutputLine {
    param([object[]]$Output)

    $last = ''
    foreach ($item in $Output) {
        $text = $item.ToString()
        if (-not [string]::IsNullOrWhiteSpace($text)) {
            $last = $text.Trim()
        }
    }
    return $last
}

function Setup-WindowsAsan {
    $programFiles = [Environment]::GetFolderPath('ProgramFiles')
    $vsParent = Join-Path $programFiles 'Microsoft Visual Studio'
    if (-not (Test-Path -LiteralPath $vsParent -PathType Container)) {
        Write-Output "WARNING: $vsParent not found; Windows fuzz build may fail"
        return
    }

    $asanDll = Get-ChildItem -LiteralPath $vsParent -File -Recurse -Filter 'clang_rt.asan_dynamic-x86_64.dll' -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $asanDll) {
        Write-Output 'WARNING: clang_rt.asan_dynamic-x86_64.dll not found under Visual Studio'
        Write-Output "Install the Visual Studio 'C++ AddressSanitizer' component if fuzz builds need ASan."
        return
    }

    $env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER = 'link.exe'
    $env:Path = "$($asanDll.Directory.FullName);$env:Path"
    if ([string]::IsNullOrWhiteSpace($env:ASAN_OPTIONS)) {
        $env:ASAN_OPTIONS = 'detect_odr_violation=0:detect_leaks=0'
    }
    Write-Output "Using Windows ASan runtime at $($asanDll.Directory.FullName)"
}

function Invoke-Sanity {
    param(
        [string[]]$SelectedTargets,
        [string]$FuzzDirectory,
        [string]$CrashDirectory
    )

    $buildFailures = 0
    foreach ($target in $SelectedTargets) {
        Write-Output "  Building $target..."
        $buildArgs = @('+nightly', 'fuzz', 'build', '--fuzz-dir', $FuzzDirectory, $target)
        $buildOutput = @(& cargo @buildArgs 2>&1)
        $buildExit = $LASTEXITCODE
        $lastLine = Get-LastOutputLine $buildOutput
        if ($lastLine) {
            Write-Output $lastLine
        }
        if ($buildExit -ne 0) {
            Write-Output "$target build=fail crashes=0 replay=not_run"
            $buildFailures++
        }
    }

    if ($buildFailures -gt 0) {
        Write-Output "RESULT: $buildFailures fuzz target build failures found!"
        return 1
    }

    Write-Output '--- Crash regression check ---'
    $failures = 0
    foreach ($target in $SelectedTargets) {
        $targetCrashDirectory = Join-Path $CrashDirectory $target
        $crashFiles = @(
            Get-ChildItem -LiteralPath $targetCrashDirectory -File -Recurse -Filter '*.bin' -ErrorAction SilentlyContinue |
                Sort-Object FullName
        )
        if ($crashFiles.Count -eq 0) {
            Write-Output "  $target`: no committed crash .bin files to check (skip)"
            Write-Output "$target build=pass crashes=0 replay=skip"
            continue
        }

        Write-Output "  $target`: checking $($crashFiles.Count) crash files..."
        $failCount = 0
        foreach ($crashFile in $crashFiles) {
            $replayArgs = @(
                '+nightly', 'fuzz', 'run', '--fuzz-dir', $FuzzDirectory, $target,
                $crashFile.FullName, '--', '-runs=0'
            )
            $replayOutput = @(& cargo @replayArgs 2>&1)
            $replayExit = $LASTEXITCODE
            if ($replayExit -ne 0) {
                Write-Output "    FAIL: $($crashFile.FullName) still crashes!"
                Write-Output "    replay command exited with status $replayExit; last output lines:"
                $replayOutput | Select-Object -Last 40 | ForEach-Object { Write-Output "      $($_.ToString())" }
                $failCount++
            }
        }

        if ($failCount -eq 0) {
            Write-Output "    OK: all $($crashFiles.Count) crash files handled without crash"
            Write-Output "$target build=pass crashes=$($crashFiles.Count) replay=pass"
        } else {
            Write-Output "    FAILED: $failCount/$($crashFiles.Count) still crash"
            Write-Output "$target build=pass crashes=$($crashFiles.Count) replay=fail"
            $failures += $failCount
        }
    }

    if ($failures -gt 0) {
        Write-Output "RESULT: $failures crash regressions found!"
        return 1
    }
    Write-Output 'RESULT: All crash regressions pass.'
    return 0
}

function Invoke-Campaign {
    param(
        [string[]]$SelectedTargets,
        [string]$Root,
        [string]$FuzzDirectory,
        [string]$CorpusDirectory,
        [string]$CrashDirectory,
        [int]$Duration,
        [int]$Jobs,
        [int]$MaxLen
    )

    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $runDirectory = Join-Path $Root 'logs\fuzz-findings'
    $runDirectory = Join-Path $runDirectory $timestamp
    New-Item -ItemType Directory -Force -Path $runDirectory | Out-Null
    $totalFindings = 0
    $targetFailures = 0

    foreach ($target in $SelectedTargets) {
        Write-Output "--- Fuzzing: $target ($Duration`s, $Jobs jobs) ---"
        $targetCorpus = Join-Path $CorpusDirectory $target
        $targetCrashes = Join-Path $CrashDirectory $target
        $targetFindings = Join-Path $runDirectory $target
        New-Item -ItemType Directory -Force -Path $targetCorpus, $targetCrashes, $targetFindings | Out-Null

        $fuzzArgs = @(
            '+nightly', 'fuzz', 'run', '--fuzz-dir', $FuzzDirectory, $target,
            $targetCorpus, '--', "-max_total_time=$Duration", "-max_len=$MaxLen",
            "-jobs=$Jobs", "-workers=$Jobs", "-artifact_prefix=$targetFindings\",
            '-print_final_stats=1'
        )
        $logFile = Join-Path $targetFindings 'fuzz.log'
        & cargo @fuzzArgs 2>&1 | Tee-Object -FilePath $logFile
        $fuzzExit = $LASTEXITCODE
        if ($fuzzExit -ne 0) {
            Write-Output "  ERROR: $target exited with status $fuzzExit."
            [void]$targetFailures++
        }

        $artifacts = @(
            Get-ChildItem -LiteralPath $targetFindings -File -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -like 'crash-*' -or $_.Name -like 'leak-*' -or $_.Name -like 'timeout-*' -or $_.Name -like 'oom-*' }
        )
        if ($artifacts.Count -gt 0) {
            foreach ($artifact in $artifacts) {
                Copy-Item -LiteralPath $artifact.FullName -Destination (Join-Path $targetCrashes "${timestamp}_$($artifact.Name).bin") -Force
            }
            Write-Output "  FOUND $($artifacts.Count) new findings!"
        } else {
            Write-Output '  No new findings.'
        }
        $totalFindings += $artifacts.Count
        Write-Output "  ${target}: $($artifacts.Count) findings (exit=$fuzzExit)"
    }

    Write-Output '=== Fuzz run complete ==='
    Write-Output "  Run ID: $timestamp"
    Write-Output "  Total findings: $totalFindings"
    Write-Output "  Target failures: $targetFailures"
    if ($totalFindings -gt 0 -or $targetFailures -gt 0) {
        if ($totalFindings -gt 0) {
            Write-Output "Findings saved to: $runDirectory"
        }
        if ($targetFailures -gt 0) {
            Write-Output 'Fuzz targets failed; inspect the per-target logs before rerunning.'
        }
        return 1
    }
    return 0
}

try {
    $sanityOnly = $false
    $listOnly = $false
    $duration = if ($env:FUZZ_DURATION) { [int]$env:FUZZ_DURATION } else { 600 }
    $jobs = if ($env:FUZZ_JOBS) { [int]$env:FUZZ_JOBS } else { 1 }
    $maxLen = if ($env:FUZZ_MAX_LEN) { [int]$env:FUZZ_MAX_LEN } else { 16384 }
    $selectedTargets = New-Object 'System.Collections.Generic.List[string]'

    for ($index = 0; $index -lt $Arguments.Count; $index++) {
        $argument = $Arguments[$index]
        switch ($argument) {
            '--sanity' { $sanityOnly = $true; continue }
            '--check-crashes' { $sanityOnly = $true; continue }
            '--list' { $listOnly = $true; continue }
            '--no-diff' { throw '--no-diff is not supported; select Rust-only safety targets with --target' }
            '-h' { Show-Usage; exit 0 }
            '--help' { Show-Usage; exit 0 }
            '--target' {
                if ($index + 1 -ge $Arguments.Count) { throw '--target requires a value' }
                $index++
                [void]$selectedTargets.Add($Arguments[$index])
                continue
            }
            '--duration' {
                if ($index + 1 -ge $Arguments.Count) { throw '--duration requires a value' }
                $index++
                $duration = [int]$Arguments[$index]
                continue
            }
            '--jobs' {
                if ($index + 1 -ge $Arguments.Count) { throw '--jobs requires a value' }
                $index++
                $jobs = [int]$Arguments[$index]
                continue
            }
            '--max-len' {
                if ($index + 1 -ge $Arguments.Count) { throw '--max-len requires a value' }
                $index++
                $maxLen = [int]$Arguments[$index]
                continue
            }
            default { throw "unknown option: $argument" }
        }
    }

    if ($duration -lt 0 -or $jobs -lt 1 -or $maxLen -lt 1) {
        throw 'duration must be non-negative; jobs and max-len must be positive'
    }

    $scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
    $root = (Resolve-Path (Join-Path $scriptDirectory '..')).Path
    Set-Location -LiteralPath $root
    $fuzzDirectory = Join-Path $root 'tests\fuzz'
    $manifest = Join-Path $fuzzDirectory 'Cargo.toml'
    $corpusDirectory = Join-Path $fuzzDirectory 'corpus'
    $crashDirectory = Join-Path $fuzzDirectory 'crashes'
    $allTargets = @(Get-FuzzTargets $manifest)
    if ($allTargets.Count -eq 0) { throw "no fuzz targets declared in $manifest" }

    foreach ($target in $selectedTargets) {
        if ($allTargets -notcontains $target) {
            throw "fuzz target is not declared in $manifest`: $target"
        }
    }
    if ($selectedTargets.Count -eq 0) {
        foreach ($target in $allTargets) {
            [void]$selectedTargets.Add([string]$target)
        }
    }

    if ($listOnly) {
        Write-Output 'Available fuzz targets:'
        foreach ($target in $allTargets) {
            $corpusCount = Get-FileCount (Join-Path $corpusDirectory $target)
            $crashCount = Get-FileCount (Join-Path $crashDirectory $target) '*.bin'
            Write-Output ("  {0,-25}  corpus: {1,3}  crashes: {2,3}" -f $target, $corpusCount, $crashCount)
        }
        exit 0
    }

    Write-Output '=== ropus fuzz runner (Windows PowerShell) ==='
    Write-Output "  Duration per target: ${duration}s"
    Write-Output "  Parallel jobs:       $jobs"
    Write-Output "  Max input length:    $maxLen bytes"
    Write-Output "  Targets:             $($selectedTargets -join ' ')"
    if ($sanityOnly) { Write-Output '  Mode:                sanity (build + committed crash replay only)' }

    $crashDirectories = @(Get-ChildItem -LiteralPath $crashDirectory -Directory -ErrorAction SilentlyContinue)
    foreach ($crashTargetDirectory in $crashDirectories) {
        if ($allTargets -notcontains $crashTargetDirectory.Name) {
            $unexpected = @(Get-ChildItem -LiteralPath $crashTargetDirectory.FullName -File -Recurse -Filter '*.bin' -ErrorAction SilentlyContinue)
            if ($unexpected.Count -gt 0) {
                throw "undeclared fuzz crash corpus contains committed .bin files: $($crashTargetDirectory.FullName)"
            }
        }
    }

    $null = & cargo fuzz --version 2>$null
    if ($LASTEXITCODE -ne 0) { throw 'cargo-fuzz not found. Install with: cargo install cargo-fuzz' }
    $null = & cargo +nightly --version 2>$null
    if ($LASTEXITCODE -ne 0) { throw 'Rust nightly toolchain not found. Install with: rustup toolchain install nightly' }
    Setup-WindowsAsan

    if ($sanityOnly) {
        exit (Invoke-Sanity $selectedTargets.ToArray() $fuzzDirectory $crashDirectory)
    }
    exit (Invoke-Campaign $selectedTargets.ToArray() $root $fuzzDirectory $corpusDirectory $crashDirectory $duration $jobs $maxLen)
}
catch {
    [Console]::Error.WriteLine("ERROR: $($_.Exception.Message)")
    exit 1
}
