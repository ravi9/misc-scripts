# test-openvino-ci-pack.ps1
#
# Local validation for the windows-openvino release packaging fix in
# llama.cpp/.github/workflows/release.yml. Mirrors the fixed CI job:
#   1. Configure llama.cpp with -DGGML_OPENVINO=ON -DLLAMA_BUILD_BORINGSSL=ON
#      using the VS 17 2022 generator + vcpkg toolchain (same as CI).
#   2. Build into a SEPARATE build dir (build\ReleaseOV-CI) so the existing
#      build\ReleaseOV from llamacpp_openvino_build.bat is left intact.
#   3. Run the EXACT pack-step PowerShell from the fixed release.yml (copies
#      OpenVINO + TBB DLLs into bin\Release\, plus LICENSE) and zip it.
#   4. Extract the zip to a clean test folder and run llama-cli.exe -h with a
#      sanitised PATH (System32 only, no setupvars, no vcpkg, no OpenVINO).
#
# Pass: exit code 0 with help text printed -> package is self-contained.
# Fail: non-zero exit (e.g. -1073741515 = STATUS_DLL_NOT_FOUND) -> still has
#       missing-DLL dependency; inspect output to see which DLL.

[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\llamacpp-bench\llama.cpp",
    [string]$OvRoot   = "C:\Intel\openvino",
    [string]$VcpkgDir = "C:\vcpkg",
    [string]$BuildDir = "build\ReleaseOV-CI",
    [string]$ZipPath  = "C:\llamacpp-bench\test-llama-bin-win-openvino-x64.zip",
    [string]$TestDir  = "C:\llamacpp-bench\test-openvino-extracted",
    [switch]$SkipBuild   # rerun only Pack + Test against an existing $BuildDir
)

$ErrorActionPreference = 'Stop'

function Section($msg) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " $msg" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
}

# -----------------------------------------------------------------------------
# 0. Sanity checks
# -----------------------------------------------------------------------------
Section "Checking prerequisites"

if (-not (Test-Path "$RepoRoot\CMakeLists.txt"))                       { throw "llama.cpp source not found at $RepoRoot" }
if (-not (Test-Path "$OvRoot\setupvars.bat"))                          { throw "OpenVINO toolkit not found at $OvRoot"   }
if (-not (Test-Path "$VcpkgDir\scripts\buildsystems\vcpkg.cmake"))     { throw "vcpkg not found at $VcpkgDir"             }

# Locate 7z.exe: PATH, then common install locations. Fall back to built-in
# Compress-Archive / Expand-Archive cmdlets if 7z is unavailable.
$SevenZip = $null
$cmd = Get-Command 7z -ErrorAction SilentlyContinue
if ($cmd) {
    $SevenZip = $cmd.Source
} else {
    foreach ($p in @(
        "$env:ProgramFiles\7-Zip\7z.exe",
        "${env:ProgramFiles(x86)}\7-Zip\7z.exe",
        "$env:LOCALAPPDATA\Programs\7-Zip\7z.exe"
    )) {
        if (Test-Path $p) { $SevenZip = $p; break }
    }
}
if ($SevenZip) { Write-Host "  7z found  : $SevenZip" }
else           { Write-Host "  7z found  : (none - will use built-in Compress-Archive / Expand-Archive)" }

$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $VsWhere)) { throw "vswhere.exe not found - install VS 2022 / Build Tools" }
$VsPath = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $VsPath) { throw "No VS install with VC++ x64 tools found" }

Write-Host "  Repo      : $RepoRoot"
Write-Host "  OpenVINO  : $OvRoot"
Write-Host "  vcpkg     : $VcpkgDir"
Write-Host "  VS        : $VsPath"
Write-Host "  Build dir : $RepoRoot\$BuildDir  (separate from existing build\ReleaseOV)"

Set-Location $RepoRoot

# -----------------------------------------------------------------------------
# 1. Import vcvars64 + setupvars into this PowerShell process
# -----------------------------------------------------------------------------
if (-not $SkipBuild) {
    Section "Importing vcvars64 + OpenVINO setupvars into env"

    $vcvars   = "$VsPath\VC\Auxiliary\Build\vcvars64.bat"
    $tmpBat   = [System.IO.Path]::GetTempFileName() + ".bat"
    $envDump  = [System.IO.Path]::GetTempFileName()
    # Reset PATH to a minimal baseline inside the child cmd so vcvars64.bat +
    # setupvars.bat start from a clean state. Otherwise a bloated parent PATH
    # (e.g. left over from a previous run that already imported these scripts)
    # can push cmd's 8191-char command-line limit and break with
    # "The input line is too long."
    @"
@echo off
set "PATH=%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem;%SystemRoot%\System32\WindowsPowerShell\v1.0"
call "$vcvars" > nul
if errorlevel 1 exit /b 1
call "$OvRoot\setupvars.bat" > nul
if errorlevel 1 exit /b 1
set > "$envDump"
"@ | Set-Content $tmpBat -Encoding ASCII

    & cmd /c $tmpBat
    if ($LASTEXITCODE -ne 0) { throw "vcvars64.bat / setupvars.bat failed" }

    Get-Content $envDump | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }
    Remove-Item $tmpBat, $envDump -Force

    # -------------------------------------------------------------------------
    # 2. CMake configure - EXACT same flags as the fixed release.yml CI step
    # -------------------------------------------------------------------------
    Section "CMake configure (CI-equivalent flags)"

    if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }

    # cmake writes informational output to stderr. With $ErrorActionPreference='Stop'
    # and 2>&1, PowerShell turns those lines into NativeCommandError records that
    # terminate the script even on success. Relax error-action around native calls
    # and rely on $LASTEXITCODE for the real status.
    $savedEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $cfgLog = Join-Path $env:TEMP "openvino-ci-cmake-configure.log"
        # VCPKG_APPLOCAL_DEPS is left at its default (ON). vcpkg's applocal step
        # copies the imported DLLs (OpenCL.dll, etc.) into each exe's directory
        # automatically. There is a known race under heavy parallel builds where
        # multiple subprojects copy the same DLL simultaneously and one hits
        # "file in use" (exit 32). If that happens locally, re-run or pass
        # -DVCPKG_APPLOCAL_DEPS=OFF and copy OpenCL.dll explicitly in the Pack step.
        & cmake -B $BuildDir -G "Visual Studio 17 2022" `
            -A x64 `
            -DCMAKE_BUILD_TYPE=Release `
            -DGGML_OPENVINO=ON `
            -DLLAMA_BUILD_BORINGSSL=ON `
            -DLLAMA_BUILD_EXAMPLES=OFF `
            -DLLAMA_BUILD_TESTS=OFF `
            -DLLAMA_BUILD_TOOLS=ON `
            -DLLAMA_BUILD_SERVER=ON `
            -DGGML_RPC=ON `
            -DCMAKE_TOOLCHAIN_FILE="$VcpkgDir\scripts\buildsystems\vcpkg.cmake" *>&1 |
            Tee-Object -FilePath $cfgLog
        $cfgExit = $LASTEXITCODE
        if ($cfgExit -ne 0) {
            Write-Host ""
            Write-Host "CMake configure FAILED (exit $cfgExit). Full log: $cfgLog" -ForegroundColor Red
            throw "CMake configure failed"
        }

        # ---------------------------------------------------------------------
        # 3. Build
        # ---------------------------------------------------------------------
        Section "Building (this can take 15-30 min on a cold cache)"

        $buildLog = Join-Path $env:TEMP "openvino-ci-cmake-build.log"
        & cmake --build $BuildDir --config Release -- /m *>&1 |
            Tee-Object -FilePath $buildLog
        $buildExit = $LASTEXITCODE
        if ($buildExit -ne 0) {
            # Write directly to host so these aren't swallowed by any pipeline redirection
            [Console]::Error.WriteLine("")
            [Console]::Error.WriteLine("================================================================")
            [Console]::Error.WriteLine("CMake build FAILED (exit $buildExit)")
            [Console]::Error.WriteLine("Full log: $buildLog")
            [Console]::Error.WriteLine("================================================================")

            # 1. Grep for actual error markers with context
            [Console]::Error.WriteLine("")
            [Console]::Error.WriteLine("--- error/fatal lines with 3-line context ---")
            $matches = Select-String -Path $buildLog -Pattern 'error |error:|fatal|FAILED\.|exited with code [^0]|MSB\d+:.*error|LNK\d+:|: error C\d+|: error LNK' -Context 3 -CaseSensitive:$false
            if ($matches) {
                $matches | ForEach-Object {
                    [Console]::Error.WriteLine("--- $($_.Filename):$($_.LineNumber) ---")
                    $_.Context.PreContext  | ForEach-Object { [Console]::Error.WriteLine("  $_") }
                    [Console]::Error.WriteLine("> $($_.Line)")
                    $_.Context.PostContext | ForEach-Object { [Console]::Error.WriteLine("  $_") }
                }
            } else {
                [Console]::Error.WriteLine("(no error markers matched)")
            }

            # 2. Print last 150 lines unconditionally
            [Console]::Error.WriteLine("")
            [Console]::Error.WriteLine("--- last 150 lines of build log ---")
            Get-Content $buildLog -Tail 150 | ForEach-Object { [Console]::Error.WriteLine($_) }

            throw "CMake build failed"
        }
    } finally {
        $ErrorActionPreference = $savedEAP
    }
}

# -----------------------------------------------------------------------------
# 4. Pack - byte-for-byte same logic as release.yml windows-openvino Pack step
# -----------------------------------------------------------------------------
Section "Pack artifacts (mirrors release.yml windows-openvino Pack step)"

$dest = ".\$BuildDir\bin\Release"
if (-not (Test-Path $dest)) { throw "Expected binaries at $dest - did the build succeed?" }

$ovBin = Join-Path $OvRoot 'runtime\bin\intel64\Release'
Copy-Item -Path (Join-Path $ovBin '*.dll')       -Destination $dest -Force
Copy-Item -Path (Join-Path $ovBin 'cache.json')  -Destination $dest -Force

$tbbBin = Join-Path $OvRoot 'runtime\3rdparty\tbb\bin'
Copy-Item -Path (Join-Path $tbbBin 'tbb*.dll') -Destination $dest -Force

Copy-Item LICENSE $dest

if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
if ($SevenZip) {
    & $SevenZip a -snl $ZipPath ".\$BuildDir\bin\*" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "7z failed" }
} else {
    Compress-Archive -Path ".\$BuildDir\bin\*" -DestinationPath $ZipPath -Force
}
Write-Host "  Zip created: $ZipPath ($([math]::Round((Get-Item $ZipPath).Length/1MB,1)) MB)"

# -----------------------------------------------------------------------------
# 5. Extract zip to a clean test folder
# -----------------------------------------------------------------------------
Section "Extracting zip to clean test folder"

if (Test-Path $TestDir) { Remove-Item -Recurse -Force $TestDir }
New-Item -ItemType Directory -Path $TestDir -Force | Out-Null
if ($SevenZip) {
    & $SevenZip x $ZipPath -o"$TestDir" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "7z extract failed" }
} else {
    Expand-Archive -Path $ZipPath -DestinationPath $TestDir -Force
}

$testExe = Join-Path $TestDir "Release\llama-cli.exe"
if (-not (Test-Path $testExe)) { throw "llama-cli.exe not found at $testExe after extract" }
Write-Host "  Extracted to: $TestDir"
Write-Host "  Files: $(@(Get-ChildItem -Path (Join-Path $TestDir 'Release') -File).Count)"

# -----------------------------------------------------------------------------
# 6. Run llama-cli.exe -h with a SANITISED PATH (System32 only)
#    This proves the package works on a machine with no OpenVINO + no vcpkg.
# -----------------------------------------------------------------------------
Section "Running llama-cli.exe -h with sanitised PATH (System32 only)"

$savedPath = $env:PATH
try {
    $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
    Push-Location (Join-Path $TestDir "Release")

    # Capture stdout + stderr separately for clearer diagnostics
    $stdoutFile = [System.IO.Path]::GetTempFileName()
    $stderrFile = [System.IO.Path]::GetTempFileName()
    $proc = Start-Process -FilePath ".\llama-cli.exe" -ArgumentList "-h" `
                          -NoNewWindow -Wait -PassThru `
                          -RedirectStandardOutput $stdoutFile `
                          -RedirectStandardError  $stderrFile
    $exitCode = $proc.ExitCode
    $stdout = (Get-Content $stdoutFile -Raw) -as [string]
    $stderr = (Get-Content $stderrFile -Raw) -as [string]
    Remove-Item $stdoutFile, $stderrFile -Force

    Pop-Location
} finally {
    $env:PATH = $savedPath
}

Write-Host ""
Write-Host "  Exit code : $exitCode"
Write-Host "  PATH used : $env:SystemRoot\System32;$env:SystemRoot"
Write-Host ""
Write-Host "--- stdout (first 30 lines) ---" -ForegroundColor DarkGray
if ($stdout) { ($stdout -split "`r?`n" | Select-Object -First 30) | ForEach-Object { Write-Host $_ } }
else         { Write-Host "(empty)" }
if ($stderr) {
    Write-Host ""
    Write-Host "--- stderr ---" -ForegroundColor DarkGray
    Write-Host $stderr
}

# -----------------------------------------------------------------------------
# 7. Verdict
# -----------------------------------------------------------------------------
Section "Verdict"

if ($exitCode -eq 0 -and $stdout -and $stdout.Length -gt 0) {
    Write-Host "  PASS - package is self-contained (no OpenVINO/vcpkg install required)" -ForegroundColor Green
    exit 0
} elseif ($exitCode -eq -1073741515 -or $exitCode -eq 3221225781) {
    Write-Host "  FAIL - STATUS_DLL_NOT_FOUND (0xC0000135)" -ForegroundColor Red
    Write-Host "  A DLL dependency is still missing from the zip." -ForegroundColor Red
    Write-Host "  Run this to identify which one:" -ForegroundColor Yellow
    Write-Host "    dumpbin /dependents `"$testExe`"" -ForegroundColor Yellow
    Write-Host "    dumpbin /dependents `"$(Join-Path $TestDir 'Release\ggml-openvino.dll')`"" -ForegroundColor Yellow
    exit 1
} else {
    Write-Host "  FAIL - unexpected exit code $exitCode" -ForegroundColor Red
    exit 1
}
