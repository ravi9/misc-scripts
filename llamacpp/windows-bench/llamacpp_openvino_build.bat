@echo off
setlocal enabledelayedexpansion

REM ============================================
REM llama.cpp OpenVINO Build Script (Ninja)
REM Source: docs/backend/OPENVINO.md
REM ============================================

set "OPENVINO_VERSION_MAJOR=2026.2"
set "OPENVINO_VERSION_FULL=2026.2.0.21903.52ddc073857"

set "SCRIPT_DIR=%~dp0"
set "VCPKG_DIR=C:\vcpkg"
set "OPENVINO_INSTALL_DIR=C:\Intel\openvino_%OPENVINO_VERSION_MAJOR%"
set "OPENVINO_LINK_DIR=C:\Intel\openvino"
set "OPENVINO_ZIP=%SCRIPT_DIR%openvino.zip"
set "OPENVINO_EXTRACT_TMP=%SCRIPT_DIR%openvino_extract_tmp"
set "OPENVINO_URL=https://storage.openvinotoolkit.org/repositories/openvino/packages/%OPENVINO_VERSION_MAJOR%/windows/openvino_toolkit_windows_%OPENVINO_VERSION_FULL%_x86_64.zip"

echo ============================================
echo Installing prerequisites...
echo ============================================
winget install --id Git.Git -e --accept-source-agreements --accept-package-agreements 2>nul
winget install --id Ninja-build.Ninja -e --accept-source-agreements --accept-package-agreements 2>nul
winget install --id Kitware.CMake -e --accept-source-agreements --accept-package-agreements 2>nul

REM Ensure Visual Studio Build Tools are installed.
echo Checking for Visual Studio Build Tools...
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS_INSTALLED="
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do (
        set "VS_INSTALLED=%%i"
    )
)
if defined VS_INSTALLED (
    echo Visual Studio with VC++ x86/x64 tools already present at "!VS_INSTALLED!". Skipping winget install.
) else (
    winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo WARNING: winget could not install Visual Studio Build Tools automatically.
        echo Install manually from https://aka.ms/vs/17/release/vs_BuildTools.exe ^(select the "Desktop development with C++" workload^)
        echo and re-run this script from a "Developer Command Prompt for VS 2022".
    )
)

echo ============================================
echo Installing OpenCL via vcpkg...
echo ============================================
if not exist "%VCPKG_DIR%" (
    git clone https://github.com/microsoft/vcpkg "%VCPKG_DIR%"
    cd /d "%VCPKG_DIR%"
    call bootstrap-vcpkg.bat
    call vcpkg integrate install
)
cd /d "%VCPKG_DIR%"
call vcpkg install opencl

cd /d "%SCRIPT_DIR%"

REM ============================================
REM Clone llama.cpp if missing
REM ============================================
if not exist "llama.cpp\CMakeLists.txt" (
    echo Cloning llama.cpp...
    git clone https://github.com/ggml-org/llama.cpp
)

cd /d "llama.cpp"
set "SCRIPT_DIR=%CD%"

REM ============================================
REM Setup OpenVINO: download & extract to C:\Intel\openvino_%OPENVINO_VERSION_MAJOR%,
REM then point C:\Intel\openvino at it via a directory junction (mklink /J).
REM ============================================

if exist "%OPENVINO_INSTALL_DIR%\setupvars.bat" (
    echo OpenVINO %OPENVINO_VERSION_MAJOR% already installed at "%OPENVINO_INSTALL_DIR%". Skipping download.
) else (
    echo OpenVINO not found at "%OPENVINO_INSTALL_DIR%". Starting download...

    curl -L -o "%OPENVINO_ZIP%" "%OPENVINO_URL%"
    if errorlevel 1 (
        echo ERROR: Download failed.
        exit /b 1
    )

    echo Extracting OpenVINO...
    if exist "%OPENVINO_EXTRACT_TMP%" rmdir /s /q "%OPENVINO_EXTRACT_TMP%"
    mkdir "%OPENVINO_EXTRACT_TMP%"
    tar -xf "%OPENVINO_ZIP%" -C "%OPENVINO_EXTRACT_TMP%"
    if errorlevel 1 (
        echo ERROR: Extraction failed.
        exit /b 1
    )

    REM Move the single top-level folder contents into the versioned install dir.
    REM NOTE: delayed expansion (!VAR!) is required because the surrounding else( ... )
    REM block is parsed once up-front, so %OPENVINO_EXTRACTED% would expand to "" here
    REM and xcopy would then treat "\*" as C:\* and fail with "Cannot perform a cyclic copy".
    set "OPENVINO_EXTRACTED="
    for /d %%i in ("%OPENVINO_EXTRACT_TMP%\*") do set "OPENVINO_EXTRACTED=%%i"
    if not defined OPENVINO_EXTRACTED (
        echo ERROR: Could not locate extracted OpenVINO folder under "%OPENVINO_EXTRACT_TMP%".
        exit /b 1
    )
    if not exist "%OPENVINO_INSTALL_DIR%" mkdir "%OPENVINO_INSTALL_DIR%"
    xcopy /e /i /y /q "!OPENVINO_EXTRACTED!\*" "%OPENVINO_INSTALL_DIR%\" >nul
    if errorlevel 1 (
        echo ERROR: Failed to copy OpenVINO from "!OPENVINO_EXTRACTED!" to "%OPENVINO_INSTALL_DIR%".
        echo Re-run this script from an elevated Command Prompt ^(Run as administrator^) if access is denied.
        exit /b 1
    )

    rmdir /s /q "%OPENVINO_EXTRACT_TMP%"
    del "%OPENVINO_ZIP%"
)

REM Refresh junction: C:\Intel\openvino -> C:\Intel\openvino_<version>.
REM `mklink /J` creates a directory junction (no admin / Developer Mode required).
if exist "%OPENVINO_LINK_DIR%" rmdir "%OPENVINO_LINK_DIR%"
mklink /J "%OPENVINO_LINK_DIR%" "%OPENVINO_INSTALL_DIR%" >nul
if errorlevel 1 (
    echo ERROR: Failed to create junction "%OPENVINO_LINK_DIR%" -^> "%OPENVINO_INSTALL_DIR%".
    echo If "%OPENVINO_LINK_DIR%" already exists as a regular non-empty folder, remove it manually and re-run.
    exit /b 1
)

set "OPENVINO_ROOT=%OPENVINO_LINK_DIR%"
echo OpenVINO Ready: %OPENVINO_ROOT% -^> %OPENVINO_INSTALL_DIR%


echo ============================================
echo Setting up compiler environment...
echo ============================================
REM Locate Visual Studio Build Tools vcvars64.bat
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products Microsoft.VisualStudio.Product.BuildTools -property installationPath`) do (
        set "VS_PATH=%%i"
    )
)
if defined VS_PATH (
    call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
) else (
    echo WARNING: Visual Studio Build Tools not found. Compiler may be missing.
)

REM ============================================
REM Clean old build cache
REM ============================================
if exist "build\ReleaseOV" (
    echo Removing old build directory ...
    rmdir /s /q "build\ReleaseOV"
)

echo ============================================
echo Configuring with CMake...
echo ============================================
call "%OPENVINO_ROOT%\setupvars.bat" >nul 2>nul

cmake -B build\ReleaseOV -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_OPENVINO=ON ^
    -DCMAKE_TOOLCHAIN_FILE="%VCPKG_DIR%\scripts\buildsystems\vcpkg.cmake"

if errorlevel 1 (
    echo If you continue to face CMAKE errors, make sure to install:
    echo   winget install Microsoft.VisualStudio.2022.BuildTools
    echo   Then run the "Developer Command Prompt for VS 2022" and launch this script from there.
    exit /b 1
)

cmake --build build\ReleaseOV --config Release
if errorlevel 1 exit /b 1

echo ============================================
echo Build completed successfully!
echo ============================================
echo Binaries: %CD%\build\ReleaseOV\bin
echo.
echo NOTE: To run, source setupvars.bat and pick a device:
echo   call "C:\Intel\openvino\setupvars.bat"
echo   set GGML_OPENVINO_DEVICE=CPU   ^&^& REM or GPU / NPU
echo   build\ReleaseOV\bin\llama-cli.exe -m model.gguf
echo.

endlocal
