@echo off
setlocal enabledelayedexpansion

REM ============================================
REM llama.cpp Vulkan Build Script (Ninja)
REM Source: docs/build.md (Vulkan / Windows)
REM
REM Prereqs auto-installed via winget:
REM   - Git
REM   - Ninja
REM   - CMake
REM   - Visual Studio 2022 Build Tools (Desktop development with C++)
REM   - LunarG Vulkan SDK
REM ============================================

set "SCRIPT_DIR=%~dp0"

echo ============================================
echo Installing prerequisites...
echo ============================================
winget install --id Git.Git                       -e --accept-source-agreements --accept-package-agreements 2>nul
winget install --id Ninja-build.Ninja             -e --accept-source-agreements --accept-package-agreements 2>nul
winget install --id Kitware.CMake                 -e --accept-source-agreements --accept-package-agreements 2>nul
winget install --id KhronosGroup.VulkanSDK        -e --accept-source-agreements --accept-package-agreements 2>nul

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

cd /d "%SCRIPT_DIR%"

REM ============================================
REM Clone llama.cpp if missing
REM ============================================
if not exist "llama.cpp\CMakeLists.txt" (
    echo Cloning llama.cpp...
    git clone https://github.com/ggml-org/llama.cpp
    if errorlevel 1 exit /b 1
)

cd /d "llama.cpp"

REM ============================================
REM Locate / activate Vulkan SDK
REM Try VULKAN_SDK env first; otherwise probe the standard install root C:\VulkanSDK\*
REM ============================================
if not defined VULKAN_SDK (
    if exist "C:\VulkanSDK" (
        for /f "delims=" %%i in ('dir /b /ad /o-n "C:\VulkanSDK" 2^>nul') do (
            if not defined VULKAN_SDK set "VULKAN_SDK=C:\VulkanSDK\%%i"
        )
    )
)

if not defined VULKAN_SDK (
    echo ERROR: Could not locate the Vulkan SDK.
    echo        Install it from https://vulkan.lunarg.com/sdk/home#windows and re-run,
    echo        or open a new shell so the VULKAN_SDK environment variable is picked up.
    exit /b 1
)

echo Using Vulkan SDK: %VULKAN_SDK%
set "PATH=%VULKAN_SDK%\Bin;%PATH%"

echo ============================================
echo Setting up compiler environment...
echo ============================================
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS_PATH="
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do (
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
if exist "build\ReleaseVK" (
    echo Removing old build directory ...
    rmdir /s /q "build\ReleaseVK"
)

echo ============================================
echo Configuring with CMake...
echo ============================================
cmake -B build\ReleaseVK -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_VULKAN=ON

if errorlevel 1 (
    echo If you continue to face CMAKE errors, make sure to install:
    echo   winget install Microsoft.VisualStudio.2022.BuildTools
    echo   winget install KhronosGroup.VulkanSDK
    echo   Then run the "Developer Command Prompt for VS 2022" and launch this script from there.
    exit /b 1
)

cmake --build build\ReleaseVK --config Release
if errorlevel 1 exit /b 1

echo ============================================
echo Build completed successfully!
echo ============================================
echo Binaries: %CD%\build\ReleaseVK\bin
echo.
echo NOTE: To run, offload layers to the Vulkan device with -ngl:
echo   build\ReleaseVK\bin\llama-cli.exe -m model.gguf -ngl 99 -c 1024
echo.
echo   List visible Vulkan devices:
echo     build\ReleaseVK\bin\llama-cli.exe --list-devices
echo.

endlocal
