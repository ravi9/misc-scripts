@echo off
setlocal enabledelayedexpansion

REM Install Prereq
winget install --id Git.Git -e --accept-source-agreements --accept-package-agreements
winget install --id Ninja-build.Ninja -e --accept-source-agreements --accept-package-agreements
winget install --id Kitware.CMake -e --accept-source-agreements --accept-package-agreements
cd /d C:\vcpkg
call vcpkg install opencl

REM Install OV Nightly via Archives
curl -L -o openvino.zip https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/2026.5.0-23005-9b1d5c9494e/openvino_toolkit_windows_2026.5.0.dev20260903_x86_64.zip

if exist "C:\Intel\openvino_nightly" rmdir /s /q "C:\Intel\openvino_nightly"
mkdir C:\Intel\openvino_nightly
tar -xf openvino.zip -C C:\Intel\openvino_nightly --strip-components=1
del openvino.zip

if exist "C:\Intel\openvino" rmdir "C:\Intel\openvino"
mklink /J C:\Intel\openvino C:\Intel\openvino_nightly

REM Git Clone custom repo of npu optimizations support
git clone https://github.com/zhaixuejun1993/llama.cpp.git
cd llama.cpp
git checkout xuejun/npu_profiling_v5

call C:\Intel\openvino\setupvars.bat

cmake -B build\ReleaseOV -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENVINO=ON -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build build\ReleaseOV --config Release --parallel

REM Download model
if not exist "C:\models" mkdir "C:\models"
curl -L -o "C:\models\microsoft_Phi-4-mini-instruct-Q4_K_M.gguf" https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf

REM Run benchmark

set GGML_OPENVINO_DEVICE=NPU
set GGML_OPENVINO_NPU_COMPILER_TYPE=DRIVER
set GGML_OPENVINO_NPU_REQUANT_POLICY=channel-wise
set GGML_OPENVINO_NPU_CONFIG="NPU_TILES=3,NPUW_SLICE_OUT=YES"
set GGML_OPENVINO_NPU_COMPILATION_MODE_PARAMS=optimization-level=3
set GGML_OPENVINO_NPU_FAST_MASK=1
set GGML_OPENVINO_NPU_L0_HOST_TENSORS=1
set GGML_OPENVINO_PREFILL_CHUNK_SIZE=1024
set GGML_OPENVINO_NPU_KV_SLICE=1
 
build\ReleaseOV\bin\llama-bench.exe -m "C:\models\microsoft_Phi-4-mini-instruct-Q4_K_M.gguf" -fa 1

endlocal
