# llama.cpp Vulkan vs OpenVINO GPU Benchmark (Windows)

End-to-end recipe for building llama.cpp twice on the same Windows machine — once with the **Vulkan** backend and once with the **OpenVINO** backend — and comparing the two on an Intel GPU using `llama-bench` with the `bartowski/gemma-4-12B-it-Q4_K_M` model.

References:

- [docs/build.md → Vulkan](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#vulkan) (canonical Vulkan instructions)
- [docs/backend/OPENVINO.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md) (canonical OpenVINO instructions and validated models)
- [tools/llama-bench/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) (benchmark CLI flags)

---

## Contents

- [1. Prerequisites](#1-prerequisites)
- [2. Layout used by this guide](#2-layout-used-by-this-guide)
- [3. Build with Vulkan](#3-build-with-vulkan)
- [4. Build with OpenVINO](#4-build-with-openvino)
- [5. Download the gemma-4-12B-it Q4_K_M model](#5-download-the-gemma-4-12b-it-q4_k_m-model)
- [6. Confirm each backend sees the GPU](#6-confirm-each-backend-sees-the-gpu)
- [7. Benchmark each backend](#7-benchmark-each-backend)
- [8. Compare results](#8-compare-results)
- [9. Troubleshooting](#9-troubleshooting)

---

## Quick Start

Copy-paste runnable. Open a fresh **`cmd`** window (a **Developer Command Prompt for VS 2022** is safest) and run the blocks below in order. The Vulkan and OpenVINO build scripts are hosted at:

- https://raw.githubusercontent.com/ravi9/misc-scripts/refs/heads/main/llamacpp/llamacpp_vulkan_build.bat
- https://raw.githubusercontent.com/ravi9/misc-scripts/refs/heads/main/llamacpp/llamacpp_openvino_build.bat

> [!TIP]
> Sections [1](#1-prerequisites)–[9](#9-troubleshooting) below explain every step in detail, including what each script installs, environment variables, and how to interpret results.

### 1) Create the working directory and grab both `.bat` scripts

```cmd
:: Create folders for the build trees and the model.
mkdir C:\llamacpp-bench
mkdir C:\models
cd /d C:\llamacpp-bench

:: Download the two build scripts.
curl -L -o llamacpp_vulkan_build.bat   https://raw.githubusercontent.com/ravi9/misc-scripts/refs/heads/main/llamacpp/llamacpp_vulkan_build.bat
curl -L -o llamacpp_openvino_build.bat https://raw.githubusercontent.com/ravi9/misc-scripts/refs/heads/main/llamacpp/llamacpp_openvino_build.bat
```

PowerShell equivalent for the two downloads:

```powershell
mkdir C:\llamacpp-bench, C:\models -Force
Set-Location C:\llamacpp-bench
Invoke-WebRequest -Uri https://raw.githubusercontent.com/ravi9/misc-scripts/refs/heads/main/llamacpp/llamacpp_vulkan_build.bat   -OutFile llamacpp_vulkan_build.bat
Invoke-WebRequest -Uri https://raw.githubusercontent.com/ravi9/misc-scripts/refs/heads/main/llamacpp/llamacpp_openvino_build.bat -OutFile llamacpp_openvino_build.bat
```

### 2) Build both backends

Both scripts auto-install their prerequisites via `winget`, clone `llama.cpp` into `C:\llamacpp-bench\llama.cpp`, and produce separate Ninja Release builds (`build\ReleaseVK` and `build\ReleaseOV`).

```cmd
cd /d C:\llamacpp-bench

:: Vulkan build  -> C:\llamacpp-bench\llama.cpp\build\ReleaseVK\bin
llamacpp_vulkan_build.bat

:: OpenVINO build -> C:\llamacpp-bench\llama.cpp\build\ReleaseOV\bin
llamacpp_openvino_build.bat
```

> First run only: if `llamacpp_openvino_build.bat` fails to copy into `C:\Intel\` or to create the junction, re-run it from an **elevated** Command Prompt.

### 3) Download the gemma-4-12B-it Q4_K_M model to `C:\models`

```cmd
curl -L ^
  -o C:\models\gemma-4-12B-it-Q4_K_M.gguf ^
  https://huggingface.co/bartowski/gemma-4-12B-it-GGUF/resolve/main/gemma-4-12B-it-Q4_K_M.gguf
```

Expected size: ~7.3 GB. If the model is gated, prepend `-H "Authorization: Bearer %HF_TOKEN%"` to the `curl` command.

### 4) Run both GPU benchmarks

```cmd
cd /d C:\llamacpp-bench\llama.cpp

:: ---- Vulkan GPU ----
build\ReleaseVK\bin\llama-bench.exe ^
    -m C:\models\gemma-4-12B-it-Q4_K_M.gguf ^
    -ngl 99 -fa 1 -p 512 -n 128 -r 5 ^
    -o md > vulkan_gpu_gemma4_12b.md

:: ---- OpenVINO GPU ----
call "C:\Intel\openvino\setupvars.bat"
set GGML_OPENVINO_DEVICE=GPU
set GGML_OPENVINO_STATEFUL_EXECUTION=1
set GGML_OPENVINO_CACHE_DIR=C:\tmp\ov_cache

build\ReleaseOV\bin\llama-bench.exe ^
    -m C:\models\gemma-4-12B-it-Q4_K_M.gguf ^
    -fa 1 -p 512 -n 128 -r 5 ^
    -o md > openvino_gpu_gemma4_12b.md
```

Run the OpenVINO command **twice** and use the second run's numbers — the first run pays a one-time graph-compile cost that the cache eliminates afterwards.

### 5) View the side-by-side results

```cmd
type vulkan_gpu_gemma4_12b.md
type openvino_gpu_gemma4_12b.md
```

That's the whole pipeline. The rest of this document explains every flag, alternative invocation, and failure mode in detail.

---

## 1. Prerequisites

- Windows 10/11 with an Intel GPU (iGPU on Core Ultra Series 1/2 or a discrete Intel GPU).
- Up-to-date Intel GPU driver (see [Intel GPU/NPU configuration guide](https://docs.openvino.ai/2026/get-started/install-openvino/configurations.html)).
- ~25 GB free disk space (build trees + the `gemma-4-12B-it-Q4_K_M.gguf` model is ~7.3 GB).
- A **Developer Command Prompt for VS 2022** is the safest shell. A regular `cmd` works if `vswhere.exe` can locate the VS Build Tools install — both build scripts auto-call `vcvars64.bat` from the latest install.

The two build scripts referenced below are the ones generated for this project:

- [llamacpp_vulkan_build.bat](llamacpp_vulkan_build.bat)
- [llamacpp_openvino_build.bat](llamacpp_openvino_build.bat)

Each script auto-installs its prerequisites via `winget` (Git, Ninja, CMake, VS 2022 Build Tools, plus the LunarG Vulkan SDK or OpenVINO Runtime), clones `llama.cpp` if missing, and produces a Ninja Release build.

---

## 2. Layout used by this guide

This guide assumes a single working folder, `C:\llamacpp-bench`, with both `.bat` files dropped in. Build outputs land in **separate** directories so the two backends don't stomp on each other:

```
C:\llamacpp-bench\
├── llamacpp_vulkan_build.bat
├── llamacpp_openvino_build.bat
└── llama.cpp\                     # cloned once, reused by both scripts
    └── build\
        ├── ReleaseVK\bin\         # Vulkan binaries
        └── ReleaseOV\bin\         # OpenVINO binaries
```

Models live in `C:\models\` to keep them out of the source tree.

```cmd
mkdir C:\llamacpp-bench
mkdir C:\models
```

Copy `llamacpp_vulkan_build.bat` and `llamacpp_openvino_build.bat` into `C:\llamacpp-bench`.

---

## 3. Build with Vulkan

From `C:\llamacpp-bench`:

```cmd
cd /d C:\llamacpp-bench
llamacpp_vulkan_build.bat
```

What the script does (per `docs/build.md` Vulkan section):

1. `winget install` Git, Ninja, CMake, VS 2022 Build Tools, **LunarG Vulkan SDK** (`KhronosGroup.VulkanSDK`).
2. Clones `https://github.com/ggml-org/llama.cpp` into `.\llama.cpp` if missing.
3. Resolves `VULKAN_SDK` (env var, or newest `C:\VulkanSDK\*`) and prepends `%VULKAN_SDK%\Bin` to `PATH` so `glslc` is reachable.
4. Calls `vcvars64.bat`, then:
   ```cmd
   cmake -B build\ReleaseVK -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
   cmake --build build\ReleaseVK --config Release
   ```

When the build finishes you should have `C:\llamacpp-bench\llama.cpp\build\ReleaseVK\bin\llama-bench.exe` and `llama-cli.exe`.

---

## 4. Build with OpenVINO

From the same `C:\llamacpp-bench`:

```cmd
cd /d C:\llamacpp-bench
llamacpp_openvino_build.bat
```

What the script does (per `docs/backend/OPENVINO.md` automated Windows build script):

1. `winget install` Git, Ninja, CMake, VS 2022 Build Tools.
2. Clones `https://github.com/microsoft/vcpkg` to `C:\vcpkg`, bootstraps it, and installs `opencl`.
3. Downloads OpenVINO Runtime `2026.2` and extracts it to `C:\Intel\openvino_2026.2`, then exposes a stable junction at `C:\Intel\openvino`.
4. Calls `setupvars.bat`, then:
   ```cmd
   cmake -B build\ReleaseOV -G Ninja ^
       -DCMAKE_BUILD_TYPE=Release ^
       -DGGML_OPENVINO=ON ^
       -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
   cmake --build build\ReleaseOV --config Release
   ```

After the build, `C:\llamacpp-bench\llama.cpp\build\ReleaseOV\bin\llama-bench.exe` exists alongside the Vulkan binaries.

> First-time only: re-run `llamacpp_openvino_build.bat` from an **elevated** Command Prompt if the OpenVINO copy step fails with "Access denied" — the script writes into `C:\Intel\` and creates a junction.

---

## 5. Download the gemma-4-12B-it Q4_K_M model

The benchmark target is the same `Q4_K_M` GGUF used in your Linux download script entry:

```
https://huggingface.co/bartowski/gemma-4-12B-it-GGUF/resolve/main/gemma-4-12B-it-Q4_K_M.gguf
```

**Windows Command Prompt (curl, ships in Win10/11):**

```cmd
curl -L ^
  -o C:\models\gemma-4-12B-it-Q4_K_M.gguf ^
  https://huggingface.co/bartowski/gemma-4-12B-it-GGUF/resolve/main/gemma-4-12B-it-Q4_K_M.gguf
```

**Windows PowerShell:**

```powershell
Invoke-WebRequest `
  -Uri  https://huggingface.co/bartowski/gemma-4-12B-it-GGUF/resolve/main/gemma-4-12B-it-Q4_K_M.gguf `
  -OutFile C:\models\gemma-4-12B-it-Q4_K_M.gguf
```

Expected size: ~7.3 GB.

> [!NOTE]
> Per `docs/backend/OPENVINO.md`, `gemma-4-12B-it-Q4_K_M` is validated on the OpenVINO **CPU** and **GPU** backends but is **not** validated on **NPU** (and stateful execution is unsupported for this model). This is why this guide compares Vulkan-GPU vs OpenVINO-GPU rather than involving the NPU.

---

## 6. Confirm each backend sees the GPU

Both backends expose visible devices via `--list-devices`. Run from `C:\llamacpp-bench\llama.cpp`:

**Vulkan build:**

```cmd
build\ReleaseVK\bin\llama-bench.exe --list-devices
```

You should see one or more `Vulkan` devices listed (e.g. `Vulkan0: Intel(R) Arc(R) Graphics`).

**OpenVINO build:**

```cmd
call "C:\Intel\openvino\setupvars.bat"
set GGML_OPENVINO_DEVICE=GPU
build\ReleaseOV\bin\llama-bench.exe --list-devices
```

The OpenVINO backend forwards the GGML device based on `GGML_OPENVINO_DEVICE` — leaving it unset defaults to CPU. On systems with multiple Intel GPUs use `GPU.0` / `GPU.1` to target a specific one (see [OpenVINO GPU Device docs](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html)).

---

## 7. Benchmark each backend

The standard apples-to-apples `llama-bench` invocation runs the default prompt-processing (`pp512`) and text-generation (`tg128`) tests. Per `docs/backend/OPENVINO.md` the OpenVINO backend **requires `-fa 1`** in `llama-bench`, so we pass `-fa 1` to both for a fair comparison.

Open a fresh `cmd` from `C:\llamacpp-bench\llama.cpp` for each run.

### 7a. Vulkan GPU run

```cmd
cd /d C:\llamacpp-bench\llama.cpp

build\ReleaseVK\bin\llama-bench.exe ^
    -m C:\models\gemma-4-12B-it-Q4_K_M.gguf ^
    -ngl 99 ^
    -fa 1 ^
    -p 512 -n 128 ^
    -r 5 ^
    -o md > vulkan_gpu_gemma4_12b.md
```

Notes:

- `-ngl 99` offloads every layer to the GPU. The Vulkan backend honours `-ngl` directly.
- `-fa 1` enables flash-attention (required by the OpenVINO run; kept identical here for parity).
- `-r 5` is the default — bump it to `-r 10` if you need tighter standard deviations.
- `-o md` writes the Markdown table to stdout; we redirect it to `vulkan_gpu_gemma4_12b.md` so it can be diffed against the OpenVINO run.

### 7b. OpenVINO GPU run

```cmd
cd /d C:\llamacpp-bench\llama.cpp

call "C:\Intel\openvino\setupvars.bat"
set GGML_OPENVINO_DEVICE=GPU
set GGML_OPENVINO_STATEFUL_EXECUTION=0
set GGML_OPENVINO_CACHE_DIR=C:\tmp\ov_cache

build\ReleaseOV\bin\llama-bench.exe ^
    -m C:\models\gemma-4-12B-it-Q4_K_M.gguf ^
    -fa 1 ^
    -p 512 -n 128 ^
    -r 5 ^
    -o md > openvino_gpu_gemma4_12b.md
```

Notes (all from `docs/backend/OPENVINO.md`):

- `GGML_OPENVINO_DEVICE=GPU` selects the Intel GPU (use `GPU.0` / `GPU.1` for multi-GPU systems).
- `GGML_OPENVINO_STATEFUL_EXECUTION=1` is the recommended GPU mode (the OpenVINO model owns the KV cache).
- `GGML_OPENVINO_CACHE_DIR` enables compiled-model caching across runs — the **first** run is slower because the GGML graph is translated/compiled to OpenVINO IR; the **second** run reads from cache and is the number you want to compare.
- The OpenVINO backend ignores `-ngl` because it owns the whole graph; do not pass it.
- `-fa 1` is **mandatory** for `llama-bench` with this backend.

> [!TIP]
> Run the OpenVINO command **twice** and use the second run's numbers for the comparison — that excludes the one-time graph compilation cost.

If the validated-models table shows a model as `gemma-4-12B-it-Q4_K_M` ⇒ `GPU SF: ✗`, you may need to drop `GGML_OPENVINO_STATEFUL_EXECUTION=1` and rerun in stateless mode (`GGML_OPENVINO_STATEFUL_EXECUTION=0`).

---

## 8. Compare results

Both runs print the same Markdown schema, so they can be glued together for an at-a-glance comparison.

After the two runs above, you will have:

```
C:\llamacpp-bench\llama.cpp\
├── vulkan_gpu_gemma4_12b.md
└── openvino_gpu_gemma4_12b.md
```

Each file looks roughly like the table below (numbers are illustrative — your hardware will produce different values):

| model                     | size      | params | backend  | ngl |  fa | test  |              t/s |
| ------------------------- | --------: | -----: | -------- | --: | --: | ----- | ---------------: |
| gemma 4 12B it Q4_K_M     |  7.3 GiB  | 12.2 B | Vulkan   |  99 |   1 | pp512 | xxx.xx ± y.yy    |
| gemma 4 12B it Q4_K_M     |  7.3 GiB  | 12.2 B | Vulkan   |  99 |   1 | tg128 |  xx.xx ± y.yy    |

…vs…

| model                     | size      | params | backend  | ngl |  fa | test  |              t/s |
| ------------------------- | --------: | -----: | -------- | --: | --: | ----- | ---------------: |
| gemma 4 12B it Q4_K_M     |  7.3 GiB  | 12.2 B | OpenVINO |   - |   1 | pp512 | xxx.xx ± y.yy    |
| gemma 4 12B it Q4_K_M     |  7.3 GiB  | 12.2 B | OpenVINO |   - |   1 | tg128 |  xx.xx ± y.yy    |

What to look at:

- **`pp512` (prompt-processing tokens/sec)** — how fast each backend ingests a 512-token prompt. Often where OpenVINO's graph fusion / kernel selection wins on Intel GPUs.
- **`tg128` (text-generation tokens/sec)** — sustained decode throughput. This is the user-perceived "tokens per second" during chat.
- **Standard deviation (`± …`)** — if it's >5 % of the mean, raise `-r` to 10 and rerun.

For programmatic comparison, switch both runs to JSON and diff with `jq`:

```cmd
build\ReleaseVK\bin\llama-bench.exe -m C:\models\gemma-4-12B-it-Q4_K_M.gguf -ngl 99 -fa 1 -p 512 -n 128 -r 5 -o json > vulkan.json
build\ReleaseOV\bin\llama-bench.exe -m C:\models\gemma-4-12B-it-Q4_K_M.gguf         -fa 1 -p 512 -n 128 -r 5 -o json > openvino.json
```

…then summarise with PowerShell:

```powershell
$vk = Get-Content vulkan.json   | ConvertFrom-Json
$ov = Get-Content openvino.json | ConvertFrom-Json
$vk + $ov | Select-Object model_filename, backend, n_gpu_layers, flash_attn, test, avg_ts, stddev_ts |
    Format-Table -AutoSize
```

---

## 9. Troubleshooting

| Symptom                                                                                  | Likely cause and fix                                                                                                                                                                                                          |
| ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `llama-bench` hangs or fails immediately on the OpenVINO run                             | `-fa 1` is missing. The OpenVINO backend requires flash-attention in `llama-bench`.                                                                                                                                           |
| Vulkan run shows `0 layers offloaded` / abysmal `tg128`                                  | `-ngl` is missing or too small. Use `-ngl 99` to offload all layers; verify with `--list-devices` that a `Vulkan` device is present.                                                                                          |
| OpenVINO run defaults to CPU even though `GGML_OPENVINO_DEVICE=GPU` was set              | `set` was used in a different `cmd` window than the one running `llama-bench`. Set the variable in the **same** shell, or use `setx` then open a new shell.                                                                   |
| Massive first-run latency on OpenVINO                                                    | Expected — the GGML graph is compiled to OpenVINO IR. Set `GGML_OPENVINO_CACHE_DIR=C:\tmp\ov_cache` and discard the first run.                                                                                                |
| `cmake` configure fails with "no C/C++ compiler"                                         | `vcvars64.bat` was not picked up. Re-run the script from a **Developer Command Prompt for VS 2022**, or install the "Desktop development with C++" workload via `winget install Microsoft.VisualStudio.2022.BuildTools`.       |
| `glslc` or `vulkan-1.lib` not found during the Vulkan build                              | LunarG Vulkan SDK isn't on `PATH`. Reinstall via `winget install KhronosGroup.VulkanSDK`, open a fresh shell, and re-run `llamacpp_vulkan_build.bat` (it adds `%VULKAN_SDK%\Bin` to `PATH` itself).                            |
| Out-of-memory during the gemma-4-12B run on iGPU                                          | 12 B at Q4_K_M needs ~7 GB of GPU memory. Either (a) run on a GPU with more dedicated memory, or (b) drop the prompt to `-p 256 -n 64`, or (c) fall back to the smaller `bartowski/google_gemma-4-E4B-it-Q4_K_M` model in the same directory. |

---

That's the full path: `llamacpp_vulkan_build.bat` → `llamacpp_openvino_build.bat` → download → two `llama-bench` runs → side-by-side Markdown/JSON tables — all targeting the same Intel **GPU** so the comparison is apples-to-apples.
