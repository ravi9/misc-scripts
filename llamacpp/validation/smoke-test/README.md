# llama-cli Smoke Test (OpenVINO backend)

Checks that every model in a folder loads and answers a simple question with the llama.cpp OpenVINO backend, on CPU, GPU and NPU.

- For each `*.gguf` in `MODEL_DIR`, the script runs `llama-cli` three times: CPU, GPU and NPU.
- The full model set is 34 models, so a full run has 102 cells (one cell = one model on one device).
- Each cell asks "What is the capital of France?" with `-c 64` and stateless execution (`GGML_OPENVINO_STATEFUL_EXECUTION=0`).
- A cell passes when `llama-cli` exits normally and the log contains "Paris".
- Each cell has a time limit of 300 s (`TIMEOUT_SEC`).
- The script writes one CSV row and one log file per cell.

## Files

| File | Purpose |
|------|---------|
| `download_gguf_q4_km.sh` | Downloads the 34 Q4_K_M models into `~/models_q4_km` (`DOWNLOAD_DIR` overrides) |
| `llama-cli-smoke-test.sh` | Runs the smoke test and writes the CSV and logs |

## System requirements

Measured on 2026-10-26 on the test machine: Intel Core Ultra 7 258V (Lunar Lake, 8 cores), 32 GB RAM, Ubuntu 26.04.1.

| Item | Requirement | Measured |
|------|-------------|----------|
| Hardware | Intel Core Ultra with integrated GPU and NPU. Validated on Lunar Lake (Core Ultra Series 2). | Core Ultra 7 258V |
| OS | Ubuntu 24.04 or 26.04 | Ubuntu 26.04.1, kernel 7.0 |
| RAM | 32 GB. 16 GB is not enough for the largest models. | Peak for one cell: 21.5 GiB (gemma-4-12B on NPU) |
| Free disk | 120 GB | 112 GB in total: models 109.3 GB, llama.cpp source and history 0.8 GB, build 1.4 GB, OpenVINO 0.35 GB, logs under 1 MB |
| OpenVINO | Latest release, in `/opt/intel/openvino` | 2026.4.1 |
| GPU driver | Intel compute runtime 26.35.39758.10 (`intel-opencl-icd`, `libze-intel-gpu1`), IGC 2.41.5 | Same |
| NPU driver | Intel NPU driver 1.38.0 | Same |
| Permissions | Your user is in the `render` group (access to `/dev/dri/renderD*` and `/dev/accel/accel0`) | |

Peak RAM for one cell with the two largest models. This is the drop in `MemAvailable` during the cell, so GPU and NPU buffers are included.

| Model (file size) | CPU | GPU | NPU |
|-------------------|-----|-----|-----|
| gemma-4-12B-it (7.1 GiB) | 19.4 GiB | 18.7 GiB | 21.5 GiB |
| Qwen3.5-9B (5.2 GiB) | 13.3 GiB | 13.9 GiB | 14.0 GiB (until the timeout) |

Notes:

- The script runs one cell at a time, so the peak for the full run is the peak of the largest model.
- On a 16 GB machine, the largest models fail with `OOM/SIGKILL`. Use `MODEL_DIR` to test a subset (see [Options](#options)).
- Without an NPU, the NPU cells fail. The script has no option to skip a device.
- The download script stops when free disk space is below 2 GB.

## Time

| Step | Time on the test machine |
|------|--------------------------|
| Download the 34 models (109 GB) | About 20 min. This depends on your network. |
| Clean build of llama.cpp | About 3 min |
| Smoke test, 102 cells | About 1 h 30 min: CPU 22 min, GPU 14 min, NPU 52 min |
| **Total** | **About 2 h** |

- Median time for one cell: CPU 29 s, GPU 20 s, NPU 67 s.
- The slowest passing cell took 272 s (Phi-3-mini-4k-instruct on NPU). A cell that reaches 300 s is recorded as `Timeout`.
- More timeouts make the run longer. Each timeout costs about 5 min.

## Quick start

### 1. Install drivers and OpenVINO

GPU and NPU drivers: follow [Additional Configurations for Hardware Acceleration](https://docs.openvino.ai/2026/get-started/install-openvino/configurations.html). Install the driver versions in [System requirements](#system-requirements).

OpenCL (the OpenVINO backend build needs it):

```bash
sudo apt-get install -y ocl-icd-opencl-dev opencl-headers opencl-clhpp-headers intel-opencl-icd
```

OpenVINO, from the archive:

```bash
wget https://raw.githubusercontent.com/ravi9/misc-scripts/refs/heads/main/openvino/ov-archive-install/install-openvino-from-archive.sh
bash install-openvino-from-archive.sh
```

- The script installs the latest OpenVINO release in `/opt/intel/openvino_<version>` and points the symlink `/opt/intel/openvino` to it.
- It also installs the build tools (`build-essential`, `cmake`, `ninja-build`) and the OpenVINO runtime dependencies.
- It uses `sudo`.

Check that the devices are visible:

```bash
ls /dev/dri/renderD* /dev/accel/accel0
groups | grep -w render
```

### 2. Build llama.cpp with the OpenVINO backend

Use the llama.cpp version that you want to validate: upstream [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `master`, or the OpenVINO development branch [ravi9/llama.cpp](https://github.com/ravi9/llama.cpp) `dev_backend_openvino`.

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
# To validate the OpenVINO development branch instead:
# git remote add ov https://github.com/ravi9/llama.cpp
# git fetch ov dev_backend_openvino && git checkout -B dev_backend_openvino ov/dev_backend_openvino

source /opt/intel/openvino/setupvars.sh
cmake -B build/ReleaseOV -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENVINO=ON
cmake --build build/ReleaseOV --parallel
```

[`../../llamacpp-ov-build/ubuntu-llamacpp-ov-build.sh`](../../llamacpp-ov-build/ubuntu-llamacpp-ov-build.sh) does most of steps 1 and 2 in one script: it installs the packages and the latest OpenVINO release, clones upstream llama.cpp and builds it. It installs `intel-opencl-icd` from the Ubuntu archive, which can be older than the GPU driver above, and it does not install the NPU driver.

### 3. Get the scripts and download the models

Download the two scripts into the llama.cpp root folder:

```bash
wget https://raw.githubusercontent.com/ravi9/misc-scripts/refs/heads/main/llamacpp/validation/smoke-test/download_gguf_q4_km.sh
wget https://raw.githubusercontent.com/ravi9/misc-scripts/refs/heads/main/llamacpp/validation/smoke-test/llama-cli-smoke-test.sh
```

Download the models into `~/models_q4_km`:

```bash
bash download_gguf_q4_km.sh
# Another location: DOWNLOAD_DIR=/data/models_q4_km bash download_gguf_q4_km.sh
```

The script skips files that already exist, so you can run it again after an interruption.

### 4. Run the smoke test

Run it from the llama.cpp root folder. The script finds `./build/ReleaseOV/bin/llama-cli` and writes its logs to `./llama-cli-smoke-logs/`.

The run takes about 1 h 30 min, so start it in `tmux` (`sudo apt-get install -y tmux`). Then a closed SSH connection does not stop the run.

```bash
# In the llama.cpp root folder
tmux new -s smoke
bash llama-cli-smoke-test.sh
# Detach: Ctrl-b d. Attach again: tmux attach -t smoke
```

At the end, the script prints the CSV path and the log folder:

```
llama-cli-smoke-logs/<timestamp>/llama_smoke_test_results_<timestamp>.csv
llama-cli-smoke-logs/<timestamp>/<model>.gguf/<CPU|GPU|NPU>_stateful_0.log
```

## Options

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_DIR` | `~/models_q4_km` | Folder with the `*.gguf` models. Symlinks are followed. |
| `LLAMA_CLI_PATH` | `./build/ReleaseOV/bin/llama-cli` | The `llama-cli` binary to test |
| `TIMEOUT_SEC` | `300` | Time limit for one cell, in seconds |

To test only some models, put symlinks to them in a separate folder:

```bash
mkdir -p ~/models_subset
ln -s ~/models_q4_km/bartowski-Llama-3.2-1B-Instruct-Q4_K_M.gguf ~/models_subset/
MODEL_DIR=~/models_subset bash llama-cli-smoke-test.sh
```

## Read the results

CSV columns:

```
model,device,stateful,test_description,status,exit_code,error_summary,prompt_speed,generation_speed
```

| Marker | Meaning | CSV values |
|--------|---------|------------|
| ✓ | Passed | `status` is `PASS` |
| ~ | Accuracy issue: the model runs, but the answer is wrong | `FAIL`, `error_summary` is `Missing expected content`, and `prompt_speed` and `generation_speed` are not zero |
| ✗ | Failed | Any other `FAIL`: `Timeout`, `OOM/SIGKILL`, `SIGTERM`, `Exit <code>`, `Empty log`, or `Missing expected content` with a speed of zero |

These markers are the same as in the [Validated Models](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md#validated-models) table.

Quick summary:

```bash
CSV=llama-cli-smoke-logs/<timestamp>/llama_smoke_test_results_<timestamp>.csv
grep -c '"PASS"' "$CSV"                       # number of passed cells
grep '"FAIL"' "$CSV" | cut -d, -f1,2,8-10     # failed cells: model, device, error, speeds
```

`test_description` contains a comma, so `error_summary` is field 8 for `cut`.

For a failed cell, read its log: `llama-cli-smoke-logs/<timestamp>/<model>.gguf/<device>_stateful_0.log`.

`llama-cli` samples at random. A small model can give a wrong answer sometimes. Run a single new ~ again before you report it as a regression.

## Reference result

Run on 2026-10-26 on the test machine: **93 ✓, 4 ~, 5 ✗** of 102 cells. CPU passed for all 34 models.

- Build: `ravi9/llama.cpp` `dev_backend_openvino` at `9e01c2f67`, plus https://github.com/ravi9/llama.cpp/pull/359 (merged after the run).
- OpenVINO 2026.4.1, GPU driver 26.35.39758.10, NPU driver 1.38.0.

| Model | Device | Result | Error |
|-------|--------|--------|-------|
| bartowski-Qwen_Qwen3.5-0.8B-Q4_K_M | NPU | ✗ | Missing expected content, speeds 0 |
| bartowski-Qwen_Qwen3.5-2B-Q4_K_M | NPU | ✗ | Missing expected content, speeds 0 |
| bartowski-Qwen_Qwen3.5-4B-Q4_K_M | NPU | ✗ | Missing expected content, speeds 0 |
| lmstudio-community-Qwen3.5-9B-Q4_K_M | NPU | ✗ | Timeout |
| bartowski-gemma-4-12B-it-Q4_K_M | NPU | ✗ | Missing expected content, speeds 0 |
| bartowski-google_gemma-4-E2B-it-Q4_K_M | NPU | ~ | Missing expected content |
| ibm-granite-granite-4.0-1b-Q4_K_M | GPU | ~ | Missing expected content |
| ibm-granite-granite-4.0-1b-Q4_K_M | NPU | ~ | Missing expected content |
| ibm-granite-granite-4.0-350m-Q4_K_M | GPU | ~ | Missing expected content |

All other cells passed. Compare your CSV with this list. A cell that passed here and fails for you needs a closer look.

## Resume and rerun

The script skips every model and device pair that is already in the CSV you give it. It writes new rows to that CSV and new logs to the CSV's folder.

Resume a stopped run:

```bash
bash llama-cli-smoke-test.sh \
  llama-cli-smoke-logs/<timestamp>/llama_smoke_test_results_<timestamp>.csv
```

Run some cells again: copy the CSV without those rows, then resume from the copy.

```bash
CSV=llama-cli-smoke-logs/<timestamp>/llama_smoke_test_results_<timestamp>.csv
grep -v '"NPU"' "$CSV" > "${CSV%.csv}_npu-rerun.csv"                      # all NPU cells
# grep -v '^"bartowski-Qwen_Qwen3.5-2B-Q4_K_M.gguf"' "$CSV" > ...        # one model, all devices
bash llama-cli-smoke-test.sh "${CSV%.csv}_npu-rerun.csv"
```

## Stop a run

- Press Ctrl-C in the tmux window. The script ignores SIGTERM, so `kill <pid>` does not stop it.
- Or, from another shell: `tmux kill-session -t smoke; pkill -x llama-cli`.

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| A GPU cell does not end, and `timeout` cannot stop it | The GPU driver hangs. Check with `ps -o pid,stat,wchan:30,cmd -C llama-cli`: the state is `D`, and the wait channel is often `xe_vm_bind_ioctl` or `__flush_workqueue`. Only a reboot clears it. After the reboot, resume the run with its CSV. The hung cell has no row, so it runs again. Most hangs do not occur again. |
| The same cell hangs again after a reboot | Reboot again. Then add a row for the cell, so that the script skips it, and resume: `echo '"<model>.gguf","GPU","0","GPU, stateful=0","FAIL","","GPU driver hang","",""' >> "$CSV"` |
| `OOM/SIGKILL` | Not enough RAM for this model on this device. See [System requirements](#system-requirements). |
| All NPU cells fail | No NPU, no NPU driver, or no access to `/dev/accel/accel0`. Check the `render` group. |
| All GPU cells fail | No GPU compute runtime, or no access to `/dev/dri/renderD*`. Check the `render` group. |
| `llama-cli not found at ./build/ReleaseOV/bin/llama-cli` | Run the script from the llama.cpp root folder, or set `LLAMA_CLI_PATH`. |
| `OpenVINO setupvars.sh not found` | Install OpenVINO in `/opt/intel/openvino`, or make a symlink there to your install folder. |
