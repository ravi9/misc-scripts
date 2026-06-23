# OpenVINO Release Package Testing

Validates that OpenVINO release artifacts are self-contained (no system OpenVINO install, no `setupvars.sh`, no `LD_LIBRARY_PATH`).

## Quick Start

```bash
# 1. Build the bundle locally (mirrors release.yml CI pack step)
./test-ov-release-ci.sh

# 2. Test the bundle in a clean Docker container (CPU/GPU/NPU)
./test-ov-release-package-in-docker.sh
```

## Files

| File | Purpose |
|------|---------|
| `test-ov-release-ci.sh` | Builds llama.cpp with OpenVINO, packs a self-contained tarball (Linux) |
| `test-ov-release-ci.ps1` | Same as above for Windows |
| `test-ov-release.Dockerfile` | Clean Ubuntu 24.04 + GPU/NPU drivers, no OpenVINO installed |
| `test-ov-release-package-in-docker.sh` | Builds Docker image, runs inference tests on CPU/GPU/NPU |

## What it tests

1. Binary loads without `LD_LIBRARY_PATH` (all deps resolved via RPATH)
2. `ldd` shows no missing shared libraries
3. Inference on CPU, GPU (if `/dev/dri`), and NPU (if `/dev/accel`)

## Environment Variables

- `MODEL_PATH` — path to GGUF model (auto-downloads `Llama-3.2-1B-Instruct-Q4_K_M.gguf` if missing)
- `OV_ROOT` — OpenVINO toolkit path (default: `/opt/intel/openvino_2026.2.1`)
- `SKIP_BUILD=1` — skip build, rerun only pack + test
