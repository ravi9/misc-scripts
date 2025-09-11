# OpenVINO NPU Caching Performance Benchmark Tool

## Setup
```bash
python3 -m venv ov-npu-env
source ov-npu-env/bin/activate
pip install openvino-genai

# Download a sample model
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/llmware/llama-3.2-1b-instruct-npu-ov 

# Run Test for different NPU cache styles
python test-npu-cache-perf.py -m llama-3.2-1b-instruct-npu-ov  -c ov-npu-cache
```

### Sample Output:

```console
$ python test-npu-cache-perf.py -m models/tinyllama-1.1b-chat-ov-fp16 -c models/tinyllama-cache
 
!!! Deleting existing cache directory: models/tinyllama-cache ... !!!

Created cache directory: models/tinyllama-cache/npu_cache_aot
OpenVINO version: 2025.3.0-19807-44526285f24-releases/2025/3
OpenVINO GenAI version: 2025.3.0.0-2463-3c0e2d3e7e1
CPU Name: Intel(R) Core(TM) Ultra 5 238V
NPU Driver: 1756305890
NPU_MAX_TILES: 5
NPU_DEVICE_TOTAL_MEM_SIZE: 30.47 GB
Model path: models/tinyllama-1.1b-chat-ov-fp16
Cache base directory: models/tinyllama-cache
------------------------------------------------------------
Starting OpenVINO LLM Pipeline Benchmark
============================================================

Benchmarking: No Cache
Configuration: {}
✓ Compile time: 12.26 seconds
✓ Inference time: 1.96 seconds

Benchmarking: NPUW_CACHE_DIR (1st Run)
Configuration: {'NPUW_CACHE_DIR': 'models/tinyllama-cache/npuw_cache_dir'}
✓ Compile time: 18.91 seconds
✓ Inference time: 1.96 seconds

Benchmarking: NPUW_CACHE_DIR (2nd Run)
Configuration: {'NPUW_CACHE_DIR': 'models/tinyllama-cache/npuw_cache_dir'}
✓ Compile time: 2.58 seconds
✓ Inference time: 1.95 seconds

Benchmarking: CACHE_DIR (1st Run)
Configuration: {'CACHE_DIR': 'models/tinyllama-cache/npu_cache_dir'}
✓ Compile time: 12.10 seconds
✓ Inference time: 2.11 seconds

Benchmarking: CACHE_DIR (2nd Run)
Configuration: {'CACHE_DIR': 'models/tinyllama-cache/npu_cache_dir'}
✓ Compile time: 12.64 seconds
✓ Inference time: 2.11 seconds

Benchmarking: CACHE_DIR OPT SIZE (1st Run)
Configuration: {'CACHE_DIR': 'models/tinyllama-cache/npu_cache_dir_opt_size', 'CACHE_MODE': 'OPTIMIZE_SIZE'}
✓ Compile time: 12.18 seconds
✓ Inference time: 2.09 seconds

Benchmarking: CACHE_DIR OPT SIZE (2nd Run)
Configuration: {'CACHE_DIR': 'models/tinyllama-cache/npu_cache_dir_opt_size', 'CACHE_MODE': 'OPTIMIZE_SIZE'}
✓ Compile time: 12.79 seconds
✓ Inference time: 2.10 seconds

Benchmarking: CACHE_DIR OPT SPEED (1st Run)
Configuration: {'CACHE_DIR': 'models/tinyllama-cache/npu_cache_dir_opt_speed', 'CACHE_MODE': 'OPTIMIZE_SPEED'}
✓ Compile time: 12.46 seconds
✓ Inference time: 2.23 seconds

Benchmarking: CACHE_DIR OPT SPEED (2nd Run)
Configuration: {'CACHE_DIR': 'models/tinyllama-cache/npu_cache_dir_opt_speed', 'CACHE_MODE': 'OPTIMIZE_SPEED'}
✓ Compile time: 1.62 seconds
✓ Inference time: 2.22 seconds

Benchmarking: AOT Compilation (1st Run)
Configuration: {'EXPORT_BLOB': 'YES', 'BLOB_PATH': 'models/tinyllama-cache/npu_cache_aot/compiled_model.blob', 'CACHE_MODE': 'OPTIMIZE_SPEED'}
✓ Compile time: 12.02 seconds
✓ Inference time: 2.15 seconds

Benchmarking: AOT Compilation (2nd Run)
Configuration: {'BLOB_PATH': 'models/tinyllama-cache/npu_cache_aot/compiled_model.blob', 'CACHE_MODE': 'OPTIMIZE_SPEED'}
✓ Compile time: 1.69 seconds
✓ Inference time: 2.17 seconds

============================================================
BENCHMARK SUMMARY
============================================================

LOAD/COMPILE TIMES:
---------------------------------------------------------------------------
No Cache                           :    12.26 sec

NPUW_CACHE_DIR (1st Run)           :    18.91 sec (0.6x vs No cache)
NPUW_CACHE_DIR (2nd Run)           :     2.58 sec (4.7x vs No cache; 7.3x vs 1st run)

CACHE_DIR (1st Run)                :    12.10 sec (1.0x vs No cache)
CACHE_DIR (2nd Run)                :    12.64 sec (1.0x vs No cache; 1.0x vs 1st run)

CACHE_DIR OPT SIZE (1st Run)       :    12.18 sec (1.0x vs No cache)
CACHE_DIR OPT SIZE (2nd Run)       :    12.79 sec (1.0x vs No cache; 1.0x vs 1st run)

CACHE_DIR OPT SPEED (1st Run)      :    12.46 sec (1.0x vs No cache)
CACHE_DIR OPT SPEED (2nd Run)      :     1.62 sec (7.5x vs No cache; 7.7x vs 1st run)

AOT Compilation (1st Run)          :    12.02 sec (1.0x vs No cache)
AOT Compilation (2nd Run)          :     1.69 sec (7.3x vs No cache; 7.1x vs 1st run)


INFERENCE TIMES:
---------------------------------------------------------------------------
No Cache                           :     1.96 sec

NPUW_CACHE_DIR (1st Run)           :     1.96 sec (1.0x vs No cache)
NPUW_CACHE_DIR (2nd Run)           :     1.95 sec (1.0x vs No cache; 1.0x vs 1st run)

CACHE_DIR (1st Run)                :     2.11 sec (0.9x vs No cache)
CACHE_DIR (2nd Run)                :     2.11 sec (0.9x vs No cache; 1.0x vs 1st run)

CACHE_DIR OPT SIZE (1st Run)       :     2.09 sec (0.9x vs No cache)
CACHE_DIR OPT SIZE (2nd Run)       :     2.10 sec (0.9x vs No cache; 1.0x vs 1st run)

CACHE_DIR OPT SPEED (1st Run)      :     2.23 sec (0.9x vs No cache)
CACHE_DIR OPT SPEED (2nd Run)      :     2.22 sec (0.9x vs No cache; 1.0x vs 1st run)

AOT Compilation (1st Run)          :     2.15 sec (0.9x vs No cache)
AOT Compilation (2nd Run)          :     2.17 sec (0.9x vs No cache; 1.0x vs 1st run)


============================================================
CACHE DIRECTORY INFO 
============================================================

Directories in 'models/tinyllama-cache':
------------------------------------------------------------
Directory                            Size (MB)    Files
------------------------------------------------------------
npu_cache_aot                          2120.36 MB        1
npu_cache_dir                            21.68 MB        1
npu_cache_dir_opt_size                   21.68 MB        1
npu_cache_dir_opt_speed                2120.36 MB        1
npuw_cache_dir                           21.62 MB        7

Total time taken: 137.77 seconds
```