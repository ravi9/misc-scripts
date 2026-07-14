#!/usr/bin/env bash

set -euo pipefail

git clone -b ov-op-support-list https://github.com/ravi9/llama.cpp.git
cd llama.cpp

git checkout ov-op-support-list
git reset --hard origin/dev_backend_openvino
git push origin ov-op-support-list --force-with-lease

source /opt/intel/openvino/setupvars.sh
cmake -B build/ReleaseOV -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENVINO=ON
cmake --build build/ReleaseOV --parallel

GGML_OPENVINO_DEVICE=CPU ./build/ReleaseOV/bin/test-backend-ops support --output csv  > docs/ops/OPENVINO-CPU.csv
GGML_OPENVINO_DEVICE=GPU ./build/ReleaseOV/bin/test-backend-ops support --output csv  > docs/ops/OPENVINO-GPU.csv
GGML_OPENVINO_DEVICE=NPU ./build/ReleaseOV/bin/test-backend-ops support --output csv  > docs/ops/OPENVINO-NPU.csv

./scripts/create_ops_docs.py

git add . && git commit -m "Update OV ops docs"
git push origin ov-op-support-list
