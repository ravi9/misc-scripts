#!/usr/bin/env bash  
set -euo pipefail  

# Install Prereq
sudo apt-get update  
sudo apt-get install -y build-essential libcurl4-openssl-dev libtbb12 cmake ninja-build python3-pip curl wget tar git  
sudo apt-get install -y ocl-icd-opencl-dev opencl-headers opencl-clhpp-headers intel-opencl-icd  

# Install OV Nightly via Archives
curl -L -o openvino.tgz https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/2026.4.0-22910-d8ab7345749/openvino_toolkit_ubuntu24_2026.4.0.dev20260826_x86_64.tgz  

sudo mkdir -p /opt/intel/openvino_nightly  
sudo tar -xzf openvino.tgz -C /opt/intel/openvino_nightly --strip-components=1  
rm -f openvino.tgz  
  
sudo ln -sfn /opt/intel/openvino_nightly /opt/intel/openvino  
  
cd /opt/intel/openvino  
echo "Y" | sudo -E ./install_dependencies/install_openvino_dependencies.sh  
cd -  

# Git Clone custom repo of qwen3.6-35B-A3B support
git clone https://github.com/cavusmustafa/llama.cpp  
cd llama.cpp  
git checkout moe_gathermatmul_v4  
  
set +u
source /opt/intel/openvino/setupvars.sh  
set -u

cmake -B build/ReleaseOV -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENVINO=ON  
cmake --build build/ReleaseOV --parallel  

# Download model
mkdir -p ~/models  
wget https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \  
     -O ~/models/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf  

# Run benchmark
export GGML_OPENVINO_DEVICE=GPU  
export GGML_OPENVINO_STATEFUL_EXECUTION=1  
./build/ReleaseOV/bin/llama-bench -m ~/models/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf -fa 1
