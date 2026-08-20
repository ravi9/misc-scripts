#!/usr/bin/env bash
# ============================================
# llama.cpp OpenVINO Build Script (Ninja)
# ============================================
set -euo pipefail

OPENVINO_VERSION_MAJOR="2026.3"
OPENVINO_VERSION_FULL="2026.3.0.22451.bd8d6542e3c"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENVINO_INSTALL_DIR="/opt/intel/openvino_${OPENVINO_VERSION_MAJOR}"
OPENVINO_LINK_DIR="/opt/intel/openvino"
OPENVINO_TGZ="${SCRIPT_DIR}/openvino.tgz"
OPENVINO_URL="https://storage.openvinotoolkit.org/repositories/openvino/packages/${OPENVINO_VERSION_MAJOR}/linux/openvino_toolkit_ubuntu24_${OPENVINO_VERSION_FULL}_x86_64.tgz"

echo "============================================"
echo "Installing prerequisites (apt)..."
echo "============================================"
sudo apt-get update
sudo apt-get install -y \
    build-essential libcurl4-openssl-dev libtbb12 \
    cmake ninja-build python3-pip \
    curl wget tar git

echo "============================================"
echo "Installing OpenCL runtime + headers..."
echo "============================================"
sudo apt-get install -y \
    ocl-icd-opencl-dev opencl-headers opencl-clhpp-headers intel-opencl-icd

cd "${SCRIPT_DIR}"

# ============================================
# Clone llama.cpp if missing
# ============================================
if [[ ! -f "llama.cpp/CMakeLists.txt" ]]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp
fi

# ============================================
# Setup OpenVINO: download & extract to /opt/intel/openvino_${OPENVINO_VERSION_MAJOR},
# then point /opt/intel/openvino at it via symlink so the active version is swappable.
# ============================================
if [[ -f "${OPENVINO_INSTALL_DIR}/setupvars.sh" ]]; then
    echo "OpenVINO ${OPENVINO_VERSION_MAJOR} already installed at ${OPENVINO_INSTALL_DIR}. Skipping download."
else
    echo "OpenVINO not found at ${OPENVINO_INSTALL_DIR}. Starting download..."
    curl -L -o "${OPENVINO_TGZ}" "${OPENVINO_URL}"

    echo "Extracting OpenVINO to ${OPENVINO_INSTALL_DIR}..."
    sudo mkdir -p "${OPENVINO_INSTALL_DIR}"
    sudo tar -xzf "${OPENVINO_TGZ}" -C "${OPENVINO_INSTALL_DIR}" --strip-components=1
    rm -f "${OPENVINO_TGZ}"
fi

# Refresh symlink: /opt/intel/openvino -> /opt/intel/openvino_${OPENVINO_VERSION_MAJOR}
sudo ln -sfn "${OPENVINO_INSTALL_DIR}" "${OPENVINO_LINK_DIR}"

OPENVINO_ROOT="${OPENVINO_LINK_DIR}"
echo "OpenVINO Ready: ${OPENVINO_ROOT} -> ${OPENVINO_INSTALL_DIR}"

# Install OpenVINO's own runtime dependencies (one-time per system).
if [[ -x "${OPENVINO_ROOT}/install_dependencies/install_openvino_dependencies.sh" ]]; then
    echo "============================================"
    echo "Installing OpenVINO runtime dependencies..."
    echo "============================================"
    echo "Y" | sudo -E "${OPENVINO_ROOT}/install_dependencies/install_openvino_dependencies.sh"
fi

# ============================================
# Clean old build cache
# ============================================
cd "${SCRIPT_DIR}/llama.cpp"
if [[ -d "build/ReleaseOV" ]]; then
    echo "Removing old build directory..."
    rm -rf "build/ReleaseOV"
fi

echo "============================================"
echo "Configuring with CMake..."
echo "============================================"

set +u
source "${OPENVINO_ROOT}/setupvars.sh"
set -u

cmake -B build/ReleaseOV -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENVINO=ON

cmake --build build/ReleaseOV --parallel

echo "============================================"
echo "Build completed successfully!"
echo "============================================"
echo "Binaries: $(pwd)/build/ReleaseOV/bin"
echo
echo "NOTE: To run, source setupvars.sh and pick a device:"
echo "  source /opt/intel/openvino/setupvars.sh"
echo "  export GGML_OPENVINO_DEVICE=CPU   # or GPU / NPU"
echo "  ./build/ReleaseOV/bin/llama-cli -m model.gguf"
