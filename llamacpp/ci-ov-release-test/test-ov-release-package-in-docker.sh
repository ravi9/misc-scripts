#!/bin/bash
# test-ov-release-package-in-docker.sh
# Build and test the self-contained OpenVINO release bundle in a clean Docker environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="${MODEL_PATH:-$HOME/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf}"

# Download model if not found
if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found at $MODEL_PATH, downloading..."
    mkdir -p "$(dirname "$MODEL_PATH")"
    wget --no-check-certificate -O "$MODEL_PATH" \
        https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
fi

echo "============================================================"
echo "Building test Docker image"
echo "============================================================"

# Ensure we have the extracted test bundle
if [ ! -d "test-openvino-extracted/llama-test" ]; then
    echo "ERROR: test-openvino-extracted/llama-test not found"
    echo "Run ./test-ov-release-ci.sh first to create the bundle"
    exit 1
fi

# Build the Docker image
docker build -f test-ov-release.Dockerfile \
    --build-arg http_proxy="${http_proxy:-}" \
    --build-arg https_proxy="${https_proxy:-}" \
    -t llama-openvino-bundle-test .

echo ""
echo "============================================================"
echo "Running tests in clean container"
echo "============================================================"
echo "Model: $MODEL_PATH"
echo ""

# Run the tests, mounting the model directory
DEVICE_ARGS="--device=/dev/dri:/dev/dri"
[ -e /dev/accel ] && DEVICE_ARGS="$DEVICE_ARGS --device=/dev/accel:/dev/accel"

docker run --rm \
    $DEVICE_ARGS \
    -v "$(dirname "$MODEL_PATH"):/models:ro" \
    -e MODEL_FILE="/models/$(basename "$MODEL_PATH")" \
    llama-openvino-bundle-test

echo ""
echo "============================================================"
echo "To run interactively:"
echo "  docker run --rm -it --device=/dev/dri:/dev/dri \\"
echo "    -v $(dirname "$MODEL_PATH"):/models:ro \\"
echo "    llama-openvino-bundle-test /bin/bash"
echo "============================================================"
