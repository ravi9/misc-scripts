#!/usr/bin/env bash

set -euo pipefail

# https://github.com/ggml-org/llama.cpp/releases
VERSION="b9987"

WORKDIR="$HOME/llamacpp-bins"
MODEL_DIR="$HOME/models_q4_km"
IMAGE_DIR="$HOME/test_images"

CPU_TAR="llama-${VERSION}-bin-ubuntu-x64.tar.gz"
OV_TAR="llama-${VERSION}-bin-ubuntu-openvino-2026.2.1-x64.tar.gz"

CHAT_PROMPT="Describe the image in two sentences."

CPU_URL="https://github.com/ggml-org/llama.cpp/releases/download/${VERSION}/${CPU_TAR}"
OV_URL="https://github.com/ggml-org/llama.cpp/releases/download/${VERSION}/${OV_TAR}"

MODEL="$MODEL_DIR/bartowski-google_gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ="$MODEL_DIR/mmproj-gemma-4-E2B-it-Q8_0.gguf"
IMAGE="$IMAGE_DIR/sample_image_1.jpg"

sudo apt-get update
sudo apt-get install -y wget curl tar ffmpeg

mkdir -p "$WORKDIR" "$MODEL_DIR" "$IMAGE_DIR"

cd "$WORKDIR"

[[ -f "$CPU_TAR" ]] || wget "$CPU_URL"
[[ -f "$OV_TAR" ]] || wget "$OV_URL"

mkdir -p cpu openvino

tar -xf "$CPU_TAR" -C cpu
tar -xf "$OV_TAR" -C openvino

if [[ ! -f "$MODEL" ]]; then
    wget -O "$MODEL" \
        "https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF/resolve/main/google_gemma-4-E2B-it-Q4_K_M.gguf"
fi

if [[ ! -f "$MMPROJ" ]]; then
    wget -O "$MMPROJ" \
        "https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/mmproj-gemma-4-E2B-it-Q8_0.gguf?download=true"
fi

if [[ ! -f "$IMAGE" ]]; then
    curl -L -o "$IMAGE" \
        "https://picsum.photos/id/611/500/600"
fi

echo
echo "===== GGML CPU backend ====="

"$WORKDIR/cpu/llama-${VERSION}/llama-mtmd-cli" \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    --image "$IMAGE" \
    --chat-template gemma \
    -p "$CHAT_PROMPT"

echo
echo "===== OpenVINO backend ====="

"$WORKDIR/openvino/llama-${VERSION}/llama-mtmd-cli" \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    --image "$IMAGE" \
    --chat-template gemma \
    -p "$CHAT_PROMPT"

echo
echo "Done."
