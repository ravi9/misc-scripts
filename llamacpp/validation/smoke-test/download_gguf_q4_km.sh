#!/bin/bash
# =============================================================================
# GGUF Models Q4_K_M Download Script
# All models in Q4_K_M quantization for edge / local LLM deployment
#
# Sourcing rules:
#   - Prefer original-org GGUF repos (Qwen, unsloth, ibm-research/granite,
#     ibm-granite, openbmb, microsoft, tiiuae, LiquidAI, LGAI-EXAONE)
#   - Fallback to bartowski / QuantFactory / CodeFault / community only when
#     the original-org GGUF does NOT provide the needed Q4_K_M quant
#
# Filename convention (repo-prefixed, validation-friendly):
#   Descriptive remote:  <RepoOrg>-<remote-filename>.gguf
#   Generic remote:      <RepoOrg>-<Family-ModelName-Size-Variant>-Q4_K_M.gguf
#   Every file on disk traces back to its HF org and specific GGUF file.
#
# Usage:
#   chmod +x download_gguf_q4_km.sh
#   ./download_gguf_q4_km.sh
#   DOWNLOAD_DIR="/path/to/models" ./download_gguf_q4_km.sh
#   HF_TOKEN=hf_xxx ./download_gguf_q4_km.sh
# =============================================================================

set -euo pipefail

export HF_TOKEN="${HF_TOKEN:-}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$HOME/models_q4_km}"

MIN_FREE_BYTES=$((2 * 1024 * 1024 * 1024))

# URL + exact local filename (validation-friendly, org-prefixed)
MODELS=(
  "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf bartowski-Llama-3.2-1B-Instruct-Q4_K_M.gguf"
  "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf bartowski-Llama-3.2-3B-Instruct-Q4_K_M.gguf"
  "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf bartowski-Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
  "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf Qwen-qwen2.5-1.5b-instruct-q4_k_m.gguf"
  "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf Qwen-qwen2.5-coder-7b-instruct-q4_k_m.gguf"
  "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf bartowski-Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
  "https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q4_K_M.gguf bartowski-Phi-3-mini-4k-instruct-Q4_K_M.gguf"
  "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf bartowski-DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
  "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf bartowski-DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
  "https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B-Q4_K_M.gguf bartowski-Qwen_Qwen3-0.6B-Q4_K_M.gguf"
  "https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF/resolve/main/Qwen_Qwen3-1.7B-Q4_K_M.gguf bartowski-Qwen_Qwen3-1.7B-Q4_K_M.gguf"
  "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf Qwen-Qwen3-4B-Q4_K_M.gguf"
  "https://huggingface.co/lm-kit/qwen-3-8b-instruct-gguf/resolve/main/Qwen3-8B-Q4_K_M.gguf lm-kit-Qwen3-8B-Q4_K_M.gguf"
  "https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/resolve/main/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf bartowski-Qwen_Qwen3.5-0.8B-Q4_K_M.gguf"
  "https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF/resolve/main/Qwen_Qwen3.5-2B-Q4_K_M.gguf bartowski-Qwen_Qwen3.5-2B-Q4_K_M.gguf"
  "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF/resolve/main/Qwen_Qwen3.5-4B-Q4_K_M.gguf bartowski-Qwen_Qwen3.5-4B-Q4_K_M.gguf"
  "https://huggingface.co/lmstudio-community/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf lmstudio-community-Qwen3.5-9B-Q4_K_M.gguf"
  "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf unsloth-gemma-3-4b-it-Q4_K_M.gguf"
  "https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF/resolve/main/google_gemma-4-E2B-it-Q4_K_M.gguf bartowski-google_gemma-4-E2B-it-Q4_K_M.gguf"
  "https://huggingface.co/bartowski/google_gemma-4-E4B-it-GGUF/resolve/main/google_gemma-4-E4B-it-Q4_K_M.gguf bartowski-google_gemma-4-E4B-it-Q4_K_M.gguf"
  "https://huggingface.co/bartowski/gemma-4-12B-it-GGUF/resolve/main/gemma-4-12B-it-Q4_K_M.gguf bartowski-gemma-4-12B-it-Q4_K_M.gguf"
  "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf bartowski-Phi-3.5-mini-instruct-Q4_K_M.gguf"
  "https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf bartowski-microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
  "https://huggingface.co/QuantFactory/Ministral-3b-instruct-GGUF/resolve/main/Ministral-3b-instruct.Q4_K_M.gguf QuantFactory-Ministral-3b-instruct.Q4_K_M.gguf"
  "https://huggingface.co/bartowski/Ministral-8B-Instruct-2410-GGUF/resolve/main/Ministral-8B-Instruct-2410-Q4_K_M.gguf bartowski-Ministral-8B-Instruct-2410-Q4_K_M.gguf"
  "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/resolve/main/smollm2-1.7b-instruct-q4_k_m.gguf HuggingFaceTB-smollm2-1.7b-instruct-q4_k_m.gguf"
  "https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/ggml-model-Q4_K_M.gguf openbmb-MiniCPM-V-2_6-Q4_K_M.gguf"
  "https://huggingface.co/bartowski/tencent_Hunyuan-7B-Instruct-GGUF/resolve/main/tencent_Hunyuan-7B-Instruct-Q4_K_M.gguf bartowski-tencent_Hunyuan-7B-Instruct-Q4_K_M.gguf"
  "https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF/resolve/main/EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf LGAI-EXAONE-EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf"
  "https://huggingface.co/bartowski/prism-ml_Bonsai-8B-unpacked-GGUF/resolve/main/prism-ml_Bonsai-8B-unpacked-Q4_K_M.gguf bartowski-prism-ml_Bonsai-8B-unpacked-Q4_K_M.gguf"
  "https://huggingface.co/ibm-research/granite-3.2-8b-instruct-GGUF/resolve/main/granite-3.2-8b-instruct-Q4_K_M.gguf ibm-research-granite-3.2-8b-instruct-Q4_K_M.gguf"
  "https://huggingface.co/ibm-granite/granite-4.0-350m-GGUF/resolve/main/granite-4.0-350m-Q4_K_M.gguf ibm-granite-granite-4.0-350m-Q4_K_M.gguf"
  "https://huggingface.co/ibm-granite/granite-4.0-1b-GGUF/resolve/main/granite-4.0-1b-Q4_K_M.gguf ibm-granite-granite-4.0-1b-Q4_K_M.gguf"
  "https://huggingface.co/ibm-granite/granite-4.0-micro-GGUF/resolve/main/granite-4.0-micro-Q4_K_M.gguf ibm-granite-granite-4.0-micro-Q4_K_M.gguf"
)

format_duration_hm() {
  local total_seconds="$1"
  local hours=$((total_seconds / 3600))
  local mins=$(((total_seconds % 3600) / 60))
  printf "%02d:%02d" "$hours" "$mins"
}

bytes_to_gb() {
  local bytes="$1"
  awk -v b="$bytes" 'BEGIN { printf "%.2f", b / 1024 / 1024 / 1024 }'
}

get_free_space_bytes() {
  df -B1 . | awk 'NR==2 {print $4}'
}

ensure_min_free_space() {
  local free_bytes
  free_bytes="$(get_free_space_bytes)"

  if (( free_bytes < MIN_FREE_BYTES )); then
    echo "Error: free disk space is below 2 GB ($(bytes_to_gb "$free_bytes") GB). Exiting."
    exit 1
  fi
}

download_model() {
  local index="$1"
  local total="$2"
  local url="$3"
  local out_file="$4"

  echo "=========================================================="
  echo " Downloading ${index}/${total}: ${out_file}"
  echo "=========================================================="

  ensure_min_free_space

  if [[ -f "$out_file" ]]; then
    local existing_size
    existing_size="$(stat -c%s "$out_file")"
    echo "Skipping (already exists): ${out_file}"
    echo "-> Size: $(bytes_to_gb "$existing_size") GB"
    echo ""
    ((skipped_count+=1))
    return
  fi

  local start_time
  local end_time
  local duration_seconds
  local size_bytes

  start_time="$(date +%s)"
  wget -c "$url" -O "$out_file"
  end_time="$(date +%s)"

  duration_seconds=$((end_time - start_time))

  if [[ -f "$out_file" ]]; then
    size_bytes="$(stat -c%s "$out_file")"
    echo "-> Time: $(format_duration_hm "$duration_seconds") (Hours:Mins) | Size: $(bytes_to_gb "$size_bytes") GB"
    echo ""
    ((downloaded_count+=1))
  else
    echo "-> Download failed or file missing"
    echo ""
    ((failed_count+=1))
  fi
}

mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"
DOWNLOAD_DIR_PATH="$(pwd)"

echo "============================================"
echo " Downloading Edge Models  --  Q4_K_M quant"
echo "============================================"

total_models="${#MODELS[@]}"
downloaded_count=0
skipped_count=0
failed_count=0

start_epoch="$(date +%s)"

index=1
for entry in "${MODELS[@]}"; do
  read -r url out_file <<< "$entry"
  download_model "$index" "$total_models" "$url" "$out_file"
  ((index+=1))
done

end_epoch="$(date +%s)"
download_dir_bytes="$(du -sb . | awk '{print $1}')"

total_duration_seconds=$((end_epoch - start_epoch))
free_space_bytes="$(get_free_space_bytes)"

echo ""
echo "============================================"
echo " Q4_K_M downloads complete."
echo "============================================"
echo " Configured download entries:   ${total_models}"
echo " Downloaded this run:           ${downloaded_count}"
echo " Skipped (already existed):     ${skipped_count}"
echo " Failed:                        ${failed_count}"
echo ""
echo " SUMMARY"
echo " Download Directory:            ${DOWNLOAD_DIR_PATH}"
echo " Total Time:                    $(format_duration_hm "$total_duration_seconds") (Hours:Mins)"
echo " Total Disk Space Used:         $(bytes_to_gb "$download_dir_bytes") GB"
echo " Free Disk Space Remaining:     $(bytes_to_gb "$free_space_bytes") GB"
echo "============================================"

