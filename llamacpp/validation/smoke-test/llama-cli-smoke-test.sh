#!/usr/bin/env bash

# llama-cli smoke test
#
# Purpose:
#   Run a quick correctness smoke test for each .gguf model in ~/models_q4_km
#   across CPU, GPU, and NPU devices, checking both stateful=0 and stateful=1.
#
# Usage:
#   ./llama-cli-smoke-test.sh
#   LLAMA_CLI_PATH=./build/ReleaseOV/bin/llama-cli ./llama-cli-smoke-test.sh
#   MODEL_DIR="$HOME/models_q4_km" ./llama-cli-smoke-test.sh
#   ./llama-cli-smoke-test.sh path/to/previous_results.csv   # resume a prior run
#
# What it creates:
#   - A CSV summary under ./llama-cli-smoke-logs/<timestamp>/...
#   - Per-model log files under the same log directory
#   - A small prompt file used for each device/model test
#
# Resume behavior:
#   Pass the old CSV path as the first argument. The script reuses the same log
#   directory and skips rows already present in the CSV, continuing only for the
#   remaining model/device/stateful combinations.
#
# Requirements:
#   - llama-cli built and present at LLAMA_CLI_PATH (default: ./build/ReleaseOV/bin/llama-cli)
#   - OpenVINO environment available via /opt/intel/openvino/setupvars.sh
#   - Models in MODEL_DIR as *.gguf files
#
# Example:
#   ./llama-cli-smoke-test.sh
#   ./llama-cli-smoke-test.sh ./llama-cli-smoke-logs/20260827_120000/llama_smoke_test_results_20260827_120000.csv

set -uo pipefail

MODEL_DIR="${HOME}/models_q4_km"
LLAMA_CLI_PATH="${LLAMA_CLI_PATH:-./build/ReleaseOV/bin/llama-cli}"
CTX_SIZE=64
TIMEOUT_SEC=300

# --- Resume Logic Configuration ---
# Pass the old CSV file path as an argument to resume: ./script.sh [old_csv_path]
RESUME_CSV="${1:-}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

if [[ -n "$RESUME_CSV" && -f "$RESUME_CSV" ]]; then
    echo -e "\033[0;36m[RESUME]\033[0m Found existing CSV file: $RESUME_CSV"
    CSV_OUTPUT="$RESUME_CSV"
    # Extract the original log base directory from the provided CSV path
    LOG_BASE=$(dirname "$RESUME_CSV")
else
    LOG_BASE="./llama-cli-smoke-logs/${TIMESTAMP}"
    mkdir -p "$LOG_BASE"
    CSV_OUTPUT="$LOG_BASE/llama_smoke_test_results_${TIMESTAMP}.csv"
fi

PROMPTS_FILE="$LOG_BASE/prompts.txt"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'

# choom biases the OOM killer toward the memory-hungry llama-cli child instead of this script/tmux.
CHOOM_PREFIX=""
if command -v choom >/dev/null 2>&1; then
    CHOOM_PREFIX="choom -n 500 -- "
fi

# Only a real Ctrl-C (SIGINT) should abort the whole run; SIGTERM (e.g. sent by an
# OOM killer/daemon) is logged and ignored so the loop continues to the next test.
trap 'echo -e "\n${YELLOW}Interrupted by user. Exiting...${NC}"; exit 130' SIGINT
trap 'echo -e "\n${YELLOW}Received SIGTERM (possibly OOM killer) - continuing with remaining tests...${NC}"' SIGTERM

# -----------------------------------------------------------------------------
write_csv_header() {
    # Only write a new header if we are NOT resuming an existing file
    if [[ ! -f "$CSV_OUTPUT" ]]; then
        echo "model,device,stateful,test_description,status,exit_code,error_summary,prompt_speed,generation_speed" > "$CSV_OUTPUT"
    fi
}

append_csv_row() {
    local model="$1" device="$2" stateful="$3" desc="$4" status="$5" exit_code="$6" error_summary="$7" prompt_speed="$8" gen_speed="$9"
    echo "\"$model\",\"$device\",\"$stateful\",\"$desc\",\"$status\",\"$exit_code\",\"$error_summary\",\"$prompt_speed\",\"$gen_speed\"" >> "$CSV_OUTPUT"
}

# Helper function to check if a model + device + stateful combo already ran
is_already_tested() {
    local target_model="$1"
    local target_device="$2"
    local target_stateful="$3"
    
    if [[ ! -f "$CSV_OUTPUT" ]]; then
        return 1
    fi
    
    # Matches strings wrapped in quotes: "model","device","stateful"
    if grep -q "^\"${target_model}\",\"${target_device}\",\"${target_stateful}\"" "$CSV_OUTPUT"; then
        return 0 # Already tested
    fi
    return 1 # Not tested yet
}

create_prompts_file() {
    cat > "$PROMPTS_FILE" <<EOF
What is the capital of France?
Tell me a short joke.
/exit
EOF
}

run_test() {
    local model_path="$1" model_name="$2" device="$3" stateful_val="$4" test_num="$5" total_tests_per_model="$6"
    local env_vars="GGML_OPENVINO_DEVICE=$device"
    local csv_stateful="$stateful_val"
    local log_suffix=""
    local desc

    env_vars="$env_vars GGML_OPENVINO_STATEFUL_EXECUTION=$stateful_val"
    desc="$device, stateful=$stateful_val"
    log_suffix="_stateful_${stateful_val}"

    local model_log_dir="$LOG_BASE/$model_name"
    mkdir -p "$model_log_dir"
    local log_file="$model_log_dir/${device}${log_suffix}.log"

    echo -e "${CYAN}[${test_num}/${total_tests_per_model}]${NC} $desc"

    (
        while IFS= read -r line; do
            echo "$line"
            sleep 0.1
        done < "$PROMPTS_FILE"
    ) | script --quiet --return --command "
        $env_vars ${CHOOM_PREFIX}'$LLAMA_CLI_PATH' -m '$model_path' -c $CTX_SIZE --simple-io --color off
    " "$log_file" > /dev/null 2>&1

    local exit_code=$?
    local status="FAIL"
    local error_summary=""
    local prompt_speed="" gen_speed=""

    if [[ -s "$log_file" ]]; then
        prompt_speed=$(grep -m1 -oP 'Prompt:\s*\K[\d\.]+' "$log_file" | head -1)
        gen_speed=$(grep -m1 -oP 'Generation:\s*\K[\d\.]+' "$log_file" | head -1)
        prompt_speed=${prompt_speed:-}
        gen_speed=${gen_speed:-}
    fi

    if [[ $exit_code -eq 0 ]]; then
        if [[ -s "$log_file" ]]; then
            if grep -qi "capital of france" "$log_file" && grep -qi "paris" "$log_file"; then
                echo -e "  ${GREEN}✓ PASS${NC}"
                status="PASS"
            else
                echo -e "  ${RED}✗ FAIL – expected prompts/responses not found${NC}"
                error_summary="Missing expected content"
            fi
        else
            echo -e "  ${RED}✗ FAIL – log file empty${NC}"
            error_summary="Empty log"
        fi
    elif [[ $exit_code -eq 124 ]]; then
        echo -e "  ${RED}✗ FAIL – timeout after ${TIMEOUT_SEC}s${NC}"
        error_summary="Timeout"
    elif [[ $exit_code -eq 137 ]]; then
        echo -e "  ${RED}✗ FAIL – killed (likely OOM, SIGKILL)${NC}"
        error_summary="OOM/SIGKILL"
    elif [[ $exit_code -eq 143 ]]; then
        echo -e "  ${RED}✗ FAIL – terminated (SIGTERM)${NC}"
        error_summary="SIGTERM"
    else
        echo -e "  ${RED}✗ FAIL – exit code $exit_code${NC}"
        error_summary="Exit $exit_code"
        if [[ -s "$log_file" ]]; then
            head -n 3 "$log_file" | sed 's/^/    /'
        fi
    fi

    append_csv_row "$model_name" "$device" "$csv_stateful" "$desc" "$status" "$exit_code" "$error_summary" "$prompt_speed" "$gen_speed"
    [[ "$status" == "PASS" ]] && return 0 || return 1
}

# -----------------------------------------------------------------------------
set +u
if [[ -f "/opt/intel/openvino/setupvars.sh" ]]; then
    echo -e "${CYAN}[INFO]${NC} Sourcing OpenVINO environment..."
    source /opt/intel/openvino/setupvars.sh
    echo -e "${CYAN}[INFO]${NC} $OpenVINO_DIR environment loaded."
else
    echo -e "${YELLOW}[WARN]${NC} OpenVINO setupvars.sh not found at /opt/intel/openvino/setupvars.sh"
fi
set -u

if [[ ! -x "$LLAMA_CLI_PATH" ]]; then
    echo -e "${RED}[ERROR]${NC} llama-cli not found at $LLAMA_CLI_PATH"
    exit 1
fi

mapfile -t model_paths < <(find "$MODEL_DIR" -maxdepth 1 -name "*.gguf" -type f | sort)
total_models=${#model_paths[@]}
if [[ $total_models -eq 0 ]]; then
    echo -e "${RED}[ERROR]${NC} No .gguf models in $MODEL_DIR"
    exit 1
fi

create_prompts_file
write_csv_header

failed_tests=0
total_tests=0
model_index=0

for model_path in "${model_paths[@]}"; do
    model_index=$((model_index + 1))
    model_name=$(basename "$model_path")
    echo -e "\n${CYAN}=== Testing model ${model_index}/${total_models}: ${model_name} ===${NC}"

    tests=(
        "CPU 0"
        "CPU 1"
        "GPU 0"
        "GPU 1"
        "NPU 0"
    )
    tests_per_model=${#tests[@]}
    test_num=0

    for test in "${tests[@]}"; do
        test_num=$((test_num + 1))
        read -r device stateful_val <<< "$test"
        
        # Check if row is already completed inside the CSV file
        if is_already_tested "$model_name" "$device" "$stateful_val"; then
            echo -e "${YELLOW}[SKIPPED]${NC} $device, stateful=$stateful_val (Already in CSV)"
            ((total_tests++))
            continue
        fi
        
        run_test "$model_path" "$model_name" "$device" "$stateful_val" "$test_num" "$tests_per_model" || ((failed_tests++))
        ((total_tests++))
    done
done

echo "========================================================================="
if [[ $failed_tests -eq 0 ]]; then
    echo -e "${GREEN}All $total_tests tests passed across ${total_models} model(s).${NC}"
else
    echo -e "${RED}$failed_tests out of $total_tests tests failed (includes previous iterations if resumed).${NC}"
fi
echo -e "${CYAN}[INFO]${NC} CSV: $CSV_OUTPUT"
echo -e "${CYAN}[INFO]${NC} Logs: $LOG_BASE"
exit $(( failed_tests ? 1 : 0 ))
