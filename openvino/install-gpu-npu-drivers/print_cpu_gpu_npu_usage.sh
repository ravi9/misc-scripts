#!/bin/bash

# --- Setup: Find GPU Paths ---

# 1. Find GPU Utilization Path
# For xe driver: use gtidle/idle_residency_ms (busy = 100% - idle%)
# For i915 driver: use engine/rcs0/busy_time_ns
GPU_METHOD=""
GPU_ENGINE_PATH=""
GPU_IDLE_PATH="/sys/class/drm/card0/device/tile0/gt0/gtidle/idle_residency_ms"

if [[ -f "$GPU_IDLE_PATH" ]]; then
    GPU_METHOD="xe_idle"
else
    for path in /sys/class/drm/card0/engine/*/; do
        if [[ -f "${path}busy_time_ns" && "$path" == *"rcs0"* ]]; then
            GPU_ENGINE_PATH="${path}busy_time_ns"
            GPU_METHOD="i915_engine"
            break
        fi
    done
fi

# 2. Find GPU Frequency Path
GPU_FREQ_PATH=""
if [ -f /sys/class/drm/card0/device/tile0/gt0/freq0/cur_freq ]; then
    GPU_FREQ_PATH="/sys/class/drm/card0/device/tile0/gt0/freq0/cur_freq"
elif [ -f /sys/class/drm/card0/gt_cur_freq_mhz ]; then
    GPU_FREQ_PATH="/sys/class/drm/card0/gt_cur_freq_mhz"
fi

# Helper function for CPU stats
get_cpu_stats() {
    awk '/^cpu /{sum=0; for(i=2;i<=NF;i++) sum+=$i; print sum, $5}' /proc/stat
}

# --- Initialization ---
# We read all values ONCE before the loop starts so the first print is valid.
read CPU_TOT1 CPU_IDLE1 <<< $(get_cpu_stats)
OLD_NPU_US=$(cat /sys/class/accel/accel0/device/npu_busy_time_us 2>/dev/null || echo 0)
if [[ "$GPU_METHOD" == "xe_idle" ]]; then
    OLD_GPU_IDLE_MS=$(cat "$GPU_IDLE_PATH")
elif [[ "$GPU_METHOD" == "i915_engine" ]]; then
    OLD_GPU_BUSY=$(cat "$GPU_ENGINE_PATH")
else
    OLD_GPU_BUSY=0
fi

# Calculate initial memory stats
NPU_MEM_BYTES=$(cat /sys/class/accel/accel0/device/npu_memory_utilization 2>/dev/null || echo 0)
NPU_MEM_GB=$(awk "BEGIN {printf \"%.1f\", $NPU_MEM_BYTES / 1024 / 1024 / 1024}")
read SYS_TOTAL_KB SYS_AVAIL_KB <<< $(awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{print t, a}' /proc/meminfo)
SYS_USED_GB=$(awk "BEGIN {printf \"%.1f\", ($SYS_TOTAL_KB - $SYS_AVAIL_KB) / 1024 / 1024}")
SYS_TOTAL_GB=$(awk "BEGIN {printf \"%.0f\", $SYS_TOTAL_KB / 1024 / 1024}")

# Initialize stat variables to 0 or N/A to prevent printing empty values
CPU_LOAD=0
GPU_LOAD="N/A"
GPU_FREQ="N/A"
NPU_LOAD=0
NPU_FREQ=0

# Print blank lines initially to reserve space on the screen for the first overwrite
# This prevents the "jumping" effect on startup.
printf "\n\n\n\n\n\n"

while true; do
    # 1. Capture Start Time
    START_TIME=$(awk '{print $1}' /proc/uptime)

    sleep 1

    # 2. Capture End Stats
    NEW_NPU_US=$(cat /sys/class/accel/accel0/device/npu_busy_time_us 2>/dev/null || echo 0)
    read CPU_TOT2 CPU_IDLE2 <<< $(get_cpu_stats)
    END_TIME=$(awk '{print $1}' /proc/uptime)

    # --- Calculations ---

    # NPU
    NPU_LOAD=$(( (NEW_NPU_US - OLD_NPU_US) / 10000 ))
    NPU_MEM_BYTES=$(cat /sys/class/accel/accel0/device/npu_memory_utilization)
    NPU_MEM_GB=$(awk "BEGIN {printf \"%.1f\", $NPU_MEM_BYTES / 1024 / 1024 / 1024}")
    NPU_FREQ=$(cat /sys/class/accel/accel0/device/npu_current_frequency_mhz 2>/dev/null || echo "0")
    OLD_NPU_US=$NEW_NPU_US

    # GPU
    GPU_FREQ="N/A"
    [[ -n "$GPU_FREQ_PATH" ]] && GPU_FREQ=$(cat "$GPU_FREQ_PATH" 2>/dev/null || echo "N/A")

    GPU_LOAD="N/A"
    if [[ "$GPU_METHOD" == "xe_idle" ]]; then
        NEW_GPU_IDLE_MS=$(cat "$GPU_IDLE_PATH")
        DELTA_IDLE_MS=$((NEW_GPU_IDLE_MS - OLD_GPU_IDLE_MS))
        OLD_GPU_IDLE_MS=$NEW_GPU_IDLE_MS

        DELTA_TIME_MS=$(awk "BEGIN {printf \"%.0f\", ($END_TIME - $START_TIME) * 1000}")
        if [ "$DELTA_TIME_MS" -gt 0 ]; then
            GPU_LOAD=$((100 - (100 * DELTA_IDLE_MS / DELTA_TIME_MS)))
            [ "$GPU_LOAD" -lt 0 ] && GPU_LOAD=0
        fi
    elif [[ "$GPU_METHOD" == "i915_engine" ]]; then
        NEW_GPU_BUSY=$(cat "$GPU_ENGINE_PATH")
        DELTA_GPU=$((NEW_GPU_BUSY - OLD_GPU_BUSY))
        OLD_GPU_BUSY=$NEW_GPU_BUSY
        
        DELTA_TIME_NS=$(awk "BEGIN {printf \"%.0f\", ($END_TIME - $START_TIME) * 1000000000}")
        
        if [ "$DELTA_TIME_NS" -gt 0 ]; then
             GPU_LOAD=$(( 100 * DELTA_GPU / DELTA_TIME_NS ))
        fi
    fi

    # CPU
    DIFF_TOT=$((CPU_TOT2 - CPU_TOT1))
    DIFF_IDLE=$((CPU_IDLE2 - CPU_IDLE1))
    [ "$DIFF_TOT" -gt 0 ] && CPU_LOAD=$(( 100 * (DIFF_TOT - DIFF_IDLE) / DIFF_TOT )) || CPU_LOAD=0
    CPU_TOT1=$CPU_TOT2
    CPU_IDLE1=$CPU_IDLE2

    # System Memory
    read SYS_TOTAL_KB SYS_AVAIL_KB <<< $(awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{print t, a}' /proc/meminfo)
    SYS_USED_GB=$(awk "BEGIN {printf \"%.1f\", ($SYS_TOTAL_KB - $SYS_AVAIL_KB) / 1024 / 1024}")
    SYS_TOTAL_GB=$(awk "BEGIN {printf \"%.0f\", $SYS_TOTAL_KB / 1024 / 1024}")

    # --- Output ---
    # Move cursor up 6 lines (\033[6A) to overwrite the previous block
    printf "\033[6A"
    
    echo "---------------------------------"
    # Using %s format specifier to safely handle numbers AND text
    printf "CPU:      %3s%%\n" "$CPU_LOAD"
    printf "GPU:      %3s%% @ %4s MHz\n" "$GPU_LOAD" "$GPU_FREQ"
    printf "NPU:      %3s%% @ %4s MHz (%4s GB)\n" "$NPU_LOAD" "$NPU_FREQ" "$NPU_MEM_GB"
    printf "Sys Mem:  %4s GB / %2s GB\n" "$SYS_USED_GB" "$SYS_TOTAL_GB"
    echo "---------------------------------"

done
