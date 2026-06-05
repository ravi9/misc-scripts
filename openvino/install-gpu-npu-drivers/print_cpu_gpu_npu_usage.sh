#!/bin/bash
#
# System Monitor - CPU, GPU (Intel xe/i915), NPU utilization tracker
#
# Prerequisites:
#   - clinfo         (GPU device info)       : apt install clinfo
#   - lspci          (PCI device listing)    : apt install pciutils
#   - sudo access    (GPU topology from debugfs)
#
# Usage:
#   bash print_cpu_gpu_npu_util.sh
#   Press Ctrl+C to stop.
#

# --- Setup: Find GPU Paths ---

# 0. Find the DRM card for the Intel GPU (card0, card1, ...).
# Some systems don't expose card0; iterate and pick the first card that has
# either the xe (tile0/gt0) or i915 (engine/rcs0) sysfs layout.
GPU_CARD=""
GPU_CARD_NUM=""
for card_path in /sys/class/drm/card*; do
    [[ -d "$card_path" ]] || continue
    card_name=$(basename "$card_path")
    # Skip connector entries like card1-DP-1
    [[ "$card_name" =~ ^card[0-9]+$ ]] || continue
    if [[ -f "${card_path}/device/tile0/gt0/gtidle/idle_residency_ms" ]] \
       || compgen -G "${card_path}/engine/rcs0" > /dev/null \
       || [[ -f "${card_path}/gt_cur_freq_mhz" ]]; then
        GPU_CARD="$card_name"
        GPU_CARD_NUM="${card_name#card}"
        break
    fi
done

# 1. Find GPU Utilization Path
# For xe driver: use gtidle/idle_residency_ms (busy = 100% - idle%)
# For i915 driver: use engine/rcs0/busy_time_ns
GPU_METHOD=""
GPU_ENGINE_PATH=""
GPU_IDLE_PATH=""
if [[ -n "$GPU_CARD" ]]; then
    GPU_IDLE_PATH="/sys/class/drm/${GPU_CARD}/device/tile0/gt0/gtidle/idle_residency_ms"
fi

if [[ -n "$GPU_IDLE_PATH" && -f "$GPU_IDLE_PATH" ]]; then
    GPU_METHOD="xe_idle"
elif [[ -n "$GPU_CARD" ]]; then
    for path in /sys/class/drm/${GPU_CARD}/engine/*/; do
        if [[ -f "${path}busy_time_ns" && "$path" == *"rcs0"* ]]; then
            GPU_ENGINE_PATH="${path}busy_time_ns"
            GPU_METHOD="i915_engine"
            break
        fi
    done
fi

# 2. Find GPU Frequency Path
GPU_FREQ_PATH=""
if [[ -n "$GPU_CARD" ]]; then
    if [ -f /sys/class/drm/${GPU_CARD}/device/tile0/gt0/freq0/cur_freq ]; then
        GPU_FREQ_PATH="/sys/class/drm/${GPU_CARD}/device/tile0/gt0/freq0/cur_freq"
    elif [ -f /sys/class/drm/${GPU_CARD}/gt_cur_freq_mhz ]; then
        GPU_FREQ_PATH="/sys/class/drm/${GPU_CARD}/gt_cur_freq_mhz"
    fi
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

# Initialize stat variables to 0 or N/A to prevent printing empty values
CPU_LOAD=0
GPU_LOAD="N/A"
GPU_FREQ="N/A"
NPU_LOAD=0
NPU_FREQ=0

# Running average accumulators
SAMPLES=0
CPU_LOAD_SUM=0
CPU_FREQ_SUM=0
GPU_LOAD_SUM=0
GPU_FREQ_SUM=0
NPU_LOAD_SUM=0
NPU_FREQ_SUM=0

# --- Print Header ---
CPU_MODEL=$(lscpu | awk -F: '/Model name/{gsub(/^[ \t]+/,"",$2); print $2}')
NPU_MODEL=$(lspci | awk -F: '/Processing accelerators/{gsub(/^[ \t]+/,"",$3); print $3}')
NUM_CORES=$(nproc)
CPU_MAX_MHZ=$(lscpu | awk -F: '/CPU max MHz/{gsub(/^[ \t]+|\.0+$/,"",$2); print $2}')
CPU_THREADS=$(lscpu | awk -F: '/Thread\(s\) per core/{gsub(/^[ \t]+/,"",$2); print $2}')
CPU_CACHE=$(lscpu | awk -F: '/L3 cache/{gsub(/^[ \t]+/,"",$2); print $2}')
SYS_MEM_GB=$(awk '/MemTotal/{printf "%.0f", $2/1024/1024}' /proc/meminfo)
GPU_MAX_FREQ="N/A"
[[ -n "$GPU_CARD" ]] && GPU_MAX_FREQ=$(cat /sys/class/drm/${GPU_CARD}/device/tile0/gt0/freq0/max_freq 2>/dev/null || echo "N/A")
NPU_MAX_FREQ=$(cat /sys/class/accel/accel0/device/npu_max_frequency_mhz 2>/dev/null || echo "N/A")

# OS info
OS_VERSION=$(lsb_release -ds 2>/dev/null || cat /etc/os-release 2>/dev/null | awk -F= '/PRETTY_NAME/{gsub(/"/,"",$2); print $2}')
KERNEL_VERSION=$(uname -r)

# Installed Intel GPU/NPU driver versions (GPU first, then NPU)
INTEL_PKGS=$(
    dpkg-query -W -f '${binary:Summary}|${Package}|${Version}\n' intel-igc-core-2 libze-intel-gpu1 intel-opencl-icd 2>/dev/null
    dpkg-query -W -f '${binary:Summary}|${Package}|${Version}\n' intel-level-zero-npu 2>/dev/null
) 
INTEL_PKGS=$(echo "$INTEL_PKGS" | awk -F'|' '{printf "      %-50s %-22s %s\n", $1, $2, $3}')

# Derive GPU info from clinfo
CLINFO_OUT=$(clinfo 2>/dev/null | grep -E "Device Name|Max compute units|Global memory size" | head -3)
GPU_DEVICE_NAME=$(echo "$CLINFO_OUT" | awk -F'  +' '/Device Name/{print $NF; exit}')
GPU_GLOBAL_MEM=$(echo "$CLINFO_OUT" | awk '/Global memory size/{gsub(/.*\(|\)/,"",$0); print; exit}')

# Count bits set in a hex string.
count_bits() {
    local n=$((16#${1})) count=0
    while (( n > 0 )); do (( count += n & 1, n >>= 1 )); done
    echo $count
}

# Derive Xe Cores and Vector Engines from topology (requires root)
GPU_TOPO=""
[[ -n "$GPU_CARD_NUM" ]] && GPU_TOPO=$(sudo cat /sys/kernel/debug/dri/${GPU_CARD_NUM}/gt0/topology 2>/dev/null)
if [[ -n "$GPU_TOPO" ]]; then
    GEOM_HEX=$(echo "$GPU_TOPO" | awk '/dss mask \(geometry\)/{gsub(/,/,"",$NF); print $NF}')
    XE_CORES=$(count_bits "$GEOM_HEX")
    EU_HEX=$(echo "$GPU_TOPO" | awk '/EU mask per DSS/{print $NF}')
    EUS_PER_CORE=$(count_bits "$EU_HEX")
    EU_TYPE=$(echo "$GPU_TOPO" | awk '/EU type/{print $NF}')
    TOTAL_EUS=$((XE_CORES * EUS_PER_CORE))
    if [[ "$EU_TYPE" == "simd16" ]]; then
        VEC_ENGINES=$((TOTAL_EUS * 2))
    else
        VEC_ENGINES=$TOTAL_EUS
    fi
fi

echo "========================================="
echo " Intel System Monitor"
echo "-----------------------------------------"
echo " OS:  $OS_VERSION | Kernel: $KERNEL_VERSION"
echo "-----------------------------------------"
echo " CPU: $CPU_MODEL"
echo "      Cores: $NUM_CORES | Threads/Core: $CPU_THREADS | Max Clock: $CPU_MAX_MHZ MHz"
echo "      L3 Cache: $CPU_CACHE"
echo " GPU: ${GPU_DEVICE_NAME:-N/A}"
if [[ -n "$XE_CORES" && "$XE_CORES" -gt 0 ]]; then
    echo "      Xe Cores: $XE_CORES | EUs: $TOTAL_EUS | Vec Engines: $VEC_ENGINES"
fi
echo "      Max Clock: $GPU_MAX_FREQ MHz | Memory: $GPU_GLOBAL_MEM"
echo " NPU: $NPU_MODEL"
echo "      Max Clock: $NPU_MAX_FREQ MHz"
echo " RAM: ${SYS_MEM_GB} GB"
echo "-----------------------------------------"
echo " Driver Versions:"
echo "$INTEL_PKGS"
echo "========================================="

# Print blank lines initially to reserve space on the screen for the first overwrite
# This prevents the "jumping" effect on startup.
printf "\n\n\n\n\n\n\n\n"

while true; do
    sleep 1

    # Capture End Stats
    NEW_NPU_US=$(cat /sys/class/accel/accel0/device/npu_busy_time_us 2>/dev/null || echo 0)
    read CPU_TOT2 CPU_IDLE2 <<< $(get_cpu_stats)

    # --- Calculations ---

    # NPU
    NPU_LOAD=$(( (NEW_NPU_US - OLD_NPU_US) / 10000 ))
    NPU_MEM_GB=$(awk '{printf "%.1f", $1/1024/1024/1024}' /sys/class/accel/accel0/device/npu_memory_utilization 2>/dev/null)
    NPU_FREQ=$(cat /sys/class/accel/accel0/device/npu_current_frequency_mhz 2>/dev/null || echo "0")
    OLD_NPU_US=$NEW_NPU_US

    # GPU
    GPU_FREQ="N/A"
    [[ -n "$GPU_FREQ_PATH" ]] && GPU_FREQ=$(cat "$GPU_FREQ_PATH" 2>/dev/null || echo "N/A")

    GPU_LOAD="N/A"
    if [[ "$GPU_METHOD" == "xe_idle" ]]; then
        NEW_GPU_IDLE_MS=$(cat "$GPU_IDLE_PATH" 2>/dev/null || echo 0)
        IDLE_PCT=$(( (NEW_GPU_IDLE_MS - OLD_GPU_IDLE_MS) / 10 ))
        GPU_LOAD=$(( 100 - IDLE_PCT ))
        [[ $GPU_LOAD -lt 0 ]] && GPU_LOAD=0
        OLD_GPU_IDLE_MS=$NEW_GPU_IDLE_MS
    elif [[ "$GPU_METHOD" == "i915_engine" ]]; then
        NEW_GPU_BUSY=$(cat "$GPU_ENGINE_PATH" 2>/dev/null || echo 0)
        GPU_LOAD=$(( (NEW_GPU_BUSY - OLD_GPU_BUSY) / 10000000 ))
        [[ $GPU_LOAD -lt 0 ]] && GPU_LOAD=0
        OLD_GPU_BUSY=$NEW_GPU_BUSY
    fi

    # CPU
    DIFF_TOT=$((CPU_TOT2 - CPU_TOT1))
    DIFF_IDLE=$((CPU_IDLE2 - CPU_IDLE1))
    [ "$DIFF_TOT" -gt 0 ] && CPU_LOAD=$(( 100 * (DIFF_TOT - DIFF_IDLE) / DIFF_TOT )) || CPU_LOAD=0
    CPU_FREQ=$(awk '/cpu MHz/{n++;s+=$4} END{printf "%.0f", s/n}' /proc/cpuinfo)
    CPU_TOT1=$CPU_TOT2
    CPU_IDLE1=$CPU_IDLE2

    # System Memory
    read SYS_TOTAL_KB SYS_AVAIL_KB <<< $(awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{print t, a}' /proc/meminfo)
    SYS_USED_GB=$(awk "BEGIN {printf \"%.1f\", ($SYS_TOTAL_KB - $SYS_AVAIL_KB) / 1024 / 1024}")
    SYS_TOTAL_GB=$(awk "BEGIN {printf \"%.0f\", $SYS_TOTAL_KB / 1024 / 1024}")

    # Update running averages
    SAMPLES=$((SAMPLES + 1))
    CPU_LOAD_SUM=$((CPU_LOAD_SUM + CPU_LOAD))
    CPU_FREQ_SUM=$((CPU_FREQ_SUM + CPU_FREQ))
    [[ "$GPU_LOAD" != "N/A" ]] && GPU_LOAD_SUM=$((GPU_LOAD_SUM + GPU_LOAD))
    [[ "$GPU_FREQ" != "N/A" ]] && GPU_FREQ_SUM=$((GPU_FREQ_SUM + GPU_FREQ))
    NPU_LOAD_SUM=$((NPU_LOAD_SUM + NPU_LOAD))
    NPU_FREQ_SUM=$((NPU_FREQ_SUM + NPU_FREQ))

    AVG_CPU_LOAD=$((CPU_LOAD_SUM / SAMPLES))
    AVG_CPU_FREQ=$((CPU_FREQ_SUM / SAMPLES))
    AVG_GPU_LOAD=$((GPU_LOAD_SUM / SAMPLES))
    AVG_GPU_FREQ=$((GPU_FREQ_SUM / SAMPLES))
    AVG_NPU_LOAD=$((NPU_LOAD_SUM / SAMPLES))
    AVG_NPU_FREQ=$((NPU_FREQ_SUM / SAMPLES))

    # --- Output ---
    # Move cursor up 8 lines (\033[8A) to overwrite the previous block
    printf "\033[8A"
    
    echo "-------------------------------------------"
    printf "         %5s  %8s  %5s  %8s\n" "util" "freq" "avg" "avg freq"
    printf "CPU:    %4s%%  %5s MHz  %4s%%  %5s MHz\n" "$CPU_LOAD" "$CPU_FREQ" "$AVG_CPU_LOAD" "$AVG_CPU_FREQ"
    printf "GPU:    %4s%%  %5s MHz  %4s%%  %5s MHz\n" "$GPU_LOAD" "$GPU_FREQ" "$AVG_GPU_LOAD" "$AVG_GPU_FREQ"
    printf "NPU:    %4s%%  %5s MHz  %4s%%  %5s MHz\n" "$NPU_LOAD" "$NPU_FREQ" "$AVG_NPU_LOAD" "$AVG_NPU_FREQ"
    printf "NPU Mem: %4s GB\n" "$NPU_MEM_GB"
    printf "Sys Mem: %4s GB / %2s GB  [1s interval, %d samples]\n" "$SYS_USED_GB" "$SYS_TOTAL_GB" "$SAMPLES"
    echo "-------------------------------------------"

done