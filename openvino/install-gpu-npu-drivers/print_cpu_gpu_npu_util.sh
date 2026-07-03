#!/bin/bash

: <<'INFO'
System Monitor - CPU, GPU (Intel xe/i915), NPU utilization tracker

Prerequisites:
  - clinfo         (GPU device info)       : apt install clinfo
  - lspci          (PCI device listing)    : apt install pciutils
  - sudo access    (GPU topology from debugfs)

Usage:
  wget https://raw.githubusercontent.com/ravi9/misc-scripts/refs/heads/main/openvino/install-gpu-npu-drivers/print_cpu_gpu_npu_util.sh
  bash print_cpu_gpu_npu_util.sh [Interval in seconds, default 1]
  bash print_cpu_gpu_npu_util.sh
  bash print_cpu_gpu_npu_util.sh 5
  Press Ctrl+C to stop.
INFO

# '.' decimal separator: comma-locale awk would make $(( 2,1 * 10 )) a syntax error.
export LC_ALL=C

INTERVAL="${1:-1}"
[[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]] || { echo "Interval must be a positive integer (seconds). Got: '$INTERVAL'" >&2; exit 1; }

# --- Setup: Find GPU Paths ---

# card0 isn't guaranteed; pick first card with an xe (tile0/gt0) or i915 (engine/rcs0) layout.
GPU_CARD=""
GPU_CARD_NUM=""
for card_path in /sys/class/drm/card*; do
    [[ -d "$card_path" ]] || continue
    card_name=$(basename "$card_path")
    [[ "$card_name" =~ ^card[0-9]+$ ]] || continue   # skip connectors like card1-DP-1
    if [[ -f "${card_path}/device/tile0/gt0/gtidle/idle_residency_ms" ]] \
       || compgen -G "${card_path}/engine/rcs0" > /dev/null \
       || [[ -f "${card_path}/gt_cur_freq_mhz" ]]; then
        GPU_CARD="$card_name"
        GPU_CARD_NUM="${card_name#card}"
        break
    fi
done

# Util method: xe = 100 - idle-residency%, i915 = rcs0/busy_time_ns delta.
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

GPU_FREQ_PATH=""
if [[ -n "$GPU_CARD" ]]; then
    if [ -f /sys/class/drm/${GPU_CARD}/device/tile0/gt0/freq0/cur_freq ]; then
        GPU_FREQ_PATH="/sys/class/drm/${GPU_CARD}/device/tile0/gt0/freq0/cur_freq"
    elif [ -f /sys/class/drm/${GPU_CARD}/gt_cur_freq_mhz ]; then
        GPU_FREQ_PATH="/sys/class/drm/${GPU_CARD}/gt_cur_freq_mhz"
    fi
fi

# x10-scaled int -> 1-decimal GB (21 -> "2.1")
x10fmt() { printf "%d.%d" $(( $1 / 10 )) $(( $1 % 10 )); }

# Cell formatters: keep "N/A" bare, else append unit. Column padding done by the printf caller.
pct() { [[ "$1" == "N/A" ]] && echo "N/A" || echo "$1%"; }
mhz() { [[ "$1" == "N/A" ]] && echo "N/A" || echo "$1 MHz"; }
gb()  { [[ "$1" == "N/A" ]] && echo "N/A" || echo "$1 GB"; }

# /proc/stat cpu line: total jiffies, idle jiffies
get_cpu_stats() {
    awk '/^cpu /{sum=0; for(i=2;i<=NF;i++) sum+=$i; print sum, $5}' /proc/stat
}

# --- Initialization ---
# Presence flags gate output/accumulators; absent device prints N/A, never 0.
NPU_PRESENT=false
[[ -f /sys/class/accel/accel0/device/npu_busy_time_us ]] && NPU_PRESENT=true
GPU_PRESENT=false
[[ -n "$GPU_METHOD" ]] && GPU_PRESENT=true
# Freq node is separate from util; gate freq avg on it, else a missing node averages to 0 MHz.
GPU_FREQ_PRESENT=false
[[ -n "$GPU_FREQ_PATH" ]] && GPU_FREQ_PRESENT=true

CPU_LOAD=0
GPU_LOAD="N/A"
GPU_FREQ="N/A"
NPU_LOAD=0
NPU_FREQ=0

START_TIME=$(date '+%Y-%m-%d %H:%M')
SAMPLES=0
CPU_LOAD_SUM=0
CPU_FREQ_SUM=0
GPU_LOAD_SUM=0
GPU_FREQ_SUM=0
NPU_LOAD_SUM=0
NPU_FREQ_SUM=0
# Mem sum/peak stored x10 (integer math on 1-decimal GB)
SYS_MEM_SUM=0
SYS_MEM_PEAK=0
NPU_MEM_SUM=0
NPU_MEM_PEAK=0

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

OS_VERSION=$(lsb_release -ds 2>/dev/null || cat /etc/os-release 2>/dev/null | awk -F= '/PRETTY_NAME/{gsub(/"/,"",$2); print $2}')
KERNEL_VERSION=$(uname -r)

# Intel driver versions, friendly-labeled (skip packages not installed)
INTEL_PKGS=""
for entry in "OpenCL Compiler:intel-igc-core-2" \
             "OpenCL Runtime:intel-opencl-icd" \
             "Level Zero GPU:libze-intel-gpu1" \
             "Level Zero NPU:intel-level-zero-npu"; do
    ver=$(dpkg-query -W -f '${Version}' "${entry#*:}" 2>/dev/null) || continue
    [[ -n "$ver" ]] && INTEL_PKGS+="$(printf '    %-14s %s (%s)' "${entry%%:*}" "${entry#*:}" "$ver")"$'\n'
done

CLINFO_OUT=$(clinfo 2>/dev/null | grep -E "Device Name|Max compute units|Global memory size" | head -3)
GPU_DEVICE_NAME=$(echo "$CLINFO_OUT" | awk -F'  +' '/Device Name/{print $NF; exit}')
GPU_GLOBAL_MEM=$(echo "$CLINFO_OUT" | awk '/Global memory size/{gsub(/.*\(|\)/,"",$0); print; exit}' | sed -E 's/([0-9])([A-Za-z])/\1 \2/')

count_bits() {
    local n=$((16#${1})) count=0
    while (( n > 0 )); do (( count += n & 1, n >>= 1 )); done
    echo $count
}

# Xe cores / EUs / vector engines from debugfs topology (needs root; omitted if unavailable)
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

RULE=$(printf '─%.0s' {1..80})

TITLE="INTEL SYSTEM MONITOR"
echo " $RULE"
printf " %*s%s\n" $(( (80 - ${#TITLE}) / 2 )) "" "$TITLE"
echo " $RULE"
printf "  %-7s %-30s %-8s %s\n" "OS" "$OS_VERSION" "Kernel" "$KERNEL_VERSION"
echo " $RULE"
printf "  %-7s %s\n" "CPU" "$CPU_MODEL"
printf "  %-7s Cores: %s  •  Threads: %s  •  Max Clock: %s MHz  •  L3: %s\n" "" "$NUM_CORES" "$CPU_THREADS" "$CPU_MAX_MHZ" "$CPU_CACHE"
echo
printf "  %-7s %s\n" "GPU" "${GPU_DEVICE_NAME:-N/A}"
if [[ -n "$XE_CORES" && "$XE_CORES" -gt 0 ]]; then
    printf "  %-7s Xe Cores: %s  •  EUs: %s  •  Max Clock: %s MHz  •  VRAM: %s\n" "" "$XE_CORES" "$TOTAL_EUS" "$GPU_MAX_FREQ" "$GPU_GLOBAL_MEM"
else
    printf "  %-7s Max Clock: %s MHz  •  VRAM: %s\n" "" "$GPU_MAX_FREQ" "$GPU_GLOBAL_MEM"
fi
echo
printf "  %-7s %s\n" "NPU" "$NPU_MODEL"
printf "  %-7s Max Clock: %s MHz\n" "" "$NPU_MAX_FREQ"
echo
printf "  %-7s %s\n" "RAM" "${SYS_MEM_GB} GB"
echo " $RULE"
echo "  DRIVERS"
printf "%s" "$INTEL_PKGS"
echo " $RULE"

# Baseline counters after the slow header commands (clinfo/sudo/dpkg), not at
# init — first sample divides by WINDOW=1 so a stale baseline inflates busy%.
read CPU_TOT1 CPU_IDLE1 <<< $(get_cpu_stats)
OLD_NPU_US=$(cat /sys/class/accel/accel0/device/npu_busy_time_us 2>/dev/null || echo 0)
if [[ "$GPU_METHOD" == "xe_idle" ]]; then
    OLD_GPU_IDLE_MS=$(cat "$GPU_IDLE_PATH")
elif [[ "$GPU_METHOD" == "i915_engine" ]]; then
    OLD_GPU_BUSY=$(cat "$GPU_ENGINE_PATH")
fi

# Reserve 12 lines for the in-place repaint (matches \033[12A below).
printf "\n\n\n\n\n\n\n\n\n\n\n\n"

WINDOW=1   # first sample: short 1s window so first summary appears fast
while true; do
    # Split sleep into 1s steps to tick the live elapsed clock (the bottom line).
    for ((s=0; s<WINDOW; s++)); do
        sleep 1
        ELAPSED=$(printf '%02d:%02d:%02d' $((SECONDS/3600)) $((SECONDS%3600/60)) $((SECONDS%60)))
        printf "\033[1A\033[2K"
        printf "  Sampling: %ss  •  Samples: %d  •  Elapsed: %s  •  Start: %s\n" "$INTERVAL" "$SAMPLES" "$ELAPSED" "$START_TIME"
    done

    NEW_NPU_US=$(cat /sys/class/accel/accel0/device/npu_busy_time_us 2>/dev/null || echo 0)
    read CPU_TOT2 CPU_IDLE2 <<< $(get_cpu_stats)

    # NPU
    if $NPU_PRESENT; then
        NPU_LOAD=$(( (NEW_NPU_US - OLD_NPU_US) / (10000 * WINDOW) ))
        NPU_MEM_GB=$(awk '{printf "%.1f", $1/1024/1024/1024}' /sys/class/accel/accel0/device/npu_memory_utilization 2>/dev/null)
        NPU_FREQ=$(cat /sys/class/accel/accel0/device/npu_current_frequency_mhz 2>/dev/null || echo "0")
    else
        NPU_LOAD="N/A"; NPU_FREQ="N/A"; NPU_MEM_GB="N/A"
    fi
    OLD_NPU_US=$NEW_NPU_US

    # GPU
    GPU_FREQ="N/A"
    [[ -n "$GPU_FREQ_PATH" ]] && GPU_FREQ=$(cat "$GPU_FREQ_PATH" 2>/dev/null || echo "N/A")

    GPU_LOAD="N/A"
    if [[ "$GPU_METHOD" == "xe_idle" ]]; then
        NEW_GPU_IDLE_MS=$(cat "$GPU_IDLE_PATH" 2>/dev/null || echo 0)
        IDLE_PCT=$(( (NEW_GPU_IDLE_MS - OLD_GPU_IDLE_MS) / (10 * WINDOW) ))
        GPU_LOAD=$(( 100 - IDLE_PCT ))
        [[ $GPU_LOAD -lt 0 ]] && GPU_LOAD=0
        OLD_GPU_IDLE_MS=$NEW_GPU_IDLE_MS
    elif [[ "$GPU_METHOD" == "i915_engine" ]]; then
        NEW_GPU_BUSY=$(cat "$GPU_ENGINE_PATH" 2>/dev/null || echo 0)
        GPU_LOAD=$(( (NEW_GPU_BUSY - OLD_GPU_BUSY) / (10000000 * WINDOW) ))
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

    SAMPLES=$((SAMPLES + 1))
    CPU_LOAD_SUM=$((CPU_LOAD_SUM + CPU_LOAD))
    CPU_FREQ_SUM=$((CPU_FREQ_SUM + CPU_FREQ))
    [[ "$GPU_LOAD" != "N/A" ]] && GPU_LOAD_SUM=$((GPU_LOAD_SUM + GPU_LOAD))
    [[ "$GPU_FREQ" != "N/A" ]] && GPU_FREQ_SUM=$((GPU_FREQ_SUM + GPU_FREQ))
    if $NPU_PRESENT; then
        NPU_LOAD_SUM=$((NPU_LOAD_SUM + NPU_LOAD))
        NPU_FREQ_SUM=$((NPU_FREQ_SUM + NPU_FREQ))
    fi

    AVG_CPU_LOAD=$((CPU_LOAD_SUM / SAMPLES))
    AVG_CPU_FREQ=$((CPU_FREQ_SUM / SAMPLES))
    if $GPU_PRESENT; then AVG_GPU_LOAD=$((GPU_LOAD_SUM / SAMPLES)); else AVG_GPU_LOAD="N/A"; fi
    if $GPU_FREQ_PRESENT; then AVG_GPU_FREQ=$((GPU_FREQ_SUM / SAMPLES)); else AVG_GPU_FREQ="N/A"; fi
    if $NPU_PRESENT; then
        AVG_NPU_LOAD=$((NPU_LOAD_SUM / SAMPLES)); AVG_NPU_FREQ=$((NPU_FREQ_SUM / SAMPLES))
    else
        AVG_NPU_LOAD="N/A"; AVG_NPU_FREQ="N/A"
    fi

    # Mem peak/avg: parse 1-decimal GB into x10 int (assumes exactly one decimal place)
    SYS_USED_X10=$(( ${SYS_USED_GB%.*} * 10 + ${SYS_USED_GB#*.} ))
    SYS_MEM_SUM=$((SYS_MEM_SUM + SYS_USED_X10))
    [[ $SYS_USED_X10 -gt $SYS_MEM_PEAK ]] && SYS_MEM_PEAK=$SYS_USED_X10
    SYS_MEM_AVG=$(x10fmt $((SYS_MEM_SUM / SAMPLES)))
    SYS_MEM_PEAK_GB=$(x10fmt $SYS_MEM_PEAK)
    if $NPU_PRESENT; then
        [[ "$NPU_MEM_GB" == *.* ]] || NPU_MEM_GB="0.0"   # empty read -> 0, avoid parse crash
        NPU_USED_X10=$(( ${NPU_MEM_GB%.*} * 10 + ${NPU_MEM_GB#*.} ))
        NPU_MEM_SUM=$((NPU_MEM_SUM + NPU_USED_X10))
        [[ $NPU_USED_X10 -gt $NPU_MEM_PEAK ]] && NPU_MEM_PEAK=$NPU_USED_X10
        NPU_MEM_AVG=$(x10fmt $((NPU_MEM_SUM / SAMPLES)))
        NPU_MEM_PEAK_GB=$(x10fmt $NPU_MEM_PEAK)
    else
        NPU_MEM_AVG="N/A"; NPU_MEM_PEAK_GB="N/A"
    fi

    ELAPSED=$(printf '%02d:%02d:%02d' $((SECONDS/3600)) $((SECONDS%3600/60)) $((SECONDS%60)))

    # Repaint in place: 12 = lines printed below AND the reserve count above.
    printf "\033[12A"

    printf "\n"
    printf "  %-6s%6s%9s%11s%11s\n" "DEVICE" "UTIL" "FREQ" "AVG UTIL" "AVG FREQ"
    printf "  %-6s%6s%9s%11s%11s\n" "CPU" "$(pct "$CPU_LOAD")"   "$(mhz "$CPU_FREQ")" "$(pct "$AVG_CPU_LOAD")" "$(mhz "$AVG_CPU_FREQ")"
    printf "  %-6s%6s%9s%11s%11s\n" "GPU" "$(pct "$GPU_LOAD")"   "$(mhz "$GPU_FREQ")" "$(pct "$AVG_GPU_LOAD")" "$(mhz "$AVG_GPU_FREQ")"
    printf "  %-6s%6s%9s%11s%11s\n" "NPU" "$(pct "$NPU_LOAD")"   "$(mhz "$NPU_FREQ")" "$(pct "$AVG_NPU_LOAD")" "$(mhz "$AVG_NPU_FREQ")"
    printf "\n"
    printf "  %-13s%9s%9s%11s\n" "MEMORY" "USED" "AVG" "PEAK"
    printf "  %-13s%9s%9s%11s\n" "NPU (? GB)"         "$(gb "$NPU_MEM_GB")" "$(gb "$NPU_MEM_AVG")" "$(gb "$NPU_MEM_PEAK_GB")"
    printf "  %-13s%9s%9s%11s\n" "SYS ($SYS_TOTAL_GB GB)" "$(gb "$SYS_USED_GB")" "$(gb "$SYS_MEM_AVG")" "$(gb "$SYS_MEM_PEAK_GB")"
    printf "\n"
    echo " $RULE"
    printf "  Sampling: %ss  •  Samples: %d  •  Elapsed: %s  •  Start: %s\n" "$INTERVAL" "$SAMPLES" "$ELAPSED" "$START_TIME"

    WINDOW=$INTERVAL   # subsequent samples use the full requested interval
done
