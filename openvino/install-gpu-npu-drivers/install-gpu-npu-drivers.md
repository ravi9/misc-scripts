## Intel GPU and NPU Installations (Ubuntu 24.04)

- https://github.com/intel/linux-npu-driver/releases
- https://github.com/intel/compute-runtime/releases 
---
### Verify Kernel version

```bash
# Check kernel version
uname -a

# Recommended kernel is 6.17.0-20-generic or above and OS is Ubuntu 24.04 LTS
# Install atleast 6.17.0-20-generic kernel if needed
sudo apt install linux-image-6.17.0-20-generic linux-headers-6.17.0-20-generic linux-modules-extra-6.17.0-20-generic
sudo apt install linux-generic-hwe-24.04
sudo reboot

sudo apt update -y
sudo apt upgrade -y
sudo reboot
```
---
### Intel GPU and NPU Drivers Install Script: 

```bash
mkdir ~/gpu-npu-drivers
cd ~/gpu-npu-drivers

sudo apt update -y

sudo apt install ocl-icd-libopencl1 -y
sudo apt install libtbb12 -y
sudo apt install clinfo -y

# GPU driver v26.18.38308.1
mkdir neo
cd neo
wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.34.4/intel-igc-core-2_2.34.4+21428_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.34.4/intel-igc-opencl-2_2.34.4+21428_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/intel-ocloc-dbgsym_26.18.38308.1-0_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/intel-ocloc_26.18.38308.1-0_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/intel-opencl-icd-dbgsym_26.18.38308.1-0_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/intel-opencl-icd_26.18.38308.1-0_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/libigdgmm12_22.10.0_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/libze-intel-gpu1-dbgsym_26.18.38308.1-0_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/libze-intel-gpu1_26.18.38308.1-0_amd64.deb

sudo dpkg -i *.deb

sudo apt-get install -f -y  # Fixes missing foundational dependencies

# NPU v1.33.0
cd ~/gpu-npu-drivers
mkdir npu-drivers
cd npu-drivers

sudo dpkg --purge --force-remove-reinstreq intel-driver-compiler-npu intel-fw-npu intel-level-zero-npu intel-level-zero-npu-dbgsym

wget https://github.com/intel/linux-npu-driver/releases/download/v1.33.0/linux-npu-driver-v1.33.0.20260529-26625960453-ubuntu2404.tar.gz
tar -xf linux-npu-driver-v1.33.0.20260529-26625960453-ubuntu2404.tar.gz

sudo apt update -y
sudo dpkg -i *.deb

wget https://snapshot.ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/20260324T100000Z/pool/main/l/level-zero-loader/libze1_1.27.0-1~24.04~ppa2_amd64.deb
sudo dpkg -i libze1_*.deb

# Force overwrite the old system level-zero file if above fails.
sudo dpkg -i --force-overwrite ./libze1_*.deb

# If permissions are missing, add your user to the necessary groups:
sudo usermod -aG render,video $USER

# Logout and Login

```
#### Verify Driver versions
```bash
dpkg -l | grep -i intel | grep -E 'igc|opencl|ocloc|igdgmm|level-zero|gpu'
```
Sample Output:
```console
$ dpkg -l | grep -i intel | grep -E 'igc|opencl|ocloc|igdgmm|level-zero|gpu'
ii  intel-gpu-tools                               1.28-1ubuntu2                            amd64        tools for debugging the Intel graphics driver
ii  intel-igc-core-2                              2.34.4                                   amd64        Intel(R) Graphics Compiler for OpenCL(TM)
ii  intel-igc-opencl-2                            2.34.4                                   amd64        Intel(R) Graphics Compiler for OpenCL(TM)
ii  intel-level-zero-npu                          1.33.0.20260529-26625960453~ubuntu24.04  amd64        Intel(R) Level Zero Driver for NPU hardware
ii  intel-ocloc                                   26.18.38308.1-0                          amd64        Tool for managing Intel Compute GPU devicebinary format
ii  intel-opencl-icd                              26.18.38308.1-0                          amd64        Intel graphics compute runtime for OpenCL
ii  libigdgmm12:amd64                             22.10.0                                  amd64        Intel Graphics Memory Management Library -- shared library
ii  libze-intel-gpu1                              26.18.38308.1-0                          amd64        Intel(R) Graphics Compute Runtime for oneAPI Level Zero.
```
---
### Validation and Testing:

#### Intel Devices Discovery with OpenVINO
```bash
# sudo apt install python3.12-venv -y

python3 -m venv ov-test-env
source ov-test-env/bin/activate
pip install openvino
python -c "import openvino as ov; print(ov.Core().available_devices)"
# Print with device id and full name
python -c "import openvino as ov; core = ov.Core(); [print(f'Device: {d} - {core.get_property(d, \"FULL_DEVICE_NAME\")}') for d in core.available_devices]"
```
---
#### GPU Verification Methods:
```bash
# Option 1. Verify GPU hardware is detected by the PCI bus
lspci -k | grep -A 3 -Ei "VGA|DISPLAY|3D"

# Option 2. Verify GPU hardware is detected by the PCI bus
# Look for "Kernel driver in use" to ensure it's bound to 'i915' or 'xe'.
lspci -k | grep -A 3 -Ei "VGA|DISPLAY"
# or using the common Intel address 00:02.0
lspci -k -s 00:02.0

# Option 3. Check if Intel Compute Runtime (OpenCL/Level Zero) libraries are installed
dpkg -l | grep -i intel | grep -E 'igc|opencl|ocloc|igdgmm|level-zero'

# Option 4. Confirm the OS has created the GPU render node
# 'renderD128' is typically the node used for hardware acceleration and compute tasks.
ls /dev/dri/render*

# Option 5. Verify OpenCL can see the Intel Graphics hardware. Install: sudo apt install clinfo -y
clinfo -l

# Check if your user has permission to access the GPU (render group)
groups | grep -E "render|video"
```

#### NPU Verification Methods:
```bash
# Option 1. Verify NPU hardware is detected by the PCI bus
lspci -k | grep -A 3 -i "NPU"

# Option 2. Confirm the OS has created the NPU acceleration node
ls /dev/accel/accel*

# Check if your user has permission to access the NPU (accel group)
# Note: On many systems, NPU access requires being in the 'accel' or 'render' group
groups | grep -E "accel|render"
```

---
### Monitoring GPU, NPU utilization

### Using bash script
- Monitor CPU, GPU, NPU, and memory usage with [print_cpu_gpu_npu_usage.sh](https://github.com/ravi9/misc-scripts/blob/main/openvino/install-gpu-npu-drivers/print_cpu_gpu_npu_usage.sh).
```console
$ bash print_cpu_gpu_npu_util.sh 
=========================================
 System Monitor
-----------------------------------------
 OS:  Ubuntu 24.04.4 LTS | Kernel: 6.17.0-20-generic
-----------------------------------------
 CPU: Intel(R) Core(TM) Ultra 5 236V
      Cores: 8 | Threads/Core: 1 | Max Clock: 4700 MHz
      L3 Cache: 8 MiB (1 instance)
 GPU: Intel(R) Arc(TM) Graphics
      Xe Cores: 7 | EUs: 56 | Vec Engines: 112
      Max Clock: 1850 MHz | Memory: 13.92GiB
 NPU: Intel Corporation Lunar Lake NPU (rev 04)
      Max Clock: 1950 MHz
 RAM: 15 GB
-----------------------------------------
 Driver Versions:
      Intel(R) Graphics Compiler for OpenCL(TM)          intel-igc-core-2       2.28.4
      Intel graphics compute runtime for OpenCL          intel-opencl-icd       26.05.37020.3-0
      Intel(R) Graphics Compute Runtime for oneAPI Level Zero. libze-intel-gpu1       26.05.37020.3-0
      Intel(R) Level Zero Driver for NPU hardware        intel-level-zero-npu   1.30.0.20260311-22963593310~ubuntu24.04
=========================================
-------------------------------------------
          util      freq    avg  avg freq
CPU:       0%    530 MHz     1%    544 MHz
GPU:       0%    750 MHz     0%    750 MHz
NPU:       0%      0 MHz     0%      0 MHz
NPU Mem:  0.1 GB
Sys Mem:  2.2 GB / 15 GB  [1s interval, 32 samples]
-------------------------------------------
```

- Monitor CPU, GPU, NPU, and memory usage with [print_v1_cpu_gpu_npu_usage.sh](https://github.com/ravi9/misc-scripts/blob/main/openvino/install-gpu-npu-drivers/print_v1_cpu_gpu_npu_usage.sh).
```console
$ bash print_v1_cpu_gpu_npu_util.sh 
=========================================
 System Monitor
-----------------------------------------
 CPU: Intel(R) Core(TM) Ultra 5 236V (8 cores)
 GPU: Intel Corporation Lunar Lake [Intel Graphics] (rev 04) (driver: xe_idle)
 RAM: 15 GB
=========================================
-------------------------------------------
          util      freq    avg  avg freq
CPU:       2%    521 MHz     7%   1567 MHz
GPU:       0%    750 MHz    25%   1412 MHz
NPU:       0%      0 MHz     0%      0 MHz
NPU Mem:  0.1 GB
Sys Mem:  2.1 GB / 15 GB  (samples: 45)
-------------------------------------------
```

### Using nvtop
```bash
sudo snap install nvtop
sudo snap connect nvtop:hardware-observe
sudo snap connect nvtop:system-observe
sudo nvtop
```

### Using intel_gpu_top
```bash
# Only for older Intel GPUs using  i915 drivers. For recent GPUs using xe drivers use nvtop
sudo apt update && sudo apt install intel-gpu-tools
sudo intel_gpu_top
```

### Using Command line snippets
Paste it directly in the terminal !
- Xe GPU utilization
```bash
IDLE_PATH="/sys/class/drm/card0/device/tile0/gt0/gtidle/idle_residency_ms"
FREQ_PATH="/sys/class/drm/card0/device/tile0/gt0/freq0/cur_freq"
echo "---------------------------------"
while true; do
  OLD=$(cat "$IDLE_PATH" 2>/dev/null || echo 0)
  sleep 1
  NEW=$(cat "$IDLE_PATH" 2>/dev/null || echo 0)
  # Calc Load: 100% - (Delta_Idle_ms / 1000ms * 100)
  IDLE_PCT=$(( (NEW - OLD) / 10 ))
  LOAD=$(( 100 - IDLE_PCT ))
  [[ $LOAD -lt 0 ]] && LOAD=0
  
  FREQ=$(cat "$FREQ_PATH" 2>/dev/null || echo "N/A")
  read T A < <(awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{print t" "a}' /proc/meminfo)
  USED_GB=$(awk "BEGIN {printf \"%.1f\", ($T - $A)/1024/1024}")
  TOT_GB=$(awk "BEGIN {printf \"%.0f\", $T/1024/1024}")
  printf "\rGPU (xe): %3d%% @ %4s MHz | Sys Mem: %4s GB / %2s GB" "$LOAD" "$FREQ" "$USED_GB" "$TOT_GB"
done
```

- i915 driver - Older Intel GPUs
```bash
BUSY_PATH="/sys/class/drm/card0/engine/rcs0/busy_time_ns"
FREQ_PATH="/sys/class/drm/card0/gt_cur_freq_mhz"
echo "---------------------------------"
while true; do
  OLD=$(cat "$BUSY_PATH" 2>/dev/null || echo 0)
  sleep 1
  NEW=$(cat "$BUSY_PATH" 2>/dev/null || echo 0)
  # Calc Load: (Delta_ns / 1s_ns) * 100
  LOAD=$(( (NEW - OLD) / 10000000 ))
  
  FREQ=$(cat "$FREQ_PATH" 2>/dev/null || echo "N/A")
  read T A < <(awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{print t" "a}' /proc/meminfo)
  USED_GB=$(awk "BEGIN {printf \"%.1f\", ($T - $A)/1024/1024}")
  TOT_GB=$(awk "BEGIN {printf \"%.0f\", $T/1024/1024}")
  printf "\rGPU (i915): %3d%% @ %4s MHz | Sys Mem: %4s GB / %2s GB" "$LOAD" "$FREQ" "$USED_GB" "$TOT_GB"
done
```

- NPU utilization 
```bash
echo "---------------------------------"
while true; do
  OLD=$(cat /sys/class/accel/accel0/device/npu_busy_time_us)
  sleep 1
  NEW=$(cat /sys/class/accel/accel0/device/npu_busy_time_us)
  LOAD=$(( (NEW - OLD) / 10000 ))
  FREQ=$(cat /sys/class/accel/accel0/device/npu_current_frequency_mhz)
  MEM_GB=$(awk '{printf "%.1f", $1/1024/1024/1024}' /sys/class/accel/accel0/device/npu_memory_utilization)
  read T A < <(awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{print t" "a}' /proc/meminfo)
  USED_GB=$(awk "BEGIN {printf \"%.1f\", ($T - $A)/1024/1024}")
  TOT_GB=$(awk "BEGIN {printf \"%.0f\", $T/1024/1024}")
  printf "\rNPU: %3d%% @ %4s MHz (%4s GB) | Sys Mem: %4s GB / %2s GB" "$LOAD" "$FREQ" "$MEM_GB" "$USED_GB" "$TOT_GB"
done
```
