## Intel GPU and NPU Installations (Ubuntu 24.04)

- https://github.com/intel/linux-npu-driver/releases
- https://github.com/intel/compute-runtime/releases 

```bash
mkdir ~/gpu-npu-drivers
cd ~/gpu-npu-drivers

sudo apt update -y

sudo apt install ocl-icd-libopencl1 -y
sudo apt install libtbb12 -y
sudo apt install clinfo -y

# GPU driver v26.14.37833.4
mkdir neo
cd neo
wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.32.7/intel-igc-core-2_2.32.7+21184_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.32.7/intel-igc-opencl-2_2.32.7+21184_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/26.14.37833.4/intel-ocloc-dbgsym_26.14.37833.4-0_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/26.14.37833.4/intel-ocloc_26.14.37833.4-0_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/26.14.37833.4/intel-opencl-icd-dbgsym_26.14.37833.4-0_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/26.14.37833.4/intel-opencl-icd_26.14.37833.4-0_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/26.14.37833.4/libigdgmm12_22.9.0_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/26.14.37833.4/libze-intel-gpu1-dbgsym_26.14.37833.4-0_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/26.14.37833.4/libze-intel-gpu1_26.14.37833.4-0_amd64.deb

sudo dpkg -i *.deb

# NPU v1.32.1
mkdir npu-drivers
cd npu-drivers

wget https://github.com/intel/linux-npu-driver/releases/download/v1.32.1/linux-npu-driver-v1.32.1.20260422-24767473183-ubuntu2404.tar.gz
tar -xf linux-npu-driver-v1.32.1.20260422-24767473183-ubuntu2404.tar.gz

sudo apt update -y
sudo apt install ./intel-*.deb -y

wget https://snapshot.ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/20260324T100000Z/pool/main/l/level-zero-loader/libze1_1.27.0-1~24.04~ppa2_amd64.deb
sudo apt install ./libze1_*.deb

# If permissions are missing, add your user to the necessary groups:
sudo usermod -aG render,video,accel $USER

# Logout and Login

```

## Validation and Testing:

### GPU Verification Methods:
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

### NPU Verification Methods:
```bash
# Option 1. Verify NPU hardware is detected by the PCI bus
lspci -k | grep -A 3 -i "NPU"

# Option 2. Confirm the OS has created the NPU acceleration node
ls /dev/accel/accel*

# Check if your user has permission to access the NPU (accel group)
# Note: On many systems, NPU access requires being in the 'accel' or 'render' group
groups | grep -E "accel|render"
```

### OpenVINO Device Verification
```bash
# sudo apt install python3.12-venv -y

python3 -m venv ov-test-env
source ov-test-env/bin/activate
pip install openvino
python -c "import openvino as ov; print(ov.Core().available_devices)"
# Print with device id and full name
python -c "import openvino as ov; core = ov.Core(); [print(f'Device: {d} - {core.get_property(d, \"FULL_DEVICE_NAME\")}') for d in core.available_devices]"
```

## Monitoring GPU, NPU utilization

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

### NPU utilization 
```bash
while true; do
  OLD=$(cat /sys/class/accel/accel0/device/npu_busy_time_us)
  sleep 1
  NEW=$(cat /sys/class/accel/accel0/device/npu_busy_time_us)
  DIFF=$(( (NEW - OLD) / 10000 ))
  FREQ=$(cat /sys/class/accel/accel0/device/npu_current_frequency_mhz)
  echo "NPU Load: $DIFF% | Current Freq: ${FREQ}MHz"
done
```
