## Intel GPU and NPU Installations (Ubuntu 24.04)

- https://github.com/intel/linux-npu-driver/releases
- https://github.com/intel/compute-runtime/releases 

```bash

mkdir ~/gpu-npu-drivers
cd ~/gpu-npu-drivers

sudo apt update -y

sudo apt install ocl-icd-libopencl1 -y

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

sudo apt install libtbb12 -y

sudo dpkg -i *.deb

sudo usermod -aG video $USER
sudo usermod -aG render $USER

# Logout and Login or Restart

```

### Test:

```bash
dpkg -l | grep -i intel | grep -E 'igc|opencl|ocloc|igdgmm|level-zero'
```

```bash
python3 -m venv ov-test-env
source ov-test-env/bin/activate
pip install openvino-genai
benchmark_app -h
```
