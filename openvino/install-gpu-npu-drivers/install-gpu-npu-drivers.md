## Intel GPU and NPU Installations (Ubuntu 24.04)

- https://github.com/intel/linux-npu-driver/releases
- https://github.com/intel/compute-runtime/releases 

```bash
mkdir ~/installers-drivers
cd ~/installers-drivers

sudo apt update -y

mkdir neo
cd neo
wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.18.5/intel-igc-core-2_2.18.5+19820_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.18.5/intel-igc-opencl-2_2.18.5+19820_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/25.35.35096.9/intel-ocloc-dbgsym_25.35.35096.9-0_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/25.35.35096.9/intel-ocloc_25.35.35096.9-0_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/25.35.35096.9/intel-opencl-icd-dbgsym_25.35.35096.9-0_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/25.35.35096.9/intel-opencl-icd_25.35.35096.9-0_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/25.35.35096.9/libigdgmm12_22.8.1_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/25.35.35096.9/libze-intel-gpu1-dbgsym_25.35.35096.9-0_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/25.35.35096.9/libze-intel-gpu1_25.35.35096.9-0_amd64.deb

sudo apt install ocl-icd-libopencl1 -y

sudo dpkg -i *.deb

# NPU
mkdir npu 
cd npu
wget https://github.com/intel/linux-npu-driver/releases/download/v1.23.0/linux-npu-driver-v1.23.0.20250827-17270089246-ubuntu2404.tar.gz
tar -xf linux-npu-driver-v1.23.0.20250827-17270089246-ubuntu2404.tar.gz

sudo apt install libtbb12 -y

sudo dpkg -i *.deb

sudo usermod -aG video $USER
sudo usermod -aG render $USER

# Logout and Login or Restart

```

### Test:

```bash
python3 -m venv ov-test-env
source ov-test-env/bin/activate
pip install openvino-genai
benchmark_app -h
```
