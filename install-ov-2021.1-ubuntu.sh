# !/bin/bash

sudo apt -y update
sudo apt install -y software-properties-common \
               wget \
               vim \
               python3-pip \
               libgtk-3-0 \
               gnupg2

wget https://apt.repos.intel.com/openvino/2021/GPG-PUB-KEY-INTEL-OPENVINO-2021
sudo apt-key add GPG-PUB-KEY-INTEL-OPENVINO-2021
sudo echo "deb https://apt.repos.intel.com/openvino/2021 all main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2021.list

sudo apt -y update

apt-cache search openvino | grep 2021.1

# apt install -y intel-openvino-runtime-ubuntu18-2021.1.110
sudo apt install -y intel-openvino-dev-ubuntu18-2021.1.110

pip3 install --upgrade pip

cd /opt/intel/openvino_2021/deployment_tools/model_optimizer/
pip install --no-cache-dir  -r requirements_tf.txt
pip install --no-cache-dir tensorflow -U
pip install --no-cache-dir -r requirements_onnx.txt
pip install --no-cache-dir -r requirements_mxnet.txt
pip install --no-cache-dir mxnet -U
pip install --no-cache-dir torch
pip install --no-cache-dir notebook progress tqdm matplotlib

echo 'source /opt/intel/openvino_2021/bin/setupvars.sh' >> /root/.bashrc

# Test Installation
source /opt/intel/openvino_2021/bin/setupvars.sh

python3 -c "from openvino.inference_engine import IECore, get_version as get_ov_version; print('openvino version: ' + get_ov_version())"

python3 -c "from openvino.tools.benchmark.main import main"
