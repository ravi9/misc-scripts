#!/bin/bash

cd
sudo yum -y update

# Install OV dependencies

sudo yum -y install \
      yum-utils \
      python3

sudo yum-config-manager --add-repo https://yum.repos.intel.com/openvino/2021/setup/intel-openvino-2021.repo
sudo rpm --import https://yum.repos.intel.com/openvino/2021/setup/RPM-GPG-KEY-INTEL-OPENVINO-2021
sudo yum list intel-openvino*2021.1*


sudo yum -y install \
        intel-openvino-runtime-centos7-2021.1.110.x86_64 \
        intel-openvino-model-optimizer-2021.1.110.x86_64 \
        intel-openvino-omz-tools-2021.1.110.x86_64 \
        intel-openvino-omz-dev-2021.1.110.x86_64


sudo pip3 install --upgrade pip

cd /opt/intel/openvino_2021/deployment_tools/model_optimizer && \
pip install -r requirements_tf.txt && \
pip install -r requirements_onnx.txt && \
pip install -r requirements_mxnet.txt && \
pip install mxnet -U && \
pip install tensorflow -U && \
pip install torch && \
pip install notebook progress tqdm matplotlib scipy



# Benchmark tools and others are missing, so this is a hacky fix.
cd
pip install gdown
gdown https://drive.google.com/uc?id=1WLNr69-e8-w1DzE2zbEICIdXJMKV_Hii
unzip -q ov-tools-2021.zip -d ov-tools-2021
cd ov-tools-2021
unzip -q 'deply_tools-tools-*.zip'
sudo cp -pra benchmark_tool cross_check_tool post_training_optimization_toolkit workbench /opt/intel/openvino_2021/deployment_tools/tools
unzip -q py-py3.6-ov-tools.zip -d py3.6-tools
unzip -q py-py3.7-ov-tools.zip -d py3.7-tools
sudo cp -pra py3.6-tools/tools /opt/intel/openvino_2021/python/python3.6/openvino
sudo cp -pra py3.7-tools/tools /opt/intel/openvino_2021/python/python3.7/openvino
cd
rm -rf ov-tools-2021.zip ov-tools-2021


cd
echo 'source /opt/intel/openvino_2021/bin/setupvars.sh' >> .bashrc

# Test Installation
source /opt/intel/openvino_2021/bin/setupvars.sh
python3 -c "from openvino.inference_engine import IENetwork, IECore, get_version as get_ov_version; print('openvino version: ' + get_ov_version())"

# Amazon Linux have opencv issues. libgtk cannot be installed on Amazon Linux 2
python3 -c "from openvino.tools.benchmark.main import main"

