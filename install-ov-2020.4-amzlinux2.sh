#!/bin/bash
#
# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cd
sudo yum -y update

# Install OV dependencies

sudo yum -y install \
      yum-utils \
      python3 

sudo yum-config-manager --add-repo https://yum.repos.intel.com/openvino/2020/setup/intel-openvino-2020.repo
sudo rpm --import https://yum.repos.intel.com/openvino/2020/setup/RPM-GPG-KEY-INTEL-OPENVINO-2020
sudo yum list intel-openvino*2020.4*


sudo yum -y install \
    intel-openvino-runtime-centos7-2020.4.287.x86_64 \
    intel-openvino-model-optimizer-2020.4.287.x86_64 \
    intel-openvino-omz-tools-2020.4.287.x86_64 \
    intel-openvino-omz-dev-2020.4.287.x86_64


sudo pip3 install --upgrade pip

cd /opt/intel/openvino/deployment_tools/model_optimizer && \
pip install -r requirements_tf.txt && \
pip install -r requirements_onnx.txt && \
pip install -r requirements_mxnet.txt && \
pip install ipython progress tqdm matplotlib torch

cd
echo 'source /opt/intel/openvino/bin/setupvars.sh' >> .bashrc

# Test Installation
source /opt/intel/openvino/bin/setupvars.sh
python3 -c "from openvino.inference_engine import IENetwork, IECore"

