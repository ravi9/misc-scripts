# !/bin/bash
apt -y update
apt install -y software-properties-common \
               wget \
               vim \
               python3-pip \
               libgtk-3-0 \
               gnupg2

wget https://apt.repos.intel.com/openvino/2020/GPG-PUB-KEY-INTEL-OPENVINO-2020
apt-key add GPG-PUB-KEY-INTEL-OPENVINO-2020
#echo "deb https://apt.repos.intel.com/openvino/2020 all main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2020.list
echo "deb https://apt.repos.intel.com/openvino/2020 all main" | tee /etc/apt/sources.list.d/intel-openvino-2020.list

apt -y update

apt-cache search openvino | grep 2020.4

# apt install -y intel-openvino-runtime-ubuntu18-2020.4.287
apt install -y intel-openvino-dev-ubuntu18-2020.4.287

cd /opt/intel/openvino/deployment_tools/model_optimizer/
pip3 install -r requirements_tf.txt
pip3 install ipython progress

source /opt/intel/openvino/bin/setupvars.sh

python3 -c "from openvino.inference_engine import IECore, get_version as get_ov_version; print('openvino version: ' + get_ov_version())"

python3 -c "from openvino.tools.benchmark.main import main"
