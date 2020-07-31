cd

sudo yum -y update

sudo yum -y groupinstall "Development tools"
sudo amazon-linux-extras install -y epel

# Install OV dependencies

 # CentOS 7.x
sudo yum -y install \
      gcc* \
      yum-utils \
      wget \
      which \
      boost-devel \
      glibc-static \
      glibc-devel \
      libstdc++-static \
      libstdc++-devel \
      libstdc++.i686 \
      libgcc.i686 \
      openblas-devel \
      libusbx-devel \
      gstreamer1 \
      gstreamer1-plugins-base \
      openssl-devel \
      python3 \
      python3-devel \
      opencv \
      opencv-devel \
      opencv-python

# Install ffmpeg
#  rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-1.el7.nux.noarch.rpm
#  sudo yum -y install libwayland-client
#  wget http://mirror.centos.org/centos/7/os/x86_64/Packages/libva-1.8.3-1.el7.x86_64.rpm
#  rpm -i libva-1.8.3-1.el7.x86_64.rpm
#  sudo yum -y install ffmpeg

sudo yum-config-manager --add-repo https://yum.repos.intel.com/openvino/2020/setup/intel-openvino-2020.repo
sudo rpm --import https://yum.repos.intel.com/openvino/2020/setup/RPM-GPG-KEY-INTEL-OPENVINO-2020
sudo yum repolist | grep -i openvino
sudo yum list intel-openvino-runtime*
sudo yum list intel-openvino*2020.4*


sudo yum -y install \
    intel-openvino-runtime-centos7-2020.4.287.x86_64 \
    intel-openvino-model-optimizer-2020.4.287.x86_64 \
    intel-openvino-omz-tools-2020.4.287.x86_64 \
    intel-openvino-omz-dev-2020.4.287.x86_64


pip3 install --upgrade pip

cd /opt/intel/openvino/deployment_tools/model_optimizer && \
pip install -r requirements_tf.txt && \
pip uninstall -y tensorflow && \
pip install tensorflow==1.15.3 && \
pip install -r requirements_onnx.txt && \
pip install -r requirements_mxnet.txt && \
pip install ipython progress

cd && \


# Benchmark tools and others are missing, so this is a hacky fix.
cd
pip install gdown
gdown https://drive.google.com/uc?id=1izrTn-4iVteyYYdS6ebrsT-vDSY--pv5
unzip ov-tools.zip 
cd ov-tools/
unzip tools.zip 
unzip workbench.zip 
unzip post_training_optimization_toolkit.zip 
unzip benchmark_tool.zip 
unzip cross_check_tool.zip
sudo cp -pra benchmark_tool cross_check_tool post_training_optimization_toolkit workbench /opt/intel/openvino/deployment_tools/tools
sudo cp -pra tools /opt/intel/openvino/python/python3.7/openvino
cd
rm -rf ov-tools.zip ov-tools

cd
echo 'source /opt/intel/openvino/bin/setupvars.sh' >> .bashrc

# Test Installation
source /opt/intel/openvino/bin/setupvars.sh
python3 -c "from openvino.inference_engine import IENetwork, IECore, get_version as get_ov_version; print('openvino version: ' + get_ov_version())"
python3 -c "from openvino.tools.benchmark.main import main"
