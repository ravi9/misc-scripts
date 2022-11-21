# Installing OpenVINO in Ubuntu 20.04 Docker

apt -y update

# Set timezone to avoid user prompt when installing tzdata (libgtk-3-0 dependency)
ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
              
apt install -y build-essential \
               wget \
               vim \
               python3-pip \
               libgtk-3-0 \
               libgl1-mesa-glx
               
pip3 install --upgrade pip

pip install openvino-dev[tensorflow2,pytorch,onnx]

# Test Installation
python3 -c "from openvino.inference_engine import IECore, get_version as get_ov_version; print('openvino version: ' + get_ov_version())"

python3 -c "from openvino.tools.benchmark.main import main"

mo -h

benchmark_app -h
