#!/bin/bash

python3 -m venv ov-env

source ov-env/bin/activate

pip install openvino
mkdir test-ov && cd test-ov

mkdir -p yolo-v2-tiny-vehicle-detection-0001/FP16-INT8
cd yolo-v2-tiny-vehicle-detection-0001/FP16-INT8
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/yolo-v2-tiny-vehicle-detection-0001/FP16-INT8/yolo-v2-tiny-vehicle-detection-0001.xml
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/yolo-v2-tiny-vehicle-detection-0001/FP16-INT8/yolo-v2-tiny-vehicle-detection-0001.bin

benchmark_app -m yolo-v2-tiny-vehicle-detection-0001.xml -t 5 -d CPU

# omz_downloader --name efficientnet-b0-pytorch
# omz_converter --name efficientnet-b0-pytorch
# benchmark_app -m public/efficientnet-b0-pytorch/efficientnet-b0.onnx -t 5

