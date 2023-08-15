#!/bin/bash

conda create -n ov-test python=3.8 -y

conda activate ov-test
pip install openvino-dev[tensorflow,pytorch,onnx]
mkdir test-ov && cd test-ov

omz_downloader --name yolo-v2-tiny-vehicle-detection-0001
benchmark_app -m intel/yolo-v2-tiny-vehicle-detection-0001/FP16-INT8/yolo-v2-tiny-vehicle-detection-0001.xml -t 5

omz_downloader --name efficientnet-b0-pytorch
omz_converter --name efficientnet-b0-pytorch
benchmark_app -m public/efficientnet-b0-pytorch/efficientnet-b0.onnx -t 5

