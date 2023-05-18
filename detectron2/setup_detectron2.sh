# This code is tested on AWS GPU instance with AMI: Deep Learning AMI GPU PyTorch 2.0.0 (Ubuntu 20.04) 20230517
# AMI: https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-pytorch-2-0-ubuntu-20-04/
# Any GPU instance is fine. I used: g4dn.2xlarge

# Prereqs: Before running the script initiaze the conda env running the following 2 cmds:
# conda init bash
# source ~/.bashrc

# Start of the script
conda create -n det2 python=3.8
conda activate det2
mkdir 7eleven
cd 7eleven/

pip install torch==2.0 torchvision==0.15.1 onnx opencv-python

git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# Download coco dataset for detectron2
cd ~/7eleven/detectron2/datasets/
mkdir coco
cd coco
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
unzip annotations_trainval2017.zip
unzip train2017.zip
unzip val2017.zip
#

cd ~/7eleven/detectron2/
tools/deploy/export_model.py -h

mkdir rp-output

tools/deploy/export_model.py  \
--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml     \
--export-method tracing     \
--format onnx     \
--output rp-output     \
MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl     \
MODEL.DEVICE cpu

pip install openvino-dev

mo -m rp-output/model.onnx -o IR/

# You can also specify shape if needed
# The default input shape is [C,H,W] [3,800,1202]
# The H can be one of these values (640, 672, 704, 736, 768, 800) Refer: https://github.com/facebookresearch/detectron2/blob/main/configs/Base-RCNN-FPN.yaml#L41
# THe W can be one of these (1202, 1200)
# We can specify the required shape in the benchmark_app as follows:
benchmark_app -m rp-output/IR/model.xml -t 10 -s [3,736,1200]

