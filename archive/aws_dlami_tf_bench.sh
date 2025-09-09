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

####### USAGE Examples #########
# bash aws_dlami_tf_bench.sh
# Runs Inference benchmark with resnet50 with BZ=32
#
# bash aws_dlami_tf_bench.sh -n "vgg16" -i "False"
# Runs Training benchmark with vgg16 with BZ=32
#
# bash aws_dlami_tf_bench.sh -n "vgg16"
# Runs Inference benchmark with vgg16 with BZ=32
#
# bash aws_dlami_tf_bench.sh -n "all" -i "False"
# Runs Training benchmark with inception3, resnet50, resnet152, vgg16 with
# BZ=32, 64, 128

# bash aws_dlami_tf_bench.sh -n "all"
# Runs Inference benchmark with inception3, resnet50, resnet152, vgg16 with
# BZ=32, 64, 128
#
# This script runs training with TensorFlow's CNN Benchmarks on AWS DLAMI and
# summarizes throughput increases when using Intel optimized TensorFlow.

TF_ENV_NAME=tensorflow_p36
# Set number of batches
num_batches=( 30 )

net_arg="resnet50"
inf_arg="True"

while getopts ":n:i:" arg; do
  case $arg in
    n) net_arg=$OPTARG;;
    i) inf_arg=$OPTARG;;
  esac
done

# Set the default and update with arg if passed
networks=( $net_arg )
batch_sizes=( 32 )

if [ "$net_arg" = "all" ]; then
  networks=( inception3 resnet50 resnet152 vgg16 )
  batch_sizes=( 32 64 128 )
fi

# Set the default and update with arg if passed
inf_flag="$inf_arg"

echo "Networks: ${networks}"
echo "Inference: $inf_flag"

# Check TF version so that we clone the right benchmarks
source activate ${TF_ENV_NAME}
export tfversion=$(python -c "import tensorflow as tf;print(tf.__version__)")
source deactivate
arr=(${tfversion//./ })  # Parse version and release
export version=${arr[0]}
export release=${arr[1]}

# Clone benchmark scripts for appropriate TF version
git clone -b cnn_tf_v${version}.${release}_compatible  https://github.com/tensorflow/benchmarks.git
cd benchmarks/scripts/tf_cnn_benchmarks
rm *.log # remove logs from any previous benchmark runs

# Install default tensorflow (non-Intel-optimized)
pip install tensorflow==${tfversion}

## Run benchmark scripts in the default environment
for network in "${networks[@]}" ; do
  for bs in "${batch_sizes[@]}"; do
    echo -e "\n\n #### Starting $network and batch size = $bs ####\n\n"

    time python tf_cnn_benchmarks.py \
    --data_format NHWC \
    --data_name imagenet \
    --device cpu \
    --model "$network" \
    --batch_size "$bs" \
    --num_batches "$num_batches" \
    --forward_only=$inf_flag \
    2>&1 | tee net_"$network"_bs_"$bs"_default.log

  done
done

## Run benchmark scripts in the Intel Optimized environment
source activate ${TF_ENV_NAME}

for network in "${networks[@]}" ; do
  for bs in "${batch_sizes[@]}"; do
    echo -e "\n\n #### Starting $network and batch size = $bs ####\n\n"

    time python tf_cnn_benchmarks.py \
    --data_format NCHW \
    --data_name imagenet \
    --device cpu \
    --mkl=True \
    --model "$network" \
    --batch_size "$bs" \
    --num_batches "$num_batches" \
    --forward_only=$inf_flag \
    2>&1 | tee net_"$network"_bs_"$bs"_optimized.log

  done
done

source deactivate

## Print a summary of training throughputs and relative speedups across all networks/batch sizes

speedup_track=0
runs=0

# Set headers
echo $'\n\n\n\n'
echo "######### Executive Summary #########"
echo $'\n'
echo "Inference: $inf_flag"
echo $'\n'
echo "Environment |  Network   | Batch Size | Images/Second"
echo "--------------------------------------------------------"
for network in "${networks[@]}" ; do
  for bs in "${batch_sizes[@]}"; do
    default_fps=$(grep  "total images/sec:"  net_"$network"_bs_"$bs"_default.log | cut -d ":" -f2 | xargs)
    optimized_fps=$(grep  "total images/sec:"  net_"$network"_bs_"$bs"_optimized.log | cut -d ":" -f2 | xargs)
    echo "Default     | $network |     $bs     | $default_fps"
    echo "Optimized   | $network |     $bs     | $optimized_fps"
    speedup=$((${optimized_fps%.*}/${default_fps%.*}))
    speedup_track=$((speedup_track + speedup))
    runs=$((runs+1))
  done
    echo -e "\n"
done

echo "#############################################"
echo "Average Intel Optimized speedup = $(($speedup_track / $runs))X"
echo "#############################################"
echo $'\n\n'
