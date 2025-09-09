#!/bin/bash

#usage: ./run-multinode-tf-cnn-bench.sh <NUM_NODES> <WORKERS_PER_SOCKET> <batch_size_per_worker>
#Ex: ./run-multinode-tf-cnn-bench.sh 2 1 64

PATH_TO_SINGULARITY="/usr/local/bin/singularity"
PATH_TO_SIMG="/mnt/shared/TensorFlow/tf-horovod-gcc-impi2019-libfabric-mlnx.simg"

#SING_EXEC_CMD="${PATH_TO_SINGULARITY} exec --bind /local/path/to/TF_Records:/image/path/to/TF_Records ${PATH_TO_SIMG}"
SING_EXEC_CMD="${PATH_TO_SINGULARITY} exec ${PATH_TO_SIMG}"

PATH_TO_SIMG_TF_BENCH="/opt/tensorflow-benchmarks"

#source /opt/intel/impi/2019.0.117/intel64/bin/mpivars.sh -ofi_internal=0
export PATH=/opt/intel/impi/2019.0.117/intel64/bin:$PATH
export LD_LIBRARY_PATH=/opt/intel/impi/2019.0.117/intel64/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/libfabric-debug/lib:$LD_LIBRARY_PATH

echo $LD_LIBRARY_PATH
#export FI_LOG_LEVEL=debug

HOSTNAMES="myNode0,myNode1"

#HOSTNAMES=`hostname`

NUM_NODES=${1}
WORKERS_PER_SOCKET=${2}
BATCH_SIZE=${3}

INTER_T=2

NUM_SOCKETS=`lscpu | grep "Socket(s)" | cut -d':' -f2 | xargs`
CORES_PER_SOCKET=`lscpu | grep "Core(s) per socket" | cut -d':' -f2 | xargs`

CORES_PER_WORKER=$((CORES_PER_SOCKET / WORKERS_PER_SOCKET))
INTRA_T=$CORES_PER_WORKER
OMP_NUM_THREADS=$((INTRA_T / INTER_T))
WORKERS_PER_NODE=$((WORKERS_PER_SOCKET * NUM_SOCKETS))
TOTAL_WORKERS=$((NUM_NODES * WORKERS_PER_NODE))

echo "CORES_PER_WORKER: $CORES_PER_WORKER"
echo "NUM_INTRA_THREADS: $INTRA_T"
echo "NUM_INTER_THREADS: $INTER_T"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "WORKERS_PER_NODE: $WORKERS_PER_NODE"
echo "TOTAL_WORKERS: $TOTAL_WORKERS"

export OMP_NUM_THREADS=$OMP_NUM_THREADS

args=" \
 --batch_size=${BATCH_SIZE} \
 --num_batches=60 \
 --model=resnet50 \
 --num_intra_threads=${INTRA_T} \
 --num_inter_threads=${INTER_T} \
 --kmp_blocktime=1 \
 --display_every=10 \
 --data_format=NCHW \
 --optimizer=momentum \
 --forward_only=False \
 --device=cpu \
 --mkl=TRUE \
 --variable_update=horovod \
 --horovod_device=cpu \
 --local_parameter_device=cpu \
 --data_dir=/mnt/shared/TensorFlow/ilsvrc2012_tfrecords_20of1024 \
 --data_name=imagenet "

echo "Common Args: $args"

echo "Starting in 5sec..."
sleep 5

mpiexec.hydra \
 -hosts ${HOSTNAMES} \
 -genv I_MPI_FABRICS 'shm:ofi' \
 -genv I_MPI_OFI_PROVIDER 'verbs' \
 -genv I_MPI_DEBUG 5 \
 -np ${TOTAL_WORKERS} \
 -ppn ${WORKERS_PER_NODE} \
 -genv OMP_NUM_THREADS $OMP_NUM_THREADS \
 -genv HOROVOD_FUSION_THRESHOLD 134217728 \
 ${SING_EXEC_CMD} \
 python ${PATH_TO_SIMG_TF_BENCH}/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
 $args
