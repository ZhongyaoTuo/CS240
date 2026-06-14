#!/bin/bash


export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=7200


name="base"
expname=controlnet_${name}
config_file=configs/control/${name}.yaml

save_root=/workspace/results_train/controlnet_${name}

mkdir -p $save_root/$name
HOST_GPU_NUM=1

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch \
--nproc_per_node=$HOST_GPU_NUM --nnodes=1 --master_addr=127.0.0.1 --master_port=12352 --node_rank=0 \
./main/trainer.py \
--base $config_file \
--train \
--name $expname \
--logdir $save_root \
--devices $HOST_GPU_NUM \
lightning.trainer.num_nodes=1


