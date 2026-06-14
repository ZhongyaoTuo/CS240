#!/bin/bash
# NCCL configuration
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_NET_GDR_LEVEL=3
# export NCCL_TOPO_FILE=/tmp/topo.txt
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=7200
# export WANDB__SERVICE_WAIT=500
export WANDB_KEY=84528bb21f4d95cd446df8d1e1eb965f4d47a942
export WANDB_API_KEY=84528bb21f4d95cd446df8d1e1eb965f4d47a942

# args
name="training_512_mv"
expname="run1_relpose_re_dc" 
config_file=configs/${name}/${expname}.yaml

# save root dir for logs, checkpoints, tensorboard record, etc.
save_root=./enhancer_logs/${expname}_0130

mkdir -p $save_root/$name
HOST_GPU_NUM=3

## run
python3 -m torch.distributed.launch \
--nproc_per_node=$HOST_GPU_NUM --nnodes=1 --node_rank=0 \
./main/trainer.py \
--base $config_file \
--train \
--name $expname \
--logdir $save_root \
--devices $HOST_GPU_NUM \
lightning.trainer.num_nodes=1

## debugging
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch \
# --nproc_per_node=4 --nnodes=1 --master_addr=127.0.0.1 --master_port=12352 --node_rank=0 \
# ./main/trainer.py \
# --base $config_file \
# --train \
# --name $name \
# --logdir $save_root \
# --devices 4 \
# lightning.trainer.num_nodes=1