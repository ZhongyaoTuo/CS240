# NCCL configuration
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_NET_GDR_LEVEL=3
# export NCCL_TOPO_FILE=/tmp/topo.txt

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=7200

# args
name="training_512_mv"
expname="debug_ex2" 
config_file=configs/${name}/${expname}.yaml

# save root dir for logs, checkpoints, tensorboard record, etc.
save_root="/workspace/results_train/${expname}"

mkdir -p $save_root/$name
HOST_GPU_NUM=1

# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch \
# --nproc_per_node=$HOST_GPU_NUM --nnodes=1 --master_addr=127.0.0.2 --master_port=12353 --node_rank=0 \
# ./main/vae_prepare.py \
# --base $config_file \
# --train \
# --name $expname \
# --logdir $save_root \
# --devices $HOST_GPU_NUM \
# lightning.trainer.num_nodes=1

## run
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch \
--nproc_per_node=$HOST_GPU_NUM --nnodes=1 --master_addr=127.0.0.2 --master_port=12353 --node_rank=0 \
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