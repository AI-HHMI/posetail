#!/usr/bin/env sh

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

source setup_env_vars.sh

d=$(date +d%dh%H)


# bsub -J en3-$d -e ~/logs/posetail/en3-$d.err -o ~/logs/posetail/en3-$d.out \
#     -n 48 -q gpu_h200 -R "span[hosts=1]" -gpu "num=4" -W 36:00 \
#     pixi run python train.py --config-path configs/config_encoder_3d_finetuning_nograd.toml \
#     --devices -1 --strategy ddp_find_unused_parameters_true

# bsub -J enf-$d -e ~/logs/posetail/enf-$d.err -o ~/logs/posetail/enf-$d.out \
#     -n 48 -q gpu_h200 -R "span[hosts=1]" -gpu "num=4" -W 96:00 \
#     pixi run python train.py --config-path configs/config_encoder_3d_finetuning_h100.toml \
#     --devices -1 --strategy ddp_find_unused_parameters_true

# bsub -J enf4-$d -e ~/logs/posetail/enf4-$d.err -o ~/logs/posetail/enf4-$d.out \
#     -n 72 -q gpu_h100 -R "span[hosts=1]" -gpu "num=6" -W 96:00 \
#     pixi run python train.py --config-path configs/config_encoder_3d_finetuning_2d.toml \
#     --devices -1 --strategy ddp_find_unused_parameters_true

#bsub -J enf6-$d -e ~/logs/posetail/enf6-$d.err -o ~/logs/posetail/enf6-$d.out \
#    -n 96 -q gpu_h100 -R "span[hosts=1]" -gpu "num=8" -W 96:00 \
#    /bin/bash train_script.sh configs/config_encoder_gridresid_finetune.toml

# bsub -J enm2-$d -e ~/logs/posetail/enm2-$d.err -o ~/logs/posetail/enm2-$d.out \
#     -n 96 -q gpu_h100 -R "span[hosts=1]" -gpu "num=8" -W 168:00 \
#     /bin/bash train_script.sh configs/config_ft_muonsf_8gpu_5.toml

# bsub -J enm3-$d -e ~/logs/posetail/enm3-$d.err -o ~/logs/posetail/enm3-$d.out \
#     -n 96 -q gpu_h100 -R "span[hosts=1]" -gpu "num=8" -W 168:00 \
#     /bin/bash train_script.sh configs/config_ft_muonsf_8gpu_5b.toml

bsub -J enf8-$d -e ~/logs/posetail/enm8-$d.err -o ~/logs/posetail/enf8-$d.out \
    -n 96 -q gpu_h100 -R "span[hosts=1]" -gpu "num=8" -W 96:00 \
    /bin/bash train_script.sh configs/config_encoder_finetune.toml
