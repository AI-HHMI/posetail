#!/usr/bin/env sh

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

source cluster/setup_env_vars.sh

d=$(date +d%dh%H%M)


bsub -J mem-$d -e ~/logs/posetail/mem-$d.err -o ~/logs/posetail/mem-$d.out \
    -n 48 -q gpu_a100 -R "span[hosts=1]" -gpu "num=4" -W 72:00 \
    /bin/bash cluster/train_script.sh configs/config_encoder_memory.toml
