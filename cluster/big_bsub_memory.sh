#!/usr/bin/env sh

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

source cluster/setup_env_vars.sh

d=$(date +d%dh%H%M)


# 4x H100. Peak measured at 30.2 GiB for the worst case this config can draw
# (cams_to_sample 8 x memory_num_context 16), so 80 GB is ample -- the frozen backbone
# removed the encoder backward, which is what pays for the raised M.
#
# NOTE: memory frames are now 2-frame tubelets and both memory_prob and M went up, so
# memory-frame image decodes rose ~5x. If the run turns out dataloader-bound, that is the
# first thing to look at (more cores, or memory_num_context back to [2, 12]).
bsub -J mem-$d -e ~/logs/posetail/mem-$d.err -o ~/logs/posetail/mem-$d.out \
    -n 48 -q gpu_h100 -R "span[hosts=1]" -gpu "num=4" -W 72:00 \
    /bin/bash cluster/train_script.sh configs/config_encoder_memory.toml
