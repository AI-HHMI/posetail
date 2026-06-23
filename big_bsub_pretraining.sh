#!/usr/bin/env sh

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

source setup_env_vars.sh

d=$(date +d%dh%H%M)


# bsub -J en3-$d -e ~/logs/posetail/en3-$d.err -o ~/logs/posetail/en3-$d.out \
#     -n 48 -q gpu_h200 -R "span[hosts=1]" -gpu "num=4" -W 36:00 \
#     pixi run python train.py --config-path configs/config_encoder_3d.toml \
#     --devices -1 --strategy ddp_find_unused_parameters_true

# bsub -J en2-$d -e ~/logs/posetail/en2-$d.err -o ~/logs/posetail/en2-$d.out \
#     -n 48 -q gpu_h200 -R "span[hosts=1]" -gpu "num=4" -W 48:00 \
#     pixi run python train.py --config-path configs/config_encoder_3d_2d.toml \
#     --devices -1 --strategy ddp_find_unused_parameters_true

# bsub -J en4-$d -e ~/logs/posetail/en4-$d.err -o ~/logs/posetail/en4-$d.out \
#     -n 72 -q gpu_h100 -R "span[hosts=1]" -gpu "num=6" -W 72:00 \
#     pixi run python train.py --config-path configs/config_encoder_3d_2d.toml \
#     --devices -1 --strategy ddp_find_unused_parameters_true

# bsub -J tn1-$d -e ~/logs/posetail/tn1-$d.err -o ~/logs/posetail/tn1-$d.out \
#     -n 72 -q gpu_h100 -R "span[hosts=1]" -gpu "num=6" -W 72:00 \
#     /bin/bash train_script.sh configs/config_tapnext_3d.toml

# bsub -J en5-$d -e ~/logs/posetail/en5-$d.err -o ~/logs/posetail/en5-$d.out \
#     -n 48 -q gpu_h100 -R "span[hosts=1]" -gpu "num=4:aff=yes" -W 72:00 \
#     /bin/bash train_script.sh configs/config_pretrain_base256.toml

# bsub -J en6-$d -e ~/logs/posetail/en6-$d.err -o ~/logs/posetail/en6-$d.out \
#     -n 48 -q gpu_h100 -R "span[hosts=1]" -gpu "num=4:aff=yes" -W 72:00 \
#     /bin/bash train_script.sh configs/config_pretrain_base384.toml

# VideoMAE-V2 ViT-L and DINOv3 ViT-L encoders (pixel-recon / dense-spatial features) vs V-JEPA,
# same frozen-encoder gridresid pretrain recipe -> matched A/B for the lateral-error hypothesis.
# bsub -J vmaL-$d -e ~/logs/posetail/vmaL-$d.err -o ~/logs/posetail/vmaL-$d.out \
#     -n 48 -q gpu_h100 -R "span[hosts=1]" -gpu "num=4:aff=yes" -W 72:00 \
#     /bin/bash train_script.sh configs/config_pretrain_videomae_large.toml

bsub -J reg-$d -e ~/logs/posetail/reg-$d.err -o ~/logs/posetail/reg-$d.out \
    -n 48 -q gpu_h100 -R "span[hosts=1]" -gpu "num=4:aff=yes" -W 72:00 \
    /bin/bash train_script.sh configs/config_pretrain_base256.toml

# bsub -J dn3L-$d -e ~/logs/posetail/dn3L-$d.err -o ~/logs/posetail/dn3L-$d.out \
#     -n 48 -q gpu_h100 -R "span[hosts=1]" -gpu "num=4:aff=yes" -W 72:00 \
#     /bin/bash train_script.sh configs/config_pretrain_dino3_large.toml
