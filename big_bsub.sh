#!/usr/bin/env sh

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

source setup_env_vars.sh

d=$(date +d%dh%H)

# bsub -J tb-$d -e ~/logs/posetail/tb-$d.err -o ~/logs/posetail/tb-$d.out \
#     -n 48 -q gpu_a100 -R "span[hosts=1]" -gpu "num=4" -W 48:00 \
#     pixi run python train.py --config-path configs/config_default_3d.toml --devices -1

# bsub -J mb-$d -e ~/logs/posetail/mb-$d.err -o ~/logs/posetail/mb-$d.out \
#     -n 48 -q gpu_a100 -R "span[hosts=1]" -gpu "num=4" -W 48:00 \
#     pixi run python train.py --config-path configs/config_minicubes_3d.toml --devices -1

# bsub -J tb-$d -e ~/logs/posetail/tb-$d.err -o ~/logs/posetail/tb-$d.out \
#     -n 96 -q gpu_h200 -R "span[hosts=1]" -gpu "num=8" -W 96:00 \
#     pixi run python train.py --config-path configs/config_default_3d.toml --devices -1

# bsub -J mb-$d -e ~/logs/posetail/mb1-$d.err -o ~/logs/posetail/mb1-$d.out \
#     -n 96 -q gpu_h200 -R "span[hosts=1]" -gpu "num=8" -W 96:00 \
#     pixi run python train.py --config-path configs/config_minicubes_finetuning.toml --devices -1

# bsub -J mb-$d -e ~/logs/posetail/mb2-$d.err -o ~/logs/posetail/mb2-$d.out \
#     -n 96 -q gpu_h200 -R "span[hosts=1]" -gpu "num=8" -W 96:00 \
#     pixi run python train.py --config-path configs/config_minicubes_finetuning_2.toml --devices -1

# bsub -J mb-$d -e ~/logs/posetail/mb3-$d.err -o ~/logs/posetail/mb3-$d.out \
#     -n 96 -q gpu_h200 -R "span[hosts=1]" -gpu "num=8" -W 96:00 \
#     pixi run python train.py --config-path configs/config_minicubes_3d.toml --devices -1  

# bsub -J mb-$d -e ~/logs/posetail/mb3-$d.err -o ~/logs/posetail/mb3-$d.out \
#     -n 96 -q gpu_h100 -R "span[hosts=1]" -gpu "num=8" -W 160:00 \
#     pixi run python train.py --config-path configs/config_minicubes_3d.toml --devices -1

# bsub -J en-$d -e ~/logs/posetail/en-$d.err -o ~/logs/posetail/en-$d.out \
#     -n 48 -q gpu_a100 -R "span[hosts=1]" -gpu "num=4" -W 48:00 \
#     pixi run python train.py --config-path configs/config_encoder_3d.toml --devices -1 --strategy ddp_find_unused_parameters_true

# bsub -J en-$d -e ~/logs/posetail/en-$d.err -o ~/logs/posetail/en-$d.out \
#     -n 48 -q gpu_h100 -R "span[hosts=1]" -gpu "num=4" -W 48:00 \
#     pixi run python train.py --config-path configs/config_encoder_3d.toml --devices -1 --strategy ddp_find_unused_parameters_true

# bsub -J en-$d -e ~/logs/posetail/en-$d.err -o ~/logs/posetail/en-$d.out \
#     -n 48 -q gpu_h100 -R "span[hosts=1]" -gpu "num=4" -W 72:00 \
#     pixi run python train.py --config-path configs/config_encoder_3d.toml --devices -1 --strategy ddp_find_unused_parameters_true 

# bsub -J en-$d -e ~/logs/posetail/en-$d.err -o ~/logs/posetail/en-$d.out \
#     -n 48 -q gpu_h100 -R "span[hosts=1]" -gpu "num=4" -W 48:00 \
#     pixi run python train.py --config-path configs/config_encoder_3d.toml --devices -1 --strategy ddp_find_unused_parameters_true

# bsub -J en-$d -e ~/logs/posetail/en-$d.err -o ~/logs/posetail/en-$d.out \
#     -n 96 -q gpu_h100 -R "span[hosts=1]" -gpu "num=8" -W 48:00 \
#     pixi run python train.py --config-path configs/config_encoder_3d.toml --devices -1 --strategy ddp_find_unused_parameters_true

# bsub -J en-$d -e ~/logs/posetail/en-$d.err -o ~/logs/posetail/en-$d.out \
#     -n 48 -q gpu_h100 -R "span[hosts=1]" -gpu "num=4" -W 24:00 \
#     pixi run python train.py --config-path configs/config_encoder_3d.toml --devices -1 --strategy ddp_find_unused_parameters_true

# bsub -J en-$d -e ~/logs/posetail/en-$d.err -o ~/logs/posetail/en-$d.out \
#     -n 36 -q gpu_h100 -R "span[hosts=1]" -gpu "num=3" -W 24:00 \
#     pixi run python train.py --config-path configs/config_encoder_3d_queue.toml --devices -1 --strategy ddp_find_unused_parameters_true

# bsub -J en-$d -e ~/logs/posetail/en-$d.err -o ~/logs/posetail/en-$d.out \
#     -n 64 -q gpu_l4 -R "span[hosts=1]" -gpu "num=8" -W 48:00 \
#     pixi run python train.py --config-path configs/config_encoder_3d_l4.toml \
#     --devices -1 --strategy ddp_find_unused_parameters_true


# bsub -J en3-$d -e ~/logs/posetail/en3-$d.err -o ~/logs/posetail/en3-$d.out \
#     -n 36 -q gpu_h100 -R "span[hosts=1]" -gpu "num=3" -W 60:00 \
#     pixi run python train.py --config-path configs/config_encoder_3d.toml \
#     --devices -1 --strategy ddp_find_unused_parameters_true

# bsub -J en3-$d -e ~/logs/posetail/en3-$d.err -o ~/logs/posetail/en3-$d.out \
#     -n 48 -q gpu_a100 -R "span[hosts=1]" -gpu "num=4" -W 60:00 \
#     pixi run python train.py --config-path configs/config_encoder_3d.toml \
#     --devices -1 --strategy ddp_find_unused_parameters_true

# bsub -J en2-$d -e ~/logs/posetail/en2-$d.err -o ~/logs/posetail/en2-$d.out \
#     -n 64 -q gpu_l4 -R "span[hosts=1]" -gpu "num=8" -W 60:00 \
#     pixi run python train.py --config-path configs/config_encoder_3d.toml \
#     --devices -1 --strategy ddp_find_unused_parameters_true

# bsub -J enf-$d -e ~/logs/posetail/enf-$d.err -o ~/logs/posetail/enf-$d.out \
#     -n 64 -q gpu_l4 -R "span[hosts=1]" -gpu "num=8" -W 168:00 \
#     pixi run python train.py --config-path configs/config_encoder_3d_finetuning.toml \
#     --devices -1 --strategy ddp_find_unused_parameters_true

bsub -J enf2-$d -e ~/logs/posetail/enf2-$d.err -o ~/logs/posetail/enf2-$d.out \
    -n 96 -q gpu_h100 -R "span[hosts=1]" -gpu "num=8" -W 72:00 \
    pixi run python train.py --config-path configs/config_encoder_3d_finetuning_h100.toml \
    --devices -1 --strategy ddp_find_unused_parameters_true

# bsub -J enf-$d -e ~/logs/posetail/en-$d.err -o ~/logs/posetail/enf-$d.out \
#     -n 48 -q gpu_a100 -R "span[hosts=1]" -gpu "num=4" -W 72:00 \
#     pixi run python train.py --config-path configs/config_encoder_3d_finetuning.toml \
#     --devices -1 --strategy ddp_find_unused_parameters_true

# bsub -J en-$d -e ~/logs/posetail/en-$d.err -o ~/logs/posetail/en-$d.out \
#     -n 48 -q gpu_a100 -R "span[hosts=1]" -gpu "num=4" -W 48:00 \
#     pixi run python train.py --config-path configs/config_encoder_3d_other.toml --devices -1 --strategy ddp_find_unused_parameters_true

# bsub -J en-$d -e ~/logs/posetail/en-$d.err -o ~/logs/posetail/en-$d.out \
#     -n 24 -q gpu_h100 -R "span[hosts=1]" -gpu "num=2" -W 48:00 \
#     pixi run python train.py --config-path configs/config_encoder_3d.toml --devices -1 --strategy ddp_find_unused_parameters_true

# bsub -J en-$d -e ~/logs/posetail/en-$d.err -o ~/logs/posetail/en-$d.out \
#     -n 36 -q gpu_h100 -R "span[hosts=1]" -gpu "num=3" -W 48:00 \
#     pixi run python train.py --config-path configs/config_encoder_3d.toml --devices -1 --strategy ddp_find_unused_parameters_true
