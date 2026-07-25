#!/usr/bin/env sh

source cluster/setup_env_vars.sh

d=$(date +d%dh%H%M)


# bsub -J en6-$d -e ~/logs/posetail/en6-$d.err -o ~/logs/posetail/en6-$d.out \
#     -n 48 -q gpu_h100 -R "span[hosts=1]" -gpu "num=4:aff=yes" -W 72:00 \
#     /bin/bash cluster/train_script.sh configs/config_encoder_pretrain.toml

bsub -J en7-$d -e ~/logs/posetail/en7-$d.err -o ~/logs/posetail/en7-$d.out \
    -n 48 -q gpu_a100 -R "span[hosts=1]" -gpu "num=4" -W 72:00 \
    /bin/bash cluster/train_script.sh configs/config_encoder_pretrain.toml
