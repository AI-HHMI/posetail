#!/usr/bin/env sh

source cluster/setup_env_vars.sh

d=$(date +d%dh%H%M)


bsub -J mem-$d -e ~/logs/posetail/mem-$d.err -o ~/logs/posetail/mem-$d.out \
    -n 48 -q gpu_a100 -R "span[hosts=1]" -gpu "num=4" -W 72:00 \
    /bin/bash cluster/train_script.sh configs/config_encoder_memory.toml
