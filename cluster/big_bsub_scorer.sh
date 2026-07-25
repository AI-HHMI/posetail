#!/usr/bin/env sh

source cluster/setup_env_vars.sh

d=$(date +d%dh%H%M)


bsub -J scorer-$d -e ~/logs/posetail/scorer-$d.err -o ~/logs/posetail/scorer-$d.out \
    -n 48 -q gpu_h100 -R "span[hosts=1]" -gpu "num=4:aff=yes" -W 72:00 \
    /bin/bash cluster/train_scorer_script.sh configs/config_scorer.toml

# bsub -J scorer-$d -e ~/logs/posetail/scorer-$d.err -o ~/logs/posetail/scorer-$d.out \
#     -n 64 -q gpu_l4 -R "span[hosts=1]" -gpu "num=8" -W 72:00 \
#     /bin/bash cluster/train_scorer_script.sh configs/config_scorer.toml
