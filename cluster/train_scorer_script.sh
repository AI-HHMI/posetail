#!/usr/bin/env sh

source cluster/setup_env_vars.sh

module load cuda/12.8

pixi run python train_scorer.py --config-path $1 \
    --devices -1 --strategy ddp_find_unused_parameters_true
