#!/usr/bin/env sh

# run inference on cotracker baseline (predict individually on each camera view)
pixi run python cotracker_inference.py \
    --dataset-root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v2 \
    --output-root /home/ruppk2@hhmi.org/cotracker-outputs \
    --datasets kubric-multiview dex_ycb cmupanoptic_3dgs \
    --split test \
    --checkpoint /home/ruppk2@hhmi.org/software/cotracker-weights/scaled_offline.pth \
    --device cuda:1

# triangulate the results 
pixi run python cotracker_triangulate.py \
    --dataset-root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v2 \
    --output-root /home/ruppk2@hhmi.org/cotracker-outputs \
    --datasets kubric-multiview dex_ycb cmupanoptic_3dgs \
    --split test

# compute the evaluation metrics
pixi run python cotracker_metrics.py \
    --dataset-root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v2 \
    --output-root  /home/ruppk2@hhmi.org/cotracker-outputs \
    --datasets kubric-multiview dex_ycb cmupanoptic_3dgs \
    --split test \
    --force

# combine the evaluation metrics into a summary dataframe
pixi run python ../scripts/combine_metrics.py --prefix /home/ruppk2@hhmi.org/cotracker-outputs
