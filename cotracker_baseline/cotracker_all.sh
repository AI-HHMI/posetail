#!/usr/bin/env sh

export PYTHONPATH="/home/ruppk2@hhmi.org/software/co-tracker:$PYTHONPATH"

# run inference on cotracker baseline (predict individually on each camera view)
pixi run python cotracker_inference.py \
    --dataset-root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v5 \
    --output-root /home/ruppk2@hhmi.org/cotracker-outputs-paper-final \
    --datasets kubric-multiview dex_ycb cmupanoptic_3dgs \
    --split test \
    --checkpoint /home/ruppk2@hhmi.org/software/cotracker-weights/scaled_offline.pth \
    --device cuda:0

# triangulate the results
pixi run python cotracker_triangulate.py \
    --dataset-root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v5 \
    --output-root /home/ruppk2@hhmi.org/cotracker-outputs-paper-final \
    --datasets kubric-multiview dex_ycb \
    --split test

# cmupanoptic uses 4 cameras to match posetail SETTINGS
pixi run python cotracker_triangulate.py \
    --dataset-root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v5 \
    --output-root /home/ruppk2@hhmi.org/cotracker-outputs-paper-final \
    --datasets cmupanoptic_3dgs \
    --split test \
    --n-views 4

# compute the evaluation metrics
pixi run python cotracker_metrics.py \
    --dataset-root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v5 \
    --output-root  /home/ruppk2@hhmi.org/cotracker-outputs-paper-final \
    --datasets kubric-multiview dex_ycb cmupanoptic_3dgs \
    --split test \
    --force

# combine the evaluation metrics into a summary dataframe
pixi run python /home/ruppk2@hhmi.org/posetail/scripts/combine_metrics.py --prefix /home/ruppk2@hhmi.org/cotracker-outputs-paper-final
