#!/usr/bin/env sh

# point odyssey 
# pixi run python inference_dataset.py --dataset-root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v2 --datasets point-odyssey-human --base-folder /groups/karashchuk/home/karashchukl/results/posetail-odyssey-pretrain/wandb/run-20260518_111742-y2ird8f0 --checkpoint 00399996 --n-overlap 2 --max-kpts 1200 --device cuda:1 --output-root /home/ruppk2@hhmi.org/dataset_predictions/y2ird8f0_pretrained_final
# pixi run python inference_dataset.py --dataset-root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v2 --datasets point-odyssey-human --base-folder /groups/karashchuk/home/karashchukl/results/posetail-odyssey-finetuning/wandb/run-20260519_104622-f8pai8gk --checkpoint 00799992 --n-overlap 2 --max-kpts 1200 --device cuda:1 --output-root /home/ruppk2@hhmi.org/dataset_predictions/f8pai8gk_finetuned_final

# pixi run python inference_dataset.py --dataset-root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v2 --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --base-folder /groups/karashchuk/home/karashchukl/results/posetail-odyssey-pretrain/wandb/run-20260518_111742-y2ird8f0 --checkpoint 00399996 --n-overlap 2 --max-kpts 1200 --device cuda:1 --output-root /home/ruppk2@hhmi.org/dataset_predictions/y2ird8f0_pretrained_final

# pixi run python inference_dataset.py --dataset-root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v2 --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --base-folder /groups/karashchuk/home/ruppk2/results/posetail-ablations/wandb/run-20260522_164721-sqj32nr1 --checkpoint 00399996 --n-overlap 2 --max-kpts 1200 --device cuda:1 --output-root /home/ruppk2@hhmi.org/dataset_predictions/sqj32nr1_base_encoder_size

# pixi run python inference_dataset.py --dataset-root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v2 --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --base-folder /groups/karashchuk/home/ruppk2/results/posetail-ablations/wandb/run-20260522_170103-logv7pci --checkpoint 00399996 --n-overlap 2 --max-kpts 1200 --device cuda:1 --output-root /home/ruppk2@hhmi.org/dataset_predictions/logv7pci_giant_encoder_size

# pixi run python inference_dataset.py --dataset-root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v2 --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --base-folder /groups/karashchuk/home/ruppk2/results/posetail-ablations/wandb/run-20260522_170133-a6jf0ox7 --checkpoint 00399996 --n-overlap 2 --max-kpts 1200 --device cuda:1 --output-root /home/ruppk2@hhmi.org/dataset_predictions/a6jf0ox7_gigantic_encoder_size

# pixi run python inference_dataset.py --dataset-root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v2 --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --base-folder /groups/karashchuk/home/ruppk2/results/posetail-ablations/wandb/run-20260522_170130-j05ijlgj --checkpoint 00399996 --n-overlap 2 --max-kpts 1200 --device cuda:1 --output-root /home/ruppk2@hhmi.org/dataset_predictions/j05ijlgj_pretrained_odyssey_only

# pixi run python inference_dataset.py --dataset-root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v2 --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --base-folder /groups/karashchuk/home/ruppk2/results/posetail-ablations/wandb/run-20260522_170112-grrq35ue --checkpoint 00399996 --n-overlap 2 --max-kpts 1200 --device cuda:1 --output-root /home/ruppk2@hhmi.org/dataset_predictions/grrq35ue_pretrained_kubric_only

# pixi run python inference_dataset.py --dataset-root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v2 --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --base-folder /groups/karashchuk/home/ruppk2/results/posetail-ablations/wandb/run-20260522_180258-l77n9wa8 --checkpoint 00399996 --n-overlap 2 --max-kpts 1200 --device cuda:1 --output-root /home/ruppk2@hhmi.org/dataset_predictions/l77n9wa8_no_cam_self_attention

# pixi run python inference_dataset.py --dataset-root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v2 --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --base-folder /groups/karashchuk/home/ruppk2/results/posetail-ablations/wandb/run-20260522_180342-4b9ttscw --checkpoint 00399996 --n-overlap 2 --max-kpts 1200 --device cuda:1 --output-root /home/ruppk2@hhmi.org/dataset_predictions/4b9ttscw_256_latent_dim

# pixi run python inference_dataset.py --dataset-root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v2 --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --base-folder /groups/karashchuk/home/ruppk2/results/posetail-ablations/wandb/run-20260526_113749-1v07t4ci --checkpoint 00399996 --n-overlap 2 --max-kpts 1200 --device cuda:1 --output-root /home/ruppk2@hhmi.org/dataset_predictions/1v07t4ci_1024_latent_dim


# finetuned networks - inference 

# final finetuned network
# pixi run python inference_dataset.py --dataset-root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v2 --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --base-folder /groups/karashchuk/home/karashchukl/results/posetail-odyssey-finetuning/wandb/run-20260519_104622-f8pai8gk --checkpoint 00799992 --n-overlap 2 --max-kpts 1200 --device cuda:1 --output-root /home/ruppk2@hhmi.org/dataset_predictions/f8pai8gk_finetuned_final

# finetuned, pretrained on point odyssey only 
pixi run python inference_dataset.py --dataset-root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v2 --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --base-folder /groups/karashchuk/home/ruppk2/results/posetail-ablations/wandb/run-20260526_123353-tzctn5y6 --checkpoint 00799996 --n-overlap 2 --max-kpts 1200 --device cuda:1 --output-root /home/ruppk2@hhmi.org/dataset_predictions/tzctn5y6_finetuned_odyssey_only

# finetuned, pretrained on kubric only
pixi run python inference_dataset.py --dataset-root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning-v2 --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --base-folder /groups/karashchuk/home/ruppk2/results/posetail-ablations/wandb/run-20260526_121956-hx167m9y --checkpoint 00799996 --n-overlap 2 --max-kpts 1200 --device cuda:1 --output-root /home/ruppk2@hhmi.org/dataset_predictions/hx167m9y_finetuned_kubric_only