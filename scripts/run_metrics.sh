#!/usr/bin/env sh


# pretrained networks - metrics 
pixi run python inference_metrics.py --dataset-root /home/ruppk2@hhmi.org/dataset_predictions/y2ird8f0_pretrained_final/ --input-name output.npz --output-name eval_metrics.npz --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --force

pixi run python inference_metrics.py --dataset-root /home/ruppk2@hhmi.org/dataset_predictions/sqj32nr1_base_encoder_size/ --input-name output.npz --output-name eval_metrics.npz --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --force

pixi run python inference_metrics.py --dataset-root /home/ruppk2@hhmi.org/dataset_predictions/logv7pci_giant_encoder_size/ --input-name output.npz --output-name eval_metrics.npz --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --force

pixi run python inference_metrics.py --dataset-root /home/ruppk2@hhmi.org/dataset_predictions/a6jf0ox7_gigantic_encoder_size/ --input-name output.npz --output-name eval_metrics.npz --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --force

pixi run python inference_metrics.py --dataset-root /home/ruppk2@hhmi.org/dataset_predictions/j05ijlgj_pretrained_odyssey_only/ --input-name output.npz --output-name eval_metrics.npz --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --force

pixi run python inference_metrics.py --dataset-root /home/ruppk2@hhmi.org/dataset_predictions/grrq35ue_pretrained_kubric_only/ --input-name output.npz --output-name eval_metrics.npz --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --force

pixi run python inference_metrics.py --dataset-root /home/ruppk2@hhmi.org/dataset_predictions/l77n9wa8_no_cam_self_attention/ --input-name output.npz --output-name eval_metrics.npz --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --force

pixi run python inference_metrics.py --dataset-root /home/ruppk2@hhmi.org/dataset_predictions/4b9ttscw_256_latent_dim/ --input-name output.npz --output-name eval_metrics.npz --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --force

pixi run python inference_metrics.py --dataset-root /home/ruppk2@hhmi.org/dataset_predictions/1v07t4ci_1024_latent_dim/ --input-name output.npz --output-name eval_metrics.npz --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --force


# finetuned networks - metrics 
pixi run python inference_metrics.py --dataset-root /home/ruppk2@hhmi.org/dataset_predictions/f8pai8gk_finetuned_final/ --input-name output.npz --output-name eval_metrics.npz --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --force

pixi run python inference_metrics.py --dataset-root /home/ruppk2@hhmi.org/dataset_predictions/tzctn5y6_finetuned_odyssey_only/ --input-name output.npz --output-name eval_metrics.npz --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --force

pixi run python inference_metrics.py --dataset-root /home/ruppk2@hhmi.org/dataset_predictions/hx167m9y_finetuned_kubric_only/ --input-name output.npz --output-name eval_metrics.npz --datasets kubric-multiview dex_ycb cmupanoptic_3dgs --force


# combine the results into a single dataframe
pixi run python /home/ruppk2@hhmi.org/posetail/scripts/combine_metrics.py --prefix /home/ruppk2@hhmi.org/dataset_predictions/