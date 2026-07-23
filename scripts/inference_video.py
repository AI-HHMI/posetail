"""
Example invocation (trial directory with img/ folder):

    python inference_video.py \
        --base-folder /path/to/wandb/run-YYYYMMDD_HHMMSS-XXXXXXXX \
        --trial-path /path/to/session/trial/ \
        --start-frame 0 \
        --n-frames 256 \
        --n-overlap 2 \
        --checkpoint 10000 \
        --device cuda:0 \
        --outpath /path/to/output.npz

The trial directory should contain:
    - metadata.yaml (camera calibration)
    - pose3d.npz (3D pose data, used for initial query points)
    - img/ (per-camera subdirectories of images) or vid/ (per-camera .mp4 files)
"""
import argparse

import torch

from posetail.inference.inference_utils import load_model_from_base_folder, run_inference


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--base-folder', type=str, required=True,
                        help='Wandb run folder containing files/config.toml and files/checkpoints/')
    parser.add_argument('--trial-path', type=str, required=True,
                        help='Path to a trial directory containing metadata.yaml, '
                             'pose3d.npz, and an img/ or vid/ folder')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--n-frames', type=int, default=128)
    parser.add_argument('--n-overlap', type=int, default=2)
    parser.add_argument('--n-views', type=int, default=None, help='Evaluate on a random subset of the cameras')
    parser.add_argument('--view-seed', type=int, default=None, help='Random seed for subsampling cameras')
    parser.add_argument('--max-kpts', type=int, default=None, help='Max keypoints per model forward pass.')
    parser.add_argument('--per-subject', action='store_true', default=False,
                        help='Track each subject independently instead of concatenating all keypoints')
    parser.add_argument('--no-query-first', dest='query_first', action='store_false', default=True,
                        help='disable query-first (default ON): with this flag all points are '
                             'anchored at start_frame instead of their first valid+visible frame')
    parser.add_argument('--no-motion-margin', dest='motion_margin', action='store_false', default=True,
                        help='disable the causal motion-margin crop expansion (default ON); with '
                             'this flag + --no-query-first the path is legacy-identical')
    parser.add_argument('--carry-latent', action='store_true', default=False,
                        help='thread the decoder latent across chunks (default OFF): the carried '
                             'full-N latent (b,t,N,cams,D) is immune to --max-kpts chunking and '
                             'roughly doubles peak memory on dense point sets, so it is off by '
                             'default; query re-anchoring still provides cross-chunk continuity')
    parser.add_argument('--checkpoint', type=int, default=None,
                        help='Optional checkpoint step number; if omitted, use latest checkpoint')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--pred-key-3d', type=str, default='coords_pred',
                        help="Which 3D model output to use as the prediction "
                             "(e.g. 'coords_pred' or '3d_pred_triangulate')")
    parser.add_argument('--clip-len', type=int, default=None,
                        help='Frames fed to the model per forward. Defaults to '
                             'model.n_frames (= stride_length). For a windowed model set '
                             'this > stride_length (e.g. 16) so internal windowing + the '
                             'latent carry engage per chunk; the latent is also threaded '
                             'across chunks.')
    parser.add_argument('--outpath', type=str, default=None,
                        help='Optional output .npz path')

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device is not None else None

    model, config, config_path, checkpoint_path = load_model_from_base_folder(
        args.base_folder,
        checkpoint=args.checkpoint,
        device=device,
    )

    run_inference(
        model=model,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        trial_path=args.trial_path,
        start_frame=args.start_frame,
        n_frames=args.n_frames,
        n_overlap=args.n_overlap,
        n_views=args.n_views,
        view_seed=args.view_seed,
        max_kpts=args.max_kpts,
        per_subject=args.per_subject,
        device=device,
        outpath=args.outpath,
        pred_key_3d=args.pred_key_3d,
        clip_len=args.clip_len,
        query_first=args.query_first,
        motion_margin=args.motion_margin,
        carry_latent=args.carry_latent,
    )


if __name__ == '__main__':
    main()
