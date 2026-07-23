"""
Visualize inference predictions produced by inference_video.py / inference_dataset.py.

Operates on either a single trial output.npz or a whole output tree, and can
produce two kinds of visualization (independently toggled):

    --rrd         rerun 3D .rrd  (red = prediction, green = ground truth)
    --videos-2d   per-camera .mp4 with the 3D predictions projected onto the
                  source frames (red = prediction, green = ground truth)

This is a separate, explicit step -- it never runs as part of inference_dataset.py.

Single trial (e.g. right after inference_video.py):
    python viz_predictions.py --output-npz /path/to/trial/output.npz --rrd
    python viz_predictions.py --output-npz /path/to/trial/output.npz --videos-2d --viz-2d-fps 30

Whole tree (after inference_dataset.py):
    python viz_predictions.py --output-root /path/to/predictions --rrd
    python viz_predictions.py --output-root /path/to/predictions --videos-2d \\
        --datasets cmupanoptic_3dgs --splits test --force

If neither --rrd nor --videos-2d is given, --rrd is assumed.
"""

import argparse

from viz3d import (
    viz_trial,
    viz_trial_2d,
    viz_predictions_dataset,
    viz_predictions_dataset_2d,
)


def parse_args():
    '''
    parse command line arguments
    '''
    parser = argparse.ArgumentParser()

    # input: exactly one of these
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--output-npz', type=str,
                     help='Path to a single trial output.npz')
    src.add_argument('--output-root', type=str,
                     help='Root of an inference output tree (mirrors inference_dataset.py output)')

    # what to generate (default: rrd only)
    parser.add_argument('--rrd', action='store_true',
                        help='Write a rerun 3D predictions_3d.rrd')
    parser.add_argument('--videos-2d', action='store_true',
                        help='Write per-camera 2D videos with projected predictions')

    # tree-walk filters (only used with --output-root)
    parser.add_argument('--input-name', default='output.npz',
                        help='Per-trial inference output filename (default: output.npz)')
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--splits', nargs='+', default=None)
    parser.add_argument('--trials', nargs='+', default=None)
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing .rrd / video outputs')

    # 3D (rrd) options
    parser.add_argument('--rrd-name', default='predictions_3d.rrd',
                        help='Filename of the .rrd written per trial')
    parser.add_argument('--spawn', action='store_true',
                        help='Spawn the rerun viewer instead of only saving the .rrd')
    parser.add_argument('--kpt-radius', type=float, default=0.05)
    parser.add_argument('--connection-radius', type=float, default=0.01)
    parser.add_argument('--connect-pred-to-gt', action='store_true', default=False,
                        help='Draw a line connecting each prediction to its ground-truth point')

    # 2D video options
    parser.add_argument('--videos-subdir', default='videos_2d',
                        help='Subdirectory (per trial) the 2D videos are written into')
    parser.add_argument('--viz-2d-fps', type=float, default=30)
    parser.add_argument('--viz-2d-max-frames', type=int, default=None,
                        help='Limit the number of frames rendered per 2D video')
    parser.add_argument('--viz-2d-no-skeleton', action='store_true', default=False,
                        help='Draw only points, no skeleton, in the 2D videos')
    parser.add_argument('--point-radius', type=int, default=3)
    parser.add_argument('--line-thickness', type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    # default to the 3D rrd if nothing was requested
    do_rrd = args.rrd
    do_2d = args.videos_2d
    if not do_rrd and not do_2d:
        do_rrd = True

    rrd_kwargs = dict(
        kpt_radius=args.kpt_radius,
        connection_radius=args.connection_radius,
        connect_pred_to_gt=args.connect_pred_to_gt,
    )
    vid_kwargs = dict(
        fps=args.viz_2d_fps,
        max_frames=args.viz_2d_max_frames,
        draw_skeleton=not args.viz_2d_no_skeleton,
        point_radius=args.point_radius,
        line_thickness=args.line_thickness,
    )

    if args.output_npz is not None:
        if do_rrd:
            print(f'[rrd] {args.output_npz}')
            viz_trial(args.output_npz, rrd_name=args.rrd_name, spawn=args.spawn, **rrd_kwargs)
        if do_2d:
            print(f'[videos-2d] {args.output_npz}')
            viz_trial_2d(args.output_npz, output_subdir=args.videos_subdir, **vid_kwargs)
    else:
        if do_rrd:
            print(f'[rrd] tree {args.output_root}')
            viz_predictions_dataset(
                args.output_root,
                input_name=args.input_name,
                rrd_name=args.rrd_name,
                datasets=args.datasets, splits=args.splits, trials=args.trials,
                force=args.force, spawn=args.spawn, **rrd_kwargs)
        if do_2d:
            print(f'[videos-2d] tree {args.output_root}')
            viz_predictions_dataset_2d(
                args.output_root,
                input_name=args.input_name,
                output_subdir=args.videos_subdir,
                datasets=args.datasets, splits=args.splits, trials=args.trials,
                force=args.force, **vid_kwargs)


if __name__ == '__main__':
    main()
