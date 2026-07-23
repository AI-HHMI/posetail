import argparse
import os

from torch.utils.data import DataLoader

from posetail.datasets.inference_dataset import PosetailInferenceDataset, custom_collate
from posetail.inference.inference_utils import *
from viz3d import *


''' 
a utility script for running inference on a single video given a 
model checkpoint and the config file that was used to train it. 
an .rrd file will be generated to visualize the 3d predictions.

python inference.py
'''

def parse_args():
    '''
    parse command line arguments
    ''' 
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-path', type = str)
    parser.add_argument('--split', type = str, default = 'test')
    parser.add_argument('--config-path', type = str)
    parser.add_argument('--checkpoint-path', type = str)
    parser.add_argument('---outpath', type = str, default = '../output')

    args = parser.parse_args()

    return args


def run_inference(dataset_path, config_path, checkpoint_path, outpath, 
                  split = 'test', n_frames = 25, max_kpts = 1000, 
                  debug_ix = None): 

    # load the config and model
    config = load_config(config_path)
    config.dataset.test.n_frames = n_frames
    # config.model.mode_3d = 'encoder' # TODO: remove later

    device = (torch.device(config.devices.device) if torch.cuda.is_available() else 'cpu')
    checkpoint_dict = load_checkpoint(config_path, checkpoint_path)
    model = checkpoint_dict['model']
    model.eval()

    # set seed for reproducibility
    set_seeds(config.training.seed)

    # create a dataset for one video
    dataset = PosetailInferenceDataset(
        dataset_path = dataset_path, 
        config = config, 
        split = split)

    dataloader = DataLoader(
        dataset, 
        batch_size = config.dataset.batch_size, 
        collate_fn = custom_collate,
        num_workers = config.dataset.get('num_workers', 16), 
        pin_memory = True, 
        prefetch_factor = 2, 
        persistent_workers = True)

    # use the model to predict the 3d positions
    split_outpath = os.path.join(outpath, split)
    os.makedirs(split_outpath, exist_ok = True)
    outpath = predict_on_dataset_3d(
        model, dataloader, 
        split_outpath, device,
        max_kpts = max_kpts, 
        debug_ix = debug_ix
    )  

    return split_outpath


if __name__ == '__main__':

    # args = parse_args()

    # dataset_path = args.dataset_path
    # split = args.split
    # config_path = args.config_path
    # checkpoint_path = args.checkpoint_path
    # outpath = args.outpath

    dataset_name = 'kubric-multiview' # cmupanoptic' # 'cmupanoptic_3dgs' 'dex_ycb'
    dataset_path = f'/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-finetuning/{dataset_name}' 
    split = 'val'
    n_frames = 25
    max_kpts = 1200
    
    # vjepa network finetuned
    checkpoint_path = '/groups/karashchuk/home/karashchukl/results/posetail-odyssey-finetuning/wandb/run-20260519_104622-f8pai8gk/files/checkpoints/checkpoint_00532480.pth'
    outpath = f'/home/ruppk2@hhmi.org/dataset_scripts/predictions_f8pai8gk/{dataset_name}'

    config_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), 'config.toml')

    split_outpath = run_inference(
        dataset_path = dataset_path,
        config_path = config_path, 
        checkpoint_path = checkpoint_path, 
        outpath = outpath, 
        split = split, 
        n_frames = n_frames,
        max_kpts = max_kpts)
    
    # combine predictions from each session and trial
    combine_predictions(split_outpath)
    
    # visualize the 3d predictions
    viz_predictions_3d(split_outpath, spawn = False,
                       kpt_radius = 0.05, 
                       connection_radius = 0.01, 
                       connect_pred_to_gt = False)  
    
