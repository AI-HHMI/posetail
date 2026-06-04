import os
import json
import time
import toml
import torch
import yaml

import numpy as np

# from torch.cuda.amp import GradScaler
# from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, IterableDataset

from datetime import datetime, timezone, timedelta
from easydict import EasyDict
# from pytorch_memlab import MemReporter, LineProfiler, profile

# from posetail.datasets.datasets import Rat7mIterableDataset
from posetail.datasets.utils import safe_make
from posetail.posetail.cube import get_camera_scale
from posetail.posetail.eval_metrics import get_eval_metrics
from posetail.posetail.losses import get_vis_true, unroll_batch, normalize_by_mean_depth
from posetail.posetail.tracker import Tracker 
from posetail.posetail.tracker_encoder import TrackerEncoder

from einops import rearrange


def set_seeds(seed = 3, set_backends = True):

    np.random.seed(seed)
    torch.manual_seed(seed)

    # seeds for (multi) gpu operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # seeds for nondeterministic operations - note that this
    # could make the code less efficient
    if set_backends:
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False


def load_config(config_path, easy = True): 
    ''' 
    loads and returns the toml configuration file in which
    keys can be accessed.like.this
    '''
    with open(config_path, 'r') as toml_file:
        config = toml.load(toml_file)

    if easy: 
        config = EasyDict(config)

    return config

# def load_config(config_path): 
#     ''' 
#     loads and returns the toml configuration file in which
#     keys can be accessed.like.this
#     '''
#     config = {}
#     ext = os.path.splitext(config_path)[1]

#     if ext == '.yaml':
#         with open(config_path, 'r') as yaml_file:
#             config = yaml.safe_load(yaml_file)

#     elif ext == '.toml': 
#         with open(config_path, 'r') as toml_file:
#             config = toml.load(toml_file)

#     if '_wandb' in config:
#         config.pop('_wandb')

#     config = EasyDict(config)

#     return config


def save_config(config_path, new_config_path):

    config = load_config(config_path, easy = False)

    with open(new_config_path, 'w') as toml_file:
        toml.dump(config, toml_file)
        
        
def write_json(json_path, results): 
    '''
    appends results to a json file
    '''
    with open(json_path, 'a') as json_file: 
        json_file.write(json.dumps(results) + '\n')


def save_checkpoint(model, optimizer, prefix, i): 

    checkpoint_dir = safe_make(os.path.join(prefix, 'checkpoints'))

    checkpoint_path = os.path.join(checkpoint_dir, 
        f'checkpoint_{str(i).zfill(8)}.pth')
    
    state_dict = {
        'iteration': i,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }

    torch.save(state_dict, checkpoint_path)


def load_checkpoint(config_path, checkpoint_path, model = None, 
                    optimizer = None, device = None):

    config = load_config(config_path) 

    # configure device
    if device is None: 
        device = torch.device(config.devices.device)

    if not torch.cuda.is_available(): 
        device = torch.device('cpu')

    # load the model 
    if model is None: 

        if config.model.mode_3d == 'encoder':
            model = TrackerEncoder(**config.model)
        else:
            model = Tracker(**config.model)

        model.to(device)

    print(f'loading model checkpoint {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path, map_location = device)
    param_dict = checkpoint['model_state']

    missing_keys, unexpected_keys = model.load_state_dict(param_dict, strict = False)
    print(f'received missing keys: {missing_keys}')
    print(f'received unexpected keys: {unexpected_keys}')

    checkpoint_dict = {'model': model}

    # continue training 
    if optimizer is not None: 

        # set up the optimizer (if provided)
        optimizer.load_state_dict(checkpoint['optimizer_state'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        #  laod the iteration where training left off
        iteration = checkpoint.get('iteration', 0)

        # update the return dict
        checkpoint_dict['optimizer'] = optimizer
        checkpoint_dict['iteration'] = iteration + 1

    return checkpoint_dict

def load_checkpoint_no_inductor(config_path, checkpoint_path): 

    config = load_config(config_path)

    device = torch.device(config.devices.device)

    if not torch.cuda.is_available(): 
        device = torch.device('cpu')

    if config.model.mode_3d == 'encoder':
        model = TrackerEncoder(**config.model)
    else:
        model = Tracker(**config.model) 

    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location = device)
    state_dict = checkpoint.get('model_state')
    model.load_state_dict(state_dict)

    return model


def print_memory(device): 

    if torch.cuda.is_available():
        
        memory_alloc = torch.cuda.memory_allocated(device) / 1024 ** 3
        memory_res = torch.cuda.memory_reserved(device) / 1024 ** 3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

        print(f'allocated memory: {memory_alloc:.3f} GB')
        print(f'reserved memory: {memory_res:.3f} GB')
        print(f'total memory: {memory_total:.3f} GB\n')
    
    return memory_alloc, memory_res, memory_total

def get_timestamp(): 

    tz = timezone(timedelta(hours = -8))
    timestamp = datetime.now(tz)
    timestamp_fmt = timestamp.strftime('%Y-%m-%d %H:%M:%S')

    return timestamp_fmt

def format_camera(cam, offset_dict, cam_type, device):

    cam_dict = {
        'name': cam.get_name(),
        'type': cam_type, # pinhole, fisheye
        'ext': torch.as_tensor(cam.get_extrinsics_mat(), device = device, dtype = torch.float),
        'mat': torch.as_tensor(cam.get_camera_matrix(), device = device, dtype = torch.float),
        'dist': torch.as_tensor(cam.dist, device = device, dtype = torch.float),
        'size': torch.as_tensor(cam.get_size(), device = device, dtype = torch.int), 
    }

    if offset_dict: 
        offset = offset_dict[cam_dict['name']][:2]
        cam_dict['offset'] = torch.as_tensor(offset, device = device, dtype = torch.float)
    else:
        cam_dict['offset'] = torch.as_tensor([0.0, 0.0], device = device, dtype = torch.float)

    cam_dict['ext_inv'] = torch.linalg.inv(cam_dict['ext'])
        
    R = cam_dict['ext'][:3,:3]
    t = cam_dict['ext'][:3, 3]
    cam_dict['center'] = -R.T @ t
        
    return cam_dict

def format_camera_group(camera_group, offset_dict, cam_type, device):
    return [format_camera(cam, offset_dict, cam_type, device)
            for cam in camera_group.cameras]

def dict_to_device(dd, device):

    dout = dict()

    for k, v in dd.items():
        if isinstance(v, torch.Tensor):
            dout[k] = v.to(device)
        else:
            dout[k] = v

    return dout

def total_to_per_gpu(i, world_size): 
    per_gpu = (i + world_size - 1) // world_size
    return per_gpu
    
def train_iteration(config, model, fabric, batch, 
                    optimizer, loss, scheduler = None,
                    prefix = 'train/',  evaluate = False): 

    device = model.device
    model.train()

    start_time = time.time()
    timestamp = get_timestamp()

    learning_rate = optimizer.param_groups[0]['lr']
    metric_dicts = []
    
    views = [view.to(device) for view in batch.views]
    coords = batch.coords.to(device) # (b, t, n_kpts, 2)
    vis = batch.vis
    cgroup = batch.cgroup  
    vis_2d = batch.vis_2d
    query_times = batch.query_times
    p2d = batch.p2d # (b, cams, t, n_kpts, 2)

    if p2d is None: # 3d mode
        query_coords = coords[:, query_times[0],
                              torch.arange(len(query_times[0]))]
    else:
        assert p2d.shape[1] == 1
        query_coords = p2d[:, 0, query_times[0],
                          torch.arange(len(query_times[0]))]


    if cgroup: 
        cgroup = [dict_to_device(cam_dict, device) for cam_dict in cgroup]

    optimizer.zero_grad()

    # with fabric.autocast():
    outputs = model(
        views = list(views), 
        coords = query_coords,
        query_times = query_times,
        camera_group = cgroup)

    coords_pred = outputs['coords_pred']
    vis_pred = outputs['vis_pred']

    total_loss = loss(
        model = model, 
        outputs = outputs,
        coords_true = coords, 
        vis_true = vis,
        vis_true_cams = vis_2d,
        cgroup = cgroup, 
        p2d = p2d, 
        device = coords_pred.device)

    fabric.backward(total_loss)

    # Calculate gradient norm
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.detach().data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5

    # torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm, 
    #                                error_if_nonfinite = False)

    try:
        fabric.clip_gradients(model, optimizer, 
            max_norm = config.training.max_grad_norm, 
            error_if_nonfinite = True)

        optimizer.step()
    except:
        print("ERROR BAD GRADIENTS!!!")
        print(batch.sample_info)
        
    optimizer.zero_grad()
 
    if evaluate and coords.shape[-1] == 3:
        if p2d is not None:
            C = cgroup[0]['center']
            vis_for_norm = vis.to(coords.device) if vis is not None else get_vis_true(coords)
            _, pred_md = normalize_by_mean_depth(coords_pred, vis_for_norm, C)
            _, tgt_md  = normalize_by_mean_depth(coords, vis_for_norm, C)
            coords_pred = C + (coords_pred - C) * (tgt_md / pred_md)

        metrics_dict = get_eval_metrics(
            vis_pred = vis_pred,
            vis_true = vis,
            coords_pred = coords_pred,
            coords_true = coords,
            prefix = prefix
        )
        metric_dicts.append(metrics_dict)

    if scheduler:
        scheduler.step()
        learning_rate = scheduler.get_last_lr()[0]

    loss_dict = loss.collapse_history(prefix = prefix)

    # track time of training loop
    elapsed_time = time.time() - start_time
    elapsed_time_hms = str(timedelta(seconds = elapsed_time)).split('.')[0]

    train_dict = {f'{prefix}timestamp': timestamp,
                  f'{prefix}elapsed_time': elapsed_time,
                  f'{prefix}elapsed_time_hms': elapsed_time_hms,
                  f'{prefix}learning_rate': learning_rate,
                  f'{prefix}grad_norm': grad_norm}
    train_dict.update(loss_dict)

    # average evaluation metrics if we evaluated
    if evaluate and metric_dicts:

        avg_metrics_dict = {}
        metrics = list(metric_dicts[0].keys())

        for metric in metrics:
            metric_list = [float(metric_dict[metric]) for metric_dict in metric_dicts]
            avg_metrics_dict[f'{metric}_avg'] = float(np.mean(metric_list))
            avg_metrics_dict[f'{metric}_std'] = float(np.std(metric_list))

        train_dict.update(avg_metrics_dict)

    return train_dict


# @profile
def train_epoch(config, model, fabric, dataloader, 
                optimizer, loss, scheduler = None,
                prefix = 'train/',  evaluate = False): 

    device = model.device
    model.train()
    
    start_time = time.time()
    timestamp = get_timestamp()

    learning_rate = optimizer.param_groups[0]['lr']

    n_batches = 0
    n_frames = 0
    metric_dicts = []
    grad_norms = []
    
    for j, batch in enumerate(dataloader):

        if j == config.training.debug_ix: 
            break
    
        views = [view.to(device) for view in batch.views]
        coords = batch.coords.to(device)
        vis = batch.vis
        cgroup = batch.cgroup 
        vis_2d = batch.vis_2d
        query_times = batch.query_times
        p2d = batch.p2d.to(device) if batch.p2d is not None else None

        if p2d is None:
            query_coords = coords[:, query_times[0], torch.arange(len(query_times[0]))]
        else:
            assert p2d.shape[1] == 1
            query_coords = p2d[:, 0, query_times[0], torch.arange(len(query_times[0]))]
        
        # fallback if visibilities are not provided
        # if vis is None: 
        #     vis = get_vis_true(coords)

        if cgroup: 
            cgroup = [dict_to_device(cam_dict, device) for cam_dict in cgroup]

        optimizer.zero_grad()

        outputs = model(
            views = list(views), 
            coords = query_coords,
            query_times = query_times,
            camera_group = cgroup)

        coords_pred = outputs['coords_pred']
        vis_pred = outputs['vis_pred']

        total_loss = loss(
            model = model, 
            outputs = outputs,
            coords_true = coords, 
            vis_true = vis,
            vis_true_cams = vis_2d,
            cgroup = cgroup, 
            p2d = p2d, 
            device = coords_pred.device)

        # if not torch.any(torch.isnan(total_loss)):
            # report = reporter.report()

        # if torch.any(torch.isnan(total_loss)):
        #     print(total_loss)
            
        fabric.backward(total_loss)

        # Calculate gradient norm
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.detach().data.norm(2).item() ** 2
        grad_norms.append(grad_norm ** 0.5)

        fabric.clip_gradients(model, optimizer, 
            max_norm = config.training.max_grad_norm, 
            error_if_nonfinite = True)

        optimizer.step()
        optimizer.zero_grad()
        # else:
        #     print('WARNING: nan loss')
        
        if evaluate and coords.shape[-1] == 3:
            if p2d is not None:
                C = cgroup[0]['center']
                vis_for_norm = vis.to(coords.device) if vis is not None else get_vis_true(coords)
                _, pred_md = normalize_by_mean_depth(coords_pred, vis_for_norm, C)
                _, tgt_md  = normalize_by_mean_depth(coords, vis_for_norm, C)
                coords_pred = C + (coords_pred - C) * (tgt_md / pred_md)

            metrics_dict = get_eval_metrics(
                vis_pred = vis_pred,
                vis_true = vis,
                coords_pred = coords_pred,
                coords_true = coords,
                prefix = prefix
            )
            metric_dicts.append(metrics_dict)

        n_batches += 1
        n_frames += coords.shape[1]

    # print_memory(device)

    if scheduler:
        scheduler.step()
        learning_rate = scheduler.get_last_lr()[0]

    loss_dict = loss.collapse_history(prefix = prefix)

    # track time of training loop
    elapsed_time = time.time() - start_time
    elapsed_time_hms = str(timedelta(seconds = elapsed_time)).split('.')[0]

    train_dict = {f'{prefix}timestamp': timestamp,
                  f'{prefix}elapsed_time': elapsed_time,
                  f'{prefix}elapsed_time_hms': elapsed_time_hms,
                  f'{prefix}batches_per_epoch': n_batches,
                  f'{prefix}frames_per_epoch': n_frames,
                  f'{prefix}learning_rate': learning_rate,
                  f'{prefix}grad_norm_avg': float(np.mean(grad_norms))}
    train_dict.update(loss_dict)

    # average evaluation metrics if we evaluated
    if evaluate and metric_dicts:

        avg_metrics_dict = {}
        metrics = list(metric_dicts[0].keys())

        for metric in metrics:
            metric_list = [float(metric_dict[metric]) for metric_dict in metric_dicts]
            avg_metrics_dict[f'{metric}_avg'] = float(np.mean(metric_list))
            avg_metrics_dict[f'{metric}_std'] = float(np.std(metric_list))

        train_dict.update(avg_metrics_dict)

    return train_dict


def average_metrics(dicts, prefix, name = None):
    ''' average a list of metric dicts. metric keys look like
    "{prefix}{metric}". When name is given, the per-dataset metrics are
    written to their own top-level wandb folder "{prefix_tag}_{name}/{metric}"
    (e.g. "val_<dataset>/mte"), keeping them separate from the summary
    metrics that live under the "{prefix}" folder (e.g. "val/mte"). '''
    out = {}
    for metric in dicts[0].keys():
        metric_list = [float(d[metric]) for d in dicts]
        key = metric
        if name is not None:
            prefix_tag = prefix.rstrip('/')
            key = f'{prefix_tag}_{name}/{metric[len(prefix):]}'
        out[f'{key}_avg'] = float(np.mean(metric_list))
        out[f'{key}_std'] = float(np.std(metric_list))
    return out


def test_epoch(config, model, dataloader, loss = None,
               prefix = 'test/', evaluate = False):

    device = model.device
    model.eval()

    start_time = time.time()
    timestamp = get_timestamp()

    n_batches = 0
    n_frames = 0
    metric_dicts = []
    metric_datasets = []  # dataset name per entry in metric_dicts (batch_size=1)

    for j, batch in enumerate(dataloader):

        if j == config.training.debug_ix: 
            break
    
        views = [view.to(device) for view in batch.views]
        coords = batch.coords.to(device)
        vis = batch.vis
        cgroup = batch.cgroup
        vis_2d = batch.vis_2d
        query_times = batch.query_times
        p2d = batch.p2d.to(device) if batch.p2d is not None else None

        if p2d is None:
            query_coords = coords[:, query_times[0], torch.arange(len(query_times[0]))]
        else:
            assert p2d.shape[1] == 1
            query_coords = p2d[:, 0, query_times[0], torch.arange(len(query_times[0]))]
        
        # fallback if visibilities are not provided
        # if vis is None: 
        #     vis = get_vis_true(coords)

        if cgroup: 
            cgroup = [dict_to_device(cam_dict, device) for cam_dict in cgroup]
                       
        # get model predictions
        with torch.no_grad():
            outputs = model(
                views = list(views), 
                coords = query_coords,
                query_times = query_times,
                camera_group = cgroup)
        
        coords_pred = outputs['coords_pred']
        vis_pred = outputs['vis_pred']

        if loss is not None:
            total_loss = loss(
                model = model, 
                outputs = outputs,
                coords_true = coords, 
                vis_true = vis,
                vis_true_cams = vis_2d,
                cgroup = cgroup, 
                p2d = p2d, 
                device = coords_pred.device)

        if evaluate and coords.shape[-1] == 3:
            if p2d is not None:
                C = cgroup[0]['center']
                vis_for_norm = vis.to(coords.device) if vis is not None else get_vis_true(coords)
                _, pred_md = normalize_by_mean_depth(coords_pred, vis_for_norm, C)
                _, tgt_md  = normalize_by_mean_depth(coords, vis_for_norm, C)
                coords_pred = C + (coords_pred - C) * (tgt_md / pred_md)

            metrics_dict = get_eval_metrics(
                vis_pred = vis_pred,
                vis_true = vis,
                coords_pred = coords_pred,
                coords_true = coords,
                prefix = prefix
            )
            metric_dicts.append(metrics_dict)
            metric_datasets.append(batch.sample_info.get('dataset', 'unknown'))

        n_batches += 1
        n_frames += coords.shape[1]

    # track time of eval loop
    elapsed_time = time.time() - start_time
    elapsed_time_hms = str(timedelta(seconds = elapsed_time)).split('.')[0]

    val_dict = {f'{prefix}timestamp': timestamp,
                f'{prefix}elapsed_time': elapsed_time,
                f'{prefix}elapsed_time_hms': elapsed_time_hms,
                f'{prefix}batches_per_epoch': n_batches,
                f'{prefix}frames_per_epoch': n_frames}

    if loss is not None:
        loss_dict = loss.collapse_history(prefix = prefix)
        val_dict.update(loss_dict)

    # average evaluation metrics if we evaluated
    if evaluate and metric_dicts:

        # overall averages, e.g. "val/mte_avg"
        val_dict.update(average_metrics(metric_dicts, prefix))

        # per-dataset averages in their own folder, e.g. "val_<dataset_name>/mte_avg"
        for dataset_name in sorted(set(metric_datasets)):
            dataset_dicts = [d for d, name in zip(metric_dicts, metric_datasets)
                             if name == dataset_name]
            val_dict.update(average_metrics(dataset_dicts, prefix, name = dataset_name))

    return val_dict
