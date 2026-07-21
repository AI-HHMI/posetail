import os
import json
import math
import time
import toml
import torch
import yaml
import random

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
from posetail.posetail.eval_metrics import get_eval_metrics, get_metrics_by_motion, get_direct_depth_metrics
from posetail.posetail.cube import get_camera_scale
from posetail.posetail.losses import get_vis_true, unroll_batch, normalize_by_mean_depth
from posetail.posetail.tracker import Tracker
from posetail.posetail.tracker_encoder import TrackerEncoder
from posetail.posetail.tracker_tapnext import TrackerTapNext

from schedulefree import AdamWScheduleFree

from einops import rearrange


def resolve_seed(seed):
    """0 or None => draw a fresh seed from OS entropy; otherwise pass through.

    Range [1, 2**32-1] so the result is a valid numpy seed and never re-triggers
    the sentinel. Use this when you want ``seed = 0`` in a config to mean "random".
    """
    if not seed:  # covers 0 and None
        seed = int.from_bytes(os.urandom(4), 'little') % (2**32 - 1) + 1
    return seed


def set_seeds(seed = 3, set_backends = True):

    # a falsy seed (0 / None) means "random run" -- the user doesn't care about
    # reproducibility, so don't pay for deterministic backends either.
    random_mode = not seed
    seed = resolve_seed(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # seeds for (multi) gpu operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # deterministic backends only for a fixed seed; in random mode let cudnn
    # benchmark for speed. note deterministic ops can be less efficient.
    if set_backends and not random_mode:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def make_worker_init_fn(base_seed, rank):
    """Build a DataLoader `worker_init_fn` that decorrelates RNG per (rank, worker).

    On `fork` (Linux default) every dataloader worker inherits the parent process's
    numpy RNG state verbatim; PyTorch auto-reseeds torch/`random` per worker but NOT
    numpy. Since PosetailDataset drives ~all of its sampling/augmentation through
    `np.random.*`, workers would otherwise emit correlated draws. Worse under DDP:
    every rank runs `set_seeds` with the same broadcast seed, and torch's per-worker
    seed (`torch.initial_seed()+worker_id`) is rank-independent, so even torch draws
    correlate rank-to-rank. Folding `rank` into the seed decorrelates both axes.

    Model init still uses the shared broadcast seed (untouched), so DDP parameter
    init stays identical across ranks -- only the dataloader streams are decorrelated.
    """
    def _init(worker_id):
        s = (int(base_seed) + int(rank) * 100003 + int(worker_id)) % 2**32
        np.random.seed(s)
        torch.manual_seed(s)
        random.seed(s)
    return _init


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


def save_config(config_path, new_config_path, extra=None):

    config = load_config(config_path, easy = False)

    # Merge in runtime-only fields (e.g. wandb run_id/run_dir) under [wandb] so the
    # saved config.toml records which run produced it.
    if extra:
        config.setdefault('wandb', {}).update(extra)

    with open(new_config_path, 'w') as toml_file:
        toml.dump(config, toml_file)
        
        
def write_json(json_path, results): 
    '''
    appends results to a json file
    '''
    with open(json_path, 'a') as json_file: 
        json_file.write(json.dumps(results) + '\n')


def build_optimizer_param_groups(model, config, base_lr):
    """Build the optimizer param list for `model`, matching the structure used in train.py.

    When `encoder_lr_scale != 1.0` the (pretrained) video encoder gets its own param group at a
    scaled LR, producing a 2-group optimizer. This grouping MUST be reproduced verbatim when
    reconstructing an optimizer to load a saved `optimizer_state` (e.g. for the schedule-free
    eval-weight swap), otherwise `load_state_dict` fails on a param-group mismatch. Returns
    `model.parameters()` for the default single-group case."""
    encoder_lr_scale = config.training.optimizer.get('encoder_lr_scale', 1.0)
    if encoder_lr_scale != 1.0 and hasattr(model, 'scene_encoder'):
        encoder_param_ids = {id(p) for p in model.scene_encoder.encoder.parameters()}
        encoder_params = [p for p in model.parameters() if id(p) in encoder_param_ids]
        other_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]
        return [{'params': other_params, 'lr': base_lr},
                {'params': encoder_params, 'lr': base_lr * encoder_lr_scale}]
    return model.parameters()


def apply_eval_weights(model, checkpoint, config, device):
    """Swap `model`'s params in place to the schedule-free *averaged* (eval) weights.

    The saved `model_state` holds the raw training weights; the averaged weights only exist
    inside the optimizer state. This reconstructs the schedule-free optimizer, loads the saved
    `optimizer_state`, and calls `.eval()` (which swaps the model's own parameter tensors in
    place). No-op (returns False) when the run is not schedule-free or `optimizer_state` is
    absent, and falls back gracefully (warn, return False) on any reconstruction error so an
    inference load never crashes."""
    if config.training.get('scheduler_type', None) != 'schedulefree':
        return False
    optimizer_state = checkpoint.get('optimizer_state', None)
    if optimizer_state is None:
        return False
    try:
        params = build_optimizer_param_groups(model, config, base_lr=1e-4)
        opt = AdamWScheduleFree(params, lr=1e-4)
        opt.load_state_dict(optimizer_state)
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        opt.eval()
        return True
    except Exception as e:
        print(f'  [warn] could not apply schedule-free eval-weight swap ({e}); '
              f'using raw model_state weights')
        return False


def save_checkpoint(model, optimizer, prefix, i, config = None):

    checkpoint_dir = safe_make(os.path.join(prefix, 'checkpoints'))

    checkpoint_path = os.path.join(checkpoint_dir,
        f'checkpoint_{str(i).zfill(8)}.pth')

    def _cpu_state():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    state_dict = {
        'iteration': i,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }

    # Schedule-free runs -- both AdamW-schedulefree (scheduler_type == 'schedulefree') and
    # muon_schedulefree (DualOptimizer with a ScheduleFreeWrapper'd Muon) -- maintain an averaged
    # "eval" weight point inside the optimizer that downstream inference should use. Bake BOTH
    # snapshots deterministically: force train mode and snapshot the raw (y) weights as model_state,
    # then eval mode and snapshot the averaged (x) weights as model_state_eval, then restore train
    # mode. This generalizes the previous schedulefree-only bake to muon (which was silently saving
    # the raw weights, and only by luck the averaged ones on saves that happened to land on a val
    # iteration), and removes the nondeterminism where the snapshot depended on whatever optimizer
    # mode the training loop left behind.
    sf_run = config is not None and (
        config.training.get('scheduler_type', None) == 'schedulefree'
        or config.training.optimizer.get('muon_schedulefree', False))
    sf = optimizer.optimizer if hasattr(optimizer, 'optimizer') else optimizer
    if sf_run and hasattr(sf, 'eval') and hasattr(sf, 'train'):
        sf.train()
        state_dict['model_state'] = _cpu_state()
        sf.eval()
        state_dict['model_state_eval'] = _cpu_state()
        sf.train()

    torch.save(state_dict, checkpoint_path)

    return checkpoint_path


def _widen_cross_attn_layer(Wq, bq, Wk, bk, Wv, bv, Wout, bout, num_heads, new_dim):
    """Function-preserving (Net2Net-style) widening of one cross-attention layer from the old
    attention width ``old_dim = Wq.shape[0]`` to ``new_dim`` (both = num_heads * head_dim).

    Per head, the old head is embedded into the first ``old_head_dim`` slots of the new head;
    the Q rows are scaled by ``sqrt(new_head_dim/old_head_dim)`` so the larger ``1/sqrt(head_dim)``
    softmax scale cancels and the attention scores are unchanged. Extra Q rows are zeroed and the
    extra ``out_proj`` columns are zeroed (so the new dims contribute nothing at load -> the output
    equals the old output exactly), while the extra K/V rows are small-random so gradients flow and
    the extra dims become trainable. Returns the 8 new tensors as a flat tuple in q,k,v,out order."""
    import math
    old_dim = Wq.shape[0]
    H = num_heads
    hd_old, hd_new = old_dim // H, new_dim // H
    s = math.sqrt(hd_new / hd_old)
    dev, dt = Wq.device, Wq.dtype

    def _embed(W, b, scale, rand_extra):
        in_dim = W.shape[1]
        Wn = torch.zeros(new_dim, in_dim, device=dev, dtype=dt)
        bn = torch.zeros(new_dim, device=dev, dtype=dt)
        for h in range(H):
            Wn[h * hd_new:h * hd_new + hd_old] = W[h * hd_old:(h + 1) * hd_old] * scale
            bn[h * hd_new:h * hd_new + hd_old] = b[h * hd_old:(h + 1) * hd_old] * scale
            if rand_extra:
                Wn[h * hd_new + hd_old:(h + 1) * hd_new] = \
                    torch.randn(hd_new - hd_old, in_dim, device=dev, dtype=dt) * 0.02
        return Wn, bn

    Wqn, bqn = _embed(Wq, bq, s, rand_extra=False)    # extra Q rows = 0
    Wkn, bkn = _embed(Wk, bk, 1.0, rand_extra=True)   # extra K rows small-random (bootstraps Q)
    Wvn, bvn = _embed(Wv, bv, 1.0, rand_extra=True)   # extra V rows small-random
    latent = Wout.shape[0]
    Woutn = torch.zeros(latent, new_dim, device=dev, dtype=dt)
    boutn = bout.clone()
    for h in range(H):
        Woutn[:, h * hd_new:h * hd_new + hd_old] = Wout[:, h * hd_old:(h + 1) * hd_old]
    return Wqn, bqn, Wkn, bkn, Wvn, bvn, Woutn, boutn


def _convert_cross_attn(param_dict, model):
    """Backward-compatible load-time conversion of decoder cross-attention params from the old
    ``nn.MultiheadAttention`` layout to the new ``DecoupledCrossAttention`` (split q/k/v/out
    ``nn.Linear``) layout. No-op when the checkpoint is already in the new layout, or the model
    has no new-style cross_attns (so old checkpoints/other architectures are untouched).

    nn.MultiheadAttention stored either a FUSED ``in_proj_weight`` [3E, E] (when kdim==embed_dim)
    or SPLIT ``{q,k,v}_proj_weight`` (when kdim!=embed_dim), always with a single ``in_proj_bias``
    [3E] and ``out_proj.{weight,bias}``. Decided per layer from the recovered vs target shapes:
      * cross_attn_dim == embed_dim (e.g. the j5y2ff92 finetune): EXACT split -> bit-identical
        predictions.
      * cross_attn_dim  > embed_dim: function-preserving per-head warm-start (see
        ``_widen_cross_attn_layer``) -> reproduces the old output at load, extra dims trainable.
    Mutates and returns ``param_dict``."""
    msd = model.state_dict()
    layer_ids = sorted({
        int(k.split('.')[2]) for k in msd
        if k.startswith('decoder.cross_attns.') and k.endswith('.q_proj.weight')
    })
    for i in layer_ids:
        p = f'decoder.cross_attns.{i}'
        if f'{p}.q_proj.weight' in param_dict:
            continue  # already in the new (split) layout
        if f'{p}.in_proj_weight' in param_dict:
            W = param_dict.pop(f'{p}.in_proj_weight')
            E = W.shape[0] // 3
            Wq, Wk, Wv = W[:E], W[E:2 * E], W[2 * E:]
        elif f'{p}.q_proj_weight' in param_dict:
            Wq = param_dict.pop(f'{p}.q_proj_weight')
            Wk = param_dict.pop(f'{p}.k_proj_weight')
            Wv = param_dict.pop(f'{p}.v_proj_weight')
            E = Wq.shape[0]
        else:
            continue  # no recognizable old cross-attn params for this layer
        b = param_dict.pop(f'{p}.in_proj_bias')
        bq, bk, bv = b[:E], b[E:2 * E], b[2 * E:]
        Wout = param_dict.pop(f'{p}.out_proj.weight')
        bout = param_dict.pop(f'{p}.out_proj.bias')

        new_dim = msd[f'{p}.q_proj.weight'].shape[0]   # target cross_attn_dim
        if new_dim == E:
            keys = (Wq, bq, Wk, bk, Wv, bv, Wout, bout)
        else:
            num_heads = model.decoder.cross_attns[i].num_heads
            keys = _widen_cross_attn_layer(Wq, bq, Wk, bk, Wv, bv, Wout, bout,
                                           num_heads=num_heads, new_dim=new_dim)
        (param_dict[f'{p}.q_proj.weight'], param_dict[f'{p}.q_proj.bias'],
         param_dict[f'{p}.k_proj.weight'], param_dict[f'{p}.k_proj.bias'],
         param_dict[f'{p}.v_proj.weight'], param_dict[f'{p}.v_proj.bias'],
         param_dict[f'{p}.out_proj.weight'], param_dict[f'{p}.out_proj.bias']) = keys
        print(f'  cross-attn convert {p}: MHA -> decoupled '
              f'({"exact" if new_dim == E else f"widen {E}->{new_dim}"})')
    return param_dict


def _interp_res_params(param_dict, model):
    """Backward-compatible load-time interpolation of the few resolution-dependent params of
    TrackerEncoder, so a checkpoint trained at one image_size can be loaded into a model
    configured at another (e.g. 256 -> 384 to finetune at higher resolution for finer features).
    No-op for every tensor whose shape already matches; only the known res-coupled tensors
    (encoder pos_embed, decoder heads_2d, pix_grid buffer) are touched on a shape mismatch."""
    import math
    import torch.nn.functional as F
    msd = model.state_dict()
    out = dict(param_dict)
    changed = False
    for k, v in list(param_dict.items()):
        if k not in msd or tuple(msd[k].shape) == tuple(v.shape):
            continue
        changed = True
        tgt = msd[k]
        if k.endswith('scene_encoder.pos_embed'):
            # [1, T'*H'*W', D] grid; interpolate spatially and/or temporally to the
            # model's grid. Handles an image_size change (T' fixed, the original case)
            # and an n_frames change (H'/W' fixed, e.g. a non-windowed 16-frame
            # checkpoint loaded into an 8-frame windowed model). Recover the
            # checkpoint's (T',H',W') grid -- spatial is square (H'==W') -- by trying
            # spatial-unchanged first, then temporal-unchanged.
            sT, sH, sW = model.scene_encoder._pe_grid
            D = v.shape[-1]; old_n = v.shape[1]
            if old_n % (sH * sW) == 0:
                oldT, oldHW = old_n // (sH * sW), sH      # n_frames changed
            else:
                oldT, oldHW = sT, int(round(math.sqrt(old_n / sT)))  # image_size changed
            if oldT * oldHW * oldHW != old_n:
                continue
            pe = v.reshape(1, oldT, oldHW, oldHW, D).permute(0, 4, 1, 2, 3).float()
            pe = F.interpolate(pe, size=(sT, sH, sW), mode='trilinear', align_corners=False)
            out[k] = pe.permute(0, 2, 3, 4, 1).reshape(1, sT * sH * sW, D).to(v.dtype)
            print(f'  res-interp {k}: {tuple(v.shape)} -> {tuple(out[k].shape)}')
        elif k.endswith('t_query_embed.weight') or k.endswith('t_target_embed.weight'):
            # [n_frames, D] learned per-frame position table; linearly resample the
            # frame dim (mirrors QueryEncoder._interp_time_embed's runtime resize).
            newF = tgt.shape[0]
            w = v.t().unsqueeze(0).float()                   # [1, D, oldF]
            w = F.interpolate(w, size=newF, mode='linear', align_corners=False)
            out[k] = w.squeeze(0).t().to(v.dtype)            # [newF, D]
            print(f'  res-interp {k}: {tuple(v.shape)} -> {tuple(out[k].shape)}')
        elif 'heads_2d' in k and k.endswith('.1.weight'):
            # Linear out = 2*image_size logits ([x|y] x P bins); resample the P-bin dim.
            oldP = v.shape[0] // 2; newP = tgt.shape[0] // 2; D = v.shape[1]
            w = v.reshape(2, oldP, D).permute(0, 2, 1).float()      # [2, D, oldP]
            w = F.interpolate(w, size=newP, mode='linear', align_corners=False)
            out[k] = w.permute(0, 2, 1).reshape(2 * newP, D).to(v.dtype)
            print(f'  res-interp {k}: {tuple(v.shape)} -> {tuple(out[k].shape)}')
        elif 'heads_2d' in k and k.endswith('.1.bias'):
            oldP = v.shape[0] // 2; newP = tgt.shape[0] // 2
            b = v.reshape(2, oldP).unsqueeze(1).float()             # [2, 1, oldP]
            b = F.interpolate(b, size=newP, mode='linear', align_corners=False)
            out[k] = b.squeeze(1).reshape(2 * newP).to(v.dtype)
            print(f'  res-interp {k}: {tuple(v.shape)} -> {tuple(out[k].shape)}')
        elif k.endswith('pix_grid'):
            out.pop(k)   # deterministic buffer (arange(image_size)); keep the model's own
            print(f'  res-interp drop {k} (recomputed buffer)')
    return out, changed


def _filter_shape_mismatch(param_dict, model):
    """Drop checkpoint tensors whose shape no longer matches the model (e.g. a latent_dim or
    scene_proj_dim change). Those params keep the model's fresh init -> warm-start from only the
    shape-compatible weights (typically the backbone). Returns (filtered_dict, dropped_keys)."""
    msd = model.state_dict()
    kept, dropped = {}, []
    for k, v in param_dict.items():
        if k in msd and tuple(msd[k].shape) != tuple(v.shape):
            dropped.append(k)
        else:
            kept[k] = v
    return kept, dropped


def load_checkpoint(config_path, checkpoint_path, model = None,
                    optimizer = None, device = None, eval_weights = 'auto'):
    """Load a checkpoint into a model (and optionally an optimizer for resuming training).

    eval_weights controls whether the schedule-free *averaged* (eval) weights are applied
    instead of the raw training weights stored in `model_state`:
      - 'auto' (default): apply eval weights iff `optimizer is None` (the inference/eval path).
        When an optimizer is passed (train.py resume/finetune), never swap -- training must
        continue from the raw weights and reloaded optimizer state.
      - True: always apply eval weights.
      - False: never apply eval weights (raw training weights; useful for debugging / smoke tests).
    Prefers a baked `model_state_eval` snapshot when present; otherwise reconstructs the
    optimizer to swap (works retroactively on older checkpoints)."""

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
        elif config.model.mode_3d == 'tapnext':
            model = TrackerTapNext(**config.model)
        else:
            model = Tracker(**config.model)

        model.to(device)

    print(f'loading model checkpoint {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path, map_location = device)
    param_dict = checkpoint['model_state']

    # convert old nn.MultiheadAttention cross-attn params to the DecoupledCrossAttention split
    # layout (no-op when already converted / model has no new-style cross_attns) -> old checkpoints
    # load and (at cross_attn_dim==latent_dim) predict identically
    param_dict = _convert_cross_attn(param_dict, model)

    # interpolate resolution-dependent params if the checkpoint was trained at a different
    # image_size (no-op when shapes already match -> backward-compatible)
    param_dict, res_changed = _interp_res_params(param_dict, model)

    # drop params whose shape no longer matches the model (e.g. a latent_dim / scene_proj_dim
    # change) -> warm-start from only the shape-compatible weights; the dropped ones keep the
    # model's fresh init. (strict=False ignores missing/unexpected keys but NOT size mismatches.)
    param_dict, dropped_keys = _filter_shape_mismatch(param_dict, model)
    arch_changed = len(dropped_keys) > 0
    if arch_changed:
        print(f'  [warn] {len(dropped_keys)} checkpoint params dropped (shape mismatch -> '
              f'reinitialized): {dropped_keys}')

    missing_keys, unexpected_keys = model.load_state_dict(param_dict, strict = False)
    print(f'received missing keys: {missing_keys}')
    print(f'received unexpected keys: {unexpected_keys}')

    checkpoint_dict = {'model': model}

    # apply schedule-free averaged (eval) weights for the inference/eval path. Skip entirely when
    # resolution interpolation changed any tensor shape: the optimizer's averaged buffers / baked
    # snapshot are at the original resolution and would shape-mismatch the interpolated params.
    do_swap = (eval_weights is True) or (eval_weights == 'auto' and optimizer is None)
    if do_swap and (res_changed or arch_changed):
        print('  [warn] param shapes changed (res-interp or arch mismatch); skipping eval-weight '
              'swap (using raw model_state for the loaded params)')
    elif do_swap:
        eval_param_dict = checkpoint.get('model_state_eval', None)
        if eval_param_dict is not None:
            # baked fast path: no optimizer reconstruction needed
            eval_param_dict = _convert_cross_attn(dict(eval_param_dict), model)
            eval_param_dict, _ = _interp_res_params(eval_param_dict, model)
            model.load_state_dict(eval_param_dict, strict = False)
        else:
            apply_eval_weights(model, checkpoint, config, device)

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
    elif config.model.mode_3d == 'tapnext':
        model = TrackerTapNext(**config.model)
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

def format_camera(cam, offset_dict, cam_type, device, ext_override=None):
    """Build a camera dict from an aniposelib Camera.

    Static cameras (default): 'ext'/'ext_inv' are (4,4) and 'center' is (3,).
    Moving cameras: pass ext_override of shape (T,4,4) (per-frame world->cam) and
    'ext'/'ext_inv' become (T,4,4), 'center' becomes (T,3). Intrinsics ('mat',
    'dist', 'size', 'offset') are time-invariant either way. A 'moving' bool records
    which representation this dict carries so downstream code can branch cheaply.
    """
    if ext_override is None:
        ext = torch.as_tensor(cam.get_extrinsics_mat(), device=device, dtype=torch.float)
    else:
        ext = torch.as_tensor(ext_override, device=device, dtype=torch.float)  # (T,4,4)

    cam_dict = {
        'name': cam.get_name(),
        'type': cam_type, # pinhole, fisheye
        'ext': ext,
        'mat': torch.as_tensor(cam.get_camera_matrix(), device = device, dtype = torch.float),
        'dist': torch.as_tensor(cam.dist, device = device, dtype = torch.float),
        'size': torch.as_tensor(cam.get_size(), device = device, dtype = torch.int),
    }

    if offset_dict:
        offset = offset_dict[cam_dict['name']][:2]
        cam_dict['offset'] = torch.as_tensor(offset, device = device, dtype = torch.float)
    else:
        cam_dict['offset'] = torch.as_tensor([0.0, 0.0], device = device, dtype = torch.float)

    cam_dict['ext_inv'] = torch.linalg.inv(ext)  # batched over leading T if present

    # center = -R^T t, per frame if ext is (T,4,4). '...ji,...j->...i' == -(R^T t).
    R = ext[..., :3, :3]
    t = ext[..., :3, 3]
    cam_dict['center'] = -torch.einsum('...ji,...j->...i', R, t)
    cam_dict['moving'] = ext.ndim == 3

    return cam_dict

def format_camera_group(camera_group, offset_dict, cam_type, device, moving_ext=None):
    """moving_ext: optional {cam_name: (T,4,4)} of per-frame extrinsics. Cameras absent
    from the dict (or when moving_ext is None) fall back to their static extrinsic."""
    return [format_camera(cam, offset_dict, cam_type, device,
                          ext_override=(None if moving_ext is None
                                        else moving_ext.get(cam.get_name())))
            for cam in camera_group.cameras]

def dict_to_device(dd, device):

    dout = dict()

    for k, v in dd.items():
        if isinstance(v, torch.Tensor):
            dout[k] = v.to(device)
        else:
            dout[k] = v

    return dout


def _eval_cube_scale(cgroup, query_coords):
    """median-over-cameras world-units-per-pixel (B,) for cross-dataset delta_x/jaccard
    normalization. Returns None on failure (-> raw world-unit thresholds)."""
    try:
        return torch.median(get_camera_scale(cgroup, query_coords), dim=0).values
    except Exception:
        return None

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

    # Per-camera occlusion query term (occlusion_embedding). Gated on the model attribute so
    # Tracker/TrackerTapNext (whose forward doesn't accept occlusion) are unaffected.
    model_kwargs = {}
    if getattr(model, 'occlusion_embedding', False):
        model_kwargs['occlusion'] = getattr(batch, 'query_occlusion', None)

    # with fabric.autocast():
    outputs = model(
        views = list(views),
        coords = query_coords,
        query_times = query_times,
        camera_group = cgroup,
        **model_kwargs)

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

    # Track the learnable soft-argmax temperatures (exp of unclamped log-params). An upward
    # drift is the suspected driver of isolated grad-norm spikes; log so it can be confirmed
    # (and confirmed pinned once _grid_softmax clamps the effective temperature).
    temp_dict = {}
    for name, p in model.named_parameters():
        if name.endswith('log_soft_argmax_temp'):
            temp_dict[f'{prefix}soft_argmax_temp'] = float(torch.exp(p.detach()).item())
        elif name.endswith('log_subpixel_temp'):
            temp_dict[f'{prefix}subpixel_temp'] = float(torch.exp(p.detach()).item())

    # torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm,
    #                                error_if_nonfinite = False)

    try:
        if hasattr(optimizer, '_opts'):
            # DualOptimizer (muon+adamw): fabric.clip_gradients takes a single optimizer, so clip
            # the model's grads directly (no AMP unscale needed at 32/bf16 precision).
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm,
                                           error_if_nonfinite=True)
        else:
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
            if C.ndim == 2:   # moving cam: (T,3) -> (T,1,3) to broadcast per-frame over the point axis
                C = C[:, None, :]
            vis_for_norm = vis.to(coords.device) if vis is not None else get_vis_true(coords)
            _, pred_md = normalize_by_mean_depth(coords_pred, vis_for_norm, C)
            _, tgt_md  = normalize_by_mean_depth(coords, vis_for_norm, C)
            coords_pred = C + (coords_pred - C) * (tgt_md / pred_md)

        metrics_dict = get_eval_metrics(
            vis_pred = vis_pred,
            vis_true = vis,
            coords_pred = coords_pred,
            coords_true = coords,
            prefix = prefix,
            cube_scale = _eval_cube_scale(cgroup, query_coords),
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
    train_dict.update(temp_dict)
    train_dict.update(loss_dict)

    # average evaluation metrics if we evaluated
    if evaluate and metric_dicts:

        avg_metrics_dict = {}
        metrics = list(metric_dicts[0].keys())

        for metric in metrics:
            metric_list = [float(metric_dict[metric]) for metric_dict in metric_dicts]
            avg_metrics_dict[f'{metric}_avg'] = float(np.mean(metric_list))
            # avg_metrics_dict[f'{metric}_std'] = float(np.std(metric_list))

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
                if C.ndim == 2:   # moving cam: (T,3) -> (T,1,3) to broadcast per-frame over the point axis
                    C = C[:, None, :]
                vis_for_norm = vis.to(coords.device) if vis is not None else get_vis_true(coords)
                _, pred_md = normalize_by_mean_depth(coords_pred, vis_for_norm, C)
                _, tgt_md  = normalize_by_mean_depth(coords, vis_for_norm, C)
                coords_pred = C + (coords_pred - C) * (tgt_md / pred_md)

            metrics_dict = get_eval_metrics(
                vis_pred = vis_pred,
                vis_true = vis,
                coords_pred = coords_pred,
                coords_true = coords,
                prefix = prefix,
                cube_scale = _eval_cube_scale(cgroup, query_coords),
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
            # avg_metrics_dict[f'{metric}_std'] = float(np.std(metric_list))

        train_dict.update(avg_metrics_dict)

    return train_dict


def drop_nan_motion_metrics(metrics):
    ''' Return a copy of the metrics dict with NaN-valued mte_mo_* entries removed.
    Empty motion bins produce NaN mte_mo_* values; logging them clutters the wandb
    visualizer, so drop those keys for this step. Other NaN metrics are kept, since
    a NaN there is meaningful. '''
    def _is_nan(v):
        try:
            return math.isnan(float(v))
        except (TypeError, ValueError):
            return False
    return {k: v for k, v in metrics.items()
            if not ('mte_mo' in k and _is_nan(v))}


def average_metrics(dicts, prefix, name = None, nan_safe = False):
    ''' average a list of metric dicts. metric keys look like
    "{prefix}{metric}". When name is given, the per-dataset metrics are
    written to their own top-level wandb folder "{prefix_tag}_{name}/{metric}"
    (e.g. "val_<dataset>/mte"), keeping them separate from the summary
    metrics that live under the "{prefix}" folder (e.g. "val/mte").

    When nan_safe is True, nan entries are ignored (np.nanmean/np.nanstd),
    matching how losses are aggregated in BaseLoss.collapse_history. '''
    mean_fn = np.nanmean if nan_safe else np.mean
    std_fn = np.nanstd if nan_safe else np.std
    out = {}
    for metric in dicts[0].keys():
        metric_list = [float(d[metric]) for d in dicts]
        key = metric
        if name is not None:
            prefix_tag = prefix.rstrip('/')
            key = f'{prefix_tag}_{name}/{metric[len(prefix):]}'
        out[f'{key}_avg'] = float(mean_fn(metric_list))
        # out[f'{key}_std'] = float(std_fn(metric_list))
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
    loss_dicts = []       # per-batch loss snapshots (batch_size=1)
    loss_datasets = []    # dataset name per entry in loss_dicts

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

            # snapshot this batch's just-appended losses (one value per key per
            # forward), tagged by dataset, so they can be aggregated per-dataset
            # below. Keyed with `prefix` for compatibility with average_metrics.
            batch_losses = {f'{prefix}{name}': vals[-1]
                            for name, vals in loss.loss_history.items() if vals}
            loss_dicts.append(batch_losses)
            loss_datasets.append(batch.sample_info.get('dataset', 'unknown'))

        if evaluate and coords.shape[-1] == 3:
            if p2d is not None:
                C = cgroup[0]['center']
                if C.ndim == 2:   # moving cam: (T,3) -> (T,1,3) to broadcast per-frame over the point axis
                    C = C[:, None, :]
                vis_for_norm = vis.to(coords.device) if vis is not None else get_vis_true(coords)
                _, pred_md = normalize_by_mean_depth(coords_pred, vis_for_norm, C)
                _, tgt_md  = normalize_by_mean_depth(coords, vis_for_norm, C)
                coords_pred = C + (coords_pred - C) * (tgt_md / pred_md)

            cube_scale = _eval_cube_scale(cgroup, query_coords)
            metrics_dict = get_eval_metrics(
                vis_pred = vis_pred,
                vis_true = vis,
                coords_pred = coords_pred,
                coords_true = coords,
                prefix = prefix,
                cube_scale = cube_scale,
            )
            # Fast-motion breakdown: error by cube_scale-normalized displacement-from-query,
            # so val/mte_mo_fast etc. are a watchable, cross-dataset-comparable "fast motion" signal.
            metrics_dict.update(get_metrics_by_motion(
                coords_pred = coords_pred,
                coords_true = coords,
                vis_true = vis,
                query_times = query_times,
                cube_scale = cube_scale,
                prefix = prefix,
            ))
            # Per-camera direct-head error split into along-ray (depth) vs in-plane,
            # in world units, per dataset. Exposes the depth-axis failure that mte
            # (fused) and the softmax CE hide. nan (averaged nan-safe below) for
            # non-grid / single-cam batches with no rays_c.
            metrics_dict.update(get_direct_depth_metrics(
                pred_cams_direct = outputs.get('3d_pred_cams_direct'),
                tri_pred = outputs.get('3d_pred_triangulate'),
                rays_c = (outputs.get('grid') or {}).get('rays_c'),
                coords_true = coords,
                vis_true_cams = vis_2d,
                prefix = prefix,
                cgroup = cgroup,
            ))
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

        # overall averages, e.g. "val/mte_avg". nan_safe so metrics that are nan
        # for some datasets (e.g. dir_depth_rms on 2D/single-cam batches) don't
        # poison the summary average; non-nan metrics (mte, etc.) are unchanged.
        val_dict.update(average_metrics(metric_dicts, prefix, nan_safe = True))

        # per-dataset averages in their own folder, e.g. "val_<dataset_name>/mte_avg".
        # Optionally restrict which datasets get their own folder logged via
        # dataset.<split>.split_out (a list of dataset names). When unset (None)
        # every dataset is logged; the overall averages above are unaffected.
        split_name = prefix.strip('/')
        split_out = config.dataset.get(split_name, {}).get('split_out', None)
        for dataset_name in sorted(set(metric_datasets)):
            if split_out is not None and dataset_name not in split_out:
                continue
            dataset_dicts = [d for d, name in zip(metric_dicts, metric_datasets)
                             if name == dataset_name]
            val_dict.update(average_metrics(dataset_dicts, prefix, name = dataset_name,
                                            nan_safe = True))

    # per-dataset loss averages, e.g. "val_<dataset_name>/coords_loss_avg"
    if loss is not None and loss_dicts:
        for dataset_name in sorted(set(loss_datasets)):
            dataset_loss_dicts = [d for d, name in zip(loss_dicts, loss_datasets)
                                  if name == dataset_name]
            val_dict.update(average_metrics(dataset_loss_dicts, prefix,
                                            name = dataset_name, nan_safe = True))

    return val_dict
