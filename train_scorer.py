#!/usr/bin/env python3
"""Train the track-quality scorer (ScorerEncoder) with a precision-weighted triplet loss.

Cloned from train.py. Each step turns one loaded clean sample into a (good, bad, anchor)
triplet (see datasets/scorer_corruption.py), scores all three with the ScorerEncoder
(per-point scores [b, k]), and trains with TripletScorerLoss. The V-JEPA backbone is frozen
(config video_encoder_requires_grad=false); the query encoder + decoder + pooling/score
heads are trained, warm-started from a trained tracker checkpoint.

    pixi run python train_scorer.py --config-path configs/config_scorer.toml --devices 1
"""

import warnings
warnings.filterwarnings('ignore')

import logging
for _name in ('torch._dynamo', 'torch._inductor'):
    logging.getLogger(_name).setLevel(logging.ERROR)

import argparse
import os
import time
import signal

# Import wandb BEFORE torch/lightning: its extension pulls in the pixi env's newer
# libstdc++ (CXXABI_1.3.15) first, which matplotlib/torchmetrics (via lightning) then
# reuse. Importing torch first instead loads the node's older system libstdc++, which on
# some queues (e.g. gpu_l4) lacks CXXABI_1.3.15 and crashes the matplotlib import. This
# mirrors train.py's import order.
import wandb

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from lightning.fabric import Fabric
from schedulefree import AdamWScheduleFree

from posetail.datasets.posetail_dataset import PosetailDataset, custom_collate
from posetail.datasets.scorer_corruption import make_triplet, build_corruptors
from posetail.posetail.scorer_encoder import ScorerEncoder
from posetail.posetail.losses_scorer import TripletScorerLoss
from train_utils import (load_config, save_config, set_seeds, write_json,
                         build_optimizer_param_groups, load_checkpoint, save_checkpoint,
                         total_to_per_gpu, dict_to_device, get_timestamp)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', default='./configs/config_scorer.toml')
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--devices', default=-1)
    parser.add_argument('--strategy', default='ddp_find_unused_parameters_true')
    parser.add_argument('--num-nodes', default=1)
    parser.add_argument('--precision', default='32-true')
    return parser.parse_args()


def build_optimizer(model, config, fabric, lr):
    """Optimizer setup mirroring train.py. Frozen (requires_grad=False) params are skipped
    in every branch. Supports SF-Muon, schedule-free AdamW, and plain AdamW."""
    encoder_lr_scale = config.training.optimizer.get('encoder_lr_scale', 1.0)
    wd = config.training.optimizer.weight_decay

    if config.training.scheduler_type == 'muon':
        from torch.optim import Muon as TorchMuon
        from posetail.posetail.muon import DualOptimizer
        adj = config.training.optimizer.get('muon_adjust_lr_fn', 'match_rms_adamw')
        muon_scale = config.training.optimizer.get('muon_lr_scale', 1.0)
        dec_substr = ('decoder.cross_attns', 'decoder.mlps', 'decoder.camera_attns',
                      'decoder.temporal_attns', 'decoder.latent_carry')
        scene_ids = {id(p) for p in model.scene_encoder.parameters()} \
            if hasattr(model, 'scene_encoder') else set()
        muon_dec, muon_enc, adamw_slow, adamw_base = [], [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is2d = (p.ndim == 2 and name.endswith('.weight'))
            if is2d and 'scene_encoder.encoder.blocks' in name:
                muon_enc.append(p)
            elif is2d and 'scene_encoder.kv_proj' in name:
                muon_dec.append(p)
            elif is2d and any(s in name for s in dec_substr):
                muon_dec.append(p)
            elif id(p) in scene_ids:
                adamw_slow.append(p)
            else:
                adamw_base.append(p)
        muon_groups = [{'params': muon_dec, 'lr': lr * muon_scale, 'weight_decay': wd}]
        if muon_enc:
            muon_groups.append({'params': muon_enc, 'lr': lr * encoder_lr_scale, 'weight_decay': wd})
        adamw_groups = [{'params': adamw_base, 'lr': lr, 'weight_decay': wd}]
        if adamw_slow:
            adamw_groups.append({'params': adamw_slow, 'lr': lr * encoder_lr_scale, 'weight_decay': wd})
        if fabric.is_global_zero:
            print(f"Muon ({adj}): dec {len(muon_dec)} | enc {len(muon_enc)} | "
                  f"adamw_base {len(adamw_base)} | adamw_slow {len(adamw_slow)}")
        sf = config.training.optimizer.get('muon_schedulefree', False)
        warmup = total_to_per_gpu(config.training.optimizer.get('warmup_steps', 0), fabric.world_size)
        if sf:
            from schedulefree import ScheduleFreeWrapper
            base_muon = TorchMuon(muon_groups, lr=lr, weight_decay=0.0,
                                  momentum=config.training.optimizer.get('muon_momentum', 0.95),
                                  adjust_lr_fn=adj)
            opt_adam = AdamWScheduleFree(adamw_groups, lr=lr, weight_decay=wd, warmup_steps=warmup,
                                         betas=(config.training.optimizer.get('beta1', 0.9),
                                                config.training.optimizer.get('beta2', 0.999)))
            base_muon, opt_adam = fabric.setup_optimizers(base_muon, opt_adam)
            opt_muon = ScheduleFreeWrapper(base_muon, momentum=0.9, weight_decay_at_y=wd)
        else:
            opt_muon = TorchMuon(muon_groups, lr=lr, weight_decay=wd,
                                 momentum=config.training.optimizer.get('muon_momentum', 0.95),
                                 adjust_lr_fn=adj)
            opt_adam = torch.optim.AdamW(adamw_groups, lr=lr, weight_decay=wd,
                                         betas=(config.training.optimizer.get('beta1', 0.9),
                                                config.training.optimizer.get('beta2', 0.95)))
            opt_muon, opt_adam = fabric.setup_optimizers(opt_muon, opt_adam)
        return DualOptimizer(opt_muon, opt_adam,
                             muon_warmup_steps=total_to_per_gpu(
                                 config.training.optimizer.get('muon_warmup_steps', 0),
                                 fabric.world_size))

    # non-muon: filter frozen params out of the single param list
    params = [p for p in model.parameters() if p.requires_grad]
    if config.training.scheduler_type == 'schedulefree':
        warmup = total_to_per_gpu(config.training.optimizer.get('warmup_steps', 0), fabric.world_size)
        optimizer = AdamWScheduleFree(params, lr=lr, weight_decay=wd, warmup_steps=warmup,
                                      betas=(config.training.optimizer.get('beta1', 0.9),
                                             config.training.optimizer.get('beta2', 0.999)))
    else:
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd,
                                      amsgrad=config.training.optimizer.get('amsgrad', False))
    return fabric.setup_optimizers(optimizer)


def _to_device(batch, device):
    batch.views = [v.to(device) for v in batch.views]
    batch.coords = batch.coords.to(device)
    if batch.cgroup:
        batch.cgroup = [dict_to_device(c, device) for c in batch.cgroup]
    return batch


def run(config_path, fabric):
    torch.set_float32_matmul_precision('medium')
    config = load_config(config_path)
    set_seeds(config.training.seed)

    interrupted = False
    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        print(f"\n[rank {fabric.global_rank}] signal {sig}, finishing after this iteration...")
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    train_dataset = PosetailDataset(config, split='train')
    sampler = DistributedSampler(train_dataset, num_replicas=fabric.world_size,
                                 rank=fabric.global_rank, shuffle=True,
                                 seed=config.training.get('seed', None))
    train_loader = DataLoader(train_dataset, batch_size=config.dataset.batch_size,
                              collate_fn=custom_collate, sampler=sampler, shuffle=False,
                              num_workers=config.dataset.num_workers, prefetch_factor=2,
                              persistent_workers=True, pin_memory=True)
    train_loader = fabric.setup_dataloaders(train_loader)

    val = config.dataset.val.get('split_dir', None)
    if val:
        val_dataset = PosetailDataset(config, split='val')
        val_loader = DataLoader(val_dataset, batch_size=config.dataset.batch_size,
                                collate_fn=custom_collate, shuffle=True,
                                num_workers=config.dataset.num_workers, prefetch_factor=2,
                                persistent_workers=True, pin_memory=True)
        val_loader = fabric.setup_dataloaders(val_loader)

    if fabric.is_global_zero:
        os.makedirs(config.wandb.path, exist_ok=True)
        wandb.init(project=config.wandb.project_name, dir=config.wandb.path,
                   mode=config.wandb.mode, config=config)

    exp_dir = ''
    if fabric.is_global_zero and wandb.run is not None:
        exp_dir = wandb.run.dir
        json_path = os.path.join(exp_dir, 'results.json')
        save_config(config_path, os.path.join(exp_dir, 'config.toml'))

    # --- model (frozen backbone via config) ---
    scorer_kwargs = dict(config.scorer)
    corruption_cfg = scorer_kwargs.pop('corruption', {})
    model = ScorerEncoder(pool_num_heads=scorer_kwargs.get('pool_num_heads', 8),
                          score_hidden=scorer_kwargs.get('score_hidden', 64),
                          use_precision=scorer_kwargs.get('use_precision', True),
                          **config.model)
    model = fabric.setup(model)
    # one marked forward method that scores the whole triplet -> DDP sees a single
    # forward per iteration (avoids the multi-forward reducer pitfall).
    model.mark_forward_method('score_triplet')
    model.print_summary()

    base_lr = config.training.optimizer.learning_rate
    lr = base_lr * (fabric.world_size ** 0.5)
    optimizer = build_optimizer(model, config, fabric, lr)

    # warm-start from the trained tracker checkpoint (new heads stay at init via strict=False)
    checkpoint_path = config.training.get('checkpoint_path', None)
    if checkpoint_path:
        ckpt = load_checkpoint(config_path, checkpoint_path, model=model, device='cpu')
        model = ckpt['model']

    corruptor_3d, corruptor_2d = build_corruptors(corruption_cfg)
    train_loss = TripletScorerLoss(margin=scorer_kwargs.get('triplet_margin', 0.5),
                                   precision_reg_weight=scorer_kwargs.get('precision_reg_weight', 0.01),
                                   score_reg_weight=scorer_kwargs.get('score_reg_weight', 0.0))
    val_loss = TripletScorerLoss(margin=scorer_kwargs.get('triplet_margin', 0.5),
                                 precision_reg_weight=scorer_kwargs.get('precision_reg_weight', 0.01),
                                 score_reg_weight=scorer_kwargs.get('score_reg_weight', 0.0))

    iters_per_gpu = total_to_per_gpu(config.training.n_iterations, fabric.world_size)
    checkpoint_freq = total_to_per_gpu(config.training.checkpoint_freq, fabric.world_size)
    val_freq = total_to_per_gpu(config.training.val_freq, fabric.world_size)
    print_freq = total_to_per_gpu(config.training.print_freq, fabric.world_size)

    device = model.device
    train_iter = iter(train_loader)
    iter_time = time.time()
    start_iteration = 0
    for i in range(iters_per_gpu):
        if interrupted:
            break
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        global_i = start_iteration + i * fabric.world_size + fabric.local_rank
        result_dict = {'iteration': global_i}

        if hasattr(optimizer, 'train'):
            optimizer.train()
        model.train()

        start_time = time.time()
        batch = _to_device(batch, device)
        trip = make_triplet(batch, train_dataset, corruptor_3d, corruptor_2d, corruption_cfg)

        optimizer.zero_grad()
        scores, precision, labels = model.score_triplet(trip)
        total_loss = train_loss(scores, precision, labels)
        fabric.backward(total_loss)

        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.detach().data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        try:
            if hasattr(optimizer, '_opts'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm,
                                               error_if_nonfinite=True)
            else:
                fabric.clip_gradients(model, optimizer, max_norm=config.training.max_grad_norm,
                                      error_if_nonfinite=True)
            optimizer.step()
        except Exception as e:
            print(f"ERROR BAD GRADIENTS!! {e}")
            print(batch.sample_info)
        optimizer.zero_grad()

        result_dict.update(train_loss.collapse_history(prefix='train/'))
        result_dict['train/grad_norm'] = grad_norm
        result_dict['train/elapsed_time'] = time.time() - start_time

        # validation
        if val and i % val_freq == 0:
            if hasattr(optimizer, 'eval'):
                optimizer.eval()
            model.eval()
            with torch.no_grad():
                for j, vbatch in enumerate(val_loader):
                    if j >= config.training.get('val_batches', 20):
                        break
                    vbatch = _to_device(vbatch, device)
                    vtrip = make_triplet(vbatch, val_dataset, corruptor_3d, corruptor_2d, corruption_cfg)
                    vs, vlp, vl = model.score_triplet(vtrip)
                    val_loss(vs, vlp, vl)
            result_dict.update(val_loss.collapse_history(prefix='val/'))
            val_loss.reset_history()

        if fabric.is_global_zero:
            result_dict['train/iter_time'] = time.time() - iter_time
            iter_time = time.time()
            wandb.log(result_dict)
            if wandb.run is not None:
                write_json(json_path, result_dict)
            if i % print_freq == 0:
                print(result_dict)

        if ((i % checkpoint_freq == 0) or (i + 1 == iters_per_gpu)) and fabric.is_global_zero:
            save_checkpoint(model, optimizer, prefix=exp_dir, i=global_i, config=config)

        train_loss.reset_history()

    if fabric.is_global_zero:
        wandb.finish()


if __name__ == '__main__':
    args = parse_args()
    fabric = Fabric(accelerator=args.accelerator, devices=args.devices,
                    strategy=args.strategy, num_nodes=args.num_nodes, precision=args.precision)
    fabric.launch()
    run(config_path=args.config_path, fabric=fabric)
