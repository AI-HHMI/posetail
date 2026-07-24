#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import logging
for _name in (
    'torch._dynamo',
    'torch._dynamo.convert_frame',
    'torch._dynamo.symbolic_convert',
    'torch._dynamo.guards',
    'torch._dynamo.utils',
    'torch._inductor',
    'torch._inductor.compile_fx',
):
    logging.getLogger(_name).setLevel(logging.ERROR)

import argparse
import os
import wandb
import time
import signal

import torch
# Re-assert child logger levels after torch import, since torch may install
# its own handlers / reset levels during initialization.
for _name in (
    'torch._dynamo',
    'torch._dynamo.convert_frame',
    'torch._dynamo.symbolic_convert',
    'torch._dynamo.guards',
    'torch._dynamo.utils',
    'torch._inductor',
    'torch._inductor.compile_fx',
):
    logging.getLogger(_name).setLevel(logging.ERROR)
import torch.multiprocessing as mp
import torch.optim as optim
# from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, DistributedSampler

from lightning.fabric import Fabric

from posetail.datasets.posetail_dataset import PosetailDataset, custom_collate
from posetail.posetail.losses import *
from posetail.posetail.tracker import Tracker
from posetail.posetail.tracker_encoder import TrackerEncoder
from posetail.posetail.tracker_tapnext import TrackerTapNext
from posetail.posetail.train_utils import *

from schedulefree import AdamWScheduleFree

''' 
python train.py --config-path configs/config_default_2d.toml
python train.py --config-path configs/config_default_3d.toml --devices 1
python train.py --config-path configs/config_default_3d.toml --devices 1 2
pixi run python train.py --config-path configs/config_default_3d.toml --precision 32 --devices 1 
pixi run python train.py --config-path configs/tuning/config_encoder_3d_2d.toml --precision 32 --devices 1 --strategy ddp_find_unused_parameters_true
'''

def parse_args(): 
    '''
    parse command line arguments
    ''' 
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', 
        default = './configs/config.toml', 
        help = 'path to model configuration file (.toml)')

    parser.add_argument('--accelerator', 
        default = 'gpu', 
        help = 'accelerator to use for training: cpu, gpu, tpu, auto')

    # parser.add_argument('--devices', 
    #     nargs = '*',
    #     default = 'auto', 
    #     help = 'number of gpus to use, list of gpu indices to use, or auto to use all available gpus')
    
    parser.add_argument('--devices', 
        default = -1, 
        help = 'number of gpus to train the model on')

    parser.add_argument('--strategy', 
        default = 'ddp_find_unused_parameters_true', 
        help = 'training strategy, e.g. dp, ddp, ddp_spawn, ddp_find_unused_parameters_true, xla, deepspeed, fsdp')

    parser.add_argument('--num-nodes', 
        default = 1, 
        help = 'number of nodes to train the model on')

    parser.add_argument('--precision', 
        default = '32-true', 
        help = 'precision type with the option to use mixed precision, e.g. 32, 32-true, 16-mixed, bf16-mixed')

    args = parser.parse_args()

    return args

def run(config_path, fabric):

    # mp.set_start_method('spawn', force = True)
    torch.set_float32_matmul_precision('medium')

    config = load_config(config_path)
    seed = fabric.broadcast(resolve_seed(config.training.seed), src=0)
    set_seeds(seed)
    if fabric.is_global_zero:
        print(f"[seed] using seed={seed}"
              + (" (random)" if not config.training.get('seed') else ""))

    # signal handler for graceful interrupt
    interrupted = False
    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        print(f"\n[rank {fabric.global_rank}] Received signal {sig}, finishing after current iteration...")
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # set up training dataloader
    train_dataset = PosetailDataset(config, split = 'train')

    sampler = DistributedSampler(
        train_dataset,
        num_replicas = fabric.world_size, 
        rank = fabric.global_rank,
        shuffle = True,
        seed = seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = config.dataset.batch_size,
        collate_fn = custom_collate,
        sampler = sampler,
        shuffle = False,
        num_workers = config.dataset.num_workers,
        prefetch_factor = 2,
        persistent_workers = True,
        pin_memory = True,
        # decorrelate dataloader RNG across (rank, worker): every rank runs
        # set_seeds with the same broadcast seed, and torch's per-worker seed is
        # rank-independent, so without this every GPU would apply correlated
        # augmentation/subsampling. Folding global_rank in fixes the cross-GPU case.
        worker_init_fn = make_worker_init_fn(seed, fabric.global_rank),
    )

    train_loader = fabric.setup_dataloaders(train_loader)
    # keep a handle on the actual sampler so we can advance its epoch (below);
    # fabric may wrap the loader, so read it back through the wrapper.
    train_sampler = getattr(train_loader, 'sampler', sampler)

    # set up validation dataloader 
    val = config.dataset.val.get('split_dir', None)
    # val = False

    if val: 

        val_dataset = PosetailDataset(config, split = 'val')

        val_loader = DataLoader(
            val_dataset,
            batch_size = config.dataset.batch_size,
            collate_fn = custom_collate,
            shuffle = True,
            num_workers = config.dataset.num_workers,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            worker_init_fn = make_worker_init_fn(seed, fabric.global_rank),
        )


        val_loader = fabric.setup_dataloaders(val_loader)

    torch.autograd.set_detect_anomaly(False)
    
    if fabric.is_global_zero:
        os.makedirs(config.wandb.path, exist_ok=True)
        wandb.init(
            project = config.wandb.project_name,
            dir = config.wandb.path,
            mode = config.wandb.mode,
            config = config)

        # Record run identifiers so a run maps to its checkpoint folder: run_id is the
        # short id (e.g. 'g6cy77bp'); run_dir is the full 'run-<timestamp>-<id>' folder
        # name. Added to the wandb config (sortable columns in the runs table) and to
        # the saved config.toml below.
        run_id = run_dir = None
        if wandb.run is not None:
            run_id = wandb.run.id
            run_dir = os.path.basename(os.path.dirname(wandb.run.dir))
            wandb.config.update({'run_id': run_id, 'run_dir': run_dir})

    exp_dir = ''
    if fabric.is_global_zero and wandb.run is not None:
        exp_dir = wandb.run.dir
    if fabric.is_global_zero:
        json_path = os.path.join(exp_dir, 'results.json')

        wandb_config_path = os.path.join(exp_dir, 'config.toml')
        save_config(config_path, wandb_config_path,
                    extra={'run_id': run_id, 'run_dir': run_dir} if run_id else None)
        wandb.save(wandb_config_path, base_path = exp_dir)

    # device = torch.device(config.devices.device)
    if config.model['mode_3d'] == 'encoder':
        model = TrackerEncoder(**config.model)
    elif config.model['mode_3d'] == 'tapnext':
        model = TrackerTapNext(**config.model)
    else:
        model = Tracker(**config.model)
        
    model = fabric.setup(model)

    model.print_summary()

    base_lr = config.training.optimizer.learning_rate
    lr = base_lr * (fabric.world_size ** 0.5)

    # Optionally give the pretrained video encoder a lower (discriminative) LR
    # than the freshly-initialized decoder/query encoder. `encoder_lr_scale` is
    # a multiplier on the (world-size-scaled) base LR, so the ratio is preserved
    # when base_lr or the GPU count changes. Defaults to 1.0 == single group,
    # leaving behavior unchanged. When the encoder is frozen this group simply
    # has nothing to update; it activates once unfreeze_video_encoder fires.
    encoder_lr_scale = config.training.optimizer.get('encoder_lr_scale', 1.0)
    params = build_optimizer_param_groups(model, config, base_lr=lr)
    if encoder_lr_scale != 1.0 and fabric.is_global_zero:
        print(f"Discriminative LR enabled: encoder_lr_scale={encoder_lr_scale} "
              f"-> encoder lr={lr * encoder_lr_scale:.3e}, other lr={lr:.3e}")

    # set up optimizer
    if config.training.scheduler_type == 'muon':
        # Official torch.optim.Muon (orthogonalized momentum) on the 2D transformer matrices of the
        # decoder AND the encoder blocks; AdamW on everything else (heads/embeddings/norms/biases/
        # query-encoder + the kubric-critical patch_embed[5D]/pos_embed[3D], which aren't 2D anyway).
        # match_rms_adamw scaling makes Muon REUSE the AdamW lr. Two optimizers wrapped as one.
        from torch.optim import Muon as TorchMuon
        from posetail.posetail.muon import DualOptimizer
        adj = config.training.optimizer.get('muon_adjust_lr_fn', 'match_rms_adamw')
        muon_scale = config.training.optimizer.get('muon_lr_scale', 1.0)
        wd = config.training.optimizer.weight_decay
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
                # The scene K/V projection is a fresh 2D hidden adapter (4096->scene_proj_dim),
                # not part of the pretrained backbone -> route to Muon at the decoder base LR
                # (it was previously mis-routed to adamw_slow via the scene_ids catch-all).
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
            print(f"Muon ({adj}): dec {len(muon_dec)} @ {lr*muon_scale:.2e} | enc {len(muon_enc)} @ "
                  f"{lr*encoder_lr_scale:.2e} | adamw_base {len(adamw_base)} @ {lr:.2e} | adamw_slow "
                  f"{len(adamw_slow)} @ {lr*encoder_lr_scale:.2e}")
        # muon_schedulefree: wrap Muon in the official ScheduleFreeWrapper (schedule-free averaging
        # on top of orthogonalized momentum) + AdamWScheduleFree for the rest -- the apples-to-apples
        # comparison vs the AdamW-schedule-free baseline, and the best-of-both (Muon geometry + SF avg).
        sf = config.training.optimizer.get('muon_schedulefree', False)
        warmup = total_to_per_gpu(config.training.optimizer.get('warmup_steps', 0), fabric.world_size)
        if fabric.is_global_zero:
            print(f"  muon_schedulefree={sf}")
        if sf:
            # ScheduleFreeWrapper isn't an Optimizer subclass, so fabric.setup the BASE Muon first,
            # then wrap it. AdamWScheduleFree IS an Optimizer subclass -> setup normally.
            from schedulefree import ScheduleFreeWrapper  # AdamWScheduleFree imported at top
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
        optimizer = DualOptimizer(opt_muon, opt_adam,
                                  muon_warmup_steps=total_to_per_gpu(
                                      config.training.optimizer.get('muon_warmup_steps', 0),
                                      fabric.world_size))
    elif config.training.scheduler_type == 'schedulefree':
        warmup_steps = total_to_per_gpu(
            config.training.optimizer.get('warmup_steps', 0),
            fabric.world_size)
        optimizer = AdamWScheduleFree(
            params,
            lr=lr,
            weight_decay=config.training.optimizer.weight_decay,
            warmup_steps=warmup_steps,
            betas=(config.training.optimizer.get('beta1', 0.9),
                   config.training.optimizer.get('beta2', 0.999))
        )
    else:
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=config.training.optimizer.weight_decay,
            amsgrad=config.training.optimizer.amsgrad,
            fused=True)

    # DualOptimizer (muon) sets up its two inner optimizers via fabric internally; others here.
    if not hasattr(optimizer, '_opts'):
        optimizer = fabric.setup_optimizers(optimizer)

    # optionally load a model checkpoint 
    checkpoint_path = config.training.get('checkpoint_path', None)
    finetune = config.training.get('finetune', False)
    start_iteration = 0

    if checkpoint_path:

        if finetune: 
            checkpoint_dict = load_checkpoint(
                config_path, checkpoint_path, model = model, 
                device = 'cpu')
            model = checkpoint_dict['model']

        else: 
            checkpoint_dict = load_checkpoint(
                config_path, checkpoint_path, model = model, 
                optimizer = optimizer, device = 'cpu')
            
            model = checkpoint_dict['model']
            optimizer = checkpoint_dict['optimizer']
            start_iteration = checkpoint_dict['iteration']

    # compile the model
    # model.cnn.compile()

    # model.cnn.stem.compile()
    # model.cnn.fpn.compile()

    # model.corr_mlp.compile()
    # model.tsformer.compile()

    # if model.mode_3d == 'minicubes':
    #     model.minicube_v2v.compile()
    #     # model.view_attention.compile()
    # elif model.mode_3d == 'triplane':
    #     model.triplane_cnn.compile()

    
    # if model.mode_3d == 'encoder':
    #     model.query_encoder.compile()
    #     model.decoder.compile()

    if hasattr(model, 'get_feature_loss'):
        model.mark_forward_method('get_feature_loss')
    
    # NOTE: memory profiling causes a CPU memory leak
    # profiler = LineProfiler(
    #     train_epoch, model, model.forward, 
    #     model.forward_iteration, model.cnn.forward, 
    #     model.corr_mlp.forward, model.tsformer.forward
    # )
    # profiler.enable()

    # reporter = MemReporter(model)
    # print(reporter.report())
    # print('')

    # put metrics in terms of one gpu, since all logging/checkpointing 
    # will happen on the zero rank gpu
    iters_per_gpu = total_to_per_gpu(config.training.n_iterations, fabric.world_size)
    checkpoint_freq = total_to_per_gpu(config.training.checkpoint_freq, fabric.world_size) 
    eval_metric_freq = total_to_per_gpu(config.training.eval_metric_freq, fabric.world_size)
    val_freq = total_to_per_gpu(config.training.val_freq, fabric.world_size)
    print_freq = total_to_per_gpu(config.training.print_freq, fabric.world_size) 
    
    # set up LR scheduler
    scheduler = None
    if config.training.scheduler_type == 'onecyclelr': 
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer = optimizer,
            max_lr = lr,
            total_steps = iters_per_gpu,
            **config.training.scheduler)

    elif config.training.scheduler_type == 'multisteplr': 
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer = optimizer, 
            **config.training.scheduler)

    pccs = config.model.get('per_camera_cube_scale', False)
    train_loss = TotalLoss(**config.training.losses, per_camera_cube_scale=pccs)
    val_loss   = TotalLoss(**config.training.losses, per_camera_cube_scale=pccs)
    
    # total_params = sum(p.numel() for p in model.parameters())
    # print(total_params)

    # advance the DistributedSampler epoch on every pass so the base-clip order
    # reshuffles (otherwise it repeats the same order every epoch, since set_epoch
    # is never otherwise called with the infinite iterator below).
    epoch = 0
    if hasattr(train_sampler, 'set_epoch'):
        train_sampler.set_epoch(epoch)
    train_iter = iter(train_loader)

    iter_time = time.time()
    global_i = start_iteration
    for i in range(iters_per_gpu):

        if interrupted:
            break

        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            if hasattr(train_sampler, 'set_epoch'):
                train_sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        global_i = start_iteration + i * fabric.world_size + fabric.local_rank
        train_dataset.set_progress(global_i / max(config.training.n_iterations, 1))
        result_dict = {'iteration': global_i}
        evaluate = i % eval_metric_freq == 0

        # optionally unfreeze the video encoder once we reach the configured
        # iteration (video_encoder_requires_grad set to an int in the config).
        # Use a rank-independent iteration count so every rank flips on the same
        # step, keeping DDP gradient sync consistent.
        if hasattr(model, 'unfreeze_video_encoder'):
            sync_i = start_iteration + i * fabric.world_size
            if model.unfreeze_video_encoder(sync_i) and fabric.is_global_zero:
                print(f"[iter {global_i}] unfroze video encoder gradients")

        if hasattr(optimizer, 'train'):
            optimizer.train()
            
        train_dict = train_iteration(
            config = config,
            model = model,
            fabric = fabric,
            batch = batch, 
            optimizer = optimizer,
            loss = train_loss,
            scheduler = scheduler, 
            evaluate = evaluate)

        result_dict.update(train_dict)

        # evaluate model on validation dataset
        if val and i % val_freq == 0: 
            if hasattr(optimizer, 'eval'):
                optimizer.eval()

            val_dict = test_epoch(
                config = config,
                model = model, 
                dataloader = val_loader,
                loss = val_loss,
                prefix = 'val/',
                evaluate = evaluate)

            result_dict.update(val_dict)

        # log losses and eval metrics to wandb and print to console 
        if fabric.is_global_zero:
            result_dict['train/curriculum_intensity'] = train_dataset.curriculum_intensity()
            result_dict['train/iter_time'] = time.time() - iter_time
            result_dict['train/loader_time'] = ( result_dict['train/iter_time']
                                                 - result_dict['train/elapsed_time']
                                                 - result_dict.get('val/elapsed_time', 0.0) )
            iter_time = time.time()
            wandb.log(drop_nan_motion_metrics(result_dict))
            write_json(json_path, result_dict)
            wandb.save(json_path, base_path = exp_dir)

            if i % print_freq == 0:
                print(result_dict)
                
        # save a model checkpoint when the condition is met
        checkpoint_cond = ((i % checkpoint_freq == 0) or
                           (i + 1 == iters_per_gpu))

        if checkpoint_cond and fabric.is_global_zero:
            ckpt_path = save_checkpoint(model, optimizer, prefix = exp_dir, i = global_i, config = config)
            # Track the latest checkpoint in the wandb summary -> shows as a column in
            # the runs table; last-write wins, so it ends on the final checkpoint.
            if wandb.run is not None:
                wandb.run.summary['final_checkpoint'] = os.path.basename(ckpt_path)
                wandb.run.summary['final_checkpoint_iter'] = global_i

        train_loss.reset_history()
        val_loss.reset_history()

    # save checkpoint on interrupt
    if interrupted and fabric.is_global_zero:
        ckpt_path = save_checkpoint(model, optimizer, prefix = exp_dir, i = global_i, config = config)
        if wandb.run is not None:
            wandb.run.summary['final_checkpoint'] = os.path.basename(ckpt_path)
            wandb.run.summary['final_checkpoint_iter'] = global_i

    if fabric.is_global_zero:
        wandb.finish()


if __name__ == '__main__':

    args = parse_args()

    fabric = Fabric(
        accelerator = args.accelerator,
        devices = args.devices, 
        strategy = args.strategy,
        num_nodes = args.num_nodes,
        precision = args.precision)

    fabric.launch()
    run(config_path = args.config_path, fabric = fabric)
