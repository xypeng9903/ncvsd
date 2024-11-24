# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import persistence
from torch_utils import misc
import gc


#----------------------------------------------------------------------------
# Noise conditional variational score distillation.

from .networks_edm2 import PrecondUNetDecoder, PrecondUNetEncoder

@persistence.persistent_class
class NCVSDLoss:
    def __init__(
        self, 
        P_mean_sigma  = 0.4, 
        P_std_sigma   = 2.0, 
        P_mean_t      = 0.4, 
        P_std_t       = 2.0,
        gamma         = 0.414,
        sigma_data    = 0.5
    ):
        self.P_mean_sigma = P_mean_sigma
        self.P_std_sigma = P_std_sigma
        self.P_mean_t = P_mean_t
        self.P_std_t = P_std_t
        self.gamma = gamma
        self.sigma_data = sigma_data

    def __call__(
            self, 
            precond_enc: torch.nn.parallel.DistributedDataParallel, 
            precond_ctrl: torch.nn.parallel.DistributedDataParallel, 
            precond_dec: torch.nn.parallel.DistributedDataParallel, 
            images: torch.Tensor, 
            labels: torch.Tensor = None
    ):
        # 1. sample condition y
        rnd_normal_y = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal_y * self.P_std_sigma + self.P_mean_sigma).exp()
        y = images + torch.randn_like(images) * sigma
        
        # 2. denoising posterior sampling x ~ \mu(x|y)
        precond_enc.module.set_adapter('generative_denoiser')
        precond_ctrl.module.set_adapter('generative_denoiser')
        precond_dec.module.set_adapter('generative_denoiser')

        precond_enc.train()
        precond_ctrl.train()
        precond_dec.train()

        z = torch.randn_like(images)
        sigma_hat = sigma * (1 + self.gamma)
        y_hat = y + z * (sigma_hat ** 2 - sigma ** 2) ** 0.5
        enc_x, enc_skips = precond_enc(y_hat, sigma_hat, labels)
        ctrl_x, ctrl_skips = precond_ctrl(y, sigma, labels, condition_labels=z)
        x, logvar_x = precond_dec(y_hat, enc_x, enc_skips, sigma_hat, labels, additional_x=ctrl_x, additional_skips=ctrl_skips, return_logvar=True)

        # 3. diffuse
        rnd_normal_t = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        t = (rnd_normal_t * self.P_std_t + self.P_mean_t).exp()
        weight_t = (t ** 2 + self.sigma_data ** 2) / (t * self.sigma_data) ** 2
        xt = x + torch.randn_like(x) * t

        # 4. effective noise
        sigma_eff = 1 / (1 / sigma ** 2 + 1 / t ** 2) ** 0.5
        y_eff = (y / sigma ** 2 + xt / t ** 2) * sigma_eff ** 2

        # 5. score prediction
        with torch.no_grad():
            precond_enc.eval()
            precond_ctrl.eval()
            precond_dec.eval()
            
            # data score
            precond_enc.module.disable_adapters()
            precond_ctrl.module.disable_adapters()
            precond_dec.module.disable_adapters()

            enc_x, enc_skips = precond_enc(y_eff, sigma_eff, labels)
            s0 = precond_dec(y_eff, enc_x, enc_skips, sigma_eff, labels)

            # model score
            precond_enc.module.set_adapter('model_score')
            precond_ctrl.module.set_adapter('model_score')
            precond_dec.module.set_adapter('model_score')

            enc_x, enc_skips = precond_enc(xt, t, labels)
            ctrl_x, ctrl_skips = precond_ctrl(y, sigma, labels, condition_labels=xt)
            s = precond_dec(xt, enc_x, enc_skips, t, labels, additional_x=ctrl_x, additional_skips=ctrl_skips)

        # 6. variational score distillation
        loss_vsd = (weight_t / logvar_x.exp()) * (x - (s0 - s + x).detach()) ** 2 + logvar_x

        return loss_vsd

#----------------------------------------------------------------------------
# Denoising score mathcing.

@persistence.persistent_class
class DSMLoss:
    def __init__(
        self, 
        P_mean_sigma  = 0.4, 
        P_std_sigma   = 2.0, 
        P_mean_t      = -0.8, 
        P_std_t       = 1.6,
        gamma         = 0.414,
        sigma_data    = 0.5
    ):
        self.P_mean_sigma = P_mean_sigma
        self.P_std_sigma = P_std_sigma
        self.P_mean_t = P_mean_t
        self.P_std_t = P_std_t
        self.gamma = gamma
        self.sigma_data = sigma_data

    def __call__(
            self, 
            precond_enc: torch.nn.parallel.DistributedDataParallel, 
            precond_ctrl: torch.nn.parallel.DistributedDataParallel, 
            precond_dec: torch.nn.parallel.DistributedDataParallel, 
            images: torch.Tensor, 
            labels: torch.Tensor = None
    ):
        # 1. sample condition y
        rnd_normal_y = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal_y * self.P_std_sigma + self.P_mean_sigma).exp()
        y = images + torch.randn_like(images) * sigma
        
        # 2. denoising posterior sampling x ~ \mu(x|y)
        with torch.no_grad():
            precond_enc.module.set_adapter('generative_denoiser')
            precond_ctrl.module.set_adapter('generative_denoiser')
            precond_dec.module.set_adapter('generative_denoiser')

            precond_enc.eval()
            precond_ctrl.eval()
            precond_dec.eval()

            z = torch.randn_like(images)
            sigma_hat = sigma * (1 + self.gamma)
            y_hat = y + z * (sigma_hat ** 2 - sigma ** 2) ** 0.5
            enc_x, enc_skips = precond_enc(y_hat, sigma_hat, labels)
            ctrl_x, ctrl_skips = precond_ctrl(y, sigma, labels, condition_labels=z)
            x = precond_dec(y_hat, enc_x, enc_skips, sigma_hat, labels, additional_x=ctrl_x, additional_skips=ctrl_skips)

        # 3. diffuse
        rnd_normal_t = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        t = (rnd_normal_t * self.P_std_t + self.P_mean_t).exp()
        weight_t = (t ** 2 + self.sigma_data ** 2) / (t * self.sigma_data) ** 2
        xt = x + torch.randn_like(x) * t

        # 5. denoising score matching
        precond_enc.module.set_adapter('model_score')
        precond_ctrl.module.set_adapter('model_score')
        precond_dec.module.set_adapter('model_score')

        precond_enc.train()
        precond_ctrl.train()
        precond_dec.train()

        enc_x, enc_skips = precond_enc(xt, t, labels)
        ctrl_x, ctrl_skips = precond_ctrl(y, sigma, labels, condition_labels=xt)
        s, logvar_s = precond_dec(xt, enc_x, enc_skips, t, labels, additional_x=ctrl_x, additional_skips=ctrl_skips, return_logvar=True)

        loss_dsm = (weight_t / logvar_s.exp()) * (s - x.detach()) ** 2 + logvar_s

        return loss_dsm
    
#----------------------------------------------------------------------------
# Learning rate decay schedule used in the paper "Analyzing and Improving
# the Training Dynamics of Diffusion Models".

def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=10):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr

#----------------------------------------------------------------------------
# Warmup schedule for VSD

def vsd_warmup_schedule(cur_nimg, batch_size, ref_batches=10e3):
    if ref_batches == 0:
        return 1.0
    r = min(cur_nimg / (ref_batches * batch_size), 1)
    return r

#----------------------------------------------------------------------------
# Main training loop.

def training_loop(
    net,
    network_kwargs,
    lora_kwargs         = dict(r=16, lora_alpha=8),
    dataset_kwargs      = dict(class_name='training.dataset.ImageFolderDataset', path=None),
    encoder_kwargs      = dict(class_name='training.encoders.StabilityVAEEncoder'),
    data_loader_kwargs  = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=2, prefetch_factor=2),
    vsd_loss_kwargs     = dict(class_name='training.training_loop.NCVSDLoss'),
    dsm_loss_kwargs     = dict(class_name='training.training_loop.DSMLoss'),
    optimizer_kwargs    = dict(class_name='torch.optim.Adam', betas=(0.9, 0.99)),
    lr_kwargs           = dict(func_name='training.training_loop.learning_rate_schedule'),
    vsd_warmup_kwargs   = dict(func_name='training.training_loop.vsd_warmup_schedule'),
    ema_kwargs          = dict(class_name='training.phema.PowerFunctionEMA'),

    run_dir             = '.',      # Output directory.
    seed                = 0,        # Global random seed.
    batch_size          = 2048,     # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU. None = no limit.
    total_nimg          = 8<<30,    # Train for a total of N training images.
    slice_nimg          = None,     # Train for a maximum of N training images in one invocation. None = no limit.
    status_nimg         = 128<<10,  # Report status every N training images. None = disable.
    snapshot_nimg       = 8<<20,    # Save network snapshot every N training images. None = disable.
    checkpoint_nimg     = 128<<20,  # Save state checkpoint every N training images. None = disable.

    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    force_finite        = True,     # Get rid of NaN/Inf gradients before feeding them to the optimizer.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
):
    # Initialize.
    prev_status_time = time.time()
    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Validate batch size.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    assert total_nimg % batch_size == 0
    assert slice_nimg is None or slice_nimg % batch_size == 0
    assert status_nimg is None or status_nimg % batch_size == 0
    assert snapshot_nimg is None or (snapshot_nimg % batch_size == 0 and snapshot_nimg % 1024 == 0)
    assert checkpoint_nimg is None or (checkpoint_nimg % batch_size == 0 and checkpoint_nimg % 1024 == 0)

    # Setup dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    
    # Setup models.
    dist.print0(f'Load teacher model: {net}')
    with dnnlib.util.open_url(net, verbose=True) as f:
        data = pickle.load(f)
    net = data['ema'].to(device)
    
    dist.print0('Constructing networks...')

    def print_trainable_parameters(net, name):
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        dist.print0(f'Trainable / total params in {name}: {trainable_params} / {total_params} ({trainable_params / total_params:.2%})')

    precond_enc = PrecondUNetEncoder(**network_kwargs)
    precond_enc.init_from_pretrained(net)
    precond_enc.requires_grad_(False)
    precond_enc.add_adapter('generative_denoiser', **lora_kwargs)
    precond_enc.add_adapter('model_score', **lora_kwargs)
    precond_enc.to(device)
    print_trainable_parameters(precond_enc, 'precond_enc')

    precond_ctrl = PrecondUNetEncoder(**network_kwargs, is_controlnet=True, conditioning_channels=3) # TODO: get conditioning channels from dataset
    precond_ctrl.init_from_pretrained(net)
    for name, param in precond_ctrl.named_parameters():
        if 'controlnet' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    precond_ctrl.add_adapter('generative_denoiser', **lora_kwargs)
    precond_ctrl.add_adapter('model_score', **lora_kwargs)
    precond_ctrl.to(device)
    print_trainable_parameters(precond_ctrl, 'precond_ctrl')

    precond_dec = PrecondUNetDecoder(**network_kwargs)
    precond_dec.init_from_pretrained(net)
    precond_dec.requires_grad_(False)
    precond_dec.add_adapter('generative_denoiser', **lora_kwargs)
    precond_dec.add_adapter('model_score', **lora_kwargs)
    precond_dec.to(device)
    print_trainable_parameters(precond_dec, 'precond_dec')

    # Print network summary.
    if dist.get_rank() == 0:
        misc.print_module_summary(net, [
            torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device),
            torch.ones([batch_gpu], device=device),
            torch.zeros([batch_gpu, net.label_dim], device=device),
        ], max_nesting=2)

    del net
    

    # Setup training state.
    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0)
    vsd_loss_fn = dnnlib.util.construct_class_by_name(**vsd_loss_kwargs)
    dsm_loss_fn = dnnlib.util.construct_class_by_name(**dsm_loss_kwargs)

    # optimizer
    params = list(precond_enc.parameters()) + list(precond_ctrl.parameters()) + list(precond_dec.parameters())
    optimizer = dnnlib.util.construct_class_by_name(params=params, **optimizer_kwargs)
    
    # ddp
    ddp_enc = torch.nn.parallel.DistributedDataParallel(precond_enc, device_ids=[device], broadcast_buffers=False)
    ddp_ctrl = torch.nn.parallel.DistributedDataParallel(precond_ctrl, device_ids=[device], broadcast_buffers=False)
    ddp_dec = torch.nn.parallel.DistributedDataParallel(precond_dec, device_ids=[device], broadcast_buffers=False)
    
    # ema
    ema_enc = dnnlib.util.construct_class_by_name(net=precond_enc, **ema_kwargs) if ema_kwargs is not None else None
    ema_ctrl = dnnlib.util.construct_class_by_name(net=precond_ctrl, **ema_kwargs) if ema_kwargs is not None else None
    ema_dec = dnnlib.util.construct_class_by_name(net=precond_dec, **ema_kwargs) if ema_kwargs is not None else None

    # Load previous checkpoint and decide how long to train.
    checkpoint = dist.CheckpointIO(
        state=state, 
        precond_enc=precond_enc,
        precond_ctrl=precond_ctrl,
        precond_dec=precond_dec,
        vsd_loss_fn=vsd_loss_fn, 
        dsm_loss_fn=dsm_loss_fn,
        optimizer=optimizer, 
        ema_enc=ema_enc, 
        ema_ctrl=ema_ctrl, 
        ema_dec=ema_dec
    )
    checkpoint.load_latest(run_dir)
    stop_at_nimg = total_nimg
    if slice_nimg is not None:
        granularity = checkpoint_nimg if checkpoint_nimg is not None else snapshot_nimg if snapshot_nimg is not None else batch_size
        slice_end_nimg = (state.cur_nimg + slice_nimg) // granularity * granularity # round down
        stop_at_nimg = min(stop_at_nimg, slice_end_nimg)
    assert stop_at_nimg > state.cur_nimg
    dist.print0(f'Training from {state.cur_nimg // 1000} kimg to {stop_at_nimg // 1000} kimg:')
    dist.print0()

    # Main training loop.
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nimg)
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    prev_status_nimg = state.cur_nimg
    cumulative_training_time = 0
    start_nimg = state.cur_nimg
    stats_jsonl = None
    gc.collect()
    torch.cuda.empty_cache()
    start_time = time.time()
    while True:
        done = (state.cur_nimg >= stop_at_nimg)

        # Report status.
        if status_nimg is not None and (done or state.cur_nimg % status_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
            dist.print0(' '.join(['Status:',
                'kimg',         f"{training_stats.report0('Progress/kimg',                              state.cur_nimg / 1e3):<9.1f}",
                'time',         f"{dnnlib.util.format_time(training_stats.report0('Timing/total_sec',   state.total_elapsed_time)):<12s}",
                'sec/tick',     f"{training_stats.report0('Timing/sec_per_tick',                        cur_time - prev_status_time):<8.2f}",
                'sec/kimg',     f"{training_stats.report0('Timing/sec_per_kimg',                        cumulative_training_time / max(state.cur_nimg - prev_status_nimg, 1) * 1e3):<7.3f}",
                'maintenance',  f"{training_stats.report0('Timing/maintenance_sec',                     cur_time - prev_status_time - cumulative_training_time):<7.2f}",
                'cpumem',       f"{training_stats.report0('Resources/cpu_mem_gb',                       cpu_memory_usage / 2**30):<6.2f}",
                'gpumem',       f"{training_stats.report0('Resources/peak_gpu_mem_gb',                  torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}",
                'reserved',     f"{training_stats.report0('Resources/peak_gpu_mem_reserved_gb',         torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}",
            ]))
            cumulative_training_time = 0
            prev_status_nimg = state.cur_nimg
            prev_status_time = cur_time
            torch.cuda.reset_peak_memory_stats()

            # Flush training stats.
            training_stats.default_collector.update()
            if dist.get_rank() == 0:
                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                fmt = {'Progress/tick': '%.0f', 'Progress/kimg': '%.3f', 'timestamp': '%.3f'}
                items = [(name, value.mean) for name, value in training_stats.default_collector.as_dict().items()] + [('timestamp', time.time())]
                items = [f'"{name}": ' + (fmt.get(name, '%g') % value if np.isfinite(value) else 'NaN') for name, value in items]
                stats_jsonl.write('{' + ', '.join(items) + '}\n')
                stats_jsonl.flush()

            # Update progress and check for abort.
            dist.update_progress(state.cur_nimg // 1000, stop_at_nimg // 1000)
            if state.cur_nimg == stop_at_nimg and state.cur_nimg < total_nimg:
                dist.request_suspend()
            if dist.should_stop() or dist.should_suspend():
                done = True

        # Save network snapshot.
        if snapshot_nimg is not None and state.cur_nimg % snapshot_nimg == 0 and (state.cur_nimg != start_nimg or start_nimg == 0) and dist.get_rank() == 0:
            for ema, name in zip([ema_enc, ema_ctrl, ema_dec], ['enc', 'ctrl', 'dec']):
                ema_list = ema.get()
                ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
                for ema_net, ema_suffix in ema_list:
                    data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs, loss_fn=vsd_loss_fn)
                    data.ema = copy.deepcopy(ema_net).cpu().eval().requires_grad_(False).to(torch.float16)
                    fname = f'{name}-snapshot-{state.cur_nimg//1000:07d}{ema_suffix}.pkl'
                    dist.print0(f'Saving {fname} ... ', end='', flush=True)
                    with open(os.path.join(run_dir, fname), 'wb') as f:
                        pickle.dump(data, f)
                    dist.print0('done')
                    del data # conserve memory

        # Save state checkpoint.
        if checkpoint_nimg is not None and (done or state.cur_nimg % checkpoint_nimg == 0) and state.cur_nimg != start_nimg:
            checkpoint.save(os.path.join(run_dir, f'training-state-{state.cur_nimg//1000:07d}.pt'))
            # TODO: misc.check_ddp_consistency(net) 

        # Done?
        if done:
            break

        # Evaluate loss and accumulate gradients.
        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_nimg)
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp_enc, (round_idx == num_accumulation_rounds - 1)), \
                 misc.ddp_sync(ddp_ctrl, (round_idx == num_accumulation_rounds - 1)), \
                 misc.ddp_sync(ddp_dec, (round_idx == num_accumulation_rounds - 1)):
                
                images, labels = next(dataset_iterator)
                images = encoder.encode_latents(images.to(device))
                vsd_loss = vsd_loss_fn(
                    precond_enc=ddp_enc, 
                    precond_ctrl=ddp_ctrl, 
                    precond_dec=ddp_dec, 
                    images=images, 
                    labels=labels.to(device)
                )
                dsm_loss = dsm_loss_fn(
                    precond_enc=ddp_enc, 
                    precond_ctrl=ddp_ctrl, 
                    precond_dec=ddp_dec, 
                    images=images, 
                    labels=labels.to(device)
                )
                r = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **vsd_warmup_kwargs)
                loss = vsd_loss * r + dsm_loss
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()
    
        # Run optimizer and update weights.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        for g in optimizer.param_groups:
            g['lr'] = lr
            if force_finite:
                for param in g['params']:
                    if param.grad is not None:
                        torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        
        optimizer.step()

        # Update EMA and training state.
        state.cur_nimg += batch_size
        for ema in [ema_enc, ema_ctrl, ema_dec]:
            ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)
        cumulative_training_time += time.time() - batch_start_time
        
        # print progress
        progress = state.cur_nimg / total_nimg
        estimated_time = (time.time() - start_time) / progress * (1 - progress)
        dist.print0(
            f'\rProgress: [{int(progress * 50) * "="}{(50 - int(progress * 50)) * " "}] {state.cur_nimg} / {total_nimg} ({progress * 100:.2f})%',
            f'| estimated_time: {dnnlib.util.format_time(estimated_time)} | vsd_loss: {vsd_loss.mean().item():.4f} | dsm_loss: {dsm_loss.mean().item():.4f} | vsd_warmup_ratio: {r:.8f} | lr: {lr:.8f}', 
            end='', flush=True
        )

#----------------------------------------------------------------------------
