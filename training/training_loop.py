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

from .networks_edm2 import PrecondUNetDecoder, PrecondUNetEncoder, Precond

@persistence.persistent_class
class NCVSDLoss:
    def __init__(
        self, 
        P_mean      = 0.4, 
        P_std       = 2.0,
        gamma       = 0.414,
        sigma_data  = 0.5
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.gamma = gamma
        self.sigma_data = sigma_data

    def __call__(
            self, 
            net: Precond,
            s_enc: PrecondUNetEncoder, 
            s_dec: PrecondUNetDecoder, 
            s_ctrlnet: PrecondUNetEncoder, 
            x: torch.Tensor, 
            logvar: torch.Tensor,
            y: torch.Tensor, 
            sigma: torch.Tensor,
            labels: torch.Tensor = None
    ):        
        # diffuse
        rnd_normal_t = torch.randn([y.shape[0], 1, 1, 1], device=y.device)
        t = (rnd_normal_t * self.P_std + self.P_mean).exp()
        weight_t = (t ** 2 + self.sigma_data ** 2) / (t * self.sigma_data) ** 2
        xt = x + torch.randn_like(x) * t

        # score prediction
        with torch.no_grad():   
            # data score
            sigma_eff = 1 / (1 / sigma ** 2 + 1 / t ** 2) ** 0.5
            y_eff = (y / sigma ** 2 + xt / t ** 2) * sigma_eff ** 2
            s0 = net(y_eff, sigma_eff, labels)

            # model score
            ctrl_x, ctrl_skips = s_ctrlnet(y, sigma, labels)
            enc_x, enc_skips = s_enc(xt, t, labels)
            s = s_dec(xt, enc_x, enc_skips, t, labels, additional_x=ctrl_x, additional_skips=ctrl_skips)

        # variational score distillation
        loss_vsd = (weight_t / logvar.exp()) * (x - (s0 - s + x).detach()) ** 2 + logvar

        return loss_vsd

#----------------------------------------------------------------------------
# Denoising score mathcing.

@persistence.persistent_class
class DSMLoss:
    def __init__(
        self, 
        P_mean        = -0.8, 
        P_std         = 1.6,
        sigma_data    = 0.5
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(
            self, 
            s_enc: PrecondUNetEncoder, 
            s_dec: PrecondUNetDecoder, 
            s_ctrlnet: PrecondUNetEncoder, 
            x: torch.Tensor, 
            y: torch.Tensor, 
            sigma: torch.Tensor,
            labels: torch.Tensor = None
    ):
        rnd_normal_t = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        t = (rnd_normal_t * self.P_std + self.P_mean).exp()
        weight_t = (t ** 2 + self.sigma_data ** 2) / (t * self.sigma_data) ** 2
        xt = x + torch.randn_like(x) * t

        ctrl_x, ctrl_skips = s_ctrlnet(y, sigma, labels)
        enc_x, enc_skips = s_enc(xt, t, labels)
        s, logvar = s_dec(xt, enc_x, enc_skips, t, labels, additional_x=ctrl_x, additional_skips=ctrl_skips, return_logvar=True)

        loss = (weight_t / logvar.exp()) * (s - x.detach()) ** 2 + logvar

        return loss
    
#----------------------------------------------------------------------------
# Denoising posterior sampling.

@persistence.persistent_class
class DenoisingPosteriorSampling:
    def __init__(self, gamma=0.414):
        self.gamma = gamma

    def __call__(
            self, 
            g_enc: PrecondUNetEncoder, 
            g_dec: PrecondUNetDecoder, 
            g_ctrlnet: PrecondUNetEncoder, 
            y: torch.Tensor, 
            sigma: torch.Tensor, 
            labels: torch.Tensor = None, 
            return_logvar: bool = False
    ):
        z = torch.randn_like(y)
        sigma_hat = sigma * (1 + self.gamma)
        y_hat = y + z * (sigma_hat ** 2 - sigma ** 2) ** 0.5
        ctrl_x, ctrl_skips = g_ctrlnet(y, sigma, labels)
        enc_x, enc_skips = g_enc(y_hat, sigma_hat, labels)
        out = g_dec(y_hat, enc_x, enc_skips, sigma_hat, labels, additional_x=ctrl_x, additional_skips=ctrl_skips, return_logvar=return_logvar)
        return out

    
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
    ema_kwargs          = dict(class_name='training.phema.PowerFunctionEMA'),
    sampling_kwargs     = dict(class_name='training.training_loop.DenoisingPosteriorSampling'),
    P_mean_sigma        = 0.4,      # Mean of the LogNormal sampler of noise condition.
    P_std_sigma         = 2.0,      # Standard deviation of the LogNormal sampler of noise condition.

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
    gradient_checkpoint = False,    # Use gradient checkpointing to save memory?
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
    
    dist.print0('Constructing networks...')
    # Teacher model.
    dist.print0(f'Load teacher model: {net}')
    with dnnlib.util.open_url(net, verbose=True) as f:
        data = pickle.load(f)
    net = data['ema'].eval().requires_grad_(False).to(device)
    
    # Generator.
    g_enc = PrecondUNetEncoder(**network_kwargs).init_from_pretrained(net).requires_grad_(True).to(device)
    g_dec = PrecondUNetDecoder(**network_kwargs).init_from_pretrained(net).requires_grad_(True).to(device)
    g_ctrlnet = PrecondUNetEncoder(**network_kwargs, is_controlnet=True).init_from_pretrained(net).to(device)
    
    # Model score.
    s_enc = PrecondUNetEncoder(**network_kwargs).init_from_pretrained(net).requires_grad_(True).to(device)
    s_dec = PrecondUNetDecoder(**network_kwargs).init_from_pretrained(net).requires_grad_(True).to(device)
    s_ctrlnet = PrecondUNetEncoder(**network_kwargs, is_controlnet=True).init_from_pretrained(net).to(device)
    
    # gradient checkpointing
    if gradient_checkpoint:
        g_enc.enable_gradient_checkpointing()
        g_dec.enable_gradient_checkpointing()
        g_ctrlnet.enable_gradient_checkpointing()
        s_enc.enable_gradient_checkpointing()
        s_dec.enable_gradient_checkpointing()
        s_ctrlnet.enable_gradient_checkpointing()

    # Print network summary.
    if dist.get_rank() == 0:
        misc.print_module_summary(net, [
            torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device),
            torch.ones([batch_gpu], device=device),
            torch.zeros([batch_gpu, net.label_dim], device=device),
        ], max_nesting=2)

    # Setup training state.
    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0)

    # optimizer
    g_params = list(g_enc.parameters()) + list(g_dec.parameters()) + list(g_ctrlnet.parameters())
    g_optimizer = dnnlib.util.construct_class_by_name(params=g_params, **optimizer_kwargs)
    s_params = list(s_enc.parameters()) + list(s_dec.parameters()) + list(s_ctrlnet.parameters())
    s_optimizer = dnnlib.util.construct_class_by_name(params=s_params, **optimizer_kwargs)
    
    # ddp
    ddp_g_enc = torch.nn.parallel.DistributedDataParallel(g_enc, device_ids=[device], broadcast_buffers=False, find_unused_parameters=False)
    ddp_g_dec = torch.nn.parallel.DistributedDataParallel(g_dec, device_ids=[device], broadcast_buffers=False, find_unused_parameters=False)
    ddp_g_ctrlnet = torch.nn.parallel.DistributedDataParallel(g_ctrlnet, device_ids=[device], broadcast_buffers=False, find_unused_parameters=False)
    ddp_s_enc = torch.nn.parallel.DistributedDataParallel(s_enc, device_ids=[device], broadcast_buffers=False, find_unused_parameters=False)
    ddp_s_dec = torch.nn.parallel.DistributedDataParallel(s_dec, device_ids=[device], broadcast_buffers=False, find_unused_parameters=False)
    ddp_s_ctrlnet = torch.nn.parallel.DistributedDataParallel(s_ctrlnet, device_ids=[device], broadcast_buffers=False, find_unused_parameters=False)

    # denoising posterior sampling function
    sample_fn = dnnlib.util.construct_class_by_name(**sampling_kwargs)
    vsd_loss_fn = dnnlib.util.construct_class_by_name(**vsd_loss_kwargs)
    dsm_loss_fn = dnnlib.util.construct_class_by_name(**dsm_loss_kwargs)
    
    # ema
    ema_g_enc = dnnlib.util.construct_class_by_name(net=g_enc, **ema_kwargs) if ema_kwargs is not None else None
    ema_g_dec = dnnlib.util.construct_class_by_name(net=g_dec, **ema_kwargs) if ema_kwargs is not None else None
    ema_g_ctrlnet = dnnlib.util.construct_class_by_name(net=s_ctrlnet, **ema_kwargs) if ema_kwargs is not None else None

    # Load previous checkpoint and decide how long to train.
    checkpoint = dist.CheckpointIO(
        state=state, 
        g_enc=g_enc,
        g_dec=g_dec,
        g_ctrlnet=g_ctrlnet,
        s_enc=s_enc,
        s_dec=s_dec,
        s_ctrlnet=s_ctrlnet,
        g_optimizer=g_optimizer,
        s_optimizer=s_optimizer,
        ema_g_enc=ema_g_enc, 
        ema_g_dec=ema_g_dec,
        ema_g_ctrlnet=ema_g_ctrlnet
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
            for ema, name in zip([ema_g_enc, ema_g_dec, ema_g_ctrlnet], ['g_enc', 'g_dec', 'g_ctrlnet']):
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
        
        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_nimg)

        #------------------------------------------
        # Denoising score matching.

        s_optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp_g_enc, (round_idx == num_accumulation_rounds - 1)), \
                 misc.ddp_sync(ddp_g_dec, (round_idx == num_accumulation_rounds - 1)), \
                 misc.ddp_sync(ddp_s_enc, (round_idx == num_accumulation_rounds - 1)), \
                 misc.ddp_sync(ddp_s_dec, (round_idx == num_accumulation_rounds - 1)), \
                 misc.ddp_sync(ddp_s_ctrlnet, (round_idx == num_accumulation_rounds - 1)):
                
                # Sample condition y.
                images, labels = next(dataset_iterator)
                images = encoder.encode_latents(images.to(device))
                rnd_normal_y = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
                sigma = (rnd_normal_y * P_std_sigma + P_mean_sigma).exp()
                y = images + torch.randn_like(images) * sigma

                # Denoising score matching.
                with torch.no_grad():
                    g_enc.eval(); g_dec.eval(); s_ctrlnet.eval()
                    x = sample_fn(
                        g_enc=ddp_g_enc,
                        g_dec=ddp_g_dec,
                        g_ctrlnet=ddp_g_ctrlnet,
                        y=y, 
                        sigma=sigma, 
                        labels=labels
                    )
                
                s_enc.train(); s_dec.train(); s_ctrlnet.train()
                loss = dsm_loss_fn(
                    s_enc=ddp_s_enc, 
                    s_dec=ddp_s_dec, 
                    s_ctrlnet=ddp_s_ctrlnet, 
                    x=x,
                    y=y, 
                    sigma=sigma,
                    labels=labels.to(device)
                )

                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Run optimizer and update weights.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        for g in s_optimizer.param_groups:
            g['lr'] = lr
            for param in g['params']:
                if param.grad is not None and force_finite:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)     
        
        s_optimizer.step()

        #------------------------------------------
        # variational score distillation.

        g_optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp_g_enc, (round_idx == num_accumulation_rounds - 1)), \
                 misc.ddp_sync(ddp_g_dec, (round_idx == num_accumulation_rounds - 1)), \
                 misc.ddp_sync(ddp_s_ctrlnet, (round_idx == num_accumulation_rounds - 1)):
                
                g_enc.train(); g_dec.train(); s_ctrlnet.train()
                x, logvar = sample_fn(
                    g_enc=ddp_g_enc,
                    g_dec=ddp_g_dec,
                    g_ctrlnet=ddp_s_ctrlnet,
                    y=y, 
                    sigma=sigma, 
                    labels=labels,
                    return_logvar=True
                )

                s_enc.eval(); s_dec.eval(); s_ctrlnet.eval()
                loss = vsd_loss_fn(
                    net=net,
                    s_enc=ddp_s_enc,
                    s_dec=ddp_s_dec,
                    s_ctrlnet=ddp_s_ctrlnet, 
                    x=x,
                    logvar=logvar,
                    y=y,
                    sigma=sigma,
                    labels=labels.to(device)
                )

                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Run optimizer and update weights.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        for g in g_optimizer.param_groups:
            g['lr'] = lr
            for param in g['params']:
                if param.grad is not None and force_finite:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)     
        
        g_optimizer.step()
                  
        # Update EMA and training state.
        state.cur_nimg += batch_size
        for ema in [ema_g_enc, ema_g_ctrlnet, ema_g_dec]:
            ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)
        cumulative_training_time += time.time() - batch_start_time
        
        # print progress
        progress = state.cur_nimg / total_nimg
        estimated_time = (time.time() - start_time) / progress * (1 - progress)
        dist.print0(
            f'\rProgress: [{int(progress * 50) * "="}{(50 - int(progress * 50)) * " "}] {state.cur_nimg} / {total_nimg} ({progress * 100:.2f})%',
            f'| estimated_time: {dnnlib.util.format_time(estimated_time)} | vsd_loss: {vsd_loss.mean().item():.4f} | dsm_loss: {dsm_loss.mean().item():.4f}', 
            end='', flush=True
        )

#----------------------------------------------------------------------------
