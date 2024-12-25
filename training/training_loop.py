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
import torch.nn.functional as F
import gc

from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from diffusers import DDPMScheduler

from guided_diffusion.unet import Precond, PrecondCondition, GenerativeDenoiser, DiscriminatorCondition
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion
)


#----------------------------------------------------------------------------
# DDPM noise schedule.

def ddpm_sigma_sampler(batch_size, device):
    alpha_cumprod = DDPMScheduler().alphas_cumprod.to(device)
    sigmas = (1 - alpha_cumprod) ** 0.5 / alpha_cumprod ** 0.5
    idx = torch.randint(0, len(sigmas), [batch_size])
    return sigmas[idx].view(-1, 1, 1, 1)

#----------------------------------------------------------------------------
# Karras inference sigma sampler.

def karras_sigma_sampler(batch_size, device, sigma_min=0.002, sigma_max=80.0, rho=7.0, steps=1000):
    ramp = torch.linspace(0, 1, steps, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    idx = torch.randint(0, steps, [batch_size])
    sigma = sigmas[idx].view(-1, 1, 1, 1)
    return sigma

#----------------------------------------------------------------------------
# Noise conditional variational score distillation.

@persistence.persistent_class
class NCVSDLoss:
    def __init__(self, sigma_data=0.5):
        self.sigma_data = sigma_data

    def __call__(
            self, 
            generator,
            net,
            score_model, 
            discriminator,
            y: torch.Tensor, 
            sigma: torch.Tensor,
            labels: torch.Tensor = None,
            disable_gan: bool = False
    ):        
        x, logvar = generator(y, sigma, labels, return_logvar=True)
        
        t = ddpm_sigma_sampler(y.shape[0], y.device)
        weight_t = (t ** 2 + self.sigma_data ** 2) / (t * self.sigma_data) ** 2
        xt = x + torch.randn_like(x) * t
        
        with torch.no_grad():   
            sigma_eff = 1 / (1 / sigma ** 2 + 1 / t ** 2) ** 0.5
            y_eff = (y / sigma ** 2 + xt / t ** 2) * sigma_eff ** 2
            s0 = net(y_eff, sigma_eff, labels)
            s = score_model(xt, t, y, sigma, labels)

        vsd_loss = (weight_t / logvar.exp()) * (x - (s0 - s + x).detach()) ** 2 + logvar
        if disable_gan:
            return vsd_loss, None
        
        logits = discriminator(xt, t, y, sigma, labels).view(-1, 1, 1, 1)
        gan_loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction='none')
        return vsd_loss, gan_loss

#----------------------------------------------------------------------------
# Denoising score matching.

@persistence.persistent_class
class DSMLoss:
    def __init__(self, sigma_data = 0.5):
        self.sigma_data = sigma_data

    def __call__(
            self, 
            generator,
            score_model,
            y: torch.Tensor, 
            sigma: torch.Tensor,
            labels: torch.Tensor = None
    ): 
        with torch.no_grad():
            x = generator(y, sigma, labels)
        t = ddpm_sigma_sampler(y.shape[0], y.device)
        weight_t = (t ** 2 + self.sigma_data ** 2) / (t * self.sigma_data) ** 2
        xt = x + torch.randn_like(x) * t

        s, logvar = score_model(xt, t, y, sigma, labels, return_logvar=True)        
        dsm_loss = (weight_t / logvar.exp()) * (s - x.detach()) ** 2 + logvar
        return dsm_loss

#----------------------------------------------------------------------------
# Discriminator loss.

@persistence.persistent_class
class DiscriminatorLoss:
    def __init__(self, sigma_data = 0.5):
        self.sigma_data = sigma_data

    def __call__(
            self, 
            generator,
            discriminator,
            images: torch.Tensor,
            y: torch.Tensor, 
            sigma: torch.Tensor,
            labels: torch.Tensor = None
    ): 
        with torch.no_grad():
            x = generator(y, sigma, labels)
        t = ddpm_sigma_sampler(y.shape[0], y.device)
        fake_xt = x + torch.randn_like(x) * t
        real_xt = images + torch.randn_like(images) * t

        real_logits = discriminator(real_xt, t, y, sigma, labels).view(-1, 1, 1, 1)
        fake_logits = discriminator(fake_xt, t, y, sigma, labels).view(-1, 1, 1, 1)        
        
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits), reduction='none')
        real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits), reduction='none')
        return fake_loss, real_loss
    
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
    pretrained_kwargs,
    network_kwargs,
    dataset_kwargs            = dict(class_name='training.dataset.ImageFolderDataset', path=None),
    encoder_kwargs            = dict(class_name='training.encoders.StabilityVAEEncoder'),
    data_loader_kwargs        = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=2, prefetch_factor=2),
    vsd_loss_kwargs           = dict(class_name='training.training_loop.NCVSDLoss'),
    dsm_loss_kwargs           = dict(class_name='training.training_loop.DSMLoss'),
    discriminator_loss_kwargs = dict(class_name='training.training_loop.DiscriminatorLoss'),
    optimizer_kwargs          = dict(class_name='torch.optim.Adam', betas=(0.9, 0.99)),
    lr_kwargs                 = dict(func_name='training.training_loop.learning_rate_schedule'),
    ema_kwargs                = dict(class_name='training.phema.PowerFunctionEMA', stds=[0.050, 0.100]),
    gamma                     = 0.414,     # TODO.
    init_sigma                = 80.0,      # Maximum noise level.
    eval_ts                   = None,      # inference steps for evaluation. None = no evaluation.
    num_eval_samples          = 64,        # Number of samples for evaluation.
    eval_batch_size           = 8,         # Batch size for evaluation.
    g_lr_scaling              = 1,         # Learning rate scaling factor for the generator.
    d_lr_scaling              = 1,         # Learning rate scaling factor for the discriminator.
    gan_warmup_batches        = 0,         # Number of batches to warm up the GAN loss.
 
    run_dir                   = '.',       # Output directory.
    seed                      = 0,         # Global random seed.
    batch_size                = 2048,      # Total batch size for one training iteration.
    batch_gpu                 = None,      # Limit batch size per GPU. None = no limit.
    total_nimg                = 8<<30,     # Train for a total of N training images.
    slice_nimg                = None,      # Train for a maximum of N training images in one invocation. None = no limit.
    status_nimg               = 128<<10,   # Report status every N training images. None = disable.
    snapshot_nimg             = 8<<20,     # Save network snapshot every N training images. None = disable.
    checkpoint_nimg           = 128<<20,   # Save state checkpoint every N training images. None = disable.
 
    loss_scaling              = 1,         # Loss scaling factor for reducing FP16 under/overflows.
    gan_loss_scaling          = 1,         # Scaling factor for the GAN loss.
    force_finite              = True,      # Get rid of NaN/Inf gradients before feeding them to the optimizer.
    cudnn_benchmark           = True,      # Enable torch.backends.cudnn.benchmark?
    device                    = torch.device('cuda'),
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
    assert num_eval_samples % int(num_eval_samples ** 0.5) == 0

    # Setup dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    
    # Load teacher model.
    dist.print0(f'Loading teacher model {net}')
    teacher_kwargs = model_and_diffusion_defaults()
    teacher_kwargs.update(**pretrained_kwargs)
    unet, diffusion = create_model_and_diffusion(**teacher_kwargs)
    unet.load_state_dict(torch.load(net, weights_only=True, map_location='cpu'))
    
    # Setup networks.
    dist.print0('Constructing networks...')
    net = Precond(diffusion.alphas_cumprod, **network_kwargs)
    net = net.init_from_pretrained(unet).convert_to_fp16().eval().requires_grad_(False).to(device)
    generator = PrecondCondition(diffusion.alphas_cumprod, **network_kwargs).init_from_pretrained(unet).requires_grad_(True).to(device)
    generator = GenerativeDenoiser(generator, gamma=gamma, init_sigma=init_sigma)
    score_model = PrecondCondition(diffusion.alphas_cumprod, **network_kwargs).init_from_pretrained(unet).requires_grad_(True).to(device)
    discriminator = DiscriminatorCondition(diffusion.alphas_cumprod, **network_kwargs).init_from_pretrained(unet).requires_grad_(True).to(device)
    ema_generator = dnnlib.util.construct_class_by_name(net=generator, **ema_kwargs) if ema_kwargs is not None else None

    del unet
    
    # TODO: Print network summary.

    # Setup training state.
    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0)
    ddp_generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[device])
    ddp_score_model = torch.nn.parallel.DistributedDataParallel(score_model, device_ids=[device])
    ddp_discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[device])
    g_optimizer = dnnlib.util.construct_class_by_name(params=ddp_generator.parameters(), **optimizer_kwargs)
    s_optimizer = dnnlib.util.construct_class_by_name(params=ddp_score_model.parameters(), **optimizer_kwargs)
    d_optimizer = dnnlib.util.construct_class_by_name(params=ddp_discriminator.parameters(), **optimizer_kwargs)
    vsd_loss_fn = dnnlib.util.construct_class_by_name(**vsd_loss_kwargs)
    dsm_loss_fn = dnnlib.util.construct_class_by_name(**dsm_loss_kwargs)
    discriminator_loss_fn = dnnlib.util.construct_class_by_name(**discriminator_loss_kwargs)
    
    # Load previous checkpoint and decide how long to train.
    checkpoint = dist.CheckpointIO(
        state=state, 
        generator=generator,
        score_model=score_model,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        s_optimizer=s_optimizer,
        d_optimizer=d_optimizer,
        ema_generator=ema_generator
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

    # Setup tensorboard.
    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=run_dir)

    # Main training loop.
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nimg)
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    prev_status_nimg = state.cur_nimg
    cumulative_training_time = 0
    start_nimg = state.cur_nimg
    stats_jsonl = None
    while True:
        done = (state.cur_nimg >= stop_at_nimg)

        #------------------------------------------------------------------------------------
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

        # Save network snapshot and evaluate.
        if snapshot_nimg is not None and state.cur_nimg % snapshot_nimg == 0 and (state.cur_nimg != start_nimg or start_nimg == 0) and dist.get_rank() == 0:
            gc.collect()
            torch.cuda.empty_cache()
            ema_list = ema_generator.get()
            ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
            for ema_net, ema_suffix in ema_list:
                data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs, loss_fn=vsd_loss_fn)
                data.ema = copy.deepcopy(ema_net).cpu().eval().requires_grad_(False).to(torch.float16)
                fname = f'snapshot-{state.cur_nimg//1000:07d}{ema_suffix}.pkl'
                dist.print0(f'Saving {fname} ... ', end='', flush=True)
                with open(os.path.join(run_dir, fname), 'wb') as f:
                    pickle.dump(data, f)
                dist.print0('done')
                del data # conserve memory
                
                # Export sample images.
                if dist.get_rank() == 0 and eval_ts is not None:
                    dist.print0(f'Exporting sample images for {fname}')
                    with torch.no_grad():
                        ema_net.eval()
                        z = torch.randn(num_eval_samples, ema_net.img_channels, ema_net.img_resolution, ema_net.img_resolution, device=device) * init_sigma
                        x = torch.cat([ema_net(batch, init_sigma * torch.ones(batch.shape[0], 1, 1, 1, device=device), ts=eval_ts) 
                                       for batch in z.split(eval_batch_size)])
                        x = encoder.decode(x).cpu()
                        save_image(x.float() / 255., os.path.join(run_dir, f'{fname}.png'), nrow=int(num_eval_samples ** 0.5))
                        del x
        
        # Save state checkpoint.
        if checkpoint_nimg is not None and (done or state.cur_nimg % checkpoint_nimg == 0) and state.cur_nimg != start_nimg:
            checkpoint.save(os.path.join(run_dir, f'training-state-{state.cur_nimg//1000:07d}.pt'))
            # TODO: misc.check_ddp_consistency(net) 

        # Done?
        if done:
            break
        
        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_nimg)

        #------------------------------------------------------------------------------------
        # Score model update.

        ddp_score_model.train().requires_grad_(True)
        ddp_discriminator.eval().requires_grad_(False)
        ddp_generator.eval().requires_grad_(False)
        s_optimizer.zero_grad(set_to_none=True)

        # Forward & backward.
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp_score_model, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = encoder.encode_latents(images.to(device))
                sigma = karras_sigma_sampler(images.shape[0], images.device)
                y = images + torch.randn_like(images) * sigma
                dsm_loss = dsm_loss_fn(
                    generator=ddp_generator,
                    score_model=ddp_score_model,
                    y=y, 
                    sigma=sigma,
                    labels=labels.to(device)
                )
                dsm_loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Score model optimization.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        for g in s_optimizer.param_groups:
            g['lr'] = lr
            for param in g['params']:
                if param.grad is not None and force_finite:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)     
        s_optimizer.step()

        #-----------------------------------------------------------------------------------
        # Discriminator update.

        ddp_score_model.eval().requires_grad_(False)
        ddp_discriminator.train().requires_grad_(True)
        ddp_generator.eval().requires_grad_(False)
        d_optimizer.zero_grad(set_to_none=True)

        # Forward & backward.
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp_discriminator, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = encoder.encode_latents(images.to(device))
                sigma = karras_sigma_sampler(images.shape[0], images.device)
                y = images + torch.randn_like(images) * sigma
                fake_loss, real_loss = discriminator_loss_fn(
                    generator=ddp_generator,
                    discriminator=ddp_discriminator,
                    images=images,
                    y=y, 
                    sigma=sigma,
                    labels=labels.to(device)
                )
                discriminator_loss = 0.5 * (fake_loss + real_loss)
                discriminator_loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Discriminator optimization.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        for g in d_optimizer.param_groups:
            g['lr'] = lr * d_lr_scaling
            for param in g['params']:
                if param.grad is not None and force_finite:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)     
        d_optimizer.step()

        #------------------------------------------------------------------------------------
        # Generator update.

        ddp_score_model.eval().requires_grad_(False)
        ddp_discriminator.eval().requires_grad_(False)
        ddp_generator.train().requires_grad_(True)
        g_optimizer.zero_grad(set_to_none=True)        
        disable_gan = state.cur_nimg < gan_warmup_batches * batch_size

        # Forward & backward.
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp_generator, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = encoder.encode_latents(images.to(device))
                sigma =  karras_sigma_sampler(images.shape[0], images.device)
                y = images + torch.randn_like(images) * sigma
                vsd_loss, gan_loss = vsd_loss_fn(
                    generator=ddp_generator,
                    net=net,
                    score_model=ddp_score_model, 
                    discriminator=ddp_discriminator,
                    y=y,
                    sigma=sigma,
                    labels=labels.to(device),
                    disable_gan=disable_gan
                )
                g_loss = vsd_loss if gan_loss is None else vsd_loss + gan_loss * gan_loss_scaling
                g_loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Generator optimization.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        for g in g_optimizer.param_groups:
            g['lr'] = lr * g_lr_scaling
            for param in g['params']:
                if param.grad is not None and force_finite:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)     
        g_optimizer.step()

        #------------------------------------------------------------------------------------
        # Update progress.

        state.cur_nimg += batch_size
        ema_generator.update(cur_nimg=state.cur_nimg, batch_size=batch_size)
        
        cumulative_training_time += time.time() - batch_start_time
        batch_time = time.time() - batch_start_time
        progress = state.cur_nimg / total_nimg
        estimated_time = total_nimg * (1 - progress) / batch_size * batch_time

        dist.print0(
            f'\rProgress: {state.cur_nimg} / {total_nimg} ({progress * 100:.2f} %)',
            f'| Estimated time: {dnnlib.util.format_time(estimated_time)}',
        )
        if dist.get_rank() == 0:
            writer.add_scalar('Loss/VSD', vsd_loss.mean().item(), state.cur_nimg)
            if gan_loss is not None:
                writer.add_scalar('Loss/GAN', gan_loss.mean().item(), state.cur_nimg)
            writer.add_scalar('Loss/DSM', dsm_loss.mean().item(), state.cur_nimg)
            writer.add_scalar('Loss/Fake', fake_loss.mean().item(), state.cur_nimg)
            writer.add_scalar('Loss/Real', real_loss.mean().item(), state.cur_nimg)
            writer.add_scalar('Learning Rate', lr, state.cur_nimg)
            writer.flush()

#----------------------------------------------------------------------------
