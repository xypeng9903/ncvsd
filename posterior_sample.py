# This work is developed based on the EDM2 codebase (https://github.com/NVlabs/edm2).
# We thank the authors for their great open-source project.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Zero-shot probablistic inference using PnP-NCVSD."""

import torch
import tqdm
import click
import dnnlib
import re
from training.dataset import ImageFolderDataset
import pickle
from torch_utils import distributed as dist
import yaml
from torchvision import transforms
import os
import numpy as np
from torchvision.transforms import Resize

from training.networks_edm2 import GenerativeDenoiser
from tasks import get_operator

#----------------------------------------------------------------------------
# Karras inference sigma.

def karras_sigma_sampler(steps, device, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    ramp = torch.linspace(0, 1, steps, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas

#----------------------------------------------------------------------------
# PnP-NCVSD. A plug-and-play probabilistic inference method for NCVSD.

@torch.no_grad()
def pnp_ncvsd_sampler(
    net: GenerativeDenoiser, 
    noise,
    sigmas,
    likelihood_step_fn,
    ts                  = None,
    daps                = False,
    ema_sigma           = 0,     # EMA sigma threshold. 0 means no EMA.
    ema_decay           = None,
    verbose             = False,
):
    pbar = tqdm.trange(len(sigmas) - 1) if verbose else range(len(sigmas) - 1)
    sigma = torch.ones(noise.shape[0], 1, 1, 1, device=noise.device) * sigmas[0]
    x0 = None
    u = noise * sigma
    for i in pbar:
        sigma = sigmas[i] * torch.ones(noise.shape[0], 1, 1, 1, device=noise.device)
        x = net(u, sigma, ts=ts)
        if sigmas[i] < ema_sigma and x0 is not None:
            x0 = x0 * ema_decay + x * (1 - ema_decay)
        else:
            x0 = x
        u = likelihood_step_fn(x, sigmas[i + 1])
        if daps:
            u = u + torch.randn_like(u) * sigmas[i + 1]       
    return x0

#----------------------------------------------------------------------------
# Unadjusted Langevin Algorithm for the likelihood step.
# (modified from https://github.com/zhangbingliang2019/DAPS/blob/25471a8d7c3416995b88243355dd677648ead6ef/sampler.py#L216)

@torch.enable_grad()
def lgvd_proximal_generator(
    x0, 
    sigma,  
    y, 
    operator, 
    beta, 
    c2,
    steps     = 100,
    c1        = 0.1,
    pbar      = None,
):
    lr = (c1 / (c2 / beta + 1 / sigma ** 2)).cpu().numpy()
    x = x0.clone().detach().requires_grad_(True)
    optimizer = torch.optim.SGD([x], lr)
    for _ in range(steps):
        optimizer.zero_grad()
        energy = ((operator(x) - y) ** 2).sum()
        if pbar is not None:
            pbar.set_description(f"energy: {energy.item() / x.shape[0]:.4f}")
        loss = energy / beta + ((x - x0.detach()) ** 2).sum() / (2 * sigma ** 2)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            epsilon = torch.randn_like(x)
            x.data = x.data + np.sqrt(2 * lr) * epsilon
        if torch.isnan(x).any():
            return torch.zeros_like(x)
    return x.detach()

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Main command line.

@click.group()
def cmdline():
    """Solving inverse problems using PnP-NCVSD."""

#----------------------------------------------------------------------------
# 'pixel' subcommand.

@cmdline.command()
@click.option('--net',                      help='Network pickle filename', metavar='PATH|URL',                     type=str, required=True)
@click.option('--data',                     help='Path to the dataset', metavar='ZIP|DIR',                          type=str, required=True)
@click.option('--preset',                   help='Configuration preset', metavar='STR',                             type=str, required=True)
@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                  type=str, required=True)
@click.option('--seeds',                    help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST',            type=parse_int_list, default='16-19', show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                               type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--class', 'class_idx',       help='Class label  [default: random]', metavar='INT',                   type=click.IntRange(min=0), default=None)

def pixel(**opts):
    """Inverse problem solving using PnP-NCVSD.

    TODO: Examples:

    """
    opts = dnnlib.EasyDict(opts)
    
    #----------------------------------------------------------------------------
    # Main.
    
    net        = opts.net
    preset     = opts.preset
    batch_size = opts.max_batch_size
    device     = torch.device('cuda')
    verbose    = True
    
    with open(preset) as f:
        preset = yaml.safe_load(f)
    preset = dnnlib.EasyDict(preset)
    c = dnnlib.EasyDict(preset.pixel)
    
    # Prepare data.
    dist.print0('Loading dataset...')
    dataset = ImageFolderDataset(opts.data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    # Load main network.
    if verbose:
        dist.print0(f'Loading network from {net} ...')
    with dnnlib.util.open_url(net, verbose=(verbose and dist.get_rank() == 0)) as f:
        data = pickle.load(f)
    net = data['ema'].to(device)
    encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
    
    # Prepare operator.
    dist.print0(f'Operator: {preset.operator}')
    operator = get_operator(**preset.operator, device=device)
    sigma_y = preset.noise['sigma']
    
    # Prepare annealing schedule.
    sigmas = karras_sigma_sampler(**c.annealing, device=device)
    
    # Prepare output directory.
    dist.print0(f'Create output directory {opts.outdir} ...')
    os.makedirs(opts.outdir, exist_ok=True)
    
    # Inference.
    for i, batch in enumerate(tqdm.tqdm(dataloader)):
        images, labels = batch
        images = encoder.encode_latents(images.to(device))
        y = operator.forward(images)
        y = y + torch.randn_like(y) * sigma_y
        likelihood_step_fn = lambda x0, sigma: operator.proximal_generator(x0, y, sigma_y, sigma)
        noise = torch.randn_like(images)
        x0hat = pnp_ncvsd_sampler(net, noise, sigmas, likelihood_step_fn, verbose=True, **c.sampler)
        x0hat = encoder.decode(x0hat)
        for j, out in enumerate(x0hat):
            out = transforms.ToPILImage()(out)
            out.save(f'{opts.outdir}/{i * batch_size + j}.png')
            
#----------------------------------------------------------------------------
# 'latent' subcommand.

@cmdline.command()
@click.option('--net',                      help='Network pickle filename', metavar='PATH|URL',                     type=str, required=True)
@click.option('--data',                     help='Path to the dataset', metavar='ZIP|DIR',                          type=str, required=True)
@click.option('--preset',                   help='Configuration preset', metavar='STR',                             type=str, required=True)
@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                  type=str, required=True)
@click.option('--seeds',                    help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST',            type=parse_int_list, default='16-19', show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                               type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--class', 'class_idx',       help='Class label  [default: random]', metavar='INT',                   type=click.IntRange(min=0), default=None)

def latent(**opts):
    """Inverse problem solving using PnP-NCVSD in latent space.

    TODO: Examples:

    """
    opts = dnnlib.EasyDict(opts)
    
    #----------------------------------------------------------------------------
    # Main.
    
    net        = opts.net
    preset     = opts.preset
    batch_size = opts.max_batch_size
    device     = torch.device('cuda')
    verbose    = True
    
    with open(preset) as f:
        preset = yaml.safe_load(f)
    preset = dnnlib.EasyDict(preset)
    c = dnnlib.EasyDict(preset.latent)
    
    # Prepare data.
    dist.print0('Loading dataset...')
    dataset = ImageFolderDataset(opts.data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    # Load main network.
    if verbose:
        dist.print0(f'Loading network from {net} ...')
    with dnnlib.util.open_url(net, verbose=(verbose and dist.get_rank() == 0)) as f:
        data = pickle.load(f)
    net = data['ema'].to(device)
    encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StabilityVAEEncoder')
    rgb_encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
    
    # Prepare operator.
    dist.print0(f'Operator: {preset.operator}')
    operator = get_operator(**preset.operator, device=device)
    sigma_y = preset.noise['sigma']
    
    # Prepare annealing schedule.
    sigmas = karras_sigma_sampler(**c.annealing, device=device)
    
    # Prepare output directory.
    dist.print0(f'Create output directory {opts.outdir} ...')
    os.makedirs(opts.outdir, exist_ok=True)
    
    # Inference.
    for i, batch in enumerate(tqdm.tqdm(dataloader)):
        images, labels = batch
        images = rgb_encoder.encode_latents(images.to(device))
        y = operator.forward(images)
        y = y + torch.randn_like(y) * sigma_y
        latent_operator = lambda x0: operator.forward(Resize(images.shape[-2:])(encoder.decode(x0, uint8=False)) * 2 - 1)
        likelihood_step_fn = lambda x0, sigma, **kwargs: lgvd_proximal_generator(x0, sigma, y, latent_operator, **c.lgvd, **kwargs)
        noise = torch.randn(batch_size, net.img_channels, net.img_resolution, net.img_resolution, device=device)
        x0hat = pnp_ncvsd_sampler(net, noise, sigmas, likelihood_step_fn, verbose=True, **c.sampler)
        x0hat = Resize(images.shape[-2:])(encoder.decode(x0hat))
        for j, out in enumerate(x0hat):
            out = transforms.ToPILImage()(out)
            out.save(f'{opts.outdir}/{i * batch_size + j}.png')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------