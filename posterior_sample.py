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
import json

from training.networks_edm2 import GenerativeDenoiser
from tasks import get_operator
from tasks.eval import get_eval_fn, Evaluator

#----------------------------------------------------------------------------
# Karras inference sigma.

def karras_sigmas(steps, device, sigma_min=0.002, sigma_max=80.0, rho=7.0):
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
        u = likelihood_step_fn(x, sigmas[i + 1], pbar)
        if daps:
            u = u + torch.randn_like(u) * sigmas[i + 1]       
    return x0

#----------------------------------------------------------------------------
# Unadjusted Langevin Algorithm for the likelihood step.
# (modified from https://github.com/zhangbingliang2019/DAPS/blob/25471a8d7c3416995b88243355dd677648ead6ef/sampler.py#L216)

@torch.enable_grad()
def ula_proximal_generator(
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

# Hyperparameters.
@click.option('--ema-sigma', type=float, default=None)
@click.option('--ema-decay', type=float, default=None)
@click.option('--sigma-min', type=float, default=None)
@click.option('--sigma-max', type=float, default=None)
@click.option('--rho',       type=float, default=None)
@click.option('--beta',      type=float, default=None)
@click.option('--runs',      type=int,   default=None)


def pixel(**opts):
    # Prepare options.
    opts = dnnlib.EasyDict(opts)
    net        = opts.net
    preset     = opts.preset
    batch_size = opts.max_batch_size
    device     = torch.device('cuda')
    verbose    = True
    with open(preset) as f:
        preset = yaml.safe_load(f)
    preset = dnnlib.EasyDict(preset)
    c = dnnlib.EasyDict(preset.pixel)
    
    # Update hyperparameters.
    c.annealing['sigma_min'] = float(c.annealing['sigma_min'])   if opts.sigma_min is None else opts.sigma_min
    c.annealing['sigma_max'] = float(c.annealing['sigma_max'])   if opts.sigma_max is None else opts.sigma_max
    c.annealing['rho']       = float(c.annealing['rho'])         if opts.rho       is None else opts.rho
    c.sampler['ema_sigma']   = float(c.sampler['ema_sigma'])     if opts.ema_sigma is None else opts.ema_sigma
    c.sampler['ema_decay']   = float(c.sampler['ema_decay'])     if opts.ema_decay is None else opts.ema_decay
    c.beta                   = float(c.beta)                     if opts.beta      is None else opts.beta
    c.runs                   = int(c.get('runs', 1))             if opts.runs      is None else opts.runs
    
    # Prepare output directory.
    dist.print0(f'Create output directory {opts.outdir} ...')
    os.makedirs(opts.outdir, exist_ok=True)
    
    # Save options.
    with open(os.path.join(opts.outdir, 'options.json'), 'wt') as f:
        json.dump(c, f, indent=2)
    
    # Prepare data.
    dist.print0('Loading dataset...')
    dataset = ImageFolderDataset(opts.data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    # Prepare model.
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
    sigmas = karras_sigmas(**c.annealing, device=device)
    
    # Prepare evaluator.
    eval_fn_list = []
    for eval_fn_name in ['psnr', 'ssim', 'lpips']:
        eval_fn_list.append(get_eval_fn(eval_fn_name))
    evaluator = Evaluator(eval_fn_list)
    
    # Inference.
    full_images = []
    full_samples = []
    full_ys = []
    for i, batch in enumerate(dataloader):
        dist.print0(f'Batch {i + 1} / {len(dataloader)} start, Total runs: {c.runs}')
        
        # Get measurements.
        images, labels = batch
        images = encoder.encode_latents(images.to(device))
        y = operator.forward(images)
        y = y + torch.randn_like(y) * sigma_y
        
        # PnP-NCVSD.
        if hasattr(operator, 'proximal_generator'):
            likelihood_step_fn = lambda x0, sigma, pbar: operator.proximal_generator(x0, y, (0.5 * c.beta) ** 0.5, sigma) # \beta = 2 \sigma_y^2
        else:
            likelihood_step_fn = lambda x0, sigma, pbar: ula_proximal_generator(x0, sigma, y, operator.forward, beta=c.beta, pbar=pbar, **c.lgvd)
        sampler = lambda noise: pnp_ncvsd_sampler(net, noise, sigmas, likelihood_step_fn, verbose=True, **c.sampler)
        x0hat = torch.cat([sampler(torch.randn_like(images)).unsqueeze(0) for _ in range(c.runs)])
        
        # Save samples.
        full_images.append(images.cpu())
        full_ys.append(y.cpu())
        full_samples.append(x0hat.cpu())
        for j in range(x0hat.shape[0]):
            for k in range(x0hat.shape[1]):
                x0pil = transforms.ToPILImage()(encoder.decode(x0hat[j, k]))
                rundir = os.path.join(opts.outdir, f"runs_{j}")
                os.makedirs(rundir, exist_ok=True)
                x0pil.save(os.path.join(rundir, f"{i * batch_size + k}.png"))
                        
    # Evaluation.
    full_samples = torch.cat(full_samples, dim=1) # runs, B, C, H, W
    full_ys = torch.cat(full_ys, dim=0)           # B, C, H, W
    full_images = torch.cat(full_images, dim=0)   # B, C, H, W
    results = evaluator.report(full_images, full_ys, full_samples)
    markdown_text = evaluator.display(results)
    print(markdown_text)
    
    # Save evaluation results.
    with open(os.path.join(opts.outdir, 'eval.md'), 'wt') as f:
        f.write(markdown_text)
    json.dump(results, open(os.path.join(opts.outdir, 'metrics.json'), 'w'), indent=2)
    
#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------