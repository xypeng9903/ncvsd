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

from guided_diffusion.unet import GenerativeDenoiser
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
    u = noise * sigma
    x0 = None
    for step in pbar:
        sigma = torch.ones(noise.shape[0], 1, 1, 1, device=noise.device) * sigmas[step]
        if sigma < ema_sigma and x0 is not None:
            x0 = x0 * ema_decay + net(u, sigma, ts=ts) * (1 - ema_decay) # Prior step.
        else:
            x0 = net(u, sigma, ts=ts) # Prior step.
        sigma_next = torch.ones(noise.shape[0], 1, 1, 1, device=noise.device) * sigmas[step + 1]
        u = likelihood_step_fn(x0, sigma_next) # Likelihood step.
        if daps: # Forward diffusion.
            u = u + torch.randn_like(u) * sigmas[step + 1]       
    return x0


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
# Command line interface.

@click.command()
@click.option('--net',                      help='Network pickle filename', metavar='PATH|URL',                     type=str, required=True)
@click.option('--data',                     help='Path to the dataset', metavar='ZIP|DIR',                          type=str, required=True)
@click.option('--preset',                   help='Configuration preset', metavar='STR',                             type=str, required=True)
@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                  type=str, required=True)
@click.option('--seeds',                    help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST',            type=parse_int_list, default='16-19', show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                               type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--class', 'class_idx',       help='Class label  [default: random]', metavar='INT',                   type=click.IntRange(min=0), default=None)

def cmdline(**opts):
    """Inverse problem solving using PnP-NCVSD.

    Examples:

    \b
    # Generate a couple of images and save them as out/*.png
    python generate_images.py --preset=edm2-img512-s-guid-dino --outdir=out

    """
    opts = dnnlib.EasyDict(opts)
    
    #----------------------------------------------------------------------------
    # Main.
    
    net = opts.net
    with open(opts.preset) as f:
        preset = yaml.safe_load(f)
    preset = dnnlib.EasyDict(preset)
    encoder = None
    device = torch.device('cuda')
    verbose = True
    batch_size = opts.max_batch_size
    
    # Prepare data.
    dist.print0('Loading dataset...')
    dataset = ImageFolderDataset(opts.data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    # Load main network.
    if isinstance(net, str):
        if verbose:
            dist.print0(f'Loading network from {net} ...')
        with dnnlib.util.open_url(net, verbose=(verbose and dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        net = data['ema'].to(device)
        if encoder is None:
            encoder = data.get('encoder', None)
            if encoder is None:
                encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
    assert net is not None
    net.to(torch.float32)
    net.convert_to_fp16().eval()
    
    # Prepare operator.
    dist.print0(f'Operator: {preset.operator}')
    operator = get_operator(**preset.operator, device=device)
    sigma_y = preset.noise['sigma']
    
    # Prepare annealing schedule.
    sigmas = karras_sigma_sampler(**preset.annealing, device=device)
    
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
        x0hat = pnp_ncvsd_sampler(net, noise, sigmas, likelihood_step_fn, verbose=True, **preset.sampler)
        for j, x0hat in enumerate(x0hat):
            x0hat = encoder.decode(x0hat)
            img = transforms.ToPILImage()(x0hat)
            img.save(f'{opts.outdir}/{i * batch_size + j}.png')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------