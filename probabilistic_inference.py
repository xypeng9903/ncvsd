import torch
import tqdm
import click
import dnnlib
import re
from training.dataset import ImageFolderDataset
import pickle
from torch_utils import distributed as dist
import yaml
from functools import partial

from guided_diffusion.unet import GenerativeDenoiser
from forward_operator.likelihood_step import lgvd
from forward_operator import get_operator

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
    verbose             = False,
):
    pbar = tqdm.trange(len(sigmas)) if verbose else range(len(sigmas))
    sigma = torch.ones(noise.shape[0], 1, 1, 1, device=noise.device) * sigmas[0]
    xt = noise * sigma
    for step in pbar:
        sigma = torch.ones(noise.shape[0], 1, 1, 1, device=noise.device) * sigmas[step]
        
        # Prior step.
        x0hat = net(xt, sigma, ts=ts)
        
        # Likelihood step.
        x0y = likelihood_step_fn(x0hat, sigma)
        
        # Forward diffusion.
        if daps:
            xt = x0y + torch.randn_like(x0y) * sigmas[step + 1]       
    return x0hat


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
@click.option('--ts',                       help='Inference timesteps', metavar='INT',                              type=parse_int_list, default=None)
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
    
    # Prepare data.
    dist.print0('Loading dataset...')
    dataset = ImageFolderDataset(opts.data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts.max_batch_size)
    
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
    net.convert_to_fp16()
    
    # Prepare operator.
    dist.print0(f'Operator: {preset.operator}')
    operator = get_operator(**preset.operator)
    
    # Prepare annealing schedule.
    sigmas = None # TODO
    
    # Inference.
    for batch in tqdm.tqdm(dataloader):
        images, labels = batch
        images = encoder.encode_latents(images.to(device))
        y = operator.measure(images)
        likelihood_step_fn = partial(lgvd, operator=operator, measurement=y, **preset.lgvd_config)
        sampler = pnp_ncvsd_sampler(net, images, sigmas, likelihood_step_fn, ts=opts.ts)
        x0hat = sampler()
        import ipdb; ipdb.set_trace()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------