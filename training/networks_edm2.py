# This work is developed based on the EDM2 codebase (https://github.com/NVlabs/edm2).
# We thank the authors for their great open-source project.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Improved diffusion model architecture proposed in the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models"."""

import numpy as np
import torch
from torch_utils import persistence
from torch_utils import misc
import torch.nn.functional as F

#----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

#----------------------------------------------------------------------------
# Upsample or downsample the given tensor with the given filter,
# or keep it as is.

def resample(x, f=[1,1], mode='keep'):
    if mode == 'keep':
        return x
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    f = np.outer(f, f)[np.newaxis, np.newaxis, :, :]
    f = misc.const_like(x, f)
    c = x.shape[1]
    if mode == 'down':
        return torch.nn.functional.conv2d(x, f.tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))
    assert mode == 'up'
    return torch.nn.functional.conv_transpose2d(x, (f * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))

#----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).

def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

#----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).

# def mp_sum(a, b, t=0.5):
#     return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t ** 2)
def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / ((1 - t) ** 2 + t ** 2) ** 0.5

#----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).

def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t ** 2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a , wb * b], dim=dim)

#----------------------------------------------------------------------------
# Magnitude-preserving Fourier features (Equation 75).

@persistence.persistent_class
class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)

#----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).

# @persistence.persistent_class
# class MPConv(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel):
#         super().__init__()
#         self.out_channels = out_channels
#         self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))
#         self.in_place_normalization = True

#     def forward(self, x, gain=1):
#         w = self.weight.to(torch.float32)
#         if self.training and self.in_place_normalization:
#             with torch.no_grad():
#                 self.weight.copy_(normalize(w)) # forced weight normalization
#         w = normalize(w) # traditional weight normalization
#         w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
#         w = w.to(x.dtype)
#         if w.ndim == 2:
#             return x @ w.t()
#         assert w.ndim == 4
#         return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))

#----------------------------------------------------------------------------
# Modified magnitude-preserving convolution proposed in "Adversarial Score Identity Distillation:
# Rapidly Surpassing the Teacher in One Step" by Zhou et al. (https://arxiv.org/abs/2410.14919).

@persistence.persistent_class
class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel, force_normalization=True, use_gan=False):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))
        self.force_normalization = force_normalization
        self.use_gan = use_gan
        # Register the forward pre-hook
        if self.use_gan and self.force_normalization:
            self.register_forward_pre_hook(self._apply_forced_weight_normalization)

    def _apply_forced_weight_normalization(self, module, input):
        # Only apply during training
        if self.training:
            with torch.no_grad():
                w = self.weight.to(torch.float32)
                w_normalized = normalize(w)
                self.weight.copy_(w_normalized)

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training and not self.use_gan:
            with torch.no_grad():
                self.weight.copy_(normalize(w))  # forced weight normalization
        w = normalize(w)  # Traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel()))  # Magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1] // 2))
        
#----------------------------------------------------------------------------
# U-Net encoder/decoder block with optional self-attention (Figure 21).

@persistence.persistent_class
class Block(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        emb_channels,                   # Number of embedding channels.
        flavor              = 'enc',    # Flavor: 'enc' or 'dec'.
        resample_mode       = 'keep',   # Resampling: 'keep', 'up', or 'down'.
        resample_filter     = [1,1],    # Resampling filter.
        attention           = False,    # Include self-attention?
        channels_per_head   = 64,       # Number of channels per attention head.
        dropout             = 0,        # Dropout probability.
        res_balance         = 0.3,      # Balance between main branch (0) and residual branch (1).
        attn_balance        = 0.3,      # Balance between main branch (0) and self-attention (1).
        clip_act            = 256,      # Clip output activations. None = do not clip.
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(out_channels if flavor == 'enc' else in_channels, out_channels, kernel=[3,3])
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3,3])
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1,1]) if in_channels != out_channels else None
        self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1,1]) if self.num_heads != 0 else None
        self.attn_proj = MPConv(out_channels, out_channels, kernel=[1,1]) if self.num_heads != 0 else None

    def forward(self, x, emb):
        # Main branch.
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1) # pixel norm

        # Residual branch.
        y = self.conv_res0(mp_silu(x))
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == 'dec' and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention.
        # Note: torch.nn.functional.scaled_dot_product_attention() could be used here,
        # but we haven't done sufficient testing to verify that it produces identical results.
        if self.num_heads != 0:
            y = self.attn_qkv(x)
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
            q, k, v = normalize(y, dim=2).unbind(3) # pixel norm & split
            w = torch.einsum('nhcq,nhck->nhqk', q, k / np.sqrt(q.shape[2])).softmax(dim=3)
            y = torch.einsum('nhqk,nhck->nhcq', w, v)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x

#----------------------------------------------------------------------------
# EDM2 U-Net model (Figure 21).

@persistence.persistent_class
class UNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Image channels.
        label_dim,                          # Class label dimensionality. 0 = unconditional.
        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_noise  = None,         # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb    = None,         # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [16,8],       # List of resolutions with self-attention.
        label_balance       = 0.5,          # Balance between noise embedding (0) and class embedding (1).
        concat_balance      = 0.5,          # Balance between skip connections (0) and main path (1).
        **block_kwargs,                     # Arguments for Block.
    ):
        super().__init__()
        cblock = [model_channels * x for x in channel_mult]
        cnoise = model_channels * channel_mult_noise if channel_mult_noise is not None else cblock[0]
        cemb = model_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])
        self.emb_label = MPConv(label_dim, cemb, kernel=[]) if label_dim != 0 else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = img_channels + 1
        for level, channels in enumerate(cblock):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_conv'] = MPConv(cin, cout, kernel=[3,3])
            else:
                self.enc[f'{res}x{res}_down'] = Block(cout, cout, cemb, flavor='enc', resample_mode='down', **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='enc', attention=(res in attn_resolutions), **block_kwargs)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = img_resolution >> level
            if level == len(cblock) - 1:
                self.dec[f'{res}x{res}_in0'] = Block(cout, cout, cemb, flavor='dec', attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = Block(cout, cout, cemb, flavor='dec', **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = Block(cout, cout, cemb, flavor='dec', resample_mode='up', **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='dec', attention=(res in attn_resolutions), **block_kwargs)
        self.out_conv = MPConv(cout, img_channels, kernel=[3,3])

    def forward(self, x, noise_labels, class_labels):
        # Embedding.
        emb = self.emb_noise(self.emb_fourier(noise_labels))
        if self.emb_label is not None:
            emb = mp_sum(emb, self.emb_label(class_labels * np.sqrt(class_labels.shape[1])), t=self.label_balance)
        emb = mp_silu(emb)

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, emb)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if 'block' in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)
        x = self.out_conv(x, gain=self.out_gain)
        return x

#----------------------------------------------------------------------------
# Preconditioning and uncertainty estimation.

@persistence.persistent_class
class Precond(torch.nn.Module):
    def __init__(self,
        img_resolution,         # Image resolution.
        img_channels,           # Image channels.
        label_dim,              # Class label dimensionality. 0 = unconditional.
        use_fp16        = True, # Run the model at FP16 precision?
        sigma_data      = 0.5,  # Expected standard deviation of the training data.
        logvar_channels = 128,  # Intermediate dimensionality for uncertainty estimation.
        **unet_kwargs,          # Keyword arguments for UNet.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.unet = UNet(img_resolution=img_resolution, img_channels=img_channels, label_dim=label_dim, **unet_kwargs)
        self.logvar_fourier = MPFourier(logvar_channels)
        self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])

    def forward(self, x, sigma, class_labels=None, force_fp32=False, return_logvar=False, **unet_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        # Preconditioning weights.
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4

        # Run the model.
        x_in = (c_in * x).to(dtype)
        F_x = self.unet(x_in, c_noise, class_labels, **unet_kwargs)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        # Estimate uncertainty if requested.
        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(c_noise)).reshape(-1, 1, 1, 1)
            return D_x, logvar # u(sigma) in Equation 21
        return D_x
        
"""Below are model architectures proposed in the paper
"Noise Conditional Variational Score Distillation"."""

#----------------------------------------------------------------------------
# UNet encoder.

@persistence.persistent_class
class UNetEncoder(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Image channels.
        label_dim,                          # Class label dimensionality. 0 = unconditional.
        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_noise  = None,         # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb    = None,         # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [16,8],       # List of resolutions with self-attention.
        label_balance       = 0.5,          # Balance between noise embedding (0) and class embedding (1).
        concat_balance      = 0.5,          # Balance between skip connections (0) and main path (1).
        **block_kwargs,                     # Arguments for Block.
    ):
        super().__init__()
        cblock = [model_channels * x for x in channel_mult]
        cnoise = model_channels * channel_mult_noise if channel_mult_noise is not None else cblock[0]
        cemb = model_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)
        self.label_balance = label_balance
        self.concat_balance = concat_balance

        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])
        self.emb_label = MPConv(label_dim, cemb, kernel=[]) if label_dim != 0 else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = img_channels + 1
        for level, channels in enumerate(cblock):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_conv'] = MPConv(cin, cout, kernel=[3,3])
            else:
                self.enc[f'{res}x{res}_down'] = Block(cout, cout, cemb, flavor='enc', resample_mode='down', **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='enc', attention=(res in attn_resolutions), **block_kwargs)

    def forward(self, x, noise_labels, class_labels):
        # Embedding.
        emb = self.emb_noise(self.emb_fourier(noise_labels))
        if self.emb_label is not None:
            emb = mp_sum(emb, self.emb_label(class_labels * np.sqrt(class_labels.shape[1])), t=self.label_balance)
        emb = mp_silu(emb)

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            if 'conv' in name:
                x = block(x)
            else: 
                x = block(x, emb)
            skips.append(x)
        return x, skips
    
#----------------------------------------------------------------------------
# UNet decoder.

@persistence.persistent_class
class UNetDecoder(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Image channels.
        label_dim,                          # Class label dimensionality. 0 = unconditional.
        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_noise  = None,         # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb    = None,         # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [16,8],       # List of resolutions with self-attention.
        label_balance       = 0.5,          # Balance between noise embedding (0) and class embedding (1).
        concat_balance      = 0.5,          # Balance between skip connections (0) and main path (1).
        gradient_checkpoint = False,        # Use gradient checkpointing?
        has_controlnet      = False,        # If true, add learnable mp_sum weight.
        **block_kwargs,                     # Arguments for Block.
    ):
        super().__init__()
        cblock = [model_channels * x for x in channel_mult]
        cnoise = model_channels * channel_mult_noise if channel_mult_noise is not None else cblock[0]
        cemb = model_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))
        self.gradient_checkpoint = gradient_checkpoint

        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])
        self.emb_label = MPConv(label_dim, cemb, kernel=[]) if label_dim != 0 else None

        # Encoder.
        skips = []
        cout = img_channels + 1
        for level, channels in enumerate(cblock):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = channels
                skips.append(cout)
            else:
                skips.append(cout)
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                skips.append(cout)

        # Learnable mp_sum weight.
        if has_controlnet:
            self.additional_x_weight = torch.nn.Parameter(torch.zeros(1))
            self.additional_skips_weight = torch.nn.Parameter(torch.zeros(len(skips)))

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, channels in reversed(list(enumerate(cblock))):
            res = img_resolution >> level
            if level == len(cblock) - 1:
                self.dec[f'{res}x{res}_in0'] = Block(cout, cout, cemb, flavor='dec', attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = Block(cout, cout, cemb, flavor='dec', **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = Block(cout, cout, cemb, flavor='dec', resample_mode='up', **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='dec', attention=(res in attn_resolutions), **block_kwargs)
        self.out_conv = MPConv(cout, img_channels, kernel=[3,3])

    def forward(self, x, skips, noise_labels, class_labels, additional_x=None, additional_skips=None):
        # Embedding.
        emb = self.emb_noise(self.emb_fourier(noise_labels))
        if self.emb_label is not None:
            emb = mp_sum(emb, self.emb_label(class_labels * np.sqrt(class_labels.shape[1])), t=self.label_balance)
        emb = mp_silu(emb)

        # Controlnet.
        if additional_x is not None:
            w = self.additional_x_weight.to(torch.float32)
            if self.training:
                with torch.no_grad():
                    self.additional_x_weight.copy_(w.clip(0, 1)) # forced weight in [0, 1]
            w = self.additional_x_weight.clip(0, 1).to(x.dtype)
            x = mp_sum(x, additional_x, t=w)
        if additional_skips is not None:
            w = self.additional_skips_weight.to(torch.float32)
            if self.training:
                with torch.no_grad():
                    self.additional_skips_weight.copy_(w.clip(0, 1))
            w = self.additional_skips_weight.clip(0, 1).to(x.dtype)
            skips = [mp_sum(skips[i], additional_skips[i], t=w[i]) for i in range(len(skips))]

        # Decoder.
        for name, block in self.dec.items():
            if 'block' in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)
        x = self.out_conv(x, gain=self.out_gain)
        return x
    
#----------------------------------------------------------------------------
# Conditional UNet.

@persistence.persistent_class
class PrecondCondition(torch.nn.Module):
    def __init__(self,
        img_resolution,         # Image resolution.
        img_channels,           # Image channels.
        label_dim,              # Class label dimensionality. 0 = unconditional.
        use_fp16        = True, # Run the model at FP16 precision?
        sigma_data      = 0.5,  # Expected standard deviation of the training data.
        logvar_channels = 128,  # Intermediate dimensionality for uncertainty estimation.
        **unet_kwargs,          # Keyword arguments for UNet.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        network_kwargs = dict(img_resolution=img_resolution, img_channels=img_channels, label_dim=label_dim, **unet_kwargs)
        self.ctrl = UNetEncoder(**network_kwargs)
        self.enc = UNetEncoder(**network_kwargs)
        self.dec = UNetDecoder(**network_kwargs, has_controlnet=True)
        self.logvar_fourier = MPFourier(logvar_channels)
        self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])

    def forward(self, x, sigma, condition_x, condition_sigma, class_labels=None, force_fp32=False, return_logvar=False):
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        
        # Controlnet forward.
        condition_x = condition_x.to(torch.float32)
        condition_sigma = condition_sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        c_skip, c_out, c_in, c_noise = self.preconditioning(condition_sigma)
        x_in = (c_in * condition_x).to(dtype)
        ctrl_x, ctrl_skips = self.ctrl(x_in, c_noise, class_labels)

        # UNet encoder forward.
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        c_skip, c_out, c_in, c_noise = self.preconditioning(sigma)
        x_in = (c_in * x).to(dtype)
        enc_x, enc_skips = self.enc(x_in, c_noise, class_labels)

        # UNet decoder forward.
        F_x = self.dec(enc_x, enc_skips, c_noise, class_labels, additional_x=ctrl_x, additional_skips=ctrl_skips)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        # Estimate uncertainty if requested.
        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(c_noise)).reshape(-1, 1, 1, 1)
            return D_x, logvar # u(sigma) in Equation 21
        return D_x
    
    def preconditioning(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4
        return c_skip, c_out, c_in, c_noise
    
    def init_from_pretrained(self, net: Precond):
        unet = net.unet
        self.enc.load_state_dict(unet.state_dict(), strict=False)
        self.dec.load_state_dict(unet.state_dict(), strict=False)
        self.ctrl.load_state_dict(unet.state_dict(), strict=False)
        return self
    
#----------------------------------------------------------------------------
# Generative denoiser. The multi-step sampling interface (specifying `ts` in the `forward`` method) 
# is the same as the consistency model (https://github.com/openai/consistency_models/blob/e32b69ee436d518377db86fb2127a3972d0d8716/cm/karras_diffusion.py#L657).

class GenerativeDenoiser(torch.nn.Module):
    def __init__(self,
        model: PrecondCondition,
        gamma: float = 0.414,
        init_sigma: float = 80.0,
    ):
        super().__init__()
        self.model = model
        self.img_resolution = model.img_resolution
        self.img_channels = model.img_channels
        self.label_dim = model.label_dim
        self.init_sigma = init_sigma
        self.gamma = gamma
        
    def forward(
        self, 
        y, 
        sigma, 
        labels        = None, 
        return_logvar = False, 
        ts            = None,
        t_min         = 0.002,
        t_max         = 80.0,
        rho           = 7.0,
        steps         = 40
    ):
        if ts is None:
            return self._forward(y, sigma, labels, return_logvar=return_logvar) 
  
        assert return_logvar == False, 'return_logvar is not supported for multistep inference'
        
        t_max_rho = t_max ** (1 / rho)
        t_min_rho = t_min ** (1 / rho)
        
        xt = torch.randn_like(y) * (t_max_rho + ts[0] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        for i in range(len(ts) - 1):
            t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
            sigma_eff = 1 / (1 / sigma ** 2 + 1 / t ** 2) ** 0.5
            y_eff = (y / sigma ** 2 + xt / t ** 2) * sigma_eff ** 2
            x0 = self._forward(y_eff, sigma_eff, labels)
            next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
            xt = x0 + torch.randn_like(x0) * next_t
        return x0

    def _forward(self, y, sigma, labels, return_logvar=False):
        z = torch.randn_like(y)
        sigma_hat = sigma * (1 + self.gamma)
        y_hat = y + z * (sigma_hat ** 2 - sigma ** 2) ** 0.5
        return self.model(y_hat, sigma_hat, y, sigma, labels, return_logvar=return_logvar)

#----------------------------------------------------------------------------
# Discriminator.

@persistence.persistent_class
class DiscriminatorCondition(torch.nn.Module):
    def __init__(self,
        img_resolution,                 # Image resolution.
        img_channels,                   # Image channels.
        label_dim,                      # Class label dimensionality. 0 = unconditional.
        use_fp16                = True, # Run the model at FP16 precision?
        sigma_data              = 0.5,  # Expected standard deviation of the training data.
        **unet_kwargs,                  # Keyword arguments for UNet.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        network_kwargs = dict(img_resolution=img_resolution, img_channels=img_channels, label_dim=label_dim, **unet_kwargs)
        self.ctrl = UNetEncoder(**network_kwargs)
        self.enc = UNetEncoder(**network_kwargs)
        
        # Scalar output.
        x_in = torch.randn(1, img_channels, img_resolution, img_resolution)
        c_noise = torch.randn(1, 1, 1, 1).flatten().log() / 4
        class_labels = torch.eye(1, label_dim)
        enc_x, _ = self.enc(x_in, c_noise, class_labels)
        cout = enc_x.shape[1]
        self.out_conv = MPConv(cout*2, 1, kernel=[])

        # Disable inplace normalization.
        def disable_inplace_normalization(m):
            if isinstance(m, MPConv):
                m.use_gan = True
        self.apply(disable_inplace_normalization)

    def forward(self, x, sigma, condition_x, condition_sigma, class_labels=None, force_fp32=False):
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        
        # Controlnet forward.
        condition_x = condition_x.to(torch.float32)
        condition_sigma = condition_sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        c_skip, c_out, c_in, c_noise = self.preconditioning(condition_sigma)
        x_in = (c_in * condition_x).to(dtype)
        ctrl_x, ctrl_skips = self.ctrl(x_in, c_noise, class_labels)

        # UNet encoder forward.
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        c_skip, c_out, c_in, c_noise = self.preconditioning(sigma)
        x_in = (c_in * x).to(dtype)
        enc_x, enc_skips = self.enc(x_in, c_noise, class_labels)

        # Reduce to scalar.
        logits = F.adaptive_avg_pool2d(mp_cat(ctrl_x, enc_x), (1, 1)).flatten(1)     
        logits = self.out_conv(logits)
        return logits
    
    def preconditioning(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4
        return c_skip, c_out, c_in, c_noise
    
    def init_from_pretrained(self, net: Precond):
        unet = net.unet
        self.enc.load_state_dict(unet.state_dict(), strict=False)
        self.ctrl.load_state_dict(unet.state_dict(), strict=False)
        return self