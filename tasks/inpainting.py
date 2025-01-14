import torch
from . import register_operator, LinearOperator
from PIL import Image
from torchvision.transforms import ToTensor


@register_operator(name='inpainting')
class Inpainting(LinearOperator):
    def __init__(self, mask_path: str, device): #ratio = 2 or 4
        mask = Image.open(mask_path)
        self.mask = ToTensor()(mask).to(device)
        self.mask_path = mask_path

    @property
    def display_name(self):
        return f'inpainting-{self.mask_path}'

    def forward(self, x, **kwargs):
        mask = self.mask.view(1, *self.mask.shape)
        return x * mask
    
    def transpose(self, x, **kwargs):
        return self.forward(x, **kwargs)
    
    def proximal_generator(self, x, y, sigma, rho):
        mask = self.mask.view(1, *self.mask.shape)
        mu = (y / sigma ** 2 * mask + x / rho ** 2) / (mask / sigma ** 2 + 1 / rho ** 2)
        std = 1 / (mask / sigma ** 2 + 1 / rho ** 2) ** 0.5
        return mu + std * torch.randn_like(mu)
