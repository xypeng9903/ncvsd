import torch
from . import register_operator, LinearSVDOperator
from PIL import Image
from torchvision.transforms import ToTensor


@register_operator(name='inpainting')
class Inpainting(LinearSVDOperator):
    def __init__(self, mask_path: str, device): #ratio = 2 or 4
        mask = Image.open(mask_path)
        self.mask = ToTensor()(mask).to(device)
        self.mask_path = mask_path

    @property
    def display_name(self):
        return f'inpainting-{self.mask_path}'

    def V(self, vec):
        return vec.flatten(1)

    def Vt(self, vec):
        return vec.flatten(1)

    def U(self, vec):
        c, h, w = self.mask.shape
        return vec.reshape(-1, c, h, w)

    def Ut(self, vec):
        return vec.flatten(1)

    def singulars(self):
        return self.mask.flatten(0)
    
    def add_zeros(self, vec):
        return vec

    
