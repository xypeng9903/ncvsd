import torch
from . import register_operator, NonLinearOperator


@register_operator(name='high_dynamic_range')
class HighDynamicRange(NonLinearOperator):
    def __init__(self, device='cuda', scale=2, sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.scale = scale

    def __call__(self, data):
        return torch.clip((data * self.scale), -1, 1)