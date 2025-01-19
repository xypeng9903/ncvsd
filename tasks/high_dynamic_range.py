import torch
from . import register_operator, NonLinearOperator


@register_operator(name='high_dynamic_range')
class HighDynamicRange(NonLinearOperator):
    def __init__(self, scale, device):
        super().__init__()
        self.scale = scale

    def forward(self, data):
        return torch.clip((data * self.scale), -1, 1)