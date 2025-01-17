import torch
from . import register_operator, NonLinearOperator
import yaml
import numpy as np


@register_operator(name='nonlinear_blur')
class NonlinearBlur(NonLinearOperator):
    def __init__(self, opt_yml_path, device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)
        self.blur_model.requires_grad_(False)

        np.random.seed(0)
        kernel_np = np.random.randn(1, 512, 2, 2) * 1.2
        random_kernel = (torch.from_numpy(kernel_np)).float().to(self.device)
        self.random_kernel = random_kernel

    def prepare_nonlinear_blur_model(self, opt_yml_path):
        from .bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path))
        blur_model = blur_model.to(self.device)
        self.random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        return blur_model

    def call_old(self, data):
        # random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        blurred = []
        for i in range(data.shape[0]):
            single_blurred = self.blur_model.adaptKernel(data[i:i + 1], kernel=self.random_kernel)
            blurred.append(single_blurred)
        blurred = torch.cat(blurred, dim=0)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]
        return blurred

    def __call__(self, data):
        data = (data + 1.0) / 2.0  # [-1, 1] -> [0, 1]

        random_kernel = self.random_kernel.repeat(data.shape[0], 1, 1, 1)
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]

        # blurred = []
        # for i in range(data.shape[0]):
        #     single_blurred = self.blur_model.adaptKernel(data[i:i + 1], kernel=self.random_kernel)
        #     blurred.append(single_blurred)
        # blurred = torch.cat(blurred, dim=0)
        # blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]
        return blurred