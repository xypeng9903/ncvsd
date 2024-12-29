import torch
import tqdm
import numpy as np

#----------------------------------------------------------------------------
# Langevin learning rate schedule from DAPS.

def get_lr(ratio, lr, lr_min_ratio):
    p = 1
    multiplier = (1 ** (1 / p) + ratio * (lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
    return multiplier * lr

#----------------------------------------------------------------------------
# Langevin dynamics.

def lgvd(
    x0hat, 
    operator, 
    measurement, 
    tau,
    sigma, 
    lr,
    num_steps,
    verbose     = False
):
    pbar = tqdm.trange(num_steps) if verbose else range(num_steps)
    x = x0hat.clone().detach().requires_grad_(True)
    optimizer = torch.optim.SGD([x], lr)
    for _ in pbar:
        optimizer.zero_grad()
        loss = operator.error(x, measurement).sum() / (2 * tau ** 2)
        loss += ((x - x0hat.detach()) ** 2).sum() / (2 * sigma ** 2)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            epsilon = torch.randn_like(x)
            x.data = x.data + np.sqrt(2 * lr) * epsilon
        if torch.isnan(x).any():
            return torch.zeros_like(x)
    return x.detach()