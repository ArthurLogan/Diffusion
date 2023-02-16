import torch
from torch import nn
from torch.nn import functional as F

import numpy as np


def extract(v, t, shape):
    """extract coefficients at timesteps"""
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(shape) - 1))


class DiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        """gaussian diffusion trainer"""
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for reverse process
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1.0 - alphas_bar))

    def forward(self, x_0):
        """[B, C, H, W] -> [1] from image to noise"""
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction="none")
        return loss


class DiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, eta):
        """gaussian diffusion sampler"""
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1.0 / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1.0 - alphas) / torch.sqrt(1.0 - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        """predict x_{t-1} in reverse process"""
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )
    
    def p_mean_variance(self, x_t, t):
        """predict mean & variance for sampling"""
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps = self.model(x_t, t)
        mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return mean, var

    def forward(self, x_T):
        """[B, C, H, W] -> [B, C, H, W] from noise to image"""
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t, t)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
