from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .inner_model import InnerModel, InnerModelConfig


def add_dims(input: Tensor, n: int) -> Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))


@dataclass
class Conditioners:
    c_in: Tensor
    c_out: Tensor
    c_skip: Tensor
    c_noise: Tensor


@dataclass
class SigmaDistributionConfig:
    loc: float
    scale: float
    sigma_min: float
    sigma_max: float


@dataclass
class DenoiserConfig:
    inner_model: InnerModelConfig
    sigma_data: float
    sigma_offset_noise: float


class Denoiser(nn.Module):
    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.inner_model = InnerModel(cfg.inner_model)
        self.sample_sigma_training = None

    @property
    def device(self) -> torch.device:
        return self.inner_model.noise_emb.weight.device

    def setup_training(self, cfg: SigmaDistributionConfig) -> None:
        assert self.sample_sigma_training is None

        def sample_sigma(n: int, device: torch.device):
            s = torch.randn(n, device=device) * cfg.scale + cfg.loc
            return s.exp().clip(cfg.sigma_min, cfg.sigma_max)

        self.sample_sigma_training = sample_sigma
    
    def apply_noise(self, x: Tensor, sigma: Tensor, sigma_offset_noise: float) -> Tensor:
        b, d = x.shape 
        offset_noise = sigma_offset_noise * torch.randn(b, d, device=self.device)
        return x + offset_noise + torch.randn_like(x) * add_dims(sigma, x.ndim)

    def compute_conditioners(self, sigma: Tensor) -> Conditioners:
        sigma = (sigma**2 + self.cfg.sigma_offset_noise**2).sqrt()  # [B]
        c_in = 1 / (sigma**2 + self.cfg.sigma_data**2).sqrt()        # [B]
        c_skip = self.cfg.sigma_data**2 / (sigma**2 + self.cfg.sigma_data**2)  # [B]
        c_out = sigma * c_skip.sqrt()                                # [B]
        c_noise = sigma.log() / 4                                    # [B]
        
        # Reshape for broadcasting to [B, D] (state vector)
        c_in = c_in[:, None]
        c_out = c_out[:, None]
        c_skip = c_skip[:, None]
        c_noise = c_noise[:, None]  # for noise embedding input
        
        return Conditioners(c_in=c_in, c_out=c_out, c_skip=c_skip, c_noise=c_noise)

    def compute_model_output(self, noisy_next_obs: Tensor, obs: Tensor, act: Tensor, cs: Conditioners) -> Tensor:
        rescaled_obs = obs / self.cfg.sigma_data
        rescaled_noise = noisy_next_obs * cs.c_in
        return self.inner_model(rescaled_noise, cs.c_noise, rescaled_obs, act)
    
    @torch.no_grad()
    def wrap_model_output(self, noisy_next_obs: Tensor, model_output: Tensor, cs: Conditioners) -> Tensor:
        d = cs.c_skip * noisy_next_obs + cs.c_out * model_output
        # Quantize to {0, ..., 255}, then back to [-1, 1]
        # d = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
        return d
    
    @torch.no_grad()
    def denoise(self, noisy_next_obs: Tensor, sigma: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        
        cs = self.compute_conditioners(sigma)
        model_output = self.compute_model_output(noisy_next_obs, obs, act, cs)
        denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
        return denoised

    def forward(self, batch):


        obs = batch['obs']
        next_obs = batch['next_obs']
        act = batch['act']

        loss = 0


        b, d = obs.shape

        sigma = self.sample_sigma_training(b, self.device)
        noisy_next_obs = self.apply_noise(next_obs, sigma, self.cfg.sigma_offset_noise)

        cs = self.compute_conditioners(sigma)
        model_output = self.compute_model_output(noisy_next_obs, obs, act, cs)

        target = (next_obs - cs.c_skip * noisy_next_obs) / cs.c_out
        loss += F.mse_loss(model_output, target)

        return loss, {"loss_denoising": loss.detach()}
