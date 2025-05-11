from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor

from .denoiser import Denoiser


@dataclass
class DiffusionSamplerConfig:
    num_steps_denoising: int
    sigma_min: float = 2e-3
    sigma_max: float = 5
    rho: int = 7
    order: int = 1
    s_churn: float = 0
    s_tmin: float = 0
    s_tmax: float = float("inf")
    s_noise: float = 1


class DiffusionSampler:
    def __init__(self, denoiser: Denoiser, cfg: DiffusionSamplerConfig, out_size: int) -> None:
        self.denoiser = denoiser
        self.cfg = cfg
        self.sigmas = build_sigmas(cfg.num_steps_denoising, cfg.sigma_min, cfg.sigma_max, cfg.rho, denoiser.device)
        self.out_size = out_size

    @torch.no_grad()
    def sample(self, prev_obs: Tensor, prev_act: Tensor) -> Tuple[Tensor, List[Tensor]]:
        device = prev_obs.device


        b, d = prev_obs.shape
        s_in = torch.ones(b, device=device)

        # Initialize x ~ N(0, 1)
        x = torch.randn(b, self.out_size, device=device)
        trajectory = [x]

        gamma_ = min(self.cfg.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
        # import ipdb; ipdb.set_trace()

        for sigma, next_sigma in zip(self.sigmas[:-1], self.sigmas[1:]):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sigma_hat = sigma * (1 + gamma)

            if gamma > 0:
                eps = torch.randn_like(x) * self.cfg.s_noise
                x = x + eps * ((sigma_hat**2 - sigma**2).sqrt())

            # Denoise step
            sigma_hat_batched = sigma_hat.expand(x.shape[0])
            denoised = self.denoiser.denoise(x, sigma_hat_batched, prev_obs, prev_act)
            d = (x - denoised) / sigma_hat
            dt = next_sigma - sigma_hat

            if self.cfg.order == 1 or next_sigma == 0:
                x = x + d * dt  # Euler
            else:
                # Heun's method
                x_2 = x + d * dt
                next_sigma_batched = next_sigma.expand(x_2.shape[0])
                denoised_2 = self.denoiser.denoise(x_2, next_sigma_batched * s_in, prev_obs, prev_act)
                d_2 = (x_2 - denoised_2) / next_sigma
                x = x + (d + d_2) / 2 * dt

            trajectory.append(x)

        return x, trajectory


def build_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: int, device: torch.device) -> Tensor:
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    l = torch.linspace(0, 1, num_steps, device=device)
    sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1)))

