from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Conv3x3, FourierFeatures, GroupNorm, UNet


@dataclass
class InnerModelConfig:
    state_dim: int                 
    cond_channels: int
    hidden_dims: List[int]          
    num_actions: Optional[int] = None
    num_steps_conditioning: int = 1
    learned_reward: bool = True



class InnerModel(nn.Module):
    def __init__(self, cfg: InnerModelConfig) -> None:
        super().__init__()

        self.cfg = cfg

        self.noise_emb = FourierFeatures(cfg.cond_channels)
        self.act_emb = nn.Sequential(
            nn.Linear(cfg.num_actions, cfg.cond_channels // cfg.num_steps_conditioning),
            nn.Flatten(),
        )


        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )

        input_dim = cfg.state_dim * 2 + int(cfg.learned_reward) # obs + noisy_next_obs

        layers = []
        prev_dim = input_dim + cfg.cond_channels 
        for h in cfg.hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.SiLU())
            prev_dim = h

        layers.append(nn.Linear(prev_dim, cfg.state_dim + int(cfg.learned_reward)))


        self.mlp = nn.Sequential(*layers)



    def forward(self, noisy_next_obs: Tensor, c_noise: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        """
        Inputs:
            noisy_next_obs: [B, D]
            c_noise: [B, 1]
            obs: [B, D]
            act: [B]
        Output:
            denoised state: [B, D]
        """
        c_noise = c_noise.squeeze(1)
        # import ipdb; ipdb.set_trace()

        cond = self.cond_proj(self.noise_emb(c_noise) + self.act_emb(act))  # [B, cond_channels]
        x = torch.cat([noisy_next_obs, obs], dim=1)  # [B, D + int(r) + D]
        x = torch.cat([x, cond], dim=1)              # [B, 2*D + int(r) + cond_channels]
        return self.mlp(x)
    
