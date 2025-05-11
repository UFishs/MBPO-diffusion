import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F

from mbrl.types import ModelInput
import mbrl.util.math

from .model import Model
from .util import EnsembleLinearLayer, truncated_normal_init
from .diffusion.diffusion_sampler import DiffusionSampler
from .diffusion.denoiser import Denoiser

class DiffusionBasedDynamics(Model):
    
    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        design: str,
        denoiser: omegaconf.DictConfig,
        diffusion_sampler: omegaconf.DictConfig,
        sigma_distribution: omegaconf.DictConfig,
        **kwargs,
    ):
        super().__init__(
            device=device,
        )
        self.denoiser = Denoiser(denoiser).to(self.device)
        self.diffusion_sampler = DiffusionSampler(self.denoiser, diffusion_sampler, out_size=out_size)
        self.in_size = in_size
        self.out_size = out_size

        print(f"DiffusionBasedDynamics: in_size={in_size}, out_size={out_size}, design={design}")

        self.denoiser.setup_training(sigma_distribution)
        self.obs_dim = out_size - int(denoiser.inner_model.learned_reward)
        self.act_dim = in_size - self.obs_dim


    def forward(self, x:torch.Tensor, rng:Optional[torch.Generator]=None):
        import ipdb; ipdb.set_trace()
        pass

    def loss(
        self,
        model_in: ModelInput,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        
        obs = model_in[:, :self.obs_dim]
        act = model_in[:, self.obs_dim:]
        next_obs = target
        batch = {
            'obs': obs,
            'act': act,
            'next_obs': next_obs,
        }
        
        loss, info = self.denoiser(batch)
        return loss, info

    def eval_score(  # type: ignore
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
       
        assert model_in.ndim == 2 and target is not None and target.ndim == 2
        with torch.no_grad():
            prev_obs = model_in[:, :self.obs_dim]
            prev_act = model_in[:, self.obs_dim:]
            pred_next_state, _ = self.diffusion_sampler.sample(prev_obs, prev_act)
            return F.mse_loss(pred_next_state, target, reduction="none"), {}

    def sample(
        self,
        act: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Dict[str, torch.Tensor]],
    ]:
        import ipdb; ipdb.set_trace()
        prev_obs = model_state
        prev_act = act
        pred_next_state, trajectory = self.diffusion_sampler.sample(prev_obs, prev_act)
        return pred_next_state, None, None, {"trajectory": trajectory}
        
        

    def reset(self):
        import ipdb; ipdb.set_trace()
        pass

    def reset_1d(
        self, obs: torch.Tensor, rng: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Initializes the model state for a new simulated trajectory.
        For diffusion models, we just keep the current obs in state.
        """
        return {
            "obs": obs,  # shape: [B, obs_dim]
        }

    def sample_1d(
        self,
        model_input: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Diffusion-based sampling of the next state, given current state and action.
        """
        # import ipdb; ipdb.set_trace()
        assert model_input.ndim == 2  # shape: [B, state_dim + action_dim]
        B = model_input.shape[0]

        # Split model_input back into state and action
        obs = model_input[:, :self.obs_dim]        # shape: [B, state_dim]
        act = model_input[:, self.obs_dim:]  # assume obs | act concat

        # Run diffusion sampling
        next_obs, _ = self.diffusion_sampler.sample(obs, act)

        # Return next observation and updated state (new obs)
        return next_obs, {"obs": next_obs}
