# Diffusion Model-based RL: Diffusion Dynamic Model

This repository implements MBPO with diffusion models as the dynamics model. The code is based on the original MBPO implementation in [mbrl-lib](https://github.com/facebookresearch/mbrl-lib), and the diffusion model code is based on [diamond](https://github.com/eloialonso/diamond). 



# Installation

The dependencies are almost same as the original MBPO implementation. You can refer to the original [README](https://github.com/facebookresearch/mbrl-lib/blob/main/README.md) for the installation instructions. And then you can add some additional dependencies for diffusion models when running the code. 

Or we also provide a conda environment file for you to install all the dependencies. You can create a conda environment with the following command:

```bash
conda env create -f environment.yml
```

Then you can activate the environment with the following command:

```bash
conda activate mbrl
```

# Training

For original MBPO, you can use the following command to train the model:

```bash
CUDA_VISIBLE_DEVICES=0 python -m mbrl.examples.main algorithm=mbpo overrides=mbpo_ant
```

The usage of the command is the same as the original MBPO implementation.

---

For diffusion model-based MBPO, you can use the following command to train the model:

```bash
# Ant
CUDA_VISIBLE_DEVICES=0 python -m mbrl.examples.main dynamics_model=diffusion seed=0 dynamics_model.diffusion_sampler.num_steps_denoising=3

# Hopper
CUDA_VISIBLE_DEVICES=0 python -m mbrl.examples.main dynamics_model=diffusion overrides=mbpo_hopper dynamics_model.denoiser.inner_model.num_actions=3 dynamics_model.denoiser.inner_model.state_dim=11 seed=0 dynamics_model.diffusion_sampler.num_steps_denoising=3
```

If you want to use other environments, you need to modify the `dynamics_model.denoiser.inner_model.num_actions` and `dynamics_model.denoiser.inner_model.state_dim` parameters in the command.

# Contributors
+ Zihang Rui



