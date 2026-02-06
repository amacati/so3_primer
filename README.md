# A Primer on SO(3) Action Representations in Deep RL
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![GitHub Pages](https://img.shields.io/badge/github_page-website-blue?logo=github)](https://amacati.github.io/so3_primer/)
[![arXiv](https://img.shields.io/badge/arXiv-2510.11103-b31b1b.svg)](https://arxiv.org/abs/2510.11103)


This repository contains the source code for **"A Primer on SO(3) Action Representations in Deep Reinforcement Learning"**.

## Running the Experiments

### Setup

Install dependencies using Pixi:
```bash
pixi install
```

### WandB Configuration

Hyperparameter sweeps require WandB. Place your API key at `secrets/wandb_api_key.secret` (relative to the repository root). This file is automatically excluded from version control.

### Idealized Rotation Tasks

The experiments for the idealized rotation tasks are located in the `experiments/` folder. To run a specific algorithm, use the following command:

```bash
pixi run python experiments/<algorithm>/<algorithm>.py -a <action_representation> -c <abs|rel>
```

Experiments automatically use the configuration files located in `experiments/<algorithm>/config/` to load hyperparameters.Available algorithms: `ppo`, `sac`, `td3`. To run sweeps, use the `sweep.py` scripts in the respective algorithm folders.

### Robotics Benchmarks

The robotics benchmark experiments are located in the `benchmarks/` folder. They include:
- `crazyflow`: A trajectory following task (figure 8) using crazyflie drones.
- `droneracing`: Drone racing similar to the IROS 2022 Safe Robot Learning Competition using crazyflie drones.
- `her_orient`: Fetch-like robotic arm tasks, one for orienting the end-effector into a target position and orientation, and one for picking up a block and placing it onto a target position and orientation. Solved using HER.
- `robosuite`: Our adapted version of the robotic manipulation tasks from the [RoboSuite benchmark](https://robosuite.ai/) that enables swapping out the action representation for the end-effector orientation.

We provide pixi environments for all benchmarks (see [pyproject.toml](pyproject.toml)). All benchmarks come with configuration files that specify the hyperparameters.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{
    schuck2026primer,
    title={A Primer on {SO(3)} Action Representations in Deep Reinforcement Learning},
    author={Martin Schuck and Sherif Samy and Angela P. Schoellig},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
}
```